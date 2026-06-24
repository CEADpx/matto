"""
Original FEniTop authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7

Major modifications:
- Ian Galloway (ian.galloway@mines.sdsmt.edu)
- Prashant Jha (prashant.jha@sdsmt.edu)

Major additions to sensitivity.py:
- Adjoint sensitivities for phi and theta design variables
- Multi-load-case sensitivity support
- Additional objective and constraint sensitivities
- Constraint evaluation framework for optimization
"""

import ufl
from mpi4py import MPI
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import (create_vector, create_matrix, assemble_vector, assemble_matrix, set_bc)
from petsc4py import PETSc
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.linalg import solve
import numpy as np

class Sensitivity():
    def __init__(self, comm, opt, problem, u_field, lambda_field,
                rho_phys, phi_phys, theta_phys):

        self.comm = comm
        self.opt = opt

        # ============================================================
        # Design-variable toggles (used to zero inactive gradients)
        # ============================================================
        dv_cfg = opt.get("design_variables", {})
        self.rho_active = dv_cfg.get("rho", {}).get("active", True)
        self.phi_active = dv_cfg.get("phi", {}).get("active", True)
        # theta placeholder for future extension
        self.theta_active = dv_cfg.get("theta", {}).get("active", False)


        # Volume
        self.total_volume = comm.allreduce(assemble_scalar(form(opt["total_volume"])), op=MPI.SUM)
        
        self.V_rho_form = form(opt["volume_rho"])
        dVdrho_form = form(ufl.derivative(opt["volume_rho"], rho_phys))
        self.dVdrho_vec = create_vector(dVdrho_form)
        assemble_vector(self.dVdrho_vec, dVdrho_form)
        self.dVdrho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.dVdrho_vec /= self.total_volume

        # φ-volume constraint uses φ_phys and ρ_phys (φ-volume is φ_phys * ρ_phys)
        self.V_phi_form = form(opt["volume_phi"])
     
        # Store derivative forms; actual vectors assembled each evaluate()
        self.dVdphi_form = form(ufl.derivative(opt["volume_phi"], phi_phys))
        self.dVdphi_vec = create_vector(self.dVdphi_form)

        # ============================================================
        # OBJECTIVE FORMS (generic)
        # ============================================================

        # Objective value form (ex: compliance)
        self.obj_form = form(opt["objective_form"])

        # Direct objective derivatives wrt design fields
        self.dObj_drho_form = form(opt["dObj_drho_form"])
        self.dObj_drho_vec = create_vector(self.dObj_drho_form)

        self.dObj_dphi_form = form(opt["dObj_dphi_form"])
        self.dObj_dphi_vec = create_vector(self.dObj_dphi_form)

        # Direct derivative wrt theta
        self.dObj_dtheta_form = form(opt["dObj_dtheta_form"])
        self.dObj_dtheta_vec  = create_vector(self.dObj_dtheta_form)

        # Direct derivative wrt displacement (adjoint RHS)
        self.dObj_du_form = form(opt["dObj_du_form"])
        self.dObj_du_vec  = create_vector(self.dObj_du_form)

        # Internal force Jacobians wrt u, rho, phi
        self.dfdu_form   = form(ufl.derivative(opt["f_int"], u_field))
        self.dfdu_mat    = create_matrix(self.dfdu_form)

        self.dfdrho_form = form(ufl.derivative(opt["f_int"], rho_phys))
        self.dfdrho_mat  = create_matrix(self.dfdrho_form)

        self.dfdphi_form = form(ufl.derivative(opt["f_int"], phi_phys))
        self.dfdphi_mat  = create_matrix(self.dfdphi_form)

        # Internal force Jacobian wrt theta
        self.dfdtheta_form = form(ufl.derivative(opt["f_int"], theta_phys))
        self.dfdtheta_mat  = create_matrix(self.dfdtheta_form)

        self.dAdjointTerm_rho_vec = rho_phys.x.petsc_vec.copy() 
        self.dAdjointTerm_phi_vec = phi_phys.x.petsc_vec.copy() 
        self.dAdjointTerm_theta_vec = theta_phys.x.petsc_vec.copy()

        self.rho_phys = rho_phys        # filtered & projected density
        self.phi_phys = phi_phys        # filtered φ
        self.theta_phys = theta_phys    # filtered theta

        self.problem = problem
        self.lambda_field = lambda_field


    def _total_derivative_vectors(self, dg_du_form, dg_drho_form, dg_dphi_form, dg_dtheta_form):
        """
        TOTAL derivatives of g(u, rho, phi, theta) via adjoint.
        """

        # Assemble dg/du
        dg_du_vec = create_vector(dg_du_form)
        dg_du_vec.zeroEntries()
        assemble_vector(dg_du_vec, dg_du_form)
        set_bc(dg_du_vec, self.problem.bcs)
        dg_du_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dg_du_vec.scale(-1.0)

        # Solve adjoint
        lambda_g = self.lambda_field.x.petsc_vec.copy()
        lambda_g.set(0.0)

        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(self.dfdu_mat)
        ksp.setTolerances(rtol=1e-8, atol=1e-12)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solveTranspose(dg_du_vec, lambda_g)

        # Direct terms
        def assemble_direct(form_):
            v = create_vector(form_)
            v.zeroEntries()
            assemble_vector(v, form_)
            v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            return v

        dgdrho_vec   = assemble_direct(dg_drho_form)
        dgdphi_vec   = assemble_direct(dg_dphi_form)
        dgdtheta_vec = assemble_direct(dg_dtheta_form)

        # Adjoint terms
        for mat, vec in [
            (self.dfdrho_mat, dgdrho_vec),
            (self.dfdphi_mat, dgdphi_vec),
            (self.dfdtheta_mat, dgdtheta_vec),
        ]:
            tmp = vec.copy()
            tmp.zeroEntries()
            mat.multTranspose(lambda_g, tmp)
            tmp.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            vec.axpy(1.0, tmp)

        return dgdrho_vec, dgdphi_vec, dgdtheta_vec

    
    def _zero_vec_(self, vec: PETSc.Vec) -> PETSc.Vec:
        """Return a zeroed copy of a PETSc Vec (safe for inactive design variables)."""
        out = vec.copy()
        out.zeroEntries()
        out.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)
        return out

    def evaluate(self):
        # Volume
        actual_rho_volume = self.comm.allreduce(assemble_scalar(self.V_rho_form), op=MPI.SUM)
        V_rho_value = actual_rho_volume / self.total_volume
        self.dVdrho_vec_copy = self.dVdrho_vec.copy()

        actual_phi_volume = self.comm.allreduce(assemble_scalar(self.V_phi_form), op=MPI.SUM)
        V_phi_value = actual_phi_volume / self.total_volume

        # Re-assemble φ-volume sensitivities each iteration
        # dV/dphi_phys
        self.dVdphi_vec.zeroEntries()
        assemble_vector(self.dVdphi_vec, self.dVdphi_form)
        self.dVdphi_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE
        )
        self.dVdphi_vec_copy = self.dVdphi_vec.copy()
        self.dVdphi_vec_copy /= self.total_volume

        # Objective value (generic)
        Obj_value = self.comm.allreduce(assemble_scalar(self.obj_form), op=MPI.SUM)

        # Direct derivative wrt rho
        with self.dObj_drho_vec.localForm() as loc:
            loc.set(0)
        assemble_vector(self.dObj_drho_vec, self.dObj_drho_form)
        self.dObj_drho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                    mode=PETSc.ScatterMode.REVERSE)

        # Direct derivative wrt phi
        with self.dObj_dphi_vec.localForm() as loc:
            loc.set(0)
        assemble_vector(self.dObj_dphi_vec, self.dObj_dphi_form)
        self.dObj_dphi_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                    mode=PETSc.ScatterMode.REVERSE)
        
        # Direct derivative wrt theta
        with self.dObj_dtheta_vec.localForm() as loc:
            loc.set(0)
        assemble_vector(self.dObj_dtheta_vec, self.dObj_dtheta_form)
        self.dObj_dtheta_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE
        )

        # Direct derivative wrt displacement (adjoint RHS)
        self.dObj_du_vec.zeroEntries()
        assemble_vector(self.dObj_du_vec, self.dObj_du_form)
        set_bc(self.dObj_du_vec, self.problem.bcs)
        self.dObj_du_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                    mode=PETSc.ScatterMode.REVERSE)
        self.dObj_du_vec.scale(-1.0)

        self.dfdu_mat.zeroEntries()
        assemble_matrix(self.dfdu_mat, self.dfdu_form, bcs=self.problem.bcs)
        self.dfdu_mat.assemble()

        self.dfdrho_mat.zeroEntries()
        assemble_matrix(self.dfdrho_mat, self.dfdrho_form)
        self.dfdrho_mat.assemble()  
        
        self.dfdphi_mat.zeroEntries()
        assemble_matrix(self.dfdphi_mat, self.dfdphi_form)
        self.dfdphi_mat.assemble()  

        self.dfdtheta_mat.zeroEntries()
        assemble_matrix(self.dfdtheta_mat, self.dfdtheta_form)
        self.dfdtheta_mat.assemble()
     
        # Solve (K)^T * lambda = (dObj/du)^T
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(self.dfdu_mat)
        ksp.setTolerances(rtol=1e-8, atol=1e-12)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        self.lambda_field.x.petsc_vec.set(0.0) 
        ksp.solveTranspose(self.dObj_du_vec, self.lambda_field.x.petsc_vec)
        self.lambda_field.x.scatter_forward()
       
        # Adjoint contributions for rho
        self.dAdjointTerm_rho_vec.zeroEntries()
        self.dfdrho_mat.multTranspose(self.lambda_field.x.petsc_vec, self.dAdjointTerm_rho_vec)
        self.dAdjointTerm_rho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                              mode=PETSc.ScatterMode.REVERSE)
        self.dObj_drho_vec.axpy(1.0, self.dAdjointTerm_rho_vec)


        # Adjoint contributions for phi_eff
        self.dAdjointTerm_phi_vec.zeroEntries()
        self.dfdphi_mat.multTranspose(self.lambda_field.x.petsc_vec, self.dAdjointTerm_phi_vec)
        self.dAdjointTerm_phi_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                              mode=PETSc.ScatterMode.REVERSE)
        self.dObj_dphi_vec.axpy(1.0, self.dAdjointTerm_phi_vec)

        # Adjoint contributions for theta
        self.dAdjointTerm_theta_vec.zeroEntries()
        self.dfdtheta_mat.multTranspose(
            self.lambda_field.x.petsc_vec,
            self.dAdjointTerm_theta_vec
        )
        self.dAdjointTerm_theta_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE
        )
        self.dObj_dtheta_vec.axpy(1.0, self.dAdjointTerm_theta_vec)

        # ============================================================
        # CONSTRAINTS (per-evaluate call = per current load case)
        #
        # We return a dict so topopt can assemble per-load-case
        # constraint vectors cleanly:
        #   constraints["stress"] -> {"g": scalar, "dg": (vec_rho, vec_phi)}
        #   constraints["strain"] -> {"g": scalar, "dg": (vec_rho, vec_phi)}
        #
        # If a constraint is disabled, the entry remains None.
        # ============================================================

        constraints = {
            "stress": None,
            "strain": None,
            "compliance": None,
            "disp": None
        }


        # ----------------------------
        # STRESS CONSTRAINT (p-norm VM)
        # ----------------------------
        if self.opt.get("stress_constraint", False):

            # Assemble the p-power integral S = ∫ (w * sigma_vm)^p dx
            stress_power_form = form(self.opt["stress_power_form"])
            S_value = self.comm.allreduce(assemble_scalar(stress_power_form), op=MPI.SUM)

            # Load parameters
            pnorm     = self.opt.get("stress_pnorm", 12)
            sigma_max = self.opt.get("sigma_max", 1.0)

            # Compute p-norm G = S^(1/p)
            G_value = S_value**(1.0 / pnorm)

            # Normalized constraint g = G/sigma_max - 1
            g_stress = G_value / sigma_max - 1.0

            # Total derivatives via adjoint:
            # We already stored dStress_du_form in opt in fem.py as "dStress_du_form"
            # and dStress_drho_form / dStress_dphi_form as direct partials of the p-power integral S.
            dStress_du_form   = form(self.opt["dStress_du_form"])
            dStress_drho_form = form(self.opt["dStress_drho_form"])
            dStress_dphi_form = form(self.opt["dStress_dphi_form"])

            dSdrho_vec, dSdphi_vec, dSdtheta_vec = self._total_derivative_vectors(
                dStress_du_form,
                dStress_drho_form,
                dStress_dphi_form,
                form(self.opt["dStress_dtheta_form"])
            )

            # Chain rule from S -> G/sigma_max:
            # d(G/sigma_max)/dx = (1/sigma_max) * (1/p) * S^(1/p - 1) * dS/dx
            if S_value <= 0.0:
                scale = 0.0
            else:
                scale = (1.0 / sigma_max) * (1.0 / pnorm) * (S_value**((1.0 / pnorm) - 1.0))

            dSdrho_vec.scale(scale)
            dSdphi_vec.scale(scale)
            dSdtheta_vec.scale(scale)

            dGdrho_vec   = dSdrho_vec
            dGdphi_vec   = dSdphi_vec
            dGdtheta_vec = dSdtheta_vec

            constraints["stress"] = {
                "g": g_stress,
                "dg": (dGdrho_vec, dGdphi_vec, dGdtheta_vec)
            }

        # ----------------------------
        # COMPLIANCE CONSTRAINT (raw C; normalization handled in topopt.py)
        # ----------------------------
        if self.opt.get("compliance_constraint", False):

            # Assemble compliance C = ∫ u·b dx + Σ ∫ u·t ds
            C_form = form(self.opt["compliance_form"])
            C_value = self.comm.allreduce(assemble_scalar(C_form), op=MPI.SUM)

            # Total derivatives via adjoint
            dC_du_form   = form(self.opt["dC_du_form"])
            dC_drho_form = form(self.opt["dC_drho_form"])
            dC_dphi_form = form(self.opt["dC_dphi_form"])

            dC_dtheta_form = form(self.opt["dC_dtheta_form"])

            dCdrho_vec, dCdphi_vec, dCdtheta_vec = self._total_derivative_vectors(
                dC_du_form,
                dC_drho_form,
                dC_dphi_form,
                dC_dtheta_form
            )

            constraints["compliance"] = {
                "g": C_value,               # raw compliance scalar
                "dg": (dCdrho_vec, dCdphi_vec, dCdtheta_vec)
            }


        # ----------------------------
        # STRAIN-ENERGY CONSTRAINT
        # ----------------------------
        if self.opt.get("strain_constraint", False):

            # Assemble global strain energy U = ∫ W_elastic dx
            U_form = form(self.opt["strain_energy_form"])
            U_value = self.comm.allreduce(assemble_scalar(U_form), op=MPI.SUM)

            # Load user parameters
            U_max = self.opt.get("U_max_active", self.opt.get("U_max", 1.0))

            # Constraint: g = U/U_max - 1
            g_strain = U_value / U_max - 1.0

            # Total derivatives via adjoint:
            dU_du_form   = form(self.opt["dU_du_form"])
            dU_drho_form = form(self.opt["dU_drho_form"])
            dU_dphi_form = form(self.opt["dU_dphi_form"])

            dU_dtheta_form = form(self.opt["dU_dtheta_form"])

            dUdrho_vec, dUdphi_vec, dUdtheta_vec = self._total_derivative_vectors(
                dU_du_form,
                dU_drho_form,
                dU_dphi_form,
                dU_dtheta_form
            )

            # g = U/U_max - 1  =>  dg/dx = (1/U_max) dU/dx
            dUdrho_vec.scale(1.0 / U_max)
            dUdphi_vec.scale(1.0 / U_max)
            dUdtheta_vec.scale(1.0 / U_max)

            constraints["strain"] = {
                "g": g_strain,
                "dg": (dUdrho_vec, dUdphi_vec, dUdtheta_vec)
            }

        # ----------------------------
        # TIP DISPLACEMENT CONSTRAINT (up-only)
        #
        # Enforce: u_tip >= u_min
        #
        # Use the better-posed ratio form:
        #   g = 1 - (u_tip / u_min) <= 0
        #
        # Notes:
        # - u_tip is the AVERAGE vertical displacement over the tip boundary
        # - The raw boundary integral form is stored in fem.py as opt["u_tip_form"]
        # - The boundary measure |Γ_tip| is provided by topopt.py as opt["tip_measure"]
        # - u_min is provided by topopt.py as opt["u_min_active"] (or base opt["u_min"])
        # ----------------------------
        if self.opt.get("disp_constraint", False):

            # Assemble raw integral: ∫ u_y ds
            u_tip_form = form(self.opt["u_tip_form"])
            u_tip_int = self.comm.allreduce(assemble_scalar(u_tip_form), op=MPI.SUM)

            # Boundary measure |Γ_tip| (must be set in topopt.py)
            tip_measure = float(self.opt.get("tip_measure", 0.0))
            if tip_measure <= 0.0:
                raise RuntimeError("tip_measure must be set in topopt.py and be > 0 for disp_constraint")

            # Average tip displacement (up is positive)
            u_tip_value = u_tip_int / tip_measure

            # Active target (ramping handled in topopt.py)
            u_min = self.opt.get("u_min_active", self.opt.get("u_min", 1.0))
            u_min = float(u_min)
            if u_min <= 0.0:
                raise RuntimeError("u_min (or u_min_active) must be > 0 for disp_constraint")

            # Constraint: g = 1 - u_tip/u_min
            g_disp = 1.0 - (u_tip_value / u_min)

            # Total derivatives via adjoint:
            # u_tip_value = (1/|Γ|) * ∫ u_y ds
            dUtip_du_form     = form(self.opt["dUtip_du_form"])
            dUtip_drho_form   = form(self.opt["dUtip_drho_form"])
            dUtip_dphi_form   = form(self.opt["dUtip_dphi_form"])
            dUtip_dtheta_form = form(self.opt["dUtip_dtheta_form"])

            dUtipdrho_vec, dUtipdphi_vec, dUtipdtheta_vec = self._total_derivative_vectors(
                dUtip_du_form,
                dUtip_drho_form,
                dUtip_dphi_form,
                dUtip_dtheta_form
            )

            # u_tip_value = u_tip_int / tip_measure
            # => d(u_tip_value)/dx = (1/tip_measure) * d(u_tip_int)/dx
            dUtipdrho_vec.scale(1.0 / tip_measure)
            dUtipdphi_vec.scale(1.0 / tip_measure)
            dUtipdtheta_vec.scale(1.0 / tip_measure)

            # g = 1 - u_tip_value/u_min  =>  dg/dx = -(1/u_min) * d(u_tip_value)/dx
            dUtipdrho_vec.scale(-1.0 / u_min)
            dUtipdphi_vec.scale(-1.0 / u_min)
            dUtipdtheta_vec.scale(-1.0 / u_min)

            constraints["disp"] = {
                "g": g_disp,
                "dg": (dUtipdrho_vec, dUtipdphi_vec, dUtipdtheta_vec),
                "u_tip": u_tip_value
            }


        # ============================================================
        # RETURN VALUES (backward compatible + future-proof dict grads)
        # ============================================================

        func_values = [Obj_value, V_rho_value, V_phi_value]

        # If a design variable is inactive, return zero gradients for it.
        # This keeps topopt logic clean and makes toggles "real" in the sensitivities layer.
        dObj_drho_out = self.dObj_drho_vec if self.rho_active else self._zero_vec_(self.dObj_drho_vec)
        dObj_dphi_out = self.dObj_dphi_vec if self.phi_active else self._zero_vec_(self.dObj_dphi_vec)

        dVdrho_out = self.dVdrho_vec_copy if self.rho_active else self._zero_vec_(self.dVdrho_vec_copy)
        dVdphi_out = self.dVdphi_vec_copy if self.phi_active else self._zero_vec_(self.dVdphi_vec_copy)

        # Backward-compatible list output (what topopt.py currently expects)
        sensitivities = [
            dObj_drho_out,
            dObj_dphi_out,
            dVdrho_out,
            dVdphi_out
        ]

        # New dict-style gradients (what we will switch topopt.py to later)
        grads = {
            "objective": {
                "rho": dObj_drho_out,
                "phi": dObj_dphi_out,
                "theta": self.dObj_dtheta_vec if self.theta_active else self._zero_vec_(self.dObj_dtheta_vec),
            },
            "volume": {
                "rho": dVdrho_out,
                "phi": dVdphi_out,
                "theta": None,  # placeholder
            }
        }

        # Return constraints as a dict so topopt can assemble per-load-case constraints
        # NOTE: We append grads as a 4th return item (backward-compatible with current topopt indexing).
        return func_values, sensitivities, constraints, grads

