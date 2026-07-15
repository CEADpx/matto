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
- Prashant K. Jha (pjha.sci@gmail.com)

Major additions to sensitivity.py:
- Adjoint sensitivities for phi and theta design variables
- Generic objective sensitivity evaluation
- Active design-variable handling
"""

import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import assemble_scalar, form
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
    set_bc,
)

class Sensitivity:
    def __init__(self, comm, opt, problem, u_field, lambda_field,
                rho_phys, phi_phys, theta_phys):

        self.comm = comm

        # ============================================================
        # Design-variable toggles (used to zero inactive gradients)
        # ============================================================
        dv_cfg = opt.get("design_variables", {})
        self.rho_active = dv_cfg.get("rho", {}).get("active", True)
        self.phi_active = dv_cfg.get("phi", {}).get("active", True)
        self.theta_active = dv_cfg.get("theta", {}).get("active", False)


        # Volume
        self.total_volume = comm.allreduce(assemble_scalar(form(opt["total_volume"])), op=MPI.SUM)
        
        self.V_rho_form = form(opt["volume_rho"])
        dVdrho_form = form(ufl.derivative(opt["volume_rho"], rho_phys))
        self.dVdrho_vec = create_vector(dVdrho_form)
        assemble_vector(self.dVdrho_vec, dVdrho_form)
        self.dVdrho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.dVdrho_vec /= self.total_volume

        # Particle-volume constraint uses phi_phys directly
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

        self.problem = problem
        self.lambda_field = lambda_field
    
    def _zero_vector(self, vector: PETSc.Vec) -> PETSc.Vec:
        """Return a zeroed copy for an inactive design variable."""
        result = vector.copy()
        result.zeroEntries()
        result.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD,
        )
        return result

    def evaluate(self):
        # Volume
        actual_rho_volume = self.comm.allreduce(assemble_scalar(self.V_rho_form), op=MPI.SUM)
        V_rho_value = actual_rho_volume / self.total_volume
        dVdrho_vector = self.dVdrho_vec.copy()

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
        dVdphi_vector = self.dVdphi_vec.copy()
        dVdphi_vector /= self.total_volume

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
        self.dObj_du_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        set_bc(self.dObj_du_vec, self.problem.bcs)
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


        # Adjoint contribution for phi_phys
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
        # Results
        # ============================================================
        function_values = (
            Obj_value,
            V_rho_value,
            V_phi_value,
        )

        dObj_drho_out = (
            self.dObj_drho_vec
            if self.rho_active
            else self._zero_vector(self.dObj_drho_vec)
        )
        dObj_dphi_out = (
            self.dObj_dphi_vec
            if self.phi_active
            else self._zero_vector(self.dObj_dphi_vec)
        )
        dObj_dtheta_out = (
            self.dObj_dtheta_vec
            if self.theta_active
            else self._zero_vector(self.dObj_dtheta_vec)
        )

        dVdrho_out = (
            dVdrho_vector
            if self.rho_active
            else self._zero_vector(dVdrho_vector)
        )
        dVdphi_out = (
            dVdphi_vector
            if self.phi_active
            else self._zero_vector(dVdphi_vector)
        )

        gradients = {
            "objective": {
                "rho": dObj_drho_out,
                "phi": dObj_dphi_out,
                "theta": dObj_dtheta_out,
            },
            "volume": {
                "rho": dVdrho_out,
                "phi": dVdphi_out,
            },
        }

        return function_values, gradients

