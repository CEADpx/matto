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

Major additions to topopt.py:
- Simultaneous optimization of rho, phi, and theta
- Active design-variable selection and management
- Multi-load-case optimization support
- Additional optimization constraints
- Design export and post-processing utilities
"""

import os
import numpy as np
import time
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.linalg import solve
import dolfinx.io
from dolfinx import fem, mesh
from dolfinx.mesh import create_box, CellType, locate_entities_boundary, meshtags, create_rectangle
from dolfinx.fem import (Function, Constant, dirichletbc, locate_dofs_topological, 
                        form, assemble_scalar, functionspace)
from dolfinx import la
from dolfinx.fem.petsc import (create_vector, create_matrix, assemble_vector, assemble_matrix, set_bc)
import pyvista
pyvista.set_jupyter_backend('html')
import basix
from basix.ufl import element
from ufl import variable, inner, grad, det, tr, Identity, outer, dev, sym

from fenitop.fem import form_fem
from fenitop.parameterize import DensityFilter, Heaviside
from fenitop.sensitivity import Sensitivity
from fenitop.optimize import optimality_criteria, mma_optimizer
from fenitop.utility import Communicator, Plotter, save_xdmf, plot_design

def topopt(fem_params, opt, design_variables=None):

    """Main function for topology optimization."""
    
    # Initialization
    comm = MPI.COMM_WORLD

    # ============================================================
    # Design Variable Configuration
    # ============================================================

    if design_variables is None:
        design_variables = {
            "rho":   {"active": True},
            "phi":   {"active": True},
        }

    # Sanity checks
    if not isinstance(design_variables, dict):
        raise TypeError("design_variables must be a dict")

    for key, cfg in design_variables.items():
        if "active" not in cfg:
            raise KeyError(f"design_variables['{key}'] missing 'active' flag")
        if not isinstance(cfg["active"], bool):
            raise TypeError(f"design_variables['{key}']['active'] must be bool")

    # Active design variables (ordering defines MMA layout)
    active_design_vars = [
        name for name, cfg in design_variables.items() if cfg["active"]
    ]

    if len(active_design_vars) == 0:
        raise RuntimeError("At least one design variable must be active")

    if comm.rank == 0:
        print(f"[topopt] Active design variables: {active_design_vars}", flush=True)
    
    # Expose configuration for FEM, sensitivity, and optimizer
    opt["design_variables"] = design_variables
    opt["active_design_vars"] = active_design_vars

    # ============================================================
    # FEM Problem Construction
    # ============================================================
    (
        femProblem,
        u_field,
        lambda_field,
        rho_field,
        rho_phys_field,
        phi_field,
        phi_phys_field,
        phi_eff_field,
        theta_phys_field,
        traction_constants,
        ds,
    ) = form_fem(fem_params, opt)

    # --- Von Mises stress field for output ---
    S0_stress = fem.functionspace(fem_params["mesh"], ("DG", 0))
    sigma_vm_field = fem.Function(S0_stress, name="sigma_vm")
    sigma_vm_expr = fem.Expression(
        opt["sigma_vm_expr"],
        S0_stress.element.interpolation_points()
    )

    # Strain energy density field (only created if required) (need for strain constraint)
    we_voidpen_obj = (opt.get("objective_type", "") == "max_disp_we_voidpen")
    if opt.get("strain_constraint", False) or opt.get("output_strain_energy_field", False) or we_voidpen_obj:
        S0_W = fem.functionspace(fem_params["mesh"], ("DG", 0))
        W_field = fem.Function(S0_W, name="strain_energy")
        W_expr = fem.Expression(
            opt["W_elastic_expr"],
            S0_W.element.interpolation_points()
        )

    # CG1 space for BP output (BP requires matching element types)
    S_stress_cg = fem.functionspace(fem_params["mesh"], ("CG", 1))
    sigma_vm_cg = fem.Function(S_stress_cg, name="sigma_vm_cg")

    # Effective magnetization field for visualization
    V_vec_cg = fem.functionspace(
        fem_params["mesh"],
        basix.ufl.element(
            "Lagrange",
            fem_params["mesh"].basix_cell(),
            1,
            shape=(fem_params["mesh"].geometry.dim,)
        )
    )
  
    m_eff_field = fem.Function(V_vec_cg, name="m_eff")

    # m_eff = phi_eff * [cos(theta), sin(theta)]
    m_eff_expr = ufl.as_vector((
        phi_eff_field * ufl.cos(opt["theta_phys_field"]),
        phi_eff_field * ufl.sin(opt["theta_phys_field"])
    ))

    ds_measure = assemble_scalar(
        form(Constant(fem_params["mesh"], PETSc.ScalarType(1)) * ds(0))
    )

    # Expose for Sensitivity (disp constraint uses average tip displacement)
    opt["tip_measure"] = float(ds_measure)

    # ============================================================
    # Density Filtering and Projection
    # ============================================================
    dv_cfg = opt.get("design_variables", {})
    active_vars = opt.get("active_design_vars", [])

    # Density variable (rho)
    if dv_cfg.get("rho", {}).get("active", True):
        rho_density_filter = DensityFilter(
            comm, rho_field, rho_phys_field,
            opt["filter_radius"], fem_params["petsc_options"]
        )
        rho_heaviside = Heaviside(rho_phys_field)
    else:
        rho_density_filter = None
        rho_heaviside = None

        # Freeze rho_phys = 1 everywhere (CG1)
        with rho_phys_field.x.petsc_vec.localForm() as loc:
            loc.set(1.0)
        rho_phys_field.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD
        )

    # Magnetic volume fraction (phi)
    if dv_cfg.get("phi", {}).get("active", True):
        phi_density_filter = DensityFilter(
            comm, phi_field, phi_phys_field,
            opt["filter_radius"], fem_params["petsc_options"]
        )
    else:
        phi_density_filter = None

        # Freeze phi_phys = 0 everywhere (CG1)
        with phi_phys_field.x.petsc_vec.localForm() as loc:
            loc.set(0.0)
        phi_phys_field.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD
        )

    # Remanence direction angle (theta)
    if dv_cfg.get("theta", {}).get("active", False):
        opt["theta_density_filter"] = DensityFilter(
            comm,
            opt["theta_field"],
            opt["theta_phys_field"],
            opt["filter_radius"],
            fem_params["petsc_options"]
        )
    else:
        opt["theta_density_filter"] = None


    # Sensitivity analysis setup
    sens_problem = Sensitivity(comm, opt, femProblem,
                            u_field, lambda_field,
                            rho_phys_field, phi_phys_field,
                            opt["theta_phys_field"])

    S_comm = Communicator(phi_phys_field.function_space, fem_params["mesh_serial"])
    
    # MMA initialization
    if comm.rank == 0:
        plotter = Plotter(fem_params["mesh_serial"])
 
    # Number of constraints:
    #   base (global): volume upper bound for each ACTIVE design variable
    #   optional (global): volume lower bound for each ACTIVE design variable (equality enforcement)
    #   optional (per-load-case): stress, compliance, strain
    rho_active   = opt.get("design_variables", {}).get("rho", {}).get("active", True)
    phi_active   = opt.get("design_variables", {}).get("phi", {}).get("active", True)
    theta_active = opt.get("design_variables", {}).get("theta", {}).get("active", False)

    num_consts = 0
    if rho_active:
        num_consts += 1
    if phi_active:
        num_consts += 1

    if opt.get("enforce_volume_equality", False):
        if rho_active:
            num_consts += 1
        if phi_active:
            num_consts += 1

    # Load cases (always a list after normalization below)
    load_cases = fem_params.get("load_cases", None)
    if load_cases is None:
        load_cases = [{"name": "single"}]
    n_lc = len(load_cases)

    if opt.get("stress_constraint", False):
        num_consts += n_lc

    if opt.get("compliance_constraint", False):
        num_consts += n_lc

    if opt.get("strain_constraint", False):
        num_consts += n_lc

    if opt.get("disp_constraint", False):
        num_consts += n_lc
       
    num_rho_elems = rho_field.x.petsc_vec.array.size
    num_phi_elems = phi_field.x.petsc_vec.array.size

    # Active-only design vector sizing (MMA x length)
    active_vars = opt.get("active_design_vars", [])
    rho_active = opt.get("design_variables", {}).get("rho", {}).get("active", True)
    phi_active = opt.get("design_variables", {}).get("phi", {}).get("active", True)

    rho_slice = None
    phi_slice = None
    theta_slice = None
    offset = 0

    if rho_active:
        rho_slice = slice(offset, offset + num_rho_elems)
        offset += num_rho_elems

    if phi_active:
        phi_slice = slice(offset, offset + num_phi_elems)
        offset += num_phi_elems
    
    if theta_active:
        num_theta_elems = opt["theta_field"].x.petsc_vec.array.size
        theta_slice = slice(offset, offset + num_theta_elems)
        offset += num_theta_elems


    design_vec_size = offset

    dvec_old1, dvec_old2 = np.zeros(design_vec_size), np.zeros(design_vec_size)
    low, upp = None, None

    # ============================================================
    # Sensitivity Backpropagation Helpers
    # ============================================================

    def _zeros_rho(n):
        return [np.zeros(num_rho_elems, dtype=float) for _ in range(n)]

    def _zeros_phi(n):
        return [np.zeros(num_phi_elems, dtype=float) for _ in range(n)]

    # Map sensitivities from physical space to design space
    def backprop_rho(vecs_phys):
        """
        vecs_phys: list of PETSc Vecs w.r.t. rho_phys_field
        returns: list of numpy arrays w.r.t. rho_field (DG0)
        """
        if rho_density_filter is None:
            return _zeros_rho(len(vecs_phys))
        # Heaviside only exists if rho is active in your setup
        if rho_heaviside is not None:
            rho_heaviside.backward(vecs_phys)
        return rho_density_filter.backward(vecs_phys)

    def backprop_phi(vecs_phys):
        """
        vecs_phys: list of PETSc Vecs w.r.t. phi_phys_field
        returns: list of numpy arrays w.r.t. phi_field (DG0)
        """
        if phi_density_filter is None:
            return _zeros_phi(len(vecs_phys))
        return phi_density_filter.backward(vecs_phys)

    def backprop_theta(vecs_phys):
        """
        vecs_phys: list of PETSc Vecs w.r.t. theta_phys_field
        returns: list of numpy arrays w.r.t. theta_field (DG0)
        """
        if not theta_active:
            return [np.zeros_like(opt["theta_field"].x.petsc_vec.array) for _ in range(len(vecs_phys))]
        # theta uses density filter only (no Heaviside)
        return opt["theta_density_filter"].backward(vecs_phys)

    # Initialize density field (rho)
    centers_rho = rho_field.function_space.tabulate_dof_coordinates()[:num_rho_elems].T

    # Backward-compatible default:
    # if rho-specific zones are not provided, use old solid_zone / void_zone.
    rho_solid = opt.get("rho_solid_zone", opt["solid_zone"])(centers_rho)
    rho_void  = opt.get("rho_void_zone",  opt["void_zone"])(centers_rho)
    rho_ini = np.full(num_rho_elems, opt["vol_frac_rho"])

    # Enforce passive rho zones at initialization only.
    # Do NOT lock rho_min/rho_max here; fixed MMA bounds can stall the update.
    rho_ini[rho_solid] = 1.0
    rho_ini[rho_void] = 0.05

    rho_min = np.full(num_rho_elems, 0.05)
    rho_max = np.ones(num_rho_elems)

    rho_field.x.petsc_vec.array[:] = np.clip(rho_ini, rho_min, rho_max)


    # Initialize magnetic fraction field (phi)
    centers_phi = phi_field.function_space.tabulate_dof_coordinates()[:num_phi_elems].T

    # Backward-compatible default:
    # if phi-specific zones are not provided, use old solid_zone / void_zone.
    phi_solid = opt.get("phi_solid_zone", opt["solid_zone"])(centers_phi)
    phi_void  = opt.get("phi_void_zone",  opt["void_zone"])(centers_phi)

    # Physical cap for magnetic material fraction
    phi_cap = opt.get("phi_cap", 0.3)

    # Initial distribution (uniform)
    phi_ini = np.full(num_phi_elems, opt["vol_frac_phi"])
    phi_ini[phi_solid] = phi_cap
    phi_ini[phi_void] = 0.0

    # Bounds for magnetic fraction (φ ∈ [0, φ_cap])
    phi_min = np.full(num_phi_elems, 0.0)
    phi_max = np.full(num_phi_elems, phi_cap)

    # Clip to bounds and assign to PETSc vector
    phi_field.x.petsc_vec.array[:] = np.clip(phi_ini, phi_min, phi_max)

    # Initialize remanence angle field (theta)
    if theta_active:
        theta_field = opt["theta_field"]

        # Initialize theta from fem_params["theta_init_dir"].
        # Default is +x if not provided.
        theta_init_dir = np.array(
            fem_params.get("theta_init_dir", (1.0, 0.0)),
            dtype=float
        )

        nrm = np.linalg.norm(theta_init_dir)
        if nrm > 0:
            theta_init_dir /= nrm

        theta0 = float(np.arctan2(theta_init_dir[1], theta_init_dir[0]))

        theta_ini = np.full_like(theta_field.x.petsc_vec.array, theta0)
        theta_field.x.petsc_vec.array[:] = theta_ini

        # Bounds: theta ∈ [-pi, pi]
        theta_min = np.full_like(theta_ini, -np.pi)
        theta_max = np.full_like(theta_ini,  np.pi)


    # sim file
    os.makedirs(opt["output_dir"], exist_ok=True)
    # Ensure path ends with a slash
    output_dir = opt["output_dir"].rstrip("/") + "/"

    # Create directory if needed
    os.makedirs(output_dir, exist_ok=True)


    # Per-load-case BP writers
    sim_bp_writers = {}

    for lc in load_cases:
        lc_name = lc.get("name", "unnamed")
        fname = os.path.join(output_dir, f"optimized_design_{lc_name}.bp")

        sim_bp_writers[lc_name] = dolfinx.io.VTXWriter(
            fem_params["mesh"].comm,
            fname,
            [rho_phys_field, phi_eff_field, m_eff_field, u_field, lambda_field, sigma_vm_cg],

            engine="BP4"
        )

    sim_file_xdmf_results = dolfinx.io.XDMFFile(
        fem_params["mesh"].comm,
        os.path.join(output_dir, "optimized_design.xdmf"),
        "w"
    )

    sim_file_xdmf_results.write_mesh(fem_params["mesh"])
    sim_file_xdmf_results.write_function(phi_eff_field, 0)
    sim_file_xdmf_results.write_function(rho_phys_field, 0)
    sim_file_xdmf_results.write_function(sigma_vm_field, 0)
    if theta_active:
        sim_file_xdmf_results.write_function(m_eff_field, 0)

    if opt.get("strain_constraint", False) or opt.get("output_strain_energy_field", False):
        sim_file_xdmf_results.write_function(W_field, 0)

    # Start topology optimization
    opt_iter, beta, change = 0, 1, 2*opt["opt_tol"]
    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:
        opt_start_time = time.perf_counter()
        opt_iter += 1
    
        # Strain-energy constraint ramping
        if opt.get("strain_constraint", False):

            ramp = opt.get("strain_ramp", {})
            if ramp.get("enabled", False):

                k0 = ramp.get("start_iter", 1)
                k1 = ramp.get("end_iter", opt["max_iter"])
                U0 = ramp.get("U_start", opt["U_max"])
                U1 = ramp.get("U_end", opt["U_max"])

                # Normalized ramp parameter
                if opt_iter <= k0:
                    theta = 0.0
                elif opt_iter >= k1:
                    theta = 1.0
                else:
                    theta = (opt_iter - k0) / max(k1 - k0, 1)

                # Ramp schedule
                if ramp.get("schedule", "linear") == "exp":
                    U_max_active = U0 * (U1 / U0) ** theta
                else:
                    U_max_active = U0 + theta * (U1 - U0)

                opt["U_max_active"] = U_max_active

            else:
                opt["U_max_active"] = opt["U_max"]

        # Tip displacement constraint ramping
        if opt.get("disp_constraint", False):

            ramp = opt.get("disp_ramp", {})
            if ramp.get("enabled", False):

                k0 = ramp.get("start_iter", 1)
                k1 = ramp.get("end_iter", opt["max_iter"])
                u0 = ramp.get("u_start", opt["u_min"])
                u1 = ramp.get("u_end", opt["u_min"])

                # Normalized ramp parameter
                if opt_iter <= k0:
                    theta = 0.0
                elif opt_iter >= k1:
                    theta = 1.0
                else:
                    theta = (opt_iter - k0) / max(k1 - k0, 1)

                # Ramp schedule
                if ramp.get("schedule", "linear") == "exp":
                    u_min_active = u0 * (u1 / u0) ** theta
                else:
                    u_min_active = u0 + theta * (u1 - u0)

                opt["u_min_active"] = float(u_min_active)

            else:
                opt["u_min_active"] = float(opt["u_min"])


        # Aggregated objective and sensitivities
        Obj_total = 0.0
        dJdrho_total = np.zeros_like(rho_field.x.petsc_vec.array)
        dJdphi_total = np.zeros_like(phi_field.x.petsc_vec.array)
        if theta_active:
            dJdtheta_total = np.zeros_like(opt["theta_field"].x.petsc_vec.array)


        # Volume values (load-independent; will be taken from first case)
        V_rho_value = None
        V_phi_value = None

        # Optional constraint aggregation containers (per-case enforcement)
        stress_enabled = opt.get("stress_constraint", False)
        strain_enabled = opt.get("strain_constraint", False)
        comp_enabled = opt.get("compliance_constraint", False)
        disp_enabled = opt.get("disp_constraint", False)

        g_stress_list = []   # one entry per load case (if enabled)
        g_strain_list = []   # one entry per load case (if enabled)

        g_comp_list = []     # one per load case (if enabled)
        dCdrho_list = []
        dCdphi_list = []
        dCdtheta_list = []

        dGdrho_list = []
        dGdphi_list = []
        dGdtheta_list = []
        dUdrho_list = []
        dUdphi_list = []
        dUdtheta_list = []

        g_disp_list = []     # one per load case (if enabled)
        u_tip_list = []      # optional diagnostics per load case
        dDispdrho_list = []
        dDispdphi_list = []
        dDispdtheta_list = []

        # Apply filtering and projection
        if rho_density_filter is not None:
            rho_density_filter.forward()
            if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
                beta *= 2
                change = opt["opt_tol"] * 2
            rho_heaviside.forward(beta)

        if phi_density_filter is not None:
            phi_density_filter.forward()

        if theta_active and opt["theta_density_filter"] is not None:
            opt["theta_density_filter"].forward()


        # --- Update phi_eff_field for sensitivities ---
        phi_eff_field.x.petsc_vec.array[:] = (
            rho_phys_field.x.petsc_vec.array *
            phi_phys_field.x.petsc_vec.array
        )
        
        N_steps = fem_params["load_steps"]
    
        # MULTI-LOAD CASE LOOP
        for load_case in load_cases:

            # Deterministic load-case name 
            lc_name = load_case.get("name", "unnamed")

            # ------------------------------------------------------------
            # Load Stepping
            # ------------------------------------------------------------
            B_app_mag_target = float(
                load_case.get(
                    "B_app_mag",
                    fem_params.get("B_app_mag", 0.0)
                )
            )

            B_app_dir_target = np.array(
                load_case.get(
                    "B_app_dir",
                    fem_params.get("B_app_dir", (0.0, 0.0))
                ),
                dtype=float
            )

            nrm = np.linalg.norm(B_app_dir_target)
            if nrm > 0:
                B_app_dir_target /= nrm
            else:
                B_app_dir_target[:] = 0.0

            # Start from ZERO field (will ramp inside load steps)
            opt["B_app"].value[:] = 0.0

            # RESET DISPLACEMENT FIELD (CRITICAL FOR PROPER LOAD STEPPING)
            with u_field.x.petsc_vec.localForm() as loc:
                loc.set(0.0)

            u_field.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT,
                mode=PETSc.ScatterMode.FORWARD
            )

            # Reset tractions for this case
            for t_const in traction_constants:
                t_const.value = np.zeros_like(t_const.value)

            # Apply load-case traction targets
            if load_case is not None:
                for bc_dict in fem_params["traction_bcs"]:
                    name = bc_dict.get("name", None)
                    if name is not None and name in load_case["tractions"]:
                        bc_dict["traction_max"] = load_case["tractions"][name]
                    else:
                        bc_dict["traction_max"] = (0.0, 0.0)

            for step in range(1, N_steps + 1):

                # --------------------------------------------
                # Ramp applied magnetic field
                # --------------------------------------------
                alpha = step / N_steps
                opt["B_app"].value[:] = alpha * B_app_mag_target * B_app_dir_target

                # --------------------------------------------
                # Ramp tractions (unchanged behavior)
                # --------------------------------------------
                for t_const, bc_dict in zip(traction_constants, fem_params["traction_bcs"]):
                    t_max = np.array(bc_dict["traction_max"], dtype=float)
                    t_const.value += (1.0 / N_steps) * t_max

                femProblem.solve_fem()


            # --- Compliance diagnostic (only if defined) ---
            if compliance_form is not None:
                C_local = assemble_scalar(form(compliance_form))
                C_val = comm.allreduce(C_local, op=MPI.SUM)

                if comm.rank == 0:
                    print(f"  [{lc_name}] compliance: {C_val:.6e}")



            # Post-solve diagnostics / fields
            sigma_vm_field.interpolate(sigma_vm_expr)
            sigma_vm_cg.interpolate(sigma_vm_field)

            # Effective magnetization vector field (for visualization)
            if theta_active:
                m_eff_field.interpolate(fem.Expression(
                    m_eff_expr,
                    V_vec_cg.element.interpolation_points()
                ))


            # Write BP output for load case 
            if opt_iter % opt["sim_output_interval"] == 0:
                sim_bp_writers[lc_name].write(opt_iter)

            # Strain energy density field (also needed for WE-voidpen objective)
            if opt.get("strain_constraint", False) or opt.get("output_strain_energy_field", False) or we_voidpen_obj:
                W_field.interpolate(W_expr)

            # --------------------------------------------------------
            # WE-voidpen weight update (frozen / lagged)
            # - We compute weights from W_field (DG0)
            # - We store them into we_w_field (DG0)
            # - We set P0 once when weights are first initialized
            #
            # Update schedule:
            #   - initialize at opt_iter == we_freeze_iter
            #   - then update every we_update_every iterations (optional)
            # --------------------------------------------------------
            if we_voidpen_obj:
                do_init = (not we_weights_initialized) and (opt_iter == we_freeze_iter)
                do_update = we_weights_initialized and (we_update_every > 0) and (opt_iter % we_update_every == 0)

                if do_init or do_update:
                    # local DG0 array of strain energy density
                    W_local = W_field.x.array
                    W_local_max = float(np.max(W_local)) if W_local.size > 0 else 0.0
                    W_max = comm.allreduce(W_local_max, op=MPI.MAX)

                    # normalize safely to [0, 1], then clamp
                    if W_max <= 1e-30:
                        w_local = np.zeros_like(W_local)
                    else:
                        w_local = W_local / (W_max + 1e-12)
                        if we_wmax < 1.0:
                            w_local = np.minimum(w_local, we_wmax)
                        else:
                            w_local = np.minimum(w_local, 1.0)

                    # write weight field (DG0)
                    we_w_field.x.petsc_vec.array[:] = w_local

                    # initialize P0 once (using current rho_phys and weights)
                    if do_init:
                        P0_val = assemble_scalar(form(opt["we_voidpen_form"]))
                        P0_val = comm.allreduce(P0_val, op=MPI.SUM)
                        if P0_val <= 1e-30:
                            P0_val = 1.0
                        we_P0_const.value = PETSc.ScalarType(P0_val)
                        we_weights_initialized = True

                        if comm.rank == 0:
                            print(f"  [we_voidpen] initialized weights at iter {opt_iter}, "
                                  f"W_max={W_max:.3e}, P0={float(P0_val):.6e}", flush=True)


            u_array = u_field.x.array
            max_disp = np.max(np.abs(u_array))
         
            # Print Displacement info (per load case)
            if comm.rank == 0:
                print(f"  [{lc_name}] max abs displacement: {max_disp:.4e}")

            # Displacement tracking diagnostics 
            if opt.get("objective_type") == "disp_track" and comm.rank == 0:

                for i, term in enumerate(opt.get("track_terms", [])):

                    # ∫ u w dx  → vector
                    ux_val = assemble_scalar(form(term["ux_form"]))
                    uy_val = assemble_scalar(form(term["uy_form"]))
                    w_val  = assemble_scalar(form(term["w_form"]))

                    if w_val > 0:
                        ux_avg = ux_val / w_val
                        uy_avg = uy_val / w_val
                    else:
                        ux_avg = 0.0
                        uy_avg = 0.0

                    ux_t, uy_t = term["target"]
                    err_x = ux_avg - ux_t
                    err_y = uy_avg - uy_t

                    print(
                        f"  track[{i}] @ {term['point']}: "
                        f"u_avg=({ux_avg:.4f}, {uy_avg:.4f}), "
                        f"target=({ux_t:.4f}, {uy_t:.4f}), "
                        f"error=({err_x:.4f}, {err_y:.4f})"
                    )

            sens_output = sens_problem.evaluate()

            # --- Base outputs ---
            [Obj_case, V_rho_case, V_phi_case] = sens_output[0]
            constraints = sens_output[2]

            # --- Dict-based sensitivities ---
            grads = sens_output[3]

            dJdrho_phys = grads["objective"]["rho"]
            dJdphi_phys = grads["objective"]["phi"]
            dJdtheta_phys = grads["objective"].get("theta", None)

            dVdrho_phys = grads["volume"]["rho"]
            dVdphi_phys = grads["volume"]["phi"]
            dVdtheta_phys = grads["volume"].get("theta", None)

            # --------------------------------------------------------
            # Compliance constraint (per load case)
            # Sensitivity returns raw C and total derivatives.
            # Normalization is applied later in topopt.py.
            # --------------------------------------------------------
            if comp_enabled:
                if constraints.get("compliance", None) is None:
                    raise RuntimeError("compliance_constraint enabled but constraints['compliance'] is None")

                C_case = float(constraints["compliance"]["g"])
                dCdrho_phys, dCdphi_phys, dCdtheta_phys = constraints["compliance"]["dg"]

                # Backprop to design space (safe with toggles)
                dCdrho_design = backprop_rho([dCdrho_phys])[0]
                dCdphi_design = backprop_phi([dCdphi_phys])[0]
                if theta_active:
                    dCdtheta_design = backprop_theta([dCdtheta_phys])[0]

                # Store
                g_comp_list.append(C_case)
                dCdrho_list.append(dCdrho_design.copy())
                dCdphi_list.append(dCdphi_design.copy())
                if theta_active:
                    dCdtheta_list.append(dCdtheta_design.copy())



            # --- Backprop objective + volume to design space (safe with toggles) ---
            dJdrho_design, dVdrho_design = backprop_rho([dJdrho_phys, dVdrho_phys])
            dJdphi_design, dVdphi_design = backprop_phi([dJdphi_phys, dVdphi_phys])
            if theta_active:
                dJdtheta_design = backprop_theta([dJdtheta_phys])[0]


            # --------------------------------------------------------
            # Constraint handling (per load case)
            # IMPORTANT: constraint gradients are wrt PHYSICAL fields.
            # We must backprop them through Heaviside + filters exactly
            # like the objective gradients.
            # --------------------------------------------------------


            # Stress constraint (per case)
            if stress_enabled:
                if constraints["stress"] is None:
                    raise RuntimeError("stress_constraint enabled but constraints['stress'] is None")
                g_stress_case = float(constraints["stress"]["g"])
                dGdrho_phys, dGdphi_phys, dGdtheta_phys = constraints["stress"]["dg"]

                # Backprop to design space (safe with toggles)
                dGdrho_design = backprop_rho([dGdrho_phys])[0]
                dGdphi_design = backprop_phi([dGdphi_phys])[0]
                if theta_active:
                    dGdtheta_design = backprop_theta([dGdtheta_phys])[0]

                # Store per-load-case
                g_stress_list.append(g_stress_case)
                dGdrho_list.append(dGdrho_design.copy())
                dGdphi_list.append(dGdphi_design.copy())
                if theta_active:
                    dGdtheta_list.append(dGdtheta_design.copy())


            # Strain constraint (per case)
            if strain_enabled:
                if constraints["strain"] is None:
                    raise RuntimeError("strain_constraint enabled but constraints['strain'] is None")
                g_strain_case = float(constraints["strain"]["g"])
                dUdrho_phys, dUdphi_phys, dUdtheta_phys = constraints["strain"]["dg"]

                # Backprop to design space (safe with toggles)
                dUdrho_design = backprop_rho([dUdrho_phys])[0]
                dUdphi_design = backprop_phi([dUdphi_phys])[0]
                if theta_active:
                    dUdtheta_design = backprop_theta([dUdtheta_phys])[0]

                # Store per-load-case
                g_strain_list.append(g_strain_case)
                dUdrho_list.append(dUdrho_design.copy())
                dUdphi_list.append(dUdphi_design.copy())
                if theta_active:
                    dUdtheta_list.append(dUdtheta_design.copy())

            # Tip displacement constraint (per case)
            if disp_enabled:
                if constraints.get("disp", None) is None:
                    raise RuntimeError("disp_constraint enabled but constraints['disp'] is None")

                g_disp_case = float(constraints["disp"]["g"])
                dDispdrho_phys, dDispdphi_phys, dDispdtheta_phys = constraints["disp"]["dg"]

                # Backprop to design space (safe with toggles)
                dDispdrho_design = backprop_rho([dDispdrho_phys])[0]
                dDispdphi_design = backprop_phi([dDispdphi_phys])[0]
                if theta_active:
                    dDispdtheta_design = backprop_theta([dDispdtheta_phys])[0]

                # Store
                g_disp_list.append(g_disp_case)
                dDispdrho_list.append(dDispdrho_design.copy())
                dDispdphi_list.append(dDispdphi_design.copy())
                if theta_active:
                    dDispdtheta_list.append(dDispdtheta_design.copy())

                # Optional diagnostic scalar (average tip displacement)
                if "u_tip" in constraints["disp"]:
                    u_tip_list.append(float(constraints["disp"]["u_tip"]))


            # Load-case weight
            w_case = 1.0 if load_case is None else float(load_case.get("weight", 1.0))

            # Accumulate weighted totals 
            Obj_total += w_case * float(Obj_case)
            dJdrho_total += w_case * dJdrho_design
            dJdphi_total += w_case * dJdphi_design
            if theta_active:
                dJdtheta_total += w_case * dJdtheta_design


            # Get volumes (only once)
            if V_rho_value is None:
                V_rho_value = V_rho_case
                V_phi_value = V_phi_case
                dVdrho = dVdrho_design
                dVdphi = dVdphi_design

        # FINAL AGGREGATED VALUES 
        Obj_value = Obj_total
        dJdrho = dJdrho_total
        dJdphi = dJdphi_total



        # ============================================================
        # Active-only MMA design vector / bounds / objective gradient
        # ============================================================
        x_parts = []
        xmin_parts = []
        xmax_parts = []
        grad_parts = []

        if rho_active:
            x_parts.append(rho_field.x.petsc_vec.array.copy())
            xmin_parts.append(rho_min)
            xmax_parts.append(rho_max)
            grad_parts.append(dJdrho)

        if phi_active:
            x_parts.append(phi_field.x.petsc_vec.array.copy())
            xmin_parts.append(phi_min)
            xmax_parts.append(phi_max)
            grad_parts.append(dJdphi)

        if theta_active:
            x_parts.append(theta_field.x.petsc_vec.array.copy())
            xmin_parts.append(theta_min)
            xmax_parts.append(theta_max)
            grad_parts.append(dJdtheta_total)


        x = np.concatenate(x_parts) if len(x_parts) > 0 else np.array([], dtype=float)
        x_min = np.concatenate(xmin_parts) if len(xmin_parts) > 0 else np.array([], dtype=float)
        x_max = np.concatenate(xmax_parts) if len(xmax_parts) > 0 else np.array([], dtype=float)

        dfdx = np.concatenate(grad_parts) if len(grad_parts) > 0 else np.array([], dtype=float)


        # BUILD CONSTRAINT VALUE VECTOR g_vec
        #
        # Order (ACTIVE-VARIABLE DEPENDENT):
        #   - Upper volume bounds for each ACTIVE design variable (rho, phi)
        #   - Lower volume bounds for each ACTIVE design variable (if equality enforced)
        #   - Compliance constraints per load case (if enabled)
        #   - Stress constraints per load case (if enabled)
        #   - Strain constraints per load case (if enabled)
        #
        # NOTE:
        #   The exact indices depend on which design variables are active.



        g_list = []

        # Upper volume bounds (global) — ACTIVE vars only
        if rho_active:
            g_list.append(V_rho_value / opt["vol_frac_rho"] - 1.0)
        if phi_active:
            g_list.append(V_phi_value / opt["vol_frac_phi"] - 1.0)

        # Lower volume bounds (global, equality enforcement) — ACTIVE vars only
        if opt.get("enforce_volume_equality", False):
            if rho_active:
                g_list.append(1.0 - V_rho_value / opt["vol_frac_rho"])
            if phi_active:
                g_list.append(1.0 - V_phi_value / opt["vol_frac_phi"])


        # Compliance constraints (per load case): g = C/C_ref - gamma
        if comp_enabled:
            if len(g_comp_list) != len(load_cases):
                raise RuntimeError(
                    f"comp_enabled but g_comp_list has {len(g_comp_list)} entries, "
                    f"expected {len(load_cases)}"
                )

            C_ref = float(opt.get("compliance_ref", 1.0))
            gamma = float(opt.get("compliance_gamma", 1.0))
            if C_ref <= 1e-30:
                raise RuntimeError("compliance_ref must be > 0 for normalized compliance constraint")

            for C_case in g_comp_list:
                g_list.append(C_case / C_ref - gamma)


        # Stress constraints (per load case)
        if stress_enabled:
            if len(g_stress_list) != len(load_cases):
                raise RuntimeError(
                    f"stress_enabled but g_stress_list has {len(g_stress_list)} entries, "
                    f"expected {len(load_cases)}"
                )
            g_list.extend(g_stress_list)

        # Strain constraints (per load case)
        if strain_enabled:
            if len(g_strain_list) != len(load_cases):
                raise RuntimeError(
                    f"strain_enabled but g_strain_list has {len(g_strain_list)} entries, "
                    f"expected {len(load_cases)}"
                )
            g_list.extend(g_strain_list)

        # Tip displacement constraint (per load case)
        if disp_enabled:
            if len(g_disp_list) != len(load_cases):
                raise RuntimeError(
                    f"disp_enabled but g_disp_list has {len(g_disp_list)} entries, "
                    f"expected {len(load_cases)}"
                )
            g_list.extend(g_disp_list)

        g_vec = np.array(g_list, dtype=float)

        # BUILD CONSTRAINT GRADIENT MATRIX dgdx
        # Each row corresponds to one g entry in g_vec.
        #
        # IMPORTANT:
        # The row order MUST match the g_list construction order above.
        dgdx_rows = []

        # ----------------------------
        # Upper volume bounds (global) — ACTIVE vars only
        # g = V - V*
        # ----------------------------
        if rho_active:
            row_parts = []
            if rho_active:
                row_parts.append(dVdrho / opt["vol_frac_rho"])
            if phi_active:
                row_parts.append(np.zeros_like(dVdphi))
            if theta_active:
                row_parts.append(np.zeros_like(opt["theta_field"].x.petsc_vec.array))
            dgdx_rows.append(np.concatenate(row_parts))

        if phi_active:
            row_parts = []
            if rho_active:
                row_parts.append(np.zeros_like(dVdrho))
            if phi_active:
                row_parts.append(dVdphi / opt["vol_frac_phi"])
            if theta_active:
                row_parts.append(np.zeros_like(opt["theta_field"].x.petsc_vec.array))
            dgdx_rows.append(np.concatenate(row_parts))

        # ----------------------------
        # Lower volume bounds (global, equality) — ACTIVE vars only
        # g = V* - V  =>  dg = -dV
        # ----------------------------
        if opt.get("enforce_volume_equality", False):
            if rho_active:
                row_parts = []
                if rho_active:
                    row_parts.append(-dVdrho / opt["vol_frac_rho"])
                if phi_active:
                    row_parts.append(np.zeros_like(dVdphi))
                if theta_active:
                    row_parts.append(np.zeros_like(opt["theta_field"].x.petsc_vec.array))
                dgdx_rows.append(np.concatenate(row_parts))

            if phi_active:
                row_parts = []
                if rho_active:
                    row_parts.append(np.zeros_like(dVdrho))
                if phi_active:
                    row_parts.append(-dVdphi / opt["vol_frac_phi"])
                if theta_active:
                    row_parts.append(np.zeros_like(opt["theta_field"].x.petsc_vec.array))
                dgdx_rows.append(np.concatenate(row_parts))

        # ----------------------------
        # Compliance gradient rows (per load case): dg/dx = (1/C_ref) dC/dx
        # ----------------------------
        if comp_enabled:
            if (len(dCdrho_list) != len(load_cases) or
                len(dCdphi_list) != len(load_cases) or
                (theta_active and len(dCdtheta_list) != len(load_cases))):
                raise RuntimeError("comp_enabled but compliance gradient lists are wrong length")

            C_ref = float(opt.get("compliance_ref", 1.0))
            if C_ref <= 1e-30:
                raise RuntimeError("compliance_ref must be > 0 for normalized compliance constraint")

            invC = 1.0 / C_ref
            for lc_i in range(len(load_cases)):
                row_parts = []
                if rho_active:
                    row_parts.append(dCdrho_list[lc_i])
                if phi_active:
                    row_parts.append(dCdphi_list[lc_i])
                if theta_active:
                    row_parts.append(dCdtheta_list[lc_i])
                dgdx_rows.append(invC * np.concatenate(row_parts))

        # ----------------------------
        # Stress gradients (per load case)
        # ----------------------------
        if stress_enabled:
            if (len(dGdrho_list) != len(load_cases) or
                len(dGdphi_list) != len(load_cases) or
                (theta_active and len(dGdtheta_list) != len(load_cases))):
                raise RuntimeError("stress_enabled but stress gradient lists are wrong length")

            for lc_i in range(len(load_cases)):
                row_parts = []
                if rho_active:
                    row_parts.append(dGdrho_list[lc_i])
                if phi_active:
                    row_parts.append(dGdphi_list[lc_i])
                if theta_active:
                    row_parts.append(dGdtheta_list[lc_i])
                dgdx_rows.append(np.concatenate(row_parts))

        # ----------------------------
        # Strain gradients (per load case)
        # ----------------------------
        if strain_enabled:
            if (len(dUdrho_list) != len(load_cases) or
                len(dUdphi_list) != len(load_cases) or
                (theta_active and len(dUdtheta_list) != len(load_cases))):
                raise RuntimeError("strain_enabled but strain gradient lists are wrong length")

            for lc_i in range(len(load_cases)):
                row_parts = []
                if rho_active:
                    row_parts.append(dUdrho_list[lc_i])
                if phi_active:
                    row_parts.append(dUdphi_list[lc_i])
                if theta_active:
                    row_parts.append(dUdtheta_list[lc_i])
                dgdx_rows.append(np.concatenate(row_parts))

        # ----------------------------
        # Tip displacement gradients (per load case)
        # (Must come last, because g_list appends g_disp at the end)
        # ----------------------------
        if disp_enabled:
            if (len(dDispdrho_list) != len(load_cases) or
                len(dDispdphi_list) != len(load_cases) or
                (theta_active and len(dDispdtheta_list) != len(load_cases))):
                raise RuntimeError("disp_enabled but displacement gradient lists are wrong length")

            for lc_i in range(len(load_cases)):
                row_parts = []
                if rho_active:
                    row_parts.append(dDispdrho_list[lc_i])
                if phi_active:
                    row_parts.append(dDispdphi_list[lc_i])
                if theta_active:
                    row_parts.append(dDispdtheta_list[lc_i])
                dgdx_rows.append(np.concatenate(row_parts))

        dgdx = np.vstack(dgdx_rows)

        # --- MMA update ---
        x_new, change, low, upp = mma_optimizer(
            num_consts, design_vec_size, opt_iter,
            x, x_min, x_max,
            dvec_old1, dvec_old2,
            dfdx, g_vec, dgdx,
            low, upp, opt["move"]
        )

        # --- Shift history and unpack new fields ---
        dvec_old2 = dvec_old1.copy()
        dvec_old1 = x.copy()

        # --- Unpack active-only MMA vector back into fields ---
        if rho_active:
            rho_new = x_new[rho_slice].copy()

            # Re-enforce passive rho zones after MMA update
            rho_new[rho_solid] = 1.0
            rho_new[rho_void] = 0.05

            rho_field.x.petsc_vec.array[:] = rho_new

        if phi_active:
            phi_new = x_new[phi_slice].copy()

            # Re-enforce passive phi zones after MMA update
            phi_new[phi_solid] = phi_cap
            phi_new[phi_void] = 0.0

            phi_field.x.petsc_vec.array[:] = phi_new

        if theta_active:
            theta_field.x.petsc_vec.array[:] = x_new[theta_slice].copy()


        # Output the histories
        opt_time = time.perf_counter() - opt_start_time

        if comm.rank == 0:
            print(f"opt_iter: {opt_iter}, opt_time: {opt_time:.3g} (s), "
                    f"beta: {beta}, Obj: {Obj_value:.3f}, "
                    f"V_rho: {V_rho_value:.3f}, V_phi: {V_phi_value:.3f}, "
                    f"change: {change:.3f}", flush=True)
            if stress_enabled:
                for i, val in enumerate(g_stress_list):
                    print(f"      g_stress[{i}]: {val:.3e}")
            if strain_enabled:
                for i, val in enumerate(g_strain_list):
                    print(f"      g_strain[{i}]: {val:.3e}")
            if comp_enabled:
                C_ref = float(opt.get("compliance_ref", 1.0))
                gamma = float(opt.get("compliance_gamma", 1.0))
                for i, C_case in enumerate(g_comp_list):
                    print(f"      g_comp[{i}]: {C_case / C_ref - gamma:.3e}")
            if disp_enabled:
                for i, val in enumerate(g_disp_list):
                    print(f"      g_disp[{i}]: {val:.3e}")
                if len(u_tip_list) == len(g_disp_list):
                    for i, ut in enumerate(u_tip_list):
                        print(f"      u_tip[{i}]: {ut:.6f}  (avg over right boundary)")

        # --- Print average tip displacement (useful for tuning) ---
        if opt.get("disp_constraint", False) and comm.rank == 0:
            if len(u_tip_list) > 0:
                # Use the last (only) load case value
                print(f"  tip displacement (avg over right boundary): {u_tip_list[-1]:.6f}")
        
        values = S_comm.gather(phi_eff_field)
        if comm.rank == 0 and opt_iter % opt["sim_image_output_interval"] == 0:
            plot_design(fem_params["mesh_serial"], values, str(opt_iter), opt["output_dir"], pv_or_image="image")
            
        if opt_iter % opt["sim_output_interval"] == 0:
            sim_file_xdmf_results.write_function(rho_phys_field, opt_iter)
            sim_file_xdmf_results.write_function(phi_eff_field, opt_iter)
            sim_file_xdmf_results.write_function(sigma_vm_field, opt_iter)
            if theta_active:
                sim_file_xdmf_results.write_function(m_eff_field, opt_iter)
            if opt.get("strain_constraint", False) or opt.get("output_strain_energy_field", False):
                sim_file_xdmf_results.write_function(W_field, opt_iter)
    
    # ============================================================
    # Final Output
    # ============================================================
    if comm.rank == 0:
        plot_design(fem_params["mesh_serial"], values, None, opt["output_dir"], pv_or_image="image")
        print(f"FINAL max abs displacement: {max_disp:.4e}")
        print(f"FINAL objective value: {Obj_value:.4f}")
    
        # Build a report 
        final_report = (
            f"FINAL max abs displacement: {max_disp:.4e}\n"
            f"FINAL objective value: {Obj_value:.4f}\n"
            f"FINAL volumes -> rho: {V_rho_value:.4f}, phi: {V_phi_value:.4f}\n"
        )
    
        # Write to file
        with open(os.path.join(opt["output_dir"], "final_results.txt"), "w") as f:
            f.write(final_report)
    
    # Save final design fields
    if comm.rank == 0:
        phi_array = phi_eff_field.x.array
        np.save(os.path.join(opt["output_dir"], "final_phi_eff.npy"), phi_array)
        rho_array = rho_phys_field.x.array
        np.save(os.path.join(opt["output_dir"], "final_rho_phys.npy"), rho_array)

        # --- Save final phi_phys_field array ---
        phi_phys_array = phi_phys_field.x.array
        np.save(
            os.path.join(opt["output_dir"], "final_phi_phys.npy"),
            phi_phys_array
        )

        # --- Save final theta_phys_field array (if present) ---
        theta_phys_field = opt.get("theta_phys_field", None)
        if theta_phys_field is not None:
            theta_phys_array = theta_phys_field.x.array
            np.save(
                os.path.join(opt["output_dir"], "final_theta_phys.npy"),
                theta_phys_array
            )

        print(f"Saved final phi_eff_field array to {opt['output_dir']}/final_phi_eff.npy")

        print(f"Saved final fields to {opt['output_dir']}:")
        print("  - final_rho_phys.npy")
        print("  - final_phi_eff.npy")
        print("  - final_phi_phys.npy")
        if theta_phys_field is not None:
            print("  - final_theta_phys.npy")


    # Close output files 
    sim_file_xdmf_results.close()

    for writer in sim_bp_writers.values():
        writer.close()
