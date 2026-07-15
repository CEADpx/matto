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
- Volume-constrained MMA optimization
- Design export and post-processing utilities
"""

import os
import time

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx.io
from dolfinx import fem

from fenitop.fem import form_fem
from fenitop.optimize import mma_optimizer
from fenitop.parameterize import DensityFilter, Heaviside
from fenitop.sensitivity import Sensitivity
from fenitop.utility import Communicator, plot_design

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

    # Active design variables used for validation and logging
    active_design_vars = [
        name for name, cfg in design_variables.items() if cfg["active"]
    ]

    if len(active_design_vars) == 0:
        raise RuntimeError("At least one design variable must be active")

    if comm.rank == 0:
        print(f"[topopt] Active design variables: {active_design_vars}", flush=True)
    
    # Expose the design-variable configuration to FEM and sensitivity modules
    opt["design_variables"] = design_variables

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
        _ds,
    ) = form_fem(fem_params, opt)

    # Von Mises stress field for output
    S0_stress = fem.functionspace(fem_params["mesh"], ("DG", 0))
    sigma_vm_field = fem.Function(S0_stress, name="sigma_vm")
    sigma_vm_expr = fem.Expression(
        opt["sigma_vm_expr"],
        S0_stress.element.interpolation_points(),
    )

    # Convert stress to CG1 for BP output
    S_stress_cg = fem.functionspace(fem_params["mesh"], ("CG", 1))
    sigma_vm_cg = fem.Function(S_stress_cg, name="sigma_vm_cg")

    # Effective magnetic material-direction field
    V_vec_cg = fem.functionspace(
        fem_params["mesh"],
        basix.ufl.element(
            "Lagrange",
            fem_params["mesh"].basix_cell(),
            1,
            shape=(fem_params["mesh"].geometry.dim,),
        ),
    )
    m_eff_field = fem.Function(V_vec_cg, name="m_eff")
    m_eff_expr = ufl.as_vector((
        phi_eff_field * ufl.cos(theta_phys_field),
        phi_eff_field * ufl.sin(theta_phys_field),
    ))

    # ============================================================
    # Density Filtering and Projection
    # ============================================================
    dv_cfg = opt.get("design_variables", {})

    rho_active = dv_cfg.get("rho", {}).get("active", True)
    phi_active = dv_cfg.get("phi", {}).get("active", True)
    theta_active = dv_cfg.get("theta", {}).get("active", False)

    # Density variable (rho)
    if rho_active:
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
    if phi_active:
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
    if theta_active:
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
 
    # One upper-volume constraint for each active material variable
    num_consts = int(rho_active) + int(phi_active)

    load_cases = fem_params["load_cases"]
    if len(load_cases) == 0:
        raise ValueError("fem_params['load_cases'] must not be empty.")
       
    num_rho_elems = rho_field.x.petsc_vec.array.size
    num_phi_elems = phi_field.x.petsc_vec.array.size

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

    # Use rho-specific passive zones when provided.
    rho_solid = opt.get("rho_solid_zone", opt["solid_zone"])(centers_rho)
    rho_void  = opt.get("rho_void_zone",  opt["void_zone"])(centers_rho)
    rho_ini = np.full(num_rho_elems, opt["vol_frac_rho"])

    # Initialize passive rho zones; they are re-enforced after each MMA update.
    rho_ini[rho_solid] = 1.0
    rho_ini[rho_void] = 0.05

    rho_min = np.full(num_rho_elems, 0.05)
    rho_max = np.ones(num_rho_elems)

    rho_field.x.petsc_vec.array[:] = np.clip(rho_ini, rho_min, rho_max)


    # Initialize magnetic fraction field (phi)
    centers_phi = phi_field.function_space.tabulate_dof_coordinates()[:num_phi_elems].T

    # Use phi-specific passive zones when provided.
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

    # Output files
    output_dir = os.path.abspath(opt["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Per-load-case BP writers
    sim_bp_writers = {}

    for lc in load_cases:
        lc_name = lc.get("name", "unnamed")
        fname = os.path.join(output_dir, f"optimized_design_{lc_name}.bp")

        sim_bp_writers[lc_name] = dolfinx.io.VTXWriter(
            fem_params["mesh"].comm,
            fname,
            [
                rho_phys_field,
                phi_eff_field,
                m_eff_field,
                u_field,
                sigma_vm_cg,
            ],

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

    # Start topology optimization
    opt_iter, beta, change = 0, 1, 2*opt["opt_tol"]
    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:
        opt_start_time = time.perf_counter()
        opt_iter += 1

        # Aggregated objective and sensitivities
        Obj_total = 0.0
        dJdrho_total = np.zeros_like(rho_field.x.petsc_vec.array)
        dJdphi_total = np.zeros_like(phi_field.x.petsc_vec.array)
        if theta_active:
            dJdtheta_total = np.zeros_like(opt["theta_field"].x.petsc_vec.array)


        # Volume values (load-independent; will be taken from first case)
        V_rho_value = None
        V_phi_value = None

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

            # Apply this load case's traction targets
            load_case_tractions = load_case.get("tractions", {})
            for bc_dict in fem_params["traction_bcs"]:
                name = bc_dict.get("name")
                bc_dict["traction_max"] = load_case_tractions.get(
                    name,
                    (0.0, 0.0),
                )

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

            u_array = u_field.x.array
            local_max_disp = float(np.max(np.abs(u_array)))
            max_disp = comm.allreduce(local_max_disp, op=MPI.MAX)
         
            # Print Displacement info (per load case)
            if comm.rank == 0:
                print(f"  [{lc_name}] max abs displacement: {max_disp:.4e}")

            function_values, gradients = sens_problem.evaluate()
            Obj_case, V_rho_case, V_phi_case = function_values

            dJdrho_phys = gradients["objective"]["rho"]
            dJdphi_phys = gradients["objective"]["phi"]
            dJdtheta_phys = gradients["objective"].get("theta")

            dVdrho_phys = gradients["volume"]["rho"]
            dVdphi_phys = gradients["volume"]["phi"]

            # Map physical-field gradients back to design variables
            dJdrho_design, dVdrho_design = backprop_rho([
                dJdrho_phys,
                dVdrho_phys,
            ])
            dJdphi_design, dVdphi_design = backprop_phi([
                dJdphi_phys,
                dVdphi_phys,
            ])

            if theta_active:
                dJdtheta_design = backprop_theta([
                    dJdtheta_phys,
                ])[0]

            # Load-case weight
            w_case = float(load_case.get("weight", 1.0))

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

        # ============================================================
        # Volume constraints and gradients
        # ============================================================
        g_list = []
        dgdx_rows = []

        if rho_active:
            g_list.append(
                V_rho_value / opt["vol_frac_rho"] - 1.0
            )

            row_parts = [
                dVdrho / opt["vol_frac_rho"],
            ]
            if phi_active:
                row_parts.append(np.zeros_like(dVdphi))
            if theta_active:
                row_parts.append(
                    np.zeros_like(theta_field.x.petsc_vec.array)
                )
            dgdx_rows.append(np.concatenate(row_parts))

        if phi_active:
            g_list.append(
                V_phi_value / opt["vol_frac_phi"] - 1.0
            )

            row_parts = []
            if rho_active:
                row_parts.append(np.zeros_like(dVdrho))
            row_parts.append(
                dVdphi / opt["vol_frac_phi"]
            )
            if theta_active:
                row_parts.append(
                    np.zeros_like(theta_field.x.petsc_vec.array)
                )
            dgdx_rows.append(np.concatenate(row_parts))

        g_vec = np.asarray(g_list, dtype=float)
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

        values = S_comm.gather(phi_eff_field)
        if comm.rank == 0 and opt_iter % opt["sim_image_output_interval"] == 0:
            plot_design(fem_params["mesh_serial"], values, str(opt_iter), opt["output_dir"], pv_or_image="image")
            
        if opt_iter % opt["sim_output_interval"] == 0:
            sim_file_xdmf_results.write_function(rho_phys_field, opt_iter)
            sim_file_xdmf_results.write_function(phi_eff_field, opt_iter)
            sim_file_xdmf_results.write_function(sigma_vm_field, opt_iter)
            if theta_active:
                sim_file_xdmf_results.write_function(m_eff_field, opt_iter)

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

        # Save the optimized remanence-direction field
        if theta_active:
            np.save(
                os.path.join(output_dir, "final_theta_phys.npy"),
                theta_phys_field.x.array,
            )

        print(f"Saved final phi_eff_field array to {opt['output_dir']}/final_phi_eff.npy")

        print(f"Saved final fields to {opt['output_dir']}:")
        print("  - final_rho_phys.npy")
        print("  - final_phi_eff.npy")
        print("  - final_phi_phys.npy")
        if theta_active:
            print("  - final_theta_phys.npy")

    # Close output files 
    sim_file_xdmf_results.close()

    for writer in sim_bp_writers.values():
        writer.close()
