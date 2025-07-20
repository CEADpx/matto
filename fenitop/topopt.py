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


def topopt(fem_params, opt):
    """Main function for topology optimization."""
    
    # Initialization
    comm = MPI.COMM_WORLD
    femProblem, u_field, lambda_field, rho_field, rho_phys_field, traction_constants, ds = form_fem(fem_params, opt)
    
    density_filter = DensityFilter(comm, rho_field, rho_phys_field,
                                    opt["filter_radius"], fem_params["petsc_options"])
    heaviside = Heaviside(rho_phys_field)
    sens_problem = Sensitivity(comm, opt, femProblem, u_field, lambda_field, rho_phys_field)
    S_comm = Communicator(rho_phys_field.function_space, fem_params["mesh_serial"])
    if comm.rank == 0:
        plotter = Plotter(fem_params["mesh_serial"])
    num_consts = 1 if opt["opt_compliance"] else 2
    num_elems = rho_field.x.petsc_vec.array.size
    if not opt["use_oc"]:
        rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
        low, upp = None, None
    
    # Apply passive zones
    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    rho_ini = np.full(num_elems, opt["vol_frac"])
    rho_ini[solid], rho_ini[void] = 0.995, 0.005
    rho_field.x.petsc_vec.array[:] = rho_ini
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01
    
    rho_min = np.full(num_elems, 0.01)
    rho_max = np.ones(num_elems)
    
    # sim file
    os.makedirs(opt["output_dir"], exist_ok=True)
    sim_filename = opt["output_dir"]+"optimized_design_sim_hist"
    sim_file_results = dolfinx.io.VTXWriter(fem_params["mesh"].comm, sim_filename + ".bp", \
                [rho_phys_field, u_field, lambda_field], \
                engine="BP4")
    sim_file_results.write(0)
    
    sim_file_xdmf_results = dolfinx.io.XDMFFile(fem_params["mesh"].comm, opt["output_dir"]+"optimized_design.xdmf", "w")
    sim_file_xdmf_results.write_mesh(fem_params["mesh"])
    sim_file_xdmf_results.write_function(rho_phys_field, 0)
    
    # Start topology optimization
    opt_iter, beta, change = 0, 1, 2*opt["opt_tol"]
    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:
        opt_start_time = time.perf_counter()
        opt_iter += 1
    
        # Density filter and Heaviside projection
        density_filter.forward()
        if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
            beta *= 2
            change = opt["opt_tol"] * 2
        heaviside.forward(beta)
        
        if fem_params["hyperelastic"]:
            N_steps = fem_params["load_steps"]
        
            # Start from zero traction each optimization iteration
            for t_const in traction_constants:
                t_const.value = np.zeros_like(t_const.value)
        
            # Incremental load stepping
            for step in range(1, N_steps + 1):
                for t_const, bc_dict in zip(traction_constants, fem_params["traction_bcs"]):
                    t_max = np.array(bc_dict["traction_max"], dtype=float)
                    t_const.value += (1.0 / N_steps) * t_max
        
                femProblem.solve_fem()
        
        else: # Solve linear FEM problem       
            femProblem.solve_fem() 
        
        # Get max displacement for output
        u_array = u_field.x.array
        max_disp = np.max(np.abs(u_array))
        
        # Compute function values and sensitivities
        [C_value, V_value, U_value], sensitivities = sens_problem.evaluate()
        heaviside.backward(sensitivities)
        [dCdrho, dVdrho, dUdrho] = density_filter.backward(sensitivities)
        
        # Pass optimization functions and constraints
        if opt["opt_compliance"]:
            g_vec = np.array([V_value-opt["vol_frac"]])
            dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])
        else:
            g_vec = np.array([V_value-opt["vol_frac"], C_value-opt["compliance_bound"]])
            dJdrho, dgdrho = dUdrho, np.vstack([dVdrho, dCdrho])
        rho_values = rho_field.x.petsc_vec.array.copy()
     
        if opt["opt_compliance"] and opt["use_oc"]:
            rho_new, change = optimality_criteria(
                rho_values, rho_min, rho_max, g_vec, dJdrho, dgdrho[0], opt["move"])
        else:
            rho_new, change, low, upp = mma_optimizer(
                num_consts, num_elems, opt_iter, rho_values, rho_min, rho_max,
                rho_old1, rho_old2, dJdrho, g_vec, dgdrho, low, upp, opt["move"])
            rho_old2 = rho_old1.copy()
            rho_old1 = rho_values.copy()
        rho_field.x.petsc_vec.array = rho_new.copy()
    
        # Output the histories
        opt_time = time.perf_counter() - opt_start_time
        if comm.rank == 0:
            print(f"opt_iter: {opt_iter}, opt_time: {opt_time:.3g} (s), "
                    f"beta: {beta}, C: {C_value:.3f}, V: {V_value:.3f}, "
                    f"U: {U_value:.3f}, change: {change:.3f}", flush=True)
        
        values = S_comm.gather(rho_phys_field)
        if comm.rank == 0 and opt_iter % opt["sim_image_output_interval"] == 0:
            plot_design(fem_params["mesh_serial"], values, str(opt_iter), opt["output_dir"], pv_or_image="image")
            
        if opt_iter % opt["sim_output_interval"] == 0:
            sim_file_xdmf_results.write_function(rho_phys_field, opt_iter)
            sim_file_results.write(opt_iter)
    
    # final output
    if comm.rank == 0:
        plot_design(fem_params["mesh_serial"], values, None, opt["output_dir"], pv_or_image="image")
        print(f"FINAL max abs displacement: {max_disp:.4e}")
        print(f"FINAL compliance: {C_value:.4f}")
    
        # Build a report 
        final_report = (
            f"FINAL max abs displacement: {max_disp:.4e}\n"
            f"FINAL compliance: {C_value:.4f}\n"
        )
    
        # Get traction info
        if fem_params["hyperelastic"]:
            traction = fem_params["traction_bcs"][0]["traction_max"]
        else:
            traction = fem_params["traction_bcs"][0][0]
    
        # Append traction info
        final_report += f"Max prescribed traction: {traction}\n"
    
        # Write to file
        with open(os.path.join(opt["output_dir"], "final_results.txt"), "w") as f:
            f.write(final_report)
    
    
    if opt_iter % opt["sim_output_interval"] != 0:
        sim_file_xdmf_results.write_function(rho_phys_field, opt_iter+1)
        sim_file_results.write(opt_iter+1)
    
    # close files
    #sim_file_xdmf_results.close()
    #sim_file_results.close()