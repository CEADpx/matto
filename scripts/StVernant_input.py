import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType
from fenitop.topopt import topopt

mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [90, 20]],
                        [360, 80], CellType.quadrilateral)
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(MPI.COMM_SELF, [[0, 0], [90, 20]],
                                   [360, 80], CellType.quadrilateral)
else:
    mesh_serial = None

fem_params = {  # FEM parameters
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "young's modulus": 100,
    "poisson's ratio": 0.25,
    "hyperelastic": True,   #linear if false
    "hyperModel": "stVernant",
    "disp_bc": lambda x: np.isclose(x[0], 0),
    "body_force": (0, 0),
    "quadrature_degree": 2,
}
if fem_params["hyperelastic"]:   
    fem_params["traction_bcs"] = [
        {
            "traction_max": (0.0, -.5),
            "on_boundary": lambda x: (np.isclose(x[0], 90) & np.greater(x[1], 8) & np.less(x[1], 12)),
        }
    ]
    fem_params["load_steps"] = 10
    fem_params["petsc_options"] = {
            "ksp_type": "cg",
            "pc_type": "gamg",
            "snes_max_it": "1000000",  # max Newton iterations for nonlinear solve
            "snes_error_if_not_converged": None,
    }

opt = {  # Topology optimization parameters
    "max_iter": 600,
    "opt_tol": 1e-5,
    "vol_frac": 0.5,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": 1.2,
    "beta_interval": 100,
    "beta_max": 128,
    "use_oc": False,
    "move": 0.005,
    "opt_compliance": True,
    "output_dir": "./stVernant_t5_results_res/",
    "sim_output_interval": 50,
    "sim_image_output_interval": 50
}

if __name__ == "__main__":
    topopt(fem_params, opt)
