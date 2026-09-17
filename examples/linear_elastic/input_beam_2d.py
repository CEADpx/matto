# Linear-elastic cantilever, 2D
# Compliance minimization with the density as the only design field, set
# up as the 2D cantilever in FEniTop so the two codes can be run side by
# side. The move limit is the value FEniTop's MMA runs with, not the one
# in its script, which lands in the a0 slot.

from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import CellType, create_rectangle

from matto.driver import OptimizationDriver
from matto.design import volume_constraint
from matto.materials import LinearElastic

# ============================================================
#  MESH
# ============================================================

# Cantilever 60 x 20
mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [60.0, 20.0]],
    [200, 60],
    cell_type=CellType.quadrilateral,
)

# Serial copy used for the gathered final arrays
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF,
        [[0.0, 0.0], [60.0, 20.0]],
        [200, 60],
        cell_type=CellType.quadrilateral,
    )
else:
    mesh_serial = None

# ============================================================
#  MATERIAL AND INTERPOLATION PARAMETERS
# ============================================================

material_parameters = {
    "E": 100.0,              # Young's modulus
    "nu": 0.25,              # Poisson's ratio (plane strain in 2D)

    # Structural-density interpolation
    "p_rho": 3.0,
    "eps_rho": 1.0e-6,
}

# ============================================================
#  1. DESIGN-VARIABLE SPECIFICATIONS
# ============================================================

design_variables = {
    "rho": {
        "active": True,
        "initial": 0.5,
        "bounds": (0.0, 1.0),
        "operators": [
            {
                "type": "density_filter",
                "radius": 1.2,
            },
            {
                "type": "heaviside",
                "beta_initial": 1.0,
                "beta_update_interval": 50,
                "beta_max": 128.0,
            },
        ],
    },
}

# ============================================================
#  2. BOUNDARY CONDITIONS
# ============================================================

boundary_conditions = [
    {
        "name": "clamped_left",
        "on_boundary": lambda x: np.isclose(x[0], 0.0),
        "value": (0.0, 0.0),
    },
]

traction_boundaries = {
    "tip_patch": lambda x: (
        np.isclose(x[0], 60.0) & np.greater(x[1], 8.0) & np.less(x[1], 12.0)
    ),
}

# ============================================================
#  3. LOAD STEPS AND LOAD CASES
# ============================================================

load_steps = 1

load_cases = [
    {
        "name": "tip_load",
        "weight": 1.0,

        "body_force": (0.0, 0.0),

        "tractions": {
            "tip_patch": (0.0, -0.2),
        },

        "stimuli": {},
    },
]

# ============================================================
#  4. MATERIAL
# ============================================================

material = LinearElastic(**material_parameters)

# ============================================================
#  5. OBJECTIVE
# ============================================================

def build_objective(u_field, external_work, dx):
    """Compliance."""
    return external_work

# ============================================================
#  6. CONSTRAINTS
# ============================================================

def build_constraints(design_variables, dx):
    rho_phys = design_variables["rho"].phys

    return {
        "volume": volume_constraint(rho_phys, 0.5, dx),
    }

# ============================================================
#  7. REQUESTED OUTPUT FIELDS
# ============================================================

requested_output_fields = [
    "u",
    "rho_phys",
]

# ============================================================
#  8. SOLVER AND MMA OPTIONS
# ============================================================

fem_options = {
    "quadrature_degree": 2,
    "solver_options": {
        "state": {
            "atol": 1.0e-8,
            "rtol": 1.0e-8,
            "max_it": 10,
            "petsc_options": {
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        },
        "adjoint": {
            "rtol": 1.0e-8,
            "atol": 1.0e-12,
            "petsc_options": {
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        },
        "filter": {
            "petsc_options": {
                "ksp_type": "cg",
                "pc_type": "gamg",
                "ksp_rtol": 1.0e-10,
            },
        },
    },
}

optimization_options = {
    "max_iter": 400,
    "opt_tol": 1.0e-5,
    "move": 0.05,
}

output_options = {
    "output_dir": str(
        Path(__file__).resolve().parent / "results_beam_2d"
    ),

    "sim_output_interval": 50,
}

# ============================================================
#  COMPLETE PROBLEM DEFINITION
# ============================================================

problem = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "comm": mesh.comm,

    "material_parameters": material_parameters,
    "design_variables": design_variables,

    "boundary_conditions": boundary_conditions,
    "traction_boundaries": traction_boundaries,

    "load_steps": load_steps,
    "load_cases": load_cases,

    "material": material,
    "build_objective": build_objective,
    "build_constraints": build_constraints,

    "requested_output_fields": requested_output_fields,

    "fem_options": fem_options,
    "optimization_options": optimization_options,
    "output_options": output_options,
}

# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    OptimizationDriver(problem).run()
