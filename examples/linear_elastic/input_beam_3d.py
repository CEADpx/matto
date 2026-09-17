# Linear-elastic cantilever, 3D
# Compliance minimization with the density as the only design field, set
# up as the 3D cantilever in FEniTop so the two codes can be run side by
# side: a box supported on two strips of one end face and loaded on a
# small patch at the centre of the other. The mesh is coarser than
# FEniTop's so the run takes hours rather than days serially; at this
# spacing the load patch has to be located with closed inequalities, or
# no facet has all its vertices inside it. The move limit is the value
# FEniTop's MMA runs with, not the one in its script, which lands in the
# a0 slot.
#
# The state and adjoint use MUMPS: CG/GAMG did not reach the residual
# reduction the Newton check requires. The filter keeps CG/GAMG.
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import CellType, create_box

from matto.driver import OptimizationDriver
from matto.design import volume_constraint
from matto.materials import LinearElastic

# ============================================================
#  MESH
# ============================================================

# Box 10 x 30 x 10
CELLS = [20, 60, 20]

mesh = create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [10.0, 30.0, 10.0]],
    CELLS,
    cell_type=CellType.hexahedron,
)

# Serial copy used for the gathered final arrays
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_box(
        MPI.COMM_SELF,
        [[0.0, 0.0, 0.0], [10.0, 30.0, 10.0]],
        CELLS,
        cell_type=CellType.hexahedron,
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
        "initial": 0.08,
        "bounds": (0.0, 1.0),
        "operators": [
            {
                "type": "density_filter",
                "radius": 0.6,
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
        "name": "supports",
        "on_boundary": lambda x: (
            np.isclose(x[1], 0.0) & (np.less(x[0], 1.5) | np.greater(x[0], 8.5))
        ),
        "value": (0.0, 0.0, 0.0),
    },
]

traction_boundaries = {
    "tip_patch": lambda x: (
        np.isclose(x[1], 30.0)
        & np.greater_equal(x[0], 4.5) & np.less_equal(x[0], 5.5)
        & np.greater_equal(x[2], 4.5) & np.less_equal(x[2], 5.5)
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

        "body_force": (0.0, 0.0, 0.0),

        "tractions": {
            "tip_patch": (0.0, 0.0, -2.0),
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
        "volume": volume_constraint(rho_phys, 0.08, dx),
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
                "pc_factor_mat_solver_type": "mumps",
            },
        },
        "adjoint": {
            "rtol": 1.0e-8,
            "atol": 1.0e-12,
            "petsc_options": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        },
        "filter": {
            "petsc_options": {
                "ksp_type": "cg",
                "pc_type": "gamg",
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
        Path(__file__).resolve().parent / "results_beam_3d"
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
