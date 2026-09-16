# Garai2025 isotropic MAP cantilever optimization
# Cantilever under downward end traction; optimize magnetic-material placement to minimize field-on compliance.

from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import CellType, create_rectangle

from matto.driver import OptimizationDriver
from matto.design import volume_constraint
from matto.materials import MagnetoActiveElastomer

# ============================================================
#  MESH
# ============================================================

mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [100.0, 20.0]],
    [150, 30],
    cell_type=CellType.quadrilateral,
)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF,
        [[0.0, 0.0], [100.0, 20.0]],
        [150, 30],
        cell_type=CellType.quadrilateral,
    )
else:
    mesh_serial = None

# ============================================================
#  MATERIAL AND INTERPOLATION PARAMETERS
# ============================================================

# All stress-like quantities use kPa. The prescribed magnetic-field
# magnitude h represents mu0*|H| and uses T.
material_parameters = {
    # Garai and Haldar 20% isotropic MAP
    "C1_Ga": 108.0,
    "C2_Ga": 137.0,
    "a1": 4.97,
    "a2": 9.147 * 4.0 * np.pi * 1.0e-7,
    "b1": 174.685,
    "b2": 13.024 * 4.0 * np.pi * 1.0e-7,

    # Zero-particle Sylgard 184 silicone matrix
    "C1_sil": 270.0,
    "C2_sil": 10.8,

    # Common nearly incompressible volumetric penalty
    "K": 12000.0,
    "mu0": 4.0 * np.pi * 1.0e-7,

    # Material interpolation
    "p_rho": 3.0,
    "eps_rho": 1.0e-6,
    "p_phi": 1.0,
}

# ============================================================
#  DESIGN-VARIABLE SPECIFICATIONS
# ============================================================

design_variables = {
    "rho": {
        "active": False,
        "initial": 1.0,
        "bounds": (0.05, 1.00),
        "operators": [
            {
                "type": "density_filter",
                "radius": 1.0,
            },
        ],
    },

    "phi": {
        # phi = 0 is Sylgard 184; phi = 1 is the 20% isotropic MAP.
        "active": True,
        "initial": 0.30,
        "bounds": (0.00, 1.00),
        "prescribed_value": 0.00,
        "operators": [
            {
                "type": "density_filter",
                "radius": 1.0,
            },
        ],
    },
}

# ============================================================
#  BOUNDARY CONDITIONS AND LOAD CASES
# ============================================================

boundary_conditions = [
    {
        "name": "clamped_left",
        "on_boundary": lambda x: np.isclose(x[0], 0.0),
        "value": (0.0, 0.0),
    },
]

traction_boundaries = {
    "out_right": lambda x: np.isclose(x[0], 100.0),
}

load_steps = 50

load_cases = [
    {
        "name": "field_on",
        "weight": 1.0,
        "body_force": (0.0, 0.0),
        "tractions": {
            "out_right": (0.0, -0.50),
        },
        "stimuli": {
            "h": 0.45,
        },
    },
]

# ============================================================
#  FREE-ENERGY DENSITY
# ============================================================

material = MagnetoActiveElastomer(**material_parameters)

# ============================================================
#  OBJECTIVE
# ============================================================

def build_objective(
    u_field,
    external_work,
    dx,
):
    """Minimize compliance in the field-on load case."""
    return external_work

# ============================================================
#  CONSTRAINTS
# ============================================================

def build_constraints(
    design_variables,
    dx,
):
    phi_phys = design_variables["phi"].phys
    return {
        "magnetic_material_fraction": volume_constraint(phi_phys, 0.30, dx),
    }

# ============================================================
#  REQUESTED OUTPUT FIELDS
# ============================================================

def build_output_fields(
    design_variables,
):
    rho_phys = design_variables["rho"].phys
    phi_phys = design_variables["phi"].phys

    return {
        "magnetic_material": rho_phys * phi_phys,
    }

requested_output_fields = [
    "u",
    "rho_phys",
    "phi_phys",
    "magnetic_material",
]

# ============================================================
#  SOLVER, MMA, AND OUTPUT OPTIONS
# ============================================================

fem_options = {
    "quadrature_degree": 2,
    "solver_options": {
        "state": {
            "atol": 1.0e-4,
            "rtol": 1.0e-4,
            "max_it": 50,
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
            },
        },
    },
}

optimization_options = {
    "max_iter": 50,
    "opt_tol": 1.0e-5,
    "move": 0.03,
}

output_options = {
    "output_dir": str(
        Path(__file__).resolve().parent
        / "results_beam"
    ),
    "sim_output_interval": 10,
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
    "build_output_fields": build_output_fields,
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
