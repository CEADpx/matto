# Restorative beam in 3D
# rho, phi and theta all active. theta is the in-plane (x-y) angle of the
# remanent magnetization; the applied field is out of plane so the
# magnetic torque bends the beam in z, against the traction.
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.mesh import CellType, create_box

from matto.driver import OptimizationDriver
from matto.design import volume_constraint
from matto.materials import HardMagneticSoftMaterial

# ============================================================
#  MESH
# ============================================================

# Cantilever 40 x 8 x 8 mm
LENGTH, HEIGHT, DEPTH = 40.0, 8.0, 8.0
CELLS = [24, 6, 6]

mesh = create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [LENGTH, HEIGHT, DEPTH]],
    CELLS,
    cell_type=CellType.hexahedron,
)

# Serial copy used for the gathered final arrays
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_box(
        MPI.COMM_SELF,
        [[0.0, 0.0, 0.0], [LENGTH, HEIGHT, DEPTH]],
        CELLS,
        cell_type=CellType.hexahedron,
    )
else:
    mesh_serial = None

# ============================================================
#  MATERIAL AND INTERPOLATION PARAMETERS
# ============================================================

material_parameters = {
    "G0": 100.0,             # Base shear modulus [kPa]
    "p_rho": 3.0,
    "eps_rho": 1.0e-6,
    "mu0": 1.256e3,          # Vacuum permeability [mT^2/kPa]
    "B_rem_mag": 200.0,      # Remanent magnetic flux density [mT]
    "dim": 3,
}

# ============================================================
#  1. DESIGN-VARIABLE SPECIFICATIONS
# ============================================================

design_variables = {
    "rho": {
        "active": True,
        "initial": 0.50,
        "bounds": (0.05, 1.00),
        "prescribed_value": 1.00,
        "operators": [
            {"type": "density_filter", "radius": 1.0},
            {
                "type": "heaviside",
                "beta_initial": 1.0,
                "beta_update_interval": 25,
                "beta_max": 4.0,
            },
        ],
    },
    "phi": {
        "active": True,
        "initial": 0.10,
        "bounds": (0.00, 0.30),
        "prescribed_value": 0.00,
        "operators": [
            {"type": "density_filter", "radius": 1.0},
        ],
    },
    "theta": {
        "active": True,
        "initial": 0.0,
        "bounds": (-np.pi, np.pi),
        "operators": [
            {"type": "density_filter", "radius": 1.0},
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
        "value": (0.0, 0.0, 0.0),
    },
]

traction_boundaries = {
    "out_right": lambda x: np.isclose(x[0], LENGTH),
}

# ============================================================
#  3. LOAD STEPS AND LOAD CASES
# ============================================================

load_steps = 25

load_cases = [
    {
        "name": "traction_down_B_up",
        "weight": 1.0,
        "body_force": (0.0, 0.0, 0.0),
        "tractions": {"out_right": (0.0, 0.0, -0.50)},
        "stimuli": {"B_app": (0.0, 0.0, 25.0)},
    },
    {
        "name": "traction_up_B_down",
        "weight": 1.0,
        "body_force": (0.0, 0.0, 0.0),
        "tractions": {"out_right": (0.0, 0.0, 0.50)},
        "stimuli": {"B_app": (0.0, 0.0, -25.0)},
    },
]

# ============================================================
#  4. MATERIAL
# ============================================================

material = HardMagneticSoftMaterial(**material_parameters)

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
    phi_phys = design_variables["phi"].phys

    return {
        "rho_volume": volume_constraint(rho_phys, 0.50, dx),
        "phi_volume": volume_constraint(phi_phys, 0.10, dx),
    }

# ============================================================
#  7. REQUESTED OUTPUT FIELDS
# ============================================================

def build_output_fields(design_variables):
    rho_phys = design_variables["rho"].phys
    phi_phys = design_variables["phi"].phys
    theta_phys = design_variables["theta"].phys

    phi_eff = rho_phys * phi_phys
    m_eff = phi_eff * ufl.as_vector((
        ufl.cos(theta_phys),
        ufl.sin(theta_phys),
        0.0,
    ))

    return {
        "phi_eff": phi_eff,
        "m_eff": m_eff,
    }

requested_output_fields = [
    "u",
    "rho_phys",
    "phi_phys",
    "theta_phys",
    "phi_eff",
    "m_eff",
]

# ============================================================
#  8. SOLVER AND MMA OPTIONS
# ============================================================

fem_options = {
    "quadrature_degree": 2,
    "solver_options": {
        "state": {
            "atol": 1.0e-4,
            "rtol": 1.0e-4,
            "max_it": 50,
            "petsc_options": {"ksp_type": "preonly", "pc_type": "lu"},
        },
        "adjoint": {
            "rtol": 1.0e-8,
            "atol": 1.0e-12,
            "petsc_options": {"ksp_type": "preonly", "pc_type": "lu"},
        },
        "filter": {
            "petsc_options": {"ksp_type": "cg", "pc_type": "gamg"},
        },
    },
}

optimization_options = {
    "max_iter": 100,
    "opt_tol": 1.0e-5,
    "move": 0.05,
}

output_options = {
    "output_dir": str(
        Path(__file__).resolve().parent / "results_Cantilever3D_TractionDown_Bup"
    ),
    "sim_output_interval": 25,
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
