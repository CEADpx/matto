# Restorative beam optimization
# rho,  phi, and theta 
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.mesh import CellType, create_rectangle

from matto.topopt import topopt
from material import make_build_free_energy

# ============================================================
#  MESH
# ============================================================

# Simple 2D cantilever: 100 × 20 mm
mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [100.0, 20.0]],
    [150, 30],
    cell_type=CellType.quadrilateral,
)

# Serial copy used for plotting/gathered output
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

material_parameters = {     # Fixed parameters used to construct W
    # Matrix material
    "G0": 100.0,             # Base shear modulus [kPa]

    # Structural-density interpolation
    "p_rho": 3.0,
    "eps_rho": 1.0e-6,

    # hMSM magnetic parameters
    "mu0": 1.256e3,          # Vacuum permeability [mT^2/kPa]
    "B_rem_mag": 200.0,      # Remanent magnetic flux density [mT]
}

# ============================================================
#  1. DESIGN-VARIABLE SPECIFICATIONS
# ============================================================

design_variables = {
    "rho": {
        "active": True,

        # MMA initialization and bounds
        "initial": 0.50,
        "bounds": (0.05, 1.00),

        # Value used if active=False
        "prescribed_value": 1.00,

        # Optimization and physical-field spaces
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),

        # Applied in the listed order:
        # rho.raw -> density filter -> Heaviside -> rho.phys
        "operators": [
            {
                "type": "density_filter",
                "radius": 1.0,
            },
            {
                "type": "heaviside",
                "beta_initial": 1.0,
                "beta_update_interval": 25,
                "beta_max": 4.0,
            },
        ],

        # Optional regions where the raw variable is fixed
        "fixed_regions": [],
    },

    "phi": {
        # Magnetic particle volume fraction
        "active": True,

        "initial": 0.10,
        "bounds": (0.00, 0.30),

        # Nonmagnetic material if phi is inactive
        "prescribed_value": 0.00,

        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),

        # phi.raw -> density filter -> phi.phys
        "operators": [
            {
                "type": "density_filter",
                "radius": 1.0,
            },
        ],

        "fixed_regions": [],
    },

    "theta": {
        # Remanent-magnetization direction angle
        "active": True,

        # theta = 0 corresponds to the +x direction
        "initial": 0.0,
        "bounds": (-np.pi, np.pi),

        # Prescribed +x direction if theta is inactive
        "prescribed_value": 0.0,

        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),

        # theta.raw -> density filter -> theta.phys
        "operators": [
            {
                "type": "density_filter",
                "radius": 1.0,
            },
        ],

        "fixed_regions": [],
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
    "out_right": lambda x: np.isclose(x[0], 100.0),
}

load_steps = 50

load_cases = [
    {
        "name": "traction_down_B_up",   # Load case name
        "weight": 1.0,

        "body_force": (0.0, 0.0),

        "tractions": {
            "out_right": (0.0, -0.50),
        },

        "stimuli": {
            "B_app": (0.0, 25.0),
        },
    },

    {
        "name": "traction_up_B_down",
        "weight": 1.0,

        "body_force": (0.0, 0.0),

        "tractions": {
            "out_right": (0.0, 0.50),
        },

        "stimuli": {
            "B_app": (0.0, -25.0),
        },
    },
]

# ============================================================
#  5. FREE-ENERGY DENSITY
# ============================================================

build_free_energy = make_build_free_energy(material_parameters)

# ============================================================
#  6. OBJECTIVE
# ============================================================

def build_objective(
    u_field,     # keep for displacement based objectives later
    external_work, 
    dx,
):
    """Return the compliance objective."""
    return external_work

# ============================================================
#  7. CONSTRAINTS
# ============================================================

def build_constraints(
    design_variables,
    dx,
):
    rho_phys = design_variables["rho"].phys
    phi_phys = design_variables["phi"].phys

    domain_volume = 1.0 * dx

    return {
        "rho_volume": {
            "form": rho_phys * dx,
            "normalize_by": domain_volume,
            "upper_bound": 0.50,
        },

        "phi_volume": {
            "form": phi_phys * dx,
            "normalize_by": domain_volume,
            "upper_bound": 0.10,
        },
    }

# ============================================================
#  8. REQUESTED OUTPUT FIELDS
# ============================================================

def build_output_fields(
    design_variables,
):
    rho_phys = design_variables["rho"].phys
    phi_phys = design_variables["phi"].phys
    theta_phys = design_variables["theta"].phys

    phi_eff = rho_phys * phi_phys

    m_eff = phi_eff * ufl.as_vector((
        ufl.cos(theta_phys),
        ufl.sin(theta_phys),
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
#  9. SOLVER AND MMA OPTIONS
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
    "max_iter": 100,
    "opt_tol": 1.0e-5,
    "move": 0.05,
}

output_options = {
    "output_dir": str(
        Path(__file__).resolve().parent
        / "results_Cantilever_TractionDown_Bup_MooneyN2"
    ),

    "sim_output_interval": 25,
    "sim_image_output_interval": 101,
}

# ============================================================
#  COMPLETE PROBLEM DEFINITION
# ============================================================

problem = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,

    "material_parameters": material_parameters,
    "design_variables": design_variables,

    "boundary_conditions": boundary_conditions,
    "traction_boundaries": traction_boundaries,

    "load_steps": load_steps,
    "load_cases": load_cases,

    "build_free_energy": build_free_energy,
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
    topopt(problem)