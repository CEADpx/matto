# Akbari2021 anisotropic MRE bridge optimization
# Bridge with clamped supports and central downward traction; 
# Jointly optimize structural topology, magnetic-material placement, and particle-chain direction to minimize compliance.

from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.mesh import CellType, create_rectangle

from matto.driver import OptimizationDriver
from matto.design import volume_constraint
from matto.materials import AnisotropicMagnetoActiveElastomer

# ============================================================
#  GEOMETRY
# ============================================================

geometry = {
    "width": 120.0,
    "height": 50.0,
    "nx": 96,
    "ny": 40,
    "support_width": 12.0,
    "support_height": 4.0,
    "load_width": 10.0,
    "load_height": 4.0,
}

width = geometry["width"]
height = geometry["height"]

support_width = geometry["support_width"]
support_height = geometry["support_height"]

load_width = geometry["load_width"]
load_height = geometry["load_height"]
load_x_min = 0.5 * (width - load_width)
load_x_max = 0.5 * (width + load_width)
load_y_min = height - load_height

def left_support_pad(x):
    return (
        (x[0] <= support_width)
        & (x[1] <= support_height)
    )

def right_support_pad(x):
    return (
        (x[0] >= width - support_width)
        & (x[1] <= support_height)
    )

def load_pad(x):
    return (
        (x[0] >= load_x_min)
        & (x[0] <= load_x_max)
        & (x[1] >= load_y_min)
    )

def attachment_pads(x):
    return (
        left_support_pad(x)
        | right_support_pad(x)
        | load_pad(x)
    )

def initial_theta(x):
    return np.where(
        x[0] <= 0.5 * width,
        np.deg2rad(15.0),
        np.deg2rad(-15.0),
    )

# ============================================================
#  MESH
# ============================================================

mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [width, height]],
    [geometry["nx"], geometry["ny"]],
    cell_type=CellType.quadrilateral,
)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF,
        [[0.0, 0.0], [width, height]],
        [geometry["nx"], geometry["ny"]],
        cell_type=CellType.quadrilateral,
    )
else:
    mesh_serial = None

# ============================================================
#  MATERIAL AND INTERPOLATION PARAMETERS
# ============================================================

# All stress-like quantities use kPa. The magnetic-field magnitude h uses T.
material_parameters = {
    # Akbari 20% anisotropic MRE
    "A_Ak": 326.75,
    "a_Ak": 2.785,
    "b_Ak": 3.40,
    "r": 64.02,
    "s": 318.16,
    "hs": 0.43,
    "K_Ak": 16000.0,

    # Zero-particle silicone matrix
    "A_sil": 122.0,
    "a_sil": 0.28,
    "b_sil": 0.33,
    "K_sil": 6000.0,

    # Prescribed horizontal magnetic-field direction
    "theta_M": 0.0,
    "delta_theta": 1.0e-6,

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
        "active": True,
        "initial": 0.45,
        "bounds": (0.05, 1.00),
        "prescribed_value": 1.00,
        "operators": [
            {
                "type": "density_filter",
                "radius": 2.0,
            },
            {
                "type": "heaviside",
                "beta_initial": 1.0,
                "beta_update_interval": 25,
                "beta_max": 4.0,
            },
        ],
        "fixed_regions": [
            {
                "where": attachment_pads,
                "value": 1.0,
            },
        ],
    },

    "phi": {
        # phi = 0 is silicone; phi = 1 is the 20% anisotropic MRE.
        "active": True,
        "initial": 0.30,
        "bounds": (0.00, 1.00),
        "prescribed_value": 0.00,
        "operators": [
            {
                "type": "density_filter",
                "radius": 2.0,
            },
            {
                "type": "heaviside",
                "beta_initial": 1.0,
                "beta_update_interval": 25,
                "beta_max": 4.0,
            },
        ],
        "fixed_regions": [
            {
                "where": attachment_pads,
                "value": 0.0,
            },
        ],
    },

    "theta": {
        # Mirrored +/-15-degree initialization preserves bridge symmetry.
        "active": True,
        "initial": initial_theta,
        "bounds": (-np.pi / 2.0, np.pi / 2.0),
        "prescribed_value": 0.0,
        "operators": [
            {
                "type": "density_filter",
                "radius": 1.5,
            },
        ],
    },
}

# ============================================================
#  BOUNDARY CONDITIONS AND LOAD CASES
# ============================================================

boundary_conditions = [
    {
        "name": "clamped_bottom_supports",
        "on_boundary": lambda x: (
            np.isclose(x[1], 0.0)
            & (
                (x[0] <= support_width)
                | (x[0] >= width - support_width)
            )
        ),
        "value": (0.0, 0.0),
    },
]

traction_boundaries = {
    "top_center": lambda x: (
        np.isclose(x[1], height)
        & (x[0] >= load_x_min)
        & (x[0] <= load_x_max)
    ),
}

load_steps = 25

load_cases = [
    {
        "name": "field_on",
        "weight": 1.0,
        "body_force": (0.0, 0.0),
        "tractions": {
            "top_center": (0.0, -1.0),
        },
        "stimuli": {
            "h": 0.45,
        },
    },
]

# ============================================================
#  FREE-ENERGY DENSITY
# ============================================================

material = AnisotropicMagnetoActiveElastomer(**material_parameters)

# ============================================================
#  OBJECTIVE
# ============================================================

def build_objective(
    u_field,
    external_work,
    dx,
):
    """Minimize field-on compliance."""
    return external_work

# ============================================================
#  CONSTRAINTS
# ============================================================

def build_constraints(
    design_variables,
    dx,
):
    rho_phys = design_variables["rho"].phys
    phi_phys = design_variables["phi"].phys

    domain_volume = 1.0 * dx
    magnetic_fraction_limit = 0.30

    return {
        "solid_volume": volume_constraint(rho_phys, 0.45, dx),

        "magnetic_fraction_of_solid": {
            "form": (
                rho_phys * phi_phys
                + magnetic_fraction_limit * (1.0 - rho_phys)
            ) * dx,
            "normalize_by": domain_volume,
            "upper_bound": magnetic_fraction_limit,
        },
    }

# ============================================================
#  REQUESTED OUTPUT FIELDS
# ============================================================

def build_output_fields(
    design_variables,
):
    rho_phys = design_variables["rho"].phys
    phi_phys = design_variables["phi"].phys
    theta_phys = design_variables["theta"].phys

    magnetic_material = rho_phys * phi_phys

    particle_chain = magnetic_material * ufl.as_vector((
        ufl.cos(theta_phys),
        ufl.sin(theta_phys),
    ))

    field_alignment = magnetic_material * ufl.cos(theta_phys)

    return {
        "magnetic_material": magnetic_material,
        "particle_chain": particle_chain,
        "field_alignment": field_alignment,
    }

requested_output_fields = [
    "u",
    "rho_phys",
    "phi_phys",
    "theta_phys",
    "magnetic_material",
    "particle_chain",
    "field_alignment",
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
                "ksp_rtol": 1.0e-10,
            },
        },
    },
}

optimization_options = {
    "max_iter": 100,
    "opt_tol": 1.0e-5,
    "move": 0.02,
}

output_options = {
    "output_dir": str(
        Path(__file__).resolve().parent
        / "results_bridge"
    ),
    "sim_output_interval": 20,
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
