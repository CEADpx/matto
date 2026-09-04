# Barrera2024 LCE V-pusher optimization
# V-pusher under LCE activation and downward output traction
# Jointly optimize structural topology, active-material placement, and mesogen direction to maximize upward output displacement.

from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.mesh import (
    CellType,
    create_rectangle,
    locate_entities_boundary,
    meshtags,
)

from matto.topopt import topopt
from material import make_build_free_energy

# ============================================================
#  GEOMETRY
# ============================================================

geometry = {
    "width": 12.0,
    "height": 8.0,
    "nx": 72,
    "ny": 48,

    # Passive attachment-pad dimensions
    "support_width": 2.0,
    "support_height": 2.0 / 3.0,
    "output_width": 2.0,
    "output_height": 2.0 / 3.0,
}

width = geometry["width"]
height = geometry["height"]

support_width = geometry["support_width"]
support_height = geometry["support_height"]

output_width = geometry["output_width"]
output_height = geometry["output_height"]
output_x_min = 0.5 * (width - output_width)
output_x_max = 0.5 * (width + output_width)
output_y_min = height - output_height

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

def output_pad(x):
    return (
        (x[0] >= output_x_min)
        & (x[0] <= output_x_max)
        & (x[1] >= output_y_min)
    )

def attachment_pads(x):
    """Union of the two supports and the top output pad."""
    return (
        left_support_pad(x)
        | right_support_pad(x)
        | output_pad(x)
    )

def initial_theta(x):
    """Mirrored, nonstationary initial director field: +/-30 degrees."""
    return np.where(
        x[0] <= 0.5 * width,
        np.pi / 6.0,
        -np.pi / 6.0,
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

# Serial copy used for plotting and gathered output.
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF,
        [[0.0, 0.0], [width, height]],
        [geometry["nx"], geometry["ny"]],
        cell_type=CellType.quadrilateral,
    )
else:
    mesh_serial = None

# Tag only the top boundary of the output pad for the displacement objective.
output_boundary_tag = 1
facet_dim = mesh.topology.dim - 1

output_facets = locate_entities_boundary(
    mesh,
    facet_dim,
    lambda x: (
        np.isclose(x[1], height)
        & (x[0] >= output_x_min)
        & (x[0] <= output_x_max)
    ),
)
output_facets = np.sort(output_facets)

output_facet_tags = meshtags(
    mesh,
    facet_dim,
    output_facets,
    np.full(
        output_facets.shape,
        output_boundary_tag,
        dtype=np.int32,
    ),
)

ds_output = ufl.Measure(
    "ds",
    domain=mesh,
    subdomain_data=output_facet_tags,
)

# ============================================================
#  MATERIAL AND INTERPOLATION PARAMETERS
# ============================================================

# All stress-like quantities use kPa.
material_parameters = {
    # Barrera et al. (2024) experimental LCE parameters
    "E_LCE": 9340.0,       # Young's modulus [kPa]
    "nu": 0.48,            # Poisson ratio [-]
    "beta": 575.0,         # Nematic coupling coefficient [kPa]
    "S0": 0.40,            # Reference order parameter [-]

    # Structural-density interpolation
    "p_rho": 3.0,
    "eps_rho": 1.0e-6,

    # Active/passive LCE interpolation
    "p_phi": 3.0,
}

# ============================================================
#  DESIGN-VARIABLE SPECIFICATIONS
# ============================================================

design_variables = {
    "rho": {
        "active": True,
        "initial": 0.50,
        "bounds": (0.05, 1.00),
        "prescribed_value": 1.00,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
        "operators": [
            {
                "type": "density_filter",
                "radius": 0.55,
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
                # Preserve solid support and output attachment pads.
                "where": attachment_pads,
                "value": 1.0,
            },
        ],
    },

    "phi": {
        # phi = 0 is passive/disordered LCE.
        # phi = 1 is fully programmed/aligned active LCE.
        "active": True,
        "initial": 0.50,
        "bounds": (0.00, 1.00),
        "prescribed_value": 0.00,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
        "operators": [
            {
                "type": "density_filter",
                "radius": 0.5,
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

        "active": True,
        "initial": initial_theta,
        "bounds": (-np.pi / 2.0, np.pi / 2.0),
        "prescribed_value": 0.0,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
        "operators": [
            {
                "type": "density_filter",
                "radius": 0.35,
            },
        ],
        "fixed_regions": [],
    },
}

# ============================================================
#  BOUNDARY CONDITIONS AND LOAD CASE
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
    "output_top": lambda x: (
        np.isclose(x[1], height)
        & (x[0] >= output_x_min)
        & (x[0] <= output_x_max)
    ),
}

load_steps = 50

load_cases = [
    {
        "name": "activated_pusher",
        "weight": 1.0,

        "body_force": (0.0, 0.0),

        "tractions": {
            "output_top": (0.0, -25.0),
        },

        "stimuli": {
            "activation": 1.0,
        },
    },
]

# ============================================================
#  FREE-ENERGY DENSITY
# ============================================================

build_free_energy = make_build_free_energy(material_parameters)

# ============================================================
#  OBJECTIVE
# ============================================================

def build_objective(
    u_field,
    external_work,
    dx,
):
    """Maximize average upward displacement of the output boundary."""
    # topopt minimizes. Apply all scalar normalization before ds creates the
    # completed UFL Form.
    return (
        -(1.0 / (output_width * height))
        * u_field[1]
        * ds_output(output_boundary_tag)
    )

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
    active_fraction_limit = 0.50  # volume fraction of LCE allowed in solid regions

    return {
        "rho_volume": {
            "form": rho_phys * dx,
            "normalize_by": domain_volume,
            "upper_bound": 0.50,
        },

        "active_lce_fraction": {
            "form": (
                rho_phys * phi_phys
                + active_fraction_limit * (1.0 - rho_phys)
            ) * dx,
            "normalize_by": domain_volume,
            "upper_bound": active_fraction_limit,
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

    active_lce = rho_phys * phi_phys

    director = ufl.as_vector((
        ufl.cos(theta_phys),
        ufl.sin(theta_phys),
    ))

    active_director = active_lce * director

    return {
        "active_lce": active_lce,
        "active_director": active_director,
    }

requested_output_fields = [
    "u",
    "rho_phys",
    "phi_phys",
    "theta_phys",
    "active_lce",
    "active_director",
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
    "max_iter": 150,
    "opt_tol": 1.0e-5,
    "move": 0.03,
}

output_options = {
    "output_dir": str(
        Path(__file__).resolve().parent
        / "results_pusher"
    ),
    "sim_output_interval": 25,
    "sim_image_output_interval": 151,
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
