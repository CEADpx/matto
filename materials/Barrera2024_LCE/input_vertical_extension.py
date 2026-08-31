# Barrera2024 LCE vertical-extension validation
# Clamped solid LCE strip under activation and no mechanical load
# Optimize active-material placement and mesogen direction to maximize average upward top-edge displacement.

import sys
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


# Make the repository's modules directory importable when this file is stored
# under materials/Barrera2024/.
repository_root = Path(__file__).resolve().parents[2]
modules_dir = repository_root / "modules"

if str(modules_dir) not in sys.path:
    sys.path.insert(0, str(modules_dir))

from topopt import topopt


# ============================================================
#  GEOMETRY AND MESH
# ============================================================

geometry = {
    "width": 2.0,
    "height": 10.0,
    "nx": 20,
    "ny": 100,
}

width = geometry["width"]
height = geometry["height"]

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


# Tag the top edge so the objective can use its exact boundary average.
top_boundary_tag = 1
facet_dim = mesh.topology.dim - 1

top_facets = locate_entities_boundary(
    mesh,
    facet_dim,
    lambda x: np.isclose(x[1], height),
)
top_facets = np.sort(top_facets)

top_facet_tags = meshtags(
    mesh,
    facet_dim,
    top_facets,
    np.full(top_facets.shape, top_boundary_tag, dtype=np.int32),
)

ds_top = ufl.Measure(
    "ds",
    domain=mesh,
    subdomain_data=top_facet_tags,
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
        # The complete rectangle is fixed solid material.
        "active": False,
        "initial": 1.0,
        "bounds": (0.05, 1.0),
        "prescribed_value": 1.0,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
        "operators": [
            {
                "type": "density_filter",
                "radius": 0.20,
            },
        ],
        "fixed_regions": [],
    },

    "phi": {
        # phi = 0: passive/disordered LCE.
        # phi = 1: fully programmed/aligned active LCE.
        "active": True,
        "initial": 0.50,
        "bounds": (0.00, 1.00),
        "prescribed_value": 1.00,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
        "operators": [
            {
                "type": "density_filter",
                "radius": 0.20,
            },
            {
                "type": "heaviside",
                "beta_initial": 1.0,
                "beta_update_interval": 25,
                "beta_max": 4.0,
            },
        ],
        "fixed_regions": [],
    },

    "theta": {
        "active": True,
        "initial": np.pi / 12.0,
        "bounds": (-np.pi / 2.0, np.pi / 2.0),
        "prescribed_value": 0.0,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
        "operators": [
            {
                "type": "density_filter",
                "radius": 0.20,
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
        "name": "clamped_bottom",
        "on_boundary": lambda x: np.isclose(x[1], 0.0),
        "value": (0.0, 0.0),
    },
]

traction_boundaries = {}

load_steps = 40

load_cases = [
    {
        "name": "full_activation",
        "weight": 1.0,
        "body_force": (0.0, 0.0),
        "tractions": {},
        "stimuli": {
            # activation = 0 gives S = S0 and zero coupling.
            # activation = 1 gives S = 0 and maximum coupling.
            "activation": 1.0,
        },
    },
]


# ============================================================
#  FREE-ENERGY DENSITY
# ============================================================

def build_free_energy(
    u_field,
    design_variables,
    stimuli,
):
    """Construct the 2D plane-strain Barrera2024 LCE energy."""
    rho_phys = design_variables["rho"].phys
    phi_phys = design_variables["phi"].phys
    theta_phys = design_variables["theta"].phys

    activation = stimuli["activation"]

    E_LCE = material_parameters["E_LCE"]
    nu = material_parameters["nu"]
    beta = material_parameters["beta"]
    S0 = material_parameters["S0"]

    p_rho = material_parameters["p_rho"]
    eps_rho = material_parameters["eps_rho"]
    p_phi = material_parameters["p_phi"]

    mu = E_LCE / (2.0 * (1.0 + nu))
    lam = (
        E_LCE * nu
        / ((1.0 + nu) * (1.0 - 2.0 * nu))
    )

    I = ufl.Identity(2)
    F = ufl.variable(I + ufl.grad(u_field))
    C = F.T * F
    E_GL = 0.5 * (C - I)

    # Programmed in-plane mesogen director.
    director = ufl.as_vector((
        ufl.cos(theta_phys),
        ufl.sin(theta_phys),
    ))

    Q_nem = 3.0 * ufl.outer(director, director) - I

    # The solver ramps activation upward from zero. This corresponds to
    # decreasing the order parameter from S0 to zero:
    #     S = S0 * (1 - activation)
    #     delta_S = S0 - S = S0 * activation
    delta_S = S0 * activation

    W_passive = (
        0.5 * lam * ufl.tr(E_GL)**2
        + mu * ufl.inner(E_GL, E_GL)
    )

    W_coupling = (
        -0.5
        * beta
        * delta_S
        * ufl.inner(Q_nem, E_GL)
    )

    rho_scale = (
        eps_rho
        + (1.0 - eps_rho) * rho_phys**p_rho
    )
    phi_scale = phi_phys**p_phi

    W = rho_scale * (
        W_passive
        + phi_scale * W_coupling
    )

    return W, F


# ============================================================
#  OBJECTIVE
# ============================================================

def build_objective(
    u_field,
    external_work,
    dx,
):
    """Maximize the average vertical displacement of the top edge."""
    return (
        -(1.0 / (width * height))
        * u_field[1]
        * ds_top(top_boundary_tag)
    )


# ============================================================
#  CONSTRAINTS
# ============================================================

def build_constraints(
    design_variables,
    dx,
):
    """Allow phi to fill the domain while satisfying the MMA interface."""
    phi_phys = design_variables["phi"].phys
    domain_volume = 1.0 * dx

    return {
        "phi_volume": {
            "form": phi_phys * dx,
            "normalize_by": domain_volume,
            "upper_bound": 1.0,
        },
    }


# ============================================================
#  REQUESTED OUTPUT FIELDS
# ============================================================

def build_output_fields(
    design_variables,
):
    phi_phys = design_variables["phi"].phys
    theta_phys = design_variables["theta"].phys

    director = ufl.as_vector((
        ufl.cos(theta_phys),
        ufl.sin(theta_phys),
    ))

    # Scale the director by phi so passive regions have zero active direction
    # in visualization output.
    active_director = phi_phys * director

    return {
        "active_director": active_director,
    }


requested_output_fields = [
    "u",
    "rho_phys",
    "phi_phys",
    "theta_phys",
    "active_director",
]


# ============================================================
#  SOLVER, MMA, AND OUTPUT OPTIONS
# ============================================================

fem_options = {
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "snes_max_it": "500",
        "snes_error_if_not_converged": None,
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
        / "results_vertical_extension"
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
