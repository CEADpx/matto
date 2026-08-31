# Garai2025 isotropic MAP bridge optimization
# Bridge with clamped supports and central downward traction
# Optimize structural topology and magnetic-material placement to minimize compliance.

import sys
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.mesh import CellType, create_rectangle


repository_root = Path(__file__).resolve().parents[2]
modules_dir = repository_root / "modules"

if str(modules_dir) not in sys.path:
    sys.path.insert(0, str(modules_dir))

from topopt import topopt


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
        "active": True,
        "initial": 0.45,
        "bounds": (0.05, 1.00),
        "prescribed_value": 1.00,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
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
        # phi = 0 is Sylgard 184; phi = 1 is the 20% isotropic MAP.
        "active": True,
        "initial": 0.30,
        "bounds": (0.00, 1.00),
        "prescribed_value": 0.00,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
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

def build_free_energy(
    u_field,
    design_variables,
    stimuli,
):
    """Construct the plane-strain Garai2025 isotropic MAP energy."""
    rho_phys = design_variables["rho"].phys
    phi_phys = design_variables["phi"].phys

    h = stimuli["h"]

    C1_Ga = material_parameters["C1_Ga"]
    C2_Ga = material_parameters["C2_Ga"]
    a1 = material_parameters["a1"]
    a2 = material_parameters["a2"]
    b1 = material_parameters["b1"]
    b2 = material_parameters["b2"]

    C1_sil = material_parameters["C1_sil"]
    C2_sil = material_parameters["C2_sil"]

    K = material_parameters["K"]
    mu0 = material_parameters["mu0"]

    p_rho = material_parameters["p_rho"]
    eps_rho = material_parameters["eps_rho"]
    p_phi = material_parameters["p_phi"]

    I2 = ufl.Identity(2)
    F2 = ufl.variable(I2 + ufl.grad(u_field))

    F = ufl.as_tensor((
        (F2[0, 0], F2[0, 1], 0.0),
        (F2[1, 0], F2[1, 1], 0.0),
        (0.0,      0.0,      1.0),
    ))

    J = ufl.det(F)
    C = F.T * F
    Cbar = J**(-2.0 / 3.0) * C

    I1bar = ufl.tr(Cbar)
    I2bar = 0.5 * (
        ufl.tr(Cbar)**2
        - ufl.tr(Cbar * Cbar)
    )

    # h = mu0*|H| [T], while the Garai parameters use |H| [A/m].
    H_mag = h / mu0

    zeta_2 = b1 * ufl.ln(b2 * H_mag + 1.0)
    eta_2 = a1 * (ufl.exp(a2 * H_mag) - 1.0)
    C2_hat = zeta_2 * ufl.atan(eta_2 * H_mag)

    W_Ga = (
        0.5 * K * (J - 1.0)**2
        + C1_Ga * (I1bar - 3.0)
        + (C2_Ga + C2_hat) * (I2bar - 3.0)
    )

    W_sil = (
        0.5 * K * (J - 1.0)**2
        + C1_sil * (I1bar - 3.0)
        + C2_sil * (I2bar - 3.0)
    )

    rho_scale = (
        eps_rho
        + (1.0 - eps_rho) * rho_phys**p_rho
    )
    phi_scale = phi_phys**p_phi

    W = rho_scale * (
        phi_scale * W_Ga
        + (1.0 - phi_scale) * W_sil
    )

    return W, F2


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
        "solid_volume": {
            "form": rho_phys * dx,
            "normalize_by": domain_volume,
            "upper_bound": 0.45,
        },

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
    "move": 0.02,
}


output_options = {
    "output_dir": str(
        Path(__file__).resolve().parent
        / "results_bridge"
    ),
    "sim_output_interval": 20,
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
