# Akbari2021 anisotropic MRE cantilever optimization
# Cantilever under downward end traction
# Optimize magnetic-material placement and particle-chain direction to minimize field-on compliance.

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
        "active": False,
        "initial": 1.0,
        "bounds": (0.05, 1.00),
        "prescribed_value": 1.0,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
        "operators": [
            {
                "type": "density_filter",
                "radius": 1.0,
            },
        ],
        "fixed_regions": [],
    },

    "phi": {
        # phi = 0 is silicone; phi = 1 is the 20% anisotropic MRE.
        "active": True,
        "initial": 0.30,
        "bounds": (0.00, 1.00),
        "prescribed_value": 0.00,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
        "operators": [
            {
                "type": "density_filter",
                "radius": 1.0,
            },
        ],
        "fixed_regions": [],
    },

    "theta": {
        # Particle-chain direction; initialized 15 degrees above +x.
        "active": True,
        "initial": np.deg2rad(15.0),
        "bounds": (-np.pi / 2.0, np.pi / 2.0),
        "prescribed_value": 0.0,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
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
    {
        # Diagnostic comparison only; this case does not affect the objective.
        "name": "field_off",
        "weight": 0.0,
        "body_force": (0.0, 0.0),
        "tractions": {
            "out_right": (0.0, -0.50),
        },
        "stimuli": {
            "h": 0.0,
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
    """Construct the plane-strain Akbari2021 MRE energy."""
    rho_phys = design_variables["rho"].phys
    phi_phys = design_variables["phi"].phys
    theta_phys = design_variables["theta"].phys

    h = stimuli["h"]

    A_Ak = material_parameters["A_Ak"]
    a_Ak = material_parameters["a_Ak"]
    b_Ak = material_parameters["b_Ak"]
    r = material_parameters["r"]
    s = material_parameters["s"]
    hs = material_parameters["hs"]
    K_Ak = material_parameters["K_Ak"]

    A_sil = material_parameters["A_sil"]
    a_sil = material_parameters["a_sil"]
    b_sil = material_parameters["b_sil"]
    K_sil = material_parameters["K_sil"]

    theta_M = material_parameters["theta_M"]
    delta_theta = material_parameters["delta_theta"]

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

    Nhat = ufl.as_vector((
        ufl.cos(theta_phys),
        ufl.sin(theta_phys),
        0.0,
    ))

    Mhat = ufl.as_vector((
        np.cos(theta_M),
        np.sin(theta_M),
        0.0,
    ))

    I4bar = ufl.dot(Nhat, Cbar * Nhat)

    alignment_raw = ufl.dot(Nhat, Mhat)
    alignment = (
        ufl.sqrt(alignment_raw**2 + delta_theta**2)
        - delta_theta
    ) / (
        np.sqrt(1.0 + delta_theta**2)
        - delta_theta
    )

    Astar = (
        s
        * (1.0 - ufl.exp(-(h / (2.0 * hs))**2))
        * alignment
    )

    g_Ak = (
        (1.0 / a_Ak) * ufl.exp(a_Ak * (I1bar - 3.0))
        + b_Ak
        * (I1bar - 2.0)
        * (1.0 - ufl.ln(I1bar - 2.0))
        - 1.0 / a_Ak
        - b_Ak
    )

    W_Ak = (
        0.5 * K_Ak * (J - 1.0)**2
        + (A_Ak + Astar) * g_Ak
        + r * (I4bar - 1.0)**2
    )

    g_sil = (
        (1.0 / a_sil) * ufl.exp(a_sil * (I1bar - 3.0))
        + b_sil
        * (I1bar - 2.0)
        * (1.0 - ufl.ln(I1bar - 2.0))
        - 1.0 / a_sil
        - b_sil
    )

    W_sil = (
        0.5 * K_sil * (J - 1.0)**2
        + A_sil * g_sil
    )

    rho_scale = (
        eps_rho
        + (1.0 - eps_rho) * rho_phys**p_rho
    )
    phi_scale = phi_phys**p_phi

    W = rho_scale * (
        phi_scale * W_Ak
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
    domain_volume = 1.0 * dx

    return {
        "magnetic_material_fraction": {
            "form": phi_phys * dx,
            "normalize_by": domain_volume,
            "upper_bound": 0.30,
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

    particle_chain = phi_phys * ufl.as_vector((
        ufl.cos(theta_phys),
        ufl.sin(theta_phys),
    ))

    field_alignment = phi_phys * ufl.cos(theta_phys)

    return {
        "particle_chain": particle_chain,
        "field_alignment": field_alignment,
    }


requested_output_fields = [
    "u",
    "rho_phys",
    "phi_phys",
    "theta_phys",
    "particle_chain",
    "field_alignment",
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
    "sim_image_output_interval": 51,
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
