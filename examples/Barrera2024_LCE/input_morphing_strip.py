# Barrera2024 LCE U-shape morphing optimization
# Center-supported flat strip under activation; optimize active-material placement and mesogen direction so both free arms curl upward into a U shape.

from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.mesh import CellType, create_rectangle

from matto.topopt import topopt
from material import make_build_free_energy

# ============================================================
#  GEOMETRY, TARGET SHAPE, AND MESH
# ============================================================

geometry = {
    "length": 12.0,
    "height": 0.60,
    "nx": 120,
    "ny": 12,
    # Width of the fixed pad centered on the bottom edge.
    "support_pad_width": 0.60,
    # Each free arm should turn upward through this angle.
    "target_arm_angle": np.deg2rad(45.0),
}

length = geometry["length"]
height = geometry["height"]
support_pad_width = geometry["support_pad_width"]
target_arm_angle = geometry["target_arm_angle"]
center_x = 0.5 * length
support_half_width = 0.5 * support_pad_width
free_arm_length = center_x - support_half_width

def initial_theta(x):
    """Seed positive curvature in both arms using two director layers."""
    return np.where(
        x[1] <= 0.5 * height,
        0.0,
        np.pi / 2.0,
    )

def center_support_pad(x):
    """Small clamped pad centered on the bottom boundary."""
    return (
        np.isclose(x[1], 0.0)
        & (np.abs(x[0] - center_x) <= support_half_width)
    )

mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [length, height]],
    [geometry["nx"], geometry["ny"]],
    cell_type=CellType.quadrilateral,
)

# Serial copy used for plotting and gathered output.
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF,
        [[0.0, 0.0], [length, height]],
        [geometry["nx"], geometry["ny"]],
        cell_type=CellType.quadrilateral,
    )
else:
    mesh_serial = None

def prescribed_target_displacement():
    """Map the two free arms to symmetric upward circular arcs."""
    X = ufl.SpatialCoordinate(mesh)

    horizontal_offset = X[0] - center_x

    # Signed distance along either free arm, measured outward from the edge of
    # the flat support pad. It is exactly zero throughout the support pad.
    signed_arc_coordinate = ufl.conditional(
        ufl.gt(horizontal_offset, support_half_width),
        horizontal_offset - support_half_width,
        ufl.conditional(
            ufl.lt(horizontal_offset, -support_half_width),
            horizontal_offset + support_half_width,
            0.0,
        ),
    )

    local_angle = (
        target_arm_angle
        * signed_arc_coordinate
        / free_arm_length
    )
    target_radius = free_arm_length / target_arm_angle

    # The centerline stays flat over the support pad. Each free arm begins at
    # its corresponding pad edge and follows a constant-curvature arc.
    arc_origin_x = ufl.conditional(
        ufl.gt(horizontal_offset, support_half_width),
        center_x + support_half_width,
        ufl.conditional(
            ufl.lt(horizontal_offset, -support_half_width),
            center_x - support_half_width,
            X[0],
        ),
    )

    centerline_x = (
        arc_origin_x
        + target_radius * ufl.sin(local_angle)
    )
    centerline_y = (
        0.5 * height
        + target_radius * (1.0 - ufl.cos(local_angle))
    )

    # Rotate each cross-section rigidly with the target centerline tangent.
    transverse_coordinate = X[1] - 0.5 * height
    target_position = ufl.as_vector((
        centerline_x
        - transverse_coordinate * ufl.sin(local_angle),
        centerline_y
        + transverse_coordinate * ufl.cos(local_angle),
    ))

    return target_position - X

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
        # Keep the complete strip solid: this is a material-programming problem.
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
        # In-plane mesogen direction. The two-layer seed already bends upward,
        # while the optimizer remains free to rotate the local director.
        "active": True,
        "initial": initial_theta,
        "bounds": (-np.pi / 2.0, np.pi / 2.0),
        "prescribed_value": 0.0,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
        "operators": [
            {
                "type": "density_filter",
                "radius": 0.15,
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
        "name": "clamped_center_pad",
        "on_boundary": center_support_pad,
        "value": (0.0, 0.0),
    },
]

traction_boundaries = {}

load_steps = 50

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

build_free_energy = make_build_free_energy(material_parameters)

# ============================================================
#  OBJECTIVE
# ============================================================

def build_objective(
    u_field,
    external_work,
    dx,
):
    """Minimize the mean squared error from the prescribed U shape."""
    target_displacement = prescribed_target_displacement()
    displacement_error = u_field - target_displacement
    domain_area = length * height

    # The L^2 and area factors make the objective dimensionless and mesh
    # independent. Matching the whole strip discourages tip-only motion.
    return (
        ufl.inner(displacement_error, displacement_error)
        / (length**2 * domain_area)
        * dx
    )

# ============================================================
#  CONSTRAINTS
# ============================================================

def build_constraints(
    design_variables,
    dx,
):
    """Limit programmed active LCE to half of the solid strip."""
    phi_phys = design_variables["phi"].phys
    domain_volume = 1.0 * dx

    return {
        "active_lce_fraction": {
            "form": phi_phys * dx,
            "normalize_by": domain_volume,
            "upper_bound": 0.80,
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
        "target_displacement": prescribed_target_displacement(),
    }

requested_output_fields = [
    "u",
    "rho_phys",
    "phi_phys",
    "theta_phys",
    "active_director",
    "target_displacement",
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
    "max_iter": 125,
    "opt_tol": 1.0e-5,
    "move": 0.03,
}

output_options = {
    "output_dir": str(
        Path(__file__).resolve().parent
        / "results_u_shape_morphing"
    ),
    "sim_output_interval": 20,
    "sim_image_output_interval": 126,
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
