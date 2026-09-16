# hMSM wheel rotation optimization

from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI

from matto.driver import OptimizationDriver
from matto.materials import HardMagneticSoftMaterial

# ============================================================
#  GEOMETRY
# ============================================================

wheel = {
    "R": 10.0,
    "lc": 0.2,
    "r_inner": 9.0,
    "t": 0.5,
}

# ============================================================
#  MESH
# ============================================================

def build_wheel_spokes_mesh(
    R=1.0,
    lc=0.05,
    comm=MPI.COMM_WORLD,
):
    """Build the wheel rim and four-spoke geometry."""
    import gmsh
    from dolfinx.io.gmshio import model_to_mesh

    rank = comm.rank

    if rank == 0:
        gmsh.initialize()
        gmsh.model.add("wheel_spokes_opt")

        inner_radius = wheel["r_inner"]
        spoke_half_width = wheel["t"]

        center = gmsh.model.geo.addPoint(
            0.0,
            0.0,
            0.0,
            lc,
        )
        x_spoke_start = gmsh.model.geo.addPoint(
            spoke_half_width,
            0.0,
            0.0,
            lc,
        )
        y_spoke_start = gmsh.model.geo.addPoint(
            0.0,
            spoke_half_width,
            0.0,
            lc,
        )
        outer_top = gmsh.model.geo.addPoint(
            0.0,
            R,
            0.0,
            lc,
        )
        outer_right = gmsh.model.geo.addPoint(
            R,
            0.0,
            0.0,
            lc,
        )

        spoke_corner = gmsh.model.geo.addPoint(
            spoke_half_width,
            spoke_half_width,
            0.0,
            lc,
        )
        inner_top = gmsh.model.geo.addPoint(
            spoke_half_width,
            inner_radius,
            0.0,
            lc,
        )
        inner_right = gmsh.model.geo.addPoint(
            inner_radius,
            spoke_half_width,
            0.0,
            lc,
        )

        outer_loop = gmsh.model.geo.addCurveLoop([
            gmsh.model.geo.addLine(x_spoke_start, outer_right),
            gmsh.model.geo.addCircleArc(
                outer_right,
                center,
                outer_top,
            ),
            gmsh.model.geo.addLine(outer_top, y_spoke_start),
            gmsh.model.geo.addLine(y_spoke_start, spoke_corner),
            gmsh.model.geo.addLine(spoke_corner, x_spoke_start),
        ])

        inner_loop = gmsh.model.geo.addCurveLoop([
            gmsh.model.geo.addLine(spoke_corner, inner_top),
            gmsh.model.geo.addCircleArc(
                inner_top,
                center,
                inner_right,
            ),
            gmsh.model.geo.addLine(inner_right, spoke_corner),
        ])

        surface = gmsh.model.geo.addPlaneSurface([
            outer_loop,
            inner_loop,
        ])

        for quarter_turn in (1, 2, 3):
            copied_surface = gmsh.model.geo.copy([
                (2, surface),
            ])

            gmsh.model.geo.rotate(
                copied_surface,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                quarter_turn * np.pi / 2.0,
            )

        gmsh.model.geo.synchronize()
        gmsh.model.geo.removeAllDuplicates()
        gmsh.model.geo.synchronize()

        all_surfaces = [
            tag
            for dimension, tag in gmsh.model.getEntities(2)
        ]

        gmsh.model.addPhysicalGroup(
            2,
            all_surfaces,
            1,
        )
        gmsh.model.setPhysicalName(2, 1, "domain")

        gmsh.model.mesh.generate(2)

    mesh, _, _ = model_to_mesh(
        gmsh.model,
        comm,
        0,
        gdim=2,
    )

    if rank == 0:
        gmsh.finalize()

    return mesh

mesh = build_wheel_spokes_mesh(
    R=wheel["R"],
    lc=wheel["lc"],
    comm=MPI.COMM_WORLD,
)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = build_wheel_spokes_mesh(
        R=wheel["R"],
        lc=wheel["lc"],
        comm=MPI.COMM_SELF,
    )
else:
    mesh_serial = None

# ============================================================
#  MATERIAL AND INTERPOLATION PARAMETERS
# ============================================================

material_parameters = {
    "G0": 100.0,          # Base shear modulus [kPa]
    "p_rho": 3.0,
    "eps_rho": 1.0e-6,
    "mu0": 1.256e3,       # Vacuum permeability [mT^2/kPa]
    "B_rem_mag": 100.0,   # Remanent magnetic flux density [mT]
}

# ============================================================
#  DESIGN-VARIABLE SPECIFICATIONS
# ============================================================

design_variables = {
    "rho": {
        # The entire wheel structure is fixed solid material.
        "active": False,
        "initial": 1.0,
        "bounds": (0.05, 1.0),
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
        "active": True,
        "initial": 0.30,
        "bounds": (0.0, 0.30),
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

    "theta": {
        # theta = 0 gives the legacy initial remanence direction (+x).
        "active": True,
        "initial": 0.0,
        "bounds": (-np.pi, np.pi),
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

def clamp_inner_hub(x):
    """Locate the four exposed sides of the central square hub."""
    tolerance = 1.0e-8
    half_width = wheel["t"]

    return (
        (
            (np.abs(x[0] - half_width) < tolerance)
            & (x[1] >= -half_width - tolerance)
            & (x[1] <= half_width + tolerance)
        )
        |
        (
            (np.abs(x[1] - half_width) < tolerance)
            & (x[0] >= -half_width - tolerance)
            & (x[0] <= half_width + tolerance)
        )
        |
        (
            (np.abs(x[0] + half_width) < tolerance)
            & (x[1] >= -half_width - tolerance)
            & (x[1] <= half_width + tolerance)
        )
        |
        (
            (np.abs(x[1] + half_width) < tolerance)
            & (x[0] >= -half_width - tolerance)
            & (x[0] <= half_width + tolerance)
        )
    )

boundary_conditions = [
    {
        "name": "clamped_inner_hub",
        "on_boundary": clamp_inner_hub,
        "value": (0.0, 0.0),
    },
]

# This problem has no applied tractions.
traction_boundaries = {}

load_steps = 100

load_cases = [
    {
        "name": "B_up_rotation",
        "weight": 1.0,
        "body_force": (0.0, 0.0),
        "tractions": {},
        "stimuli": {
            "B_app": (0.0, 100.0),
        },
    },
]

# ============================================================
#  FREE-ENERGY DENSITY
# ============================================================

material = HardMagneticSoftMaterial(**material_parameters)

# ============================================================
#  OBJECTIVE
# ============================================================

rotation_center = (0.0, 0.0)
rotation_radius = 0.95 * wheel["R"]
rotation_band_sigma = 0.75
rotation_sign = 1.0       # +1 rewards counterclockwise rotation
rotation_weight = 1.0

def build_objective(
    u_field,
    external_work,
    dx,
):
    """Reward tangential displacement in an annular band near the rim."""
    X = ufl.SpatialCoordinate(mesh)

    radial_x = X[0] - rotation_center[0]
    radial_y = X[1] - rotation_center[1]

    radius = ufl.sqrt(
        radial_x**2
        + radial_y**2
        + 1.0e-12
    )

    band_weight = ufl.exp(
        -((radius - rotation_radius) / rotation_band_sigma)**2
    )

    tangent = rotation_sign * ufl.as_vector((
        -radial_y / radius,
        radial_x / radius,
    ))

    return (
        -rotation_weight
        * band_weight
        * ufl.inner(u_field, tangent)
        * dx
    )

# ============================================================
#  CONSTRAINTS
# ============================================================

def build_constraints(
    design_variables,
    dx,
):
    """Constrain the domain-average magnetic particle fraction."""
    phi_phys = design_variables["phi"].phys
    domain_volume = 1.0 * dx

    return {
        "phi_volume": {
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
    "max_iter": 100,
    "opt_tol": 1.0e-5,
    "move": 0.01,
}

output_options = {
    "output_dir": str(
        Path(__file__).resolve().parent
        / "results_Wheel_Rotation_PhiTheta"
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
