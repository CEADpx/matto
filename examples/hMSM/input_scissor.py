# hMSM closed-push scissor optimization

from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI

from matto.topopt import topopt
from material import make_build_free_energy

# ============================================================
#  GEOMETRY
# ============================================================

geometry = {
    "lc": 0.08,
    "L": 10.0,
    "plate_width": 0.05,
    "arm_width": 0.45,
    "y_mid": 7.0,
    "y_amp": 5.0,
}

# ============================================================
#  MESH
# ============================================================

def build_closed_push_mesh(lc=0.08, comm=MPI.COMM_WORLD):
    """Build the connected |<>| scissor body with a diamond-shaped void."""
    import gmsh
    from dolfinx.io.gmshio import model_to_mesh

    rank = comm.rank

    if rank == 0:
        gmsh.initialize()
        gmsh.model.add("closed_push_phi_theta")

        L = geometry["L"]
        plate_width = geometry["plate_width"]
        arm_width = geometry["arm_width"]
        y_mid = geometry["y_mid"]
        y_amp = geometry["y_amp"]

        x_left = 0.0
        x_right = L
        x_center = 0.5 * L
        x_plate_start = L - plate_width

        y_upper = y_mid + y_amp
        y_lower = y_mid - y_amp
        half_arm_width = 0.5 * arm_width

        # Outer boundary of the connected body.
        point_1 = gmsh.model.geo.addPoint(
            x_left, y_mid - half_arm_width, 0.0, lc
        )
        point_2 = gmsh.model.geo.addPoint(
            x_center, y_lower - half_arm_width, 0.0, lc
        )
        point_3 = gmsh.model.geo.addPoint(
            x_plate_start, y_mid - half_arm_width, 0.0, lc
        )
        point_4 = gmsh.model.geo.addPoint(
            x_right, y_mid - half_arm_width, 0.0, lc
        )
        point_5 = gmsh.model.geo.addPoint(
            x_right, y_mid + half_arm_width, 0.0, lc
        )
        point_6 = gmsh.model.geo.addPoint(
            x_plate_start, y_mid + half_arm_width, 0.0, lc
        )
        point_7 = gmsh.model.geo.addPoint(
            x_center, y_upper + half_arm_width, 0.0, lc
        )
        point_8 = gmsh.model.geo.addPoint(
            x_left, y_mid + half_arm_width, 0.0, lc
        )

        outer_lines = [
            gmsh.model.geo.addLine(point_1, point_2),
            gmsh.model.geo.addLine(point_2, point_3),
            gmsh.model.geo.addLine(point_3, point_4),
            gmsh.model.geo.addLine(point_4, point_5),
            gmsh.model.geo.addLine(point_5, point_6),
            gmsh.model.geo.addLine(point_6, point_7),
            gmsh.model.geo.addLine(point_7, point_8),
            gmsh.model.geo.addLine(point_8, point_1),
        ]
        outer_loop = gmsh.model.geo.addCurveLoop(outer_lines)

        # Inner diamond void.
        void_1 = gmsh.model.geo.addPoint(
            x_left + arm_width, y_mid, 0.0, lc
        )
        void_2 = gmsh.model.geo.addPoint(
            x_center, y_upper - half_arm_width, 0.0, lc
        )
        void_3 = gmsh.model.geo.addPoint(
            x_plate_start - arm_width, y_mid, 0.0, lc
        )
        void_4 = gmsh.model.geo.addPoint(
            x_center, y_lower + half_arm_width, 0.0, lc
        )

        inner_lines = [
            gmsh.model.geo.addLine(void_1, void_4),
            gmsh.model.geo.addLine(void_4, void_3),
            gmsh.model.geo.addLine(void_3, void_2),
            gmsh.model.geo.addLine(void_2, void_1),
        ]
        inner_loop = gmsh.model.geo.addCurveLoop(inner_lines)

        surface = gmsh.model.geo.addPlaneSurface([
            outer_loop,
            inner_loop,
        ])

        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], 1)
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

mesh = build_closed_push_mesh(
    lc=geometry["lc"],
    comm=MPI.COMM_WORLD,
)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = build_closed_push_mesh(
        lc=geometry["lc"],
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

L = geometry["L"]
plate_x0 = L - geometry["plate_width"]
y_mid = geometry["y_mid"]

def phi_void_output_plate(x):
    """Keep magnetic particles out of the thin output plate."""
    return x[0] >= plate_x0

design_variables = {
    "rho": {
        # The entire scissor structure is fixed solid material.
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
        "active": True,
        "initial": 0.15,
        "bounds": (0.0, 0.30),
        "prescribed_value": 0.0,
        "raw_space": ("DG", 0),
        "physical_space": ("CG", 1),
        "operators": [
            {
                "type": "density_filter",
                "radius": 0.20,
            },
        ],
        "fixed_regions": [
            {
                "where": phi_void_output_plate,
                "value": 0.0,
            },
        ],
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
                "radius": 0.20,
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

# This problem has no applied tractions.
traction_boundaries = {}

load_steps = 40

load_cases = [
    {
        "name": "B_right_push",
        "weight": 1.0,
        "body_force": (0.0, 0.0),
        "tractions": {},
        "stimuli": {
            "B_app": (125.0, 0.0),
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

displacement_targets = [
    {
        "point": (L - 0.01, y_mid),
        "target": (3.0, 0.0),
        "sigma": 0.35,
        "weight": 1.0,
        "components": ("x", "y"),
    },
]

def build_objective(
    u_field,
    external_work,
    dx,
):
    """Return the legacy Gaussian-localized displacement-tracking objective."""
    X = ufl.SpatialCoordinate(mesh)
    objective = 0

    for index, target_definition in enumerate(displacement_targets):
        target_point = target_definition["point"]
        target_displacement = target_definition["target"]
        sigma = float(target_definition.get("sigma", 1.0))
        weight = float(target_definition.get("weight", 1.0))
        components = target_definition.get("components", ("x", "y"))

        if sigma <= 0.0:
            raise ValueError(
                f"displacement_targets[{index}]['sigma'] must be positive."
            )

        if not components or any(
            component not in ("x", "y")
            for component in components
        ):
            raise ValueError(
                f"displacement_targets[{index}]['components'] must contain "
                "'x', 'y', or both."
            )

        distance_squared = (
            (X[0] - float(target_point[0]))**2
            + (X[1] - float(target_point[1]))**2
        )

        localization = ufl.exp(
            -distance_squared / sigma**2
        )

        tracking_error = 0

        if "x" in components:
            tracking_error += (
                u_field[0] - float(target_displacement[0])
            )**2

        if "y" in components:
            tracking_error += (
                u_field[1] - float(target_displacement[1])
            )**2

        objective += (
            weight
            * localization
            * tracking_error
            * dx
        )

    return objective

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
            "upper_bound": 0.15,
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
        / "results_ClosedPush_PhiTheta_RhoFixed"
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
