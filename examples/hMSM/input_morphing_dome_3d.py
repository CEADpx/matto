"""
hMSM load-bearing morphing surface in 3D.

A square slab is clamped on its four side faces. Its top 0.5 mm is a
solid, non-magnetic skin; the material below the skin is the design
region for rho, phi and theta. Under a uniform applied field along z
the skin is to take a prescribed out-of-plane shape, a shallow dome or
a saddle, and to hold that shape under a downward traction on the skin.

The slab, the loads and the target are symmetric about the planes
x = 0 and y = 0, so one quadrant is modeled with symmetry conditions
on those planes.

With the remanent magnetization in the x-y plane and the field along
z, the magnetic energy density is proportional to
phi (cos(theta) du_z/dx + sin(theta) du_z/dy), so theta programs the
direction of the surface slope that the field drives.

Objective: the mean squared error of u_z from the target shape over the
top skin, normalized by the skin area and the target height squared,
under one load case: the field and the larger of the two tractions
previously used (0.50 kPa). A single load case avoids asking for the
same target shape under two different loads at once.
"""

import os
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.mesh import CellType, create_box, locate_entities_boundary, meshtags

from matto.driver import OptimizationDriver
from matto.design import volume_constraint
from matto.materials import HardMagneticSoftMaterial

# ============================================================
#  GEOMETRY AND MESH
# ============================================================

# Slab 24 x 24 x 3, skin z > THICKNESS - SKIN, clamped on |x| = |y| = HALF;
# the quadrant 0 < x, y < HALF is modeled.
HALF, THICKNESS, SKIN = 12.0, 3.0, 0.5
# Cells per direction of the quadrant; MORPHING_CELLS="12,12,6" in the
# environment overrides it so the same input runs coarser elsewhere.
CELLS = [int(n) for n in os.environ.get("MORPHING_CELLS", "24,24,6").split(",")]

corners = [[0.0, 0.0, 0.0], [HALF, HALF, THICKNESS]]
mesh = create_box(MPI.COMM_WORLD, corners, CELLS, cell_type=CellType.hexahedron)
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_box(MPI.COMM_SELF, corners, CELLS, cell_type=CellType.hexahedron)
else:
    mesh_serial = None


def on_top(x):
    return np.isclose(x[2], THICKNESS)


def on_sides(x):
    return np.isclose(x[0], HALF) | np.isclose(x[1], HALF)


def on_plane_x(x):
    return np.isclose(x[0], 0.0)


def on_plane_y(x):
    return np.isclose(x[1], 0.0)


def in_skin(x):
    return x[2] > THICKNESS - SKIN


top_tag = 1
facet_dim = mesh.topology.dim - 1
top_facets = np.sort(locate_entities_boundary(mesh, facet_dim, on_top))
top_facet_tags = meshtags(
    mesh, facet_dim, top_facets, np.full(top_facets.shape, top_tag, dtype=np.int32)
)
ds_top = ufl.Measure("ds", domain=mesh, subdomain_data=top_facet_tags)

# ============================================================
#  MATERIAL
# ============================================================

material_parameters = {
    "G0": 100.0,             # Base shear modulus [kPa]
    "p_rho": 3.0,
    "eps_rho": 1.0e-2,
    "mu0": 1.256e3,          # Vacuum permeability [mT^2/kPa]
    "B_rem_mag": 200.0,      # Remanent magnetic flux density [mT]
    "dim": 3,
}

material = HardMagneticSoftMaterial(**material_parameters)

# ============================================================
#  DESIGN VARIABLES
# ============================================================

TARGET = "dome"          # "dome" or "saddle"
TARGET_HEIGHT = 1.0      # [mm]


def target_shape(X, Y):
    """Out-of-plane target displacement of the skin, zero with zero slope at the clamp."""
    bump = (1.0 - (X / HALF)**2)**2 * (1.0 - (Y / HALF)**2)**2
    if TARGET == "dome":
        return TARGET_HEIGHT * bump
    if TARGET == "saddle":
        return TARGET_HEIGHT * (27.0 / 4.0) * (X**2 - Y**2) / HALF**2 * bump
    raise ValueError(f"TARGET must be 'dome' or 'saddle', got {TARGET!r}.")


THETA_INITIAL = "target_gradient"   # "target_gradient" or a number [rad]


def target_gradient_angle(x, step=1.0e-3):
    """Direction of the target's slope: the magnetization that the field
    drives toward the target locally. Radially inward for the dome."""
    dwdx = (target_shape(x[0] + step, x[1]) - target_shape(x[0] - step, x[1])) / (2 * step)
    dwdy = (target_shape(x[0], x[1] + step) - target_shape(x[0], x[1] - step)) / (2 * step)
    return np.arctan2(dwdy, dwdx)


theta_initial = (
    target_gradient_angle if THETA_INITIAL == "target_gradient" else float(THETA_INITIAL)
)

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
                "beta_update_interval": 30,
                "beta_max": 8.0,
            },
        ],
        "fixed_regions": [{"where": in_skin, "value": 1.0}],
    },
    "phi": {
        "active": True,
        "initial": 0.10,
        "bounds": (0.00, 0.30),
        "prescribed_value": 0.00,
        "operators": [
            {"type": "density_filter", "radius": 1.0},
        ],
        "fixed_regions": [{"where": in_skin, "value": 0.0}],
    },
    "theta": {
        # theta = 0 is remanent magnetization along +x. The start follows
        # the target's slope (radially inward for the dome); from a
        # uniform start the half where m already points the right way
        # takes all the magnetic material and the design stays one-sided.
        "active": True,
        "initial": theta_initial,
        # One turn, placed so that the inward direction along the plane
        # y = 0 (theta = -pi) is not at a bound.
        "bounds": (-1.5 * np.pi, 0.5 * np.pi),
        # theta is kept cell-wise and unfiltered. A radial field has a
        # line where the angle jumps from pi to -pi; a filter averages
        # the two to 0 and turns the magnetization along that line
        # outward, which the optimizer then answers with a void channel.
        "physical_space": ("DG", 0),
        "operators": [],
        # The skin's theta is not a design variable. It is held at the
        # start direction; phi is zero there.
        "fixed_regions": [{"where": in_skin, "value": theta_initial}],
    },
}

# ============================================================
#  BOUNDARY CONDITIONS AND LOAD CASES
# ============================================================

boundary_conditions = [
    {
        "name": "clamped_sides",
        "on_boundary": on_sides,
        "value": (0.0, 0.0, 0.0),
    },
    {
        "name": "symmetry_x",
        "on_boundary": on_plane_x,
        "value": (0.0, None, None),
    },
    {
        "name": "symmetry_y",
        "on_boundary": on_plane_y,
        "value": (None, 0.0, None),
    },
]

traction_boundaries = {"top": on_top}

load_steps = 10

B_APPLIED = 200.0       # [mT], along z
PRESSURES = (0.50,)     # [kPa], downward on the skin; the larger of the original two

load_cases = [
    {
        "name": f"B_up_load_{index + 1}",
        "weight": 1.0 / len(PRESSURES),
        "body_force": (0.0, 0.0, 0.0),
        "tractions": {"top": (0.0, 0.0, -pressure)},
        "stimuli": {"B_app": (0.0, 0.0, B_APPLIED)},
    }
    for index, pressure in enumerate(PRESSURES)
]

# ============================================================
#  OBJECTIVE
# ============================================================


def build_objective(u_field, external_work, dx):
    """Mean squared error of u_z from the target shape over the skin."""
    X = ufl.SpatialCoordinate(mesh)
    error = u_field[2] - target_shape(X[0], X[1])
    skin_area = HALF**2     # of the quadrant
    return (error**2 / (skin_area * TARGET_HEIGHT**2)) * ds_top(top_tag)


# ============================================================
#  CONSTRAINTS
# ============================================================


def build_constraints(design_variables, dx):
    rho_phys = design_variables["rho"].phys
    phi_phys = design_variables["phi"].phys
    return {
        "rho_volume": volume_constraint(rho_phys, 0.60, dx),
        "phi_volume": volume_constraint(phi_phys, 0.10, dx),
    }


# ============================================================
#  OUTPUT FIELDS
# ============================================================


def build_output_fields(design_variables):
    rho_phys = design_variables["rho"].phys
    phi_phys = design_variables["phi"].phys
    theta_phys = design_variables["theta"].phys
    phi_eff = rho_phys * phi_phys
    m_eff = phi_eff * ufl.as_vector((ufl.cos(theta_phys), ufl.sin(theta_phys), 0.0))
    return {"phi_eff": phi_eff, "m_eff": m_eff}


# theta_phys is cell-wise and cannot share the VTX file with the CG1
# fields; m_eff carries the magnetization direction instead.
requested_output_fields = ["u", "rho_phys", "phi_phys", "phi_eff", "m_eff"]

# ============================================================
#  SOLVER AND MMA OPTIONS
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
            "petsc_options": {"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1.0e-10},
        },
    },
}

optimization_options = {
    # R2 rerun, run to convergence. The change is that of the free raw
    # dofs only; the skin is not part of the MMA vector.
    "max_iter": 1000,
    "opt_tol": 1.0e-3,
    "move": 0.05,
}

_pressure_tag = "_".join(f"{p:g}kPa" for p in PRESSURES)
output_options = {
    # tagged with the pressures actually used, so this run's output never lands in the
    # same folder as the earlier two-load-case run (results_Morphing3D_dome_48x48x6)
    "output_dir": str(Path(__file__).resolve().parent
                       / f"results_Morphing3D_{TARGET}_quadrant_{CELLS[0]}x{CELLS[1]}x{CELLS[2]}_{_pressure_tag}"),
    "sim_output_interval": 25,
}

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

if __name__ == "__main__":
    OptimizationDriver(problem).run()
