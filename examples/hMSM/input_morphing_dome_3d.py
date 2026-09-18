"""
hMSM load-bearing morphing surface in 3D.

A square slab is clamped on its four side faces. Its top layer is a
solid, non-magnetic skin; the material below the skin is the design
region for rho, phi and theta. Under a uniform applied field along z
the skin is to take the shape of a shallow dome and to hold it under a
downward traction on the skin.

With the remanent magnetization in the x-y plane and the field along
z, the magnetic energy density is proportional to
phi (cos(theta) du_z/dx + sin(theta) du_z/dy), so theta sets the
direction of the surface slope that the field drives.

Objective: the mean squared error of u_z from the target shape over the
skin, normalized by the skin area and the target height squared, summed
over two load cases that share the field and the target and differ in
the traction. The sum penalizes the shape error and the change of shape
with load.

The run is followed with the postprocessors of matto.postprocess:
history.csv, and snapshots/ with the design arrays and a picture every
ten iterations. DomeMonitor shows how to subclass SnapshotPlotter.
"""

from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.mesh import CellType, create_box, locate_entities_boundary, meshtags

from matto.driver import OptimizationDriver
from matto.design import volume_constraint
from matto.materials import HardMagneticSoftMaterial
from matto.postprocess import HistoryWriter, SnapshotPlotter

# ============================================================
#  GEOMETRY AND MESH
# ============================================================

# Slab 24 x 24 x 3, clamped on |x| = |y| = HALF. The skin must be a
# whole number of cell layers.
HALF, THICKNESS, SKIN = 12.0, 3.0, 0.5
CELLS = [32, 32, 6]

corners = [[-HALF, -HALF, 0.0], [HALF, HALF, THICKNESS]]
mesh = create_box(MPI.COMM_WORLD, corners, CELLS, cell_type=CellType.hexahedron)
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_box(MPI.COMM_SELF, corners, CELLS, cell_type=CellType.hexahedron)
else:
    mesh_serial = None


def on_top(x):
    return np.isclose(x[2], THICKNESS)


def on_sides(x):
    return np.isclose(np.abs(x[0]), HALF) | np.isclose(np.abs(x[1]), HALF)


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

# eps_rho: with 1e-6 the void cells next to loaded solid invert and the
# Newton solve fails once the Heaviside projection sharpens.
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
#  TARGET SHAPE
# ============================================================

TARGET_HEIGHT = 1.0


def target_shape(X, Y):
    """Dome with zero displacement and zero slope at the clamped sides."""
    return TARGET_HEIGHT * (1.0 - (X / HALF)**2)**2 * (1.0 - (Y / HALF)**2)**2


def target_slope_angle(x, step=1.0e-3):
    """Direction of the target's slope, radially inward for the dome."""
    dwdx = (target_shape(x[0] + step, x[1]) - target_shape(x[0] - step, x[1])) / (2 * step)
    dwdy = (target_shape(x[0], x[1] + step) - target_shape(x[0], x[1] - step)) / (2 * step)
    return np.arctan2(dwdy, dwdx)


# ============================================================
#  DESIGN VARIABLES
# ============================================================

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
        # the target's slope: from a uniform start the half where m
        # already points inward takes all the magnetic material and the
        # design stays one-sided.
        "active": True,
        "initial": target_slope_angle,
        "bounds": (-np.pi, np.pi),
        # theta is cell-wise and unfiltered. A radial field has a line
        # where the angle jumps from pi to -pi; a filter averages the
        # two to 0, turns the magnetization outward along that line, and
        # the optimizer answers with a void channel.
        "physical_space": ("DG", 0),
        "operators": [],
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
]

traction_boundaries = {"top": on_top}

load_steps = 10

B_APPLIED = 100.0          # [mT], along z
PRESSURES = (0.10, 0.50)   # [kPa], downward on the skin

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
    skin_area = (2.0 * HALF)**2
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
# fields; m_eff carries the magnetization direction.
requested_output_fields = ["u", "rho_phys", "phi_phys", "phi_eff", "m_eff"]

# ============================================================
#  POSTPROCESSORS
# ============================================================


class DomeMonitor(SnapshotPlotter):
    """The design as a body from above and from below, and the skin against the target."""

    def draw(self, figure, data):
        figure.set_size_inches(13.0, 8.5)
        figure.set_layout_engine("constrained")
        grid = figure.add_gridspec(2, 6)
        for slot, camera in ((grid[0, 0:3], {"elev": 25, "azim": -60}),
                             (grid[0, 3:6], {"elev": -30, "azim": -60})):
            axis = figure.add_subplot(slot, projection="3d")
            mappable = self.draw_body(axis, data, camera)
        figure.colorbar(mappable, ax=figure.axes, fraction=0.015, label=self.body_color())
        figure.suptitle(f"iteration {data['iteration']}   objective {data['objective']:.4e}")
        if data["u"] is None:
            return

        points, u = data["u_points"], data["u"]
        top = np.isclose(points[:, 2], THICKNESS)
        x, y, uz = points[top, 0], points[top, 1], u[top, 2]
        target = target_shape(x, y)
        limit = max(np.abs(uz).max(), np.abs(target).max())
        levels = np.linspace(-limit, limit, 21)
        for slot, values, title in ((grid[1, 0:2], uz, "skin u_z, last load case"),
                                    (grid[1, 2:4], target, "target")):
            axis = figure.add_subplot(slot)
            image = axis.tricontourf(x, y, values, levels=levels, cmap="RdBu_r")
            axis.set_aspect("equal")
            axis.set_title(title)
            figure.colorbar(image, ax=axis, fraction=0.04)

        axis = figure.add_subplot(grid[1, 4:6])
        line = np.isclose(y, 0.0)
        order = np.argsort(x[line])
        axis.plot(x[line][order], uz[line][order], label="u_z(x, 0)")
        axis.plot(x[line][order], target[line][order], "k:", label="target")
        axis.set_xlabel("x")
        axis.legend()
        axis.set_title(f"rms error {np.sqrt(np.mean((uz - target)**2)):.3f}")


postprocessors = [
    HistoryWriter(),
    DomeMonitor(every=10, weight="phi", color="phi"),
]

# ============================================================
#  SOLVER AND MMA OPTIONS
# ============================================================

# MUMPS: PETSc's own LU is several times slower on this mesh in serial.
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
                "pc_factor_mat_solver_type": "mumps",
            },
        },
        "adjoint": {
            "rtol": 1.0e-8,
            "atol": 1.0e-12,
            "petsc_options": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        },
        "filter": {
            "petsc_options": {"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1.0e-10},
        },
    },
}

optimization_options = {
    "max_iter": 150,
    "opt_tol": 1.0e-5,
    "move": 0.05,
}

output_options = {
    "output_dir": str(Path(__file__).resolve().parent / "results_MorphingDome3D"),
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
    "postprocessors": postprocessors,
    "fem_options": fem_options,
    "optimization_options": optimization_options,
    "output_options": output_options,
}

if __name__ == "__main__":
    OptimizationDriver(problem).run()
