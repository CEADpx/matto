import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from mpi4py import MPI

from dolfinx import fem

# Import evaluation driver
from fenitop.evaluate import evaluate


# ============================================================
#  GRIPPER GEOMETRY PARAMETERS
# ============================================================

gripper = {

    "lc": 0.08,

    "base_width": 5.0,
    "base_height": 2.5,

    "center_divider_width": 6.0,

    "diagonal_length": 20.0,

    "arm_length": 20.0,
    "arm_width": 2.5,

    "diag_angle": np.pi/3,

    "phi_cap": 0.30,
}

# ============================================================
#  BUILD GRIPPER MESH
# ============================================================

def build_gripper_mesh(lc=0.05, comm=MPI.COMM_WORLD):

    import gmsh

    rank = comm.rank

    if rank == 0:

        gmsh.initialize()
        gmsh.model.add("gripper")

        bw = gripper["base_width"]
        bh = gripper["base_height"]

        cw = gripper["center_divider_width"]

        dl = gripper["diagonal_length"]

        al = gripper["arm_length"]
        aw = gripper["arm_width"]

        ang = gripper["diag_angle"]

        dx = dl*np.cos(ang)
        dy = dl*np.sin(ang)

        # ----------------------------------------------------
        # Right half points
        # ----------------------------------------------------

        P1 = gmsh.model.geo.addPoint(cw, 0.0, 0.0, lc)

        P2 = gmsh.model.geo.addPoint(cw + bw, 0.0, 0.0, lc)

        P3 = gmsh.model.geo.addPoint(cw + bw, bh, 0.0, lc)

        P4 = gmsh.model.geo.addPoint(
            cw + bw + dx,
            bh + dy,
            0.0,
            lc
        )

        P5 = gmsh.model.geo.addPoint(
            cw + bw + dx,
            bh + dy + al,
            0.0,
            lc
        )

        P6 = gmsh.model.geo.addPoint(
            cw + bw + dx - aw,
            bh + dy + al,
            0.0,
            lc
        )

        P7 = gmsh.model.geo.addPoint(
            cw + bw + dx - aw,
            bh + dy,
            0.0,
            lc
        )

        P8 = gmsh.model.geo.addPoint(
            cw,
            bh,
            0.0,
            lc
        )

        # ----------------------------------------------------
        # Mirrored inner divider points
        # ----------------------------------------------------

        P1m = gmsh.model.geo.addPoint(-cw, 0.0, 0.0, lc)

        P8m = gmsh.model.geo.addPoint(
            -cw,
            bh,
            0.0,
            lc
        )

        # ----------------------------------------------------
        # Mirror points
        # ----------------------------------------------------

        P2m = gmsh.model.geo.addPoint(-(cw + bw), 0.0, 0.0, lc)

        P3m = gmsh.model.geo.addPoint(-(cw + bw), bh, 0.0, lc)

        P4m = gmsh.model.geo.addPoint(
            -(cw + bw + dx),
            bh + dy,
            0.0,
            lc
        )

        P5m = gmsh.model.geo.addPoint(
            -(cw + bw + dx),
            bh + dy + al,
            0.0,
            lc
        )

        P6m = gmsh.model.geo.addPoint(
            -(cw + bw + dx) + aw,
            bh + dy + al,
            0.0,
            lc
        )

        P7m = gmsh.model.geo.addPoint(
            -(cw + bw + dx) + aw,
            bh + dy,
            0.0,
            lc
        )

        # ----------------------------------------------------
        # Lines (counter clockwise outer boundary)
        # ----------------------------------------------------

        L1 = gmsh.model.geo.addLine(P1, P2)
        L2 = gmsh.model.geo.addLine(P2, P3)
        L3 = gmsh.model.geo.addLine(P3, P4)
        L4 = gmsh.model.geo.addLine(P4, P5)
        L5 = gmsh.model.geo.addLine(P5, P6)
        L6 = gmsh.model.geo.addLine(P6, P7)
        L7 = gmsh.model.geo.addLine(P7, P8)

        L8 = gmsh.model.geo.addLine(P8, P8m)
        L9 = gmsh.model.geo.addLine(P8m, P7m)
        L10 = gmsh.model.geo.addLine(P7m, P6m)
        L11 = gmsh.model.geo.addLine(P6m, P5m)
        L12 = gmsh.model.geo.addLine(P5m, P4m)
        L13 = gmsh.model.geo.addLine(P4m, P3m)
        L14 = gmsh.model.geo.addLine(P3m, P2m)
        L15 = gmsh.model.geo.addLine(P2m, P1m)
        L16 = gmsh.model.geo.addLine(P1m, P1)

        OuterLoop = gmsh.model.geo.addCurveLoop(
            [
                L1, L2, L3, L4, L5, L6, L7, L8,
                L9, L10, L11, L12, L13, L14, L15, L16
            ]
        )

        Surface = gmsh.model.geo.addPlaneSurface([OuterLoop])

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [Surface], 1)
        gmsh.model.setPhysicalName(2, 1, "domain")

        gmsh.model.mesh.generate(2)

    from dolfinx.io.gmshio import model_to_mesh

    mesh, cell_tags, facet_tags = model_to_mesh(
        gmsh.model,
        comm,
        0,
        gdim=2
    )

    if rank == 0:
        gmsh.finalize()

    return mesh


mesh = build_gripper_mesh(lc=gripper["lc"], comm=MPI.COMM_WORLD)

# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {

    "mesh": mesh,

    "shear_modulus": 100.0, # (kPa)

    # Clamp entire base (y = 0)
    "disp_bc": lambda x: np.isclose(x[1], 0.0),

    "body_force": (0.0, 0.0),
    "traction_bcs": [],

    "load_cases": [
        {
            "name": "B_up",
            "B_app_mag": 80.0, # (mT)
            "B_app_dir": (0.0, 1.0),
            "tractions": {},
        },
    ],

    "load_steps": 50,

    "quadrature_degree": 2,

    "mu0": 1.256e3, # vacuum permeability (mT^2/kPa)
    "B_rem_mag": 40.0, # (mT)

    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "snes_max_it": "500",
        "snes_error_if_not_converged": None,
    },
}


# ============================================================
#  EVALUATION SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ============================================================
#  MEASUREMENT MARKER (RIGHT GRIPPER TIP)
# ============================================================

def right_tip_marker(x):

    bw = gripper["base_width"]
    bh = gripper["base_height"]
    cw = gripper["center_divider_width"]

    dl = gripper["diagonal_length"]
    al = gripper["arm_length"]
    aw = gripper["arm_width"]

    ang = gripper["diag_angle"]

    dx = dl*np.cos(ang)
    dy = dl*np.sin(ang)

    x_min = cw + bw + dx - aw
    x_max = cw + bw + dx
    y_tip = bh + dy + al

    return (
        np.isclose(x[1], y_tip)
        &
        (x[0] >= x_min - 1e-8)
        &
        (x[0] <= x_max + 1e-8)
    )

eval_config = {

    "G_models": ["default", "guth", "mooney", "hill", "kerner", "LP", "LPA"],
    "hyperelastic_models": ["stVenant", "neoHookean1", "neoHookean2"],

    "output_dir": RESULTS_DIR,

    "write_bp": True,
    "write_csv": True,
    "csv_name": "gripper_model_comparison.csv",

    "measurement_marker": right_tip_marker,
    "compute_compliance": False,
}


# ============================================================
#  DESIGN BUILDER (ARM MAGNETIZATION)
# ============================================================

def build_gripper_design(mesh):

    V = fem.functionspace(mesh, ("CG", 1))

    coords = V.tabulate_dof_coordinates()

    x = coords[:, 0]
    y = coords[:, 1]

    ndofs = coords.shape[0]

    rho = np.ones(ndofs)

    # ----------------------------------------------------
    # Compute hinge height (P4_y)
    # ----------------------------------------------------

    bh = gripper["base_height"]
    dl = gripper["diagonal_length"]
    ang = gripper["diag_angle"]

    P4_y = bh + dl*np.sin(ang)

    phi_cut_y = np.round(P4_y + 0.5*gripper["arm_width"], 12)

    # ----------------------------------------------------
    # Magnetic region (vertical fingers)
    # ----------------------------------------------------

    phi = np.zeros(ndofs)

    phi[y > phi_cut_y] = gripper["phi_cap"]

    # ----------------------------------------------------
    # Magnetization direction
    # ----------------------------------------------------

    theta = np.zeros(ndofs)

    # Right finger
    theta[x > 0] = 0.0

    # Left finger
    theta[x < 0] = np.pi

    return rho, phi, theta

design_source = {
    "type": "callable",
    "builder": build_gripper_design,
}

# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    evaluate(fem_params, eval_config, design_source)