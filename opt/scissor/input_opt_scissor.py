import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fenitop.topopt import topopt


# ============================================================
#  CLOSED |<>| PUSH MECHANISM PARAMETERS
# ============================================================

geom = {
    "lc": 0.08,

    "L": 10.0,

    "plate_width": 0.05,

    "arm_width": 0.45,

    "y_mid": 7.0,
    "y_amp": 5.0,

    "phi_cap": 0.30,
}


# ============================================================
#  BUILD CLOSED |<>| MESH
# ============================================================

def build_closed_push_mesh(lc=0.08, comm=MPI.COMM_WORLD):
    import gmsh
    from dolfinx.io.gmshio import model_to_mesh

    rank = comm.rank

    if rank == 0:
        gmsh.initialize()
        gmsh.model.add("closed_push_phi_theta")

        L = geom["L"]
        pw = geom["plate_width"]
        aw = geom["arm_width"]

        y_mid = geom["y_mid"]
        y_amp = geom["y_amp"]

        xL = 0.0
        xR = L
        xC = 0.5 * L
        x_plate0 = L - pw

        yU = y_mid + y_amp
        yD = y_mid - y_amp

        h = 0.5 * aw

        # ====================================================
        # OUTER BOUNDARY POINTS
        # One connected outer perimeter of the whole |<>| body
        # ====================================================

        P1 = gmsh.model.geo.addPoint(xL,       y_mid - h, 0.0, lc)
        P2 = gmsh.model.geo.addPoint(xC,       yD - h,    0.0, lc)
        P3 = gmsh.model.geo.addPoint(x_plate0, y_mid - h, 0.0, lc)
        P4 = gmsh.model.geo.addPoint(xR,       y_mid - h, 0.0, lc)
        P5 = gmsh.model.geo.addPoint(xR,       y_mid + h, 0.0, lc)
        P6 = gmsh.model.geo.addPoint(x_plate0, y_mid + h, 0.0, lc)
        P7 = gmsh.model.geo.addPoint(xC,       yU + h,    0.0, lc)
        P8 = gmsh.model.geo.addPoint(xL,       y_mid + h, 0.0, lc)

        # ====================================================
        # INNER DIAMOND VOID POINTS
        # This makes the central hole.
        # ====================================================

        Q1 = gmsh.model.geo.addPoint(xL + aw,        y_mid,     0.0, lc)
        Q2 = gmsh.model.geo.addPoint(xC,             yU - h,    0.0, lc)
        Q3 = gmsh.model.geo.addPoint(x_plate0 - aw,  y_mid,     0.0, lc)
        Q4 = gmsh.model.geo.addPoint(xC,             yD + h,    0.0, lc)

        # ====================================================
        # OUTER LOOP
        # Counter-clockwise
        # ====================================================

        L1 = gmsh.model.geo.addLine(P1, P2)
        L2 = gmsh.model.geo.addLine(P2, P3)
        L3 = gmsh.model.geo.addLine(P3, P4)
        L4 = gmsh.model.geo.addLine(P4, P5)
        L5 = gmsh.model.geo.addLine(P5, P6)
        L6 = gmsh.model.geo.addLine(P6, P7)
        L7 = gmsh.model.geo.addLine(P7, P8)
        L8 = gmsh.model.geo.addLine(P8, P1)

        OuterLoop = gmsh.model.geo.addCurveLoop(
            [L1, L2, L3, L4, L5, L6, L7, L8]
        )

        # ====================================================
        # INNER LOOP
        # Clockwise hole boundary
        # ====================================================

        H1 = gmsh.model.geo.addLine(Q1, Q4)
        H2 = gmsh.model.geo.addLine(Q4, Q3)
        H3 = gmsh.model.geo.addLine(Q3, Q2)
        H4 = gmsh.model.geo.addLine(Q2, Q1)

        InnerLoop = gmsh.model.geo.addCurveLoop(
            [H1, H2, H3, H4]
        )

        # Single connected surface with one diamond hole
        Surface = gmsh.model.geo.addPlaneSurface(
            [OuterLoop, InnerLoop]
        )

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [Surface], 1)
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


mesh = build_closed_push_mesh(lc=geom["lc"], comm=MPI.COMM_WORLD)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = build_closed_push_mesh(lc=geom["lc"], comm=MPI.COMM_SELF)
else:
    mesh_serial = None

# ============================================================
#  GEOMETRIC MARKERS
# ============================================================

L = geom["L"]
plate_width = geom["plate_width"]

plate_x0 = L - plate_width
y_mid = geom["y_mid"]

def left_clamp(x):
    return np.isclose(x[0], 0.0)

def false_zone(x):
    return np.full(x.shape[1], False)

def phi_void_output_plate(x):
    return x[0] >= plate_x0

# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,

    "shear_modulus": 100.0, # kPa

    "hyperModel": "neoHookean2", # Options: "neoHookean1", "neoHookean2" ,"stVenant"

    # --- Shear modulus microstructure model ---
    # Options: "default", "guth", "mooney", "kerner", "LP", "LPA", "hill"
    "G_model": "mooney",

    # Clamp left side of the left block
    "disp_bc": left_clamp,

    "body_force": (0.0, 0.0),
    "quadrature_degree": 2,

    "mu0": 1.256e3, # vacuum permeability (mT^2/kPa)
    "B_rem_mag": 100.0, # mT
    "theta_init_dir": (1.0, 0.0),

    "traction_bcs": [],

    "load_cases": [
        {
            "name": "B_right_push",
            "weight": 1.0,
            "B_app_mag": 125.0,
            "B_app_dir": (1.0, 0.0),
            "tractions": {},
        },
    ],

    "load_steps": 40,

    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "snes_max_it": "500",
        "snes_error_if_not_converged": None,
    },
}


# ============================================================
#  OPTIMIZATION PARAMETERS
# ============================================================

opt = {
    "max_iter": 100,
    "opt_tol": 1e-5,

    "vol_frac_rho": 1.0,

    "vol_frac_phi": 0.15,
    "phi_cap": geom["phi_cap"],

    "solid_zone": false_zone,
    "void_zone": false_zone,

    "phi_void_zone": phi_void_output_plate,

    "penalty": 3.0,
    "epsilon": 1e-6,

    "filter_radius": 0.20,

    "move": 0.01,

    "objective_type": "disp_track",

    "disp_track": [
        {
            "point": (L - 0.01, y_mid),
            "target": (3.0, 0.0),
            "sigma": 0.35,
            "weight": 1.0,
            "components": ("x", "y"),
        },
    ],

    "output_dir": str(
        Path(__file__).resolve().parent
        / "results_ClosedPush_PhiTheta_RhoFixed"
    ),

    "sim_output_interval": 25,
    "sim_image_output_interval": 101,
}


# ============================================================
#  DESIGN VARIABLE TOGGLES
# ============================================================

design_variables = {
    "rho": {
        "active": False,
        "type": "density",
    },
    "phi": {
        "active": True,
        "type": "scalar",
    },
    "theta": {
        "active": True,
        "type": "angle",
    },
}


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    topopt(fem_params, opt, design_variables=design_variables)