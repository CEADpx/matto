import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fenitop.evaluate import evaluate


# ============================================================
#  SCISSOR GEOMETRY
# ============================================================

geom = {
    "lc": 0.08,

    "L": 10.0,

    "plate_width": 0.05,

    "arm_width": 0.45,

    "y_mid": 7.0,
    "y_amp": 5.0,
}


# ============================================================
#  BUILD MESH
# ============================================================

def build_closed_push_mesh(lc=0.08, comm=MPI.COMM_WORLD):
    import gmsh
    from dolfinx.io.gmshio import model_to_mesh

    rank = comm.rank

    if rank == 0:
        gmsh.initialize()
        gmsh.model.add("closed_push_eval")

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

        P1 = gmsh.model.geo.addPoint(xL,       y_mid - h, 0.0, lc)
        P2 = gmsh.model.geo.addPoint(xC,       yD - h,    0.0, lc)
        P3 = gmsh.model.geo.addPoint(x_plate0, y_mid - h, 0.0, lc)
        P4 = gmsh.model.geo.addPoint(xR,       y_mid - h, 0.0, lc)
        P5 = gmsh.model.geo.addPoint(xR,       y_mid + h, 0.0, lc)
        P6 = gmsh.model.geo.addPoint(x_plate0, y_mid + h, 0.0, lc)
        P7 = gmsh.model.geo.addPoint(xC,       yU + h,    0.0, lc)
        P8 = gmsh.model.geo.addPoint(xL,       y_mid + h, 0.0, lc)

        Q1 = gmsh.model.geo.addPoint(xL + aw,        y_mid,     0.0, lc)
        Q2 = gmsh.model.geo.addPoint(xC,             yU - h,    0.0, lc)
        Q3 = gmsh.model.geo.addPoint(x_plate0 - aw,  y_mid,     0.0, lc)
        Q4 = gmsh.model.geo.addPoint(xC,             yD + h,    0.0, lc)

        L1 = gmsh.model.geo.addLine(P1, P2)
        L2 = gmsh.model.geo.addLine(P2, P3)
        L3 = gmsh.model.geo.addLine(P3, P4)
        L4 = gmsh.model.geo.addLine(P4, P5)
        L5 = gmsh.model.geo.addLine(P5, P6)
        L6 = gmsh.model.geo.addLine(P6, P7)
        L7 = gmsh.model.geo.addLine(P7, P8)
        L8 = gmsh.model.geo.addLine(P8, P1)

        outer_loop = gmsh.model.geo.addCurveLoop(
            [L1, L2, L3, L4, L5, L6, L7, L8]
        )

        H1 = gmsh.model.geo.addLine(Q1, Q4)
        H2 = gmsh.model.geo.addLine(Q4, Q3)
        H3 = gmsh.model.geo.addLine(Q3, Q2)
        H4 = gmsh.model.geo.addLine(Q2, Q1)

        inner_loop = gmsh.model.geo.addCurveLoop(
            [H1, H2, H3, H4]
        )

        surface = gmsh.model.geo.addPlaneSurface(
            [outer_loop, inner_loop]
        )

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface], 1)

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

# ============================================================
#  BOUNDARIES
# ============================================================

L = geom["L"]

def left_clamp(x):
    return np.isclose(x[0], 0.0)

def right_output_edge(x):
    return np.isclose(x[0], L)

# ============================================================
#  RESULTS DIRECTORY
# ============================================================

result_dir = (
    Path(__file__).resolve().parent
    / "results_ClosedPush_PhiTheta_RhoFixed"
)

# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {
    "mesh": mesh,

    "shear_modulus": 100.0, # kPa

    "G_models": ["mooney"],
    "hyperelastic_models": ["neoHookean2"],

    "disp_bc": left_clamp,

    "body_force": (0.0, 0.0),
    "quadrature_degree": 2,

    "mu0": 1.256e3, # vacuum permeability (mT^2/kPa)
    "B_rem_mag": 100.0, # mT

    "traction_bcs": [],

    # Evaluate the optimized design with the applied field reversed by 180 degrees
    "load_cases": [
        {
            "name": "B_left_close",
            "B_app_mag": 125.0, # mT
            "B_app_dir": (-1.0, 0.0),
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
#  EVALUATION SETTINGS
# ============================================================

eval_config = {
    "G_models": ["mooney"],
    "hyperelastic_models": ["neoHookean2"],

    "output_dir": str(
        result_dir / "eval_B_reversed"
    ),

    "write_bp": True,
    "write_csv": True,

    "csv_name": "reversed_field.csv",

    "measurement_marker": right_output_edge,

}

# ============================================================
#  LOAD OPTIMIZED DESIGN
# ============================================================

design_source = {
    "type": "files",
    "phi": str(result_dir / "final_phi_phys.npy"),
    "theta": str(result_dir / "final_theta_phys.npy"),
}

# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    evaluate(fem_params, eval_config, design_source)