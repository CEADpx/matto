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
#  WHEEL GEOMETRY PARAMETERS
# ============================================================

wheel = {
    "R": 10.0,
    "lc": 0.06,

    "r_inner": 9.0,
    "t": 0.5,

    "phi_cap": 0.30,
}

if wheel["r_inner"] is None:
    wheel["r_inner"] = 0.9 * wheel["R"]

if wheel["t"] is None:
    wheel["t"] = 0.025 * wheel["R"]


# ============================================================
#  BUILD WHEEL + SPOKES MESH
# ============================================================

def build_wheel_spokes_mesh(R=1.0, lc=0.05, comm=MPI.COMM_WORLD):

    import gmsh

    rank = comm.rank

    if rank == 0:
        gmsh.initialize()
        gmsh.model.add("wheel_spokes")

        r = wheel["r_inner"]
        t = wheel["t"]

        P1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
        P11 = gmsh.model.geo.addPoint(t, 0.0, 0.0, lc)
        P12 = gmsh.model.geo.addPoint(0.0, t, 0.0, lc)
        P2 = gmsh.model.geo.addPoint(0.0, R,   0.0, lc)
        P3 = gmsh.model.geo.addPoint(R,   0.0, 0.0, lc)

        P4 = gmsh.model.geo.addPoint(t, t, 0.0, lc)
        P5 = gmsh.model.geo.addPoint(t, r, 0.0, lc)
        P6 = gmsh.model.geo.addPoint(r, t, 0.0, lc)

        L1 = gmsh.model.geo.addLine(P11, P3)
        L2 = gmsh.model.geo.addCircleArc(P3, P1, P2)
        L3 = gmsh.model.geo.addLine(P2, P12)
        L31 = gmsh.model.geo.addLine(P12, P4)
        L32 = gmsh.model.geo.addLine(P4, P11)

        L4 = gmsh.model.geo.addLine(P4, P5)
        L5 = gmsh.model.geo.addCircleArc(P5, P1, P6)
        L6 = gmsh.model.geo.addLine(P6, P4)

        OuterLoop = gmsh.model.geo.addCurveLoop([L1, L2, L3, L31, L32])
        InnerLoop = gmsh.model.geo.addCurveLoop([L4, L5, L6])

        Surface = gmsh.model.geo.addPlaneSurface([OuterLoop, InnerLoop])

        surfs = [Surface]

        for k in [1, 2, 3]:
            cp = gmsh.model.geo.copy([(2, Surface)])
            gmsh.model.geo.rotate(cp, 0, 0, 0, 0, 0, 1, k*np.pi/2)
            surfs.append(cp[0][1])

        gmsh.model.geo.synchronize()

        gmsh.model.geo.removeAllDuplicates()
        gmsh.model.geo.synchronize()

        all_surfs = [tag for (dim, tag) in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, all_surfs, 1)
        gmsh.model.setPhysicalName(2, 1, "domain")

        gmsh.model.geo.synchronize()
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

mesh = build_wheel_spokes_mesh(R=wheel["R"], lc=wheel["lc"], comm=MPI.COMM_WORLD)

# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {

    "mesh": mesh,

    "shear_modulus": 100.0, # (kPa)

    # Clamp hub boundary (square void edges)
    "disp_bc": lambda x: (
        (
            (np.abs(x[0] - wheel["t"]) < 1e-8) &
            (x[1] >= -wheel["t"] - 1e-8) &
            (x[1] <=  wheel["t"] + 1e-8)
        )
        |
        (
            (np.abs(x[1] - wheel["t"]) < 1e-8) &
            (x[0] >= -wheel["t"] - 1e-8) &
            (x[0] <=  wheel["t"] + 1e-8)
        )
        |
        (
            (np.abs(x[0] + wheel["t"]) < 1e-8) &
            (x[1] >= -wheel["t"] - 1e-8) &
            (x[1] <=  wheel["t"] + 1e-8)
        )
        |
        (
            (np.abs(x[1] + wheel["t"]) < 1e-8) &
            (x[0] >= -wheel["t"] - 1e-8) &
            (x[0] <=  wheel["t"] + 1e-8)
        )
    ),

    "body_force": (0.0, 0.0),
    "traction_bcs": [],

    "load_cases": [
        {
            "name": "B_up",
            "B_app_mag": 125.0, # (mT)
            "B_app_dir": (0.0, 1.0),
            "tractions": {},
        },
    ],

    "load_steps": 100,

    "quadrature_degree": 2,

    "mu0": 1.256e3, # vacuum permeability (mT^2/kPa)
    "B_rem_mag": 100.0, # (mT)

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
#  MEASUREMENT MARKER (RIGHT OUTER RIM)
# ============================================================

def right_rim_marker(x):

    R = wheel["R"]
    t = wheel["t"]

    r = np.sqrt(x[0]**2 + x[1]**2)

    return (
        np.isclose(r, R, atol=2.0*t)
        &
        (x[0] > 0.0)
        &
        (np.abs(x[1]) <= 4.0*t)
    )

eval_config = {

    "G_models": ["default", "guth", "mooney", "hill", "kerner", "LP", "LPA"],
    "hyperelastic_models": ["stVenant", "neoHookean1", "neoHookean2"],

    "output_dir": RESULTS_DIR,

    "write_bp": True,
    "write_csv": True,
    "csv_name": "wheel_model_comparison.csv",

    "measurement_marker": right_rim_marker,
    "compute_compliance": False,
}


# ============================================================
#  DESIGN BUILDER (WHEEL MAGNETIZATION)
# ============================================================

def build_wheel_design(mesh):

    V = fem.functionspace(mesh, ("CG", 1))

    coords = V.tabulate_dof_coordinates()

    ndofs = coords.shape[0]

    # Standard density remains solid throughout the wheel
    rho = np.ones(ndofs)

    # Uniform magnetic particle volume fraction
    phi = wheel["phi_cap"] * np.ones(ndofs)

    # Uniform remanent magnetization in the +x direction
    theta = np.zeros(ndofs)

    return rho, phi, theta

design_source = {
    "type": "callable",
    "builder": build_wheel_design,
}



# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    evaluate(fem_params, eval_config, design_source)