import os
import sys

# Add project root (FinalTop) to python path so fenitop can be imported
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
    "r_hub": None,

    "phi_cap": 0.30,
}

if wheel["r_inner"] is None:
    wheel["r_inner"] = 0.9 * wheel["R"]

if wheel["t"] is None:
    wheel["t"] = 0.025 * wheel["R"]

if wheel["r_hub"] is None:
    wheel["r_hub"] = wheel["t"]


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

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = build_wheel_spokes_mesh(R=wheel["R"], lc=wheel["lc"], comm=MPI.COMM_SELF)
else:
    mesh_serial = None


# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {

    "mesh": mesh,
    "mesh_serial": mesh_serial,

    "shear_modulus": 100.0,
    "poisson's ratio": 0.49,

    "hyperelastic": True,
    "hyperModel": "stVenant",
    "G_model": "default",

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
            "weight": 1.0,
            "B_app_mag": 125.0,
            "B_app_dir": (0.0, 1.0),
            "tractions": {},
        },
    ],

    "load_steps": 100,

    "quadrature_degree": 2,

    "mu0": 1.256e3,
    "B_rem_mag": 100.0,
    "B_rem_dir": (1.0, 0.0),

    "B_app_mag": 100.0, #overwritten
    "B_app_dir": (0.0, 1.0),

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

    "G_models": ["default", "guth", "mooney", "kerner"],
    #"G_models": ["kerner"],
    "hyperelastic_models": ["stVenant", "neoHookean1", "neoHookean2"],
    #"hyperelastic_models": ["stVenant"],

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

    x = coords[:, 0]
    y = coords[:, 1]

    r = np.sqrt(x**2 + y**2)

    R = wheel["R"]
    r_inner = wheel["r_inner"]

    ndofs = coords.shape[0]

    rho = np.ones(ndofs)

    #phi = np.zeros(ndofs)
    #phi[(r > r_inner) & (r <= R)] = wheel["phi_cap"]
    phi = wheel["phi_cap"] * np.ones(ndofs)

    # RADIAL MAGNETIZATION
    #theta = np.arctan2(y, x)
    # CONSTANT UPWARD MAGNETIZATION
    #theta = (np.pi / 2.0) * np.ones_like(x)
    theta = np.zeros_like(x)

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