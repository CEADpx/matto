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
#  BEAM GEOMETRY PARAMETERS
# ============================================================

beam = {
    "L": 10.0,
    "H": 2.0,
    "lc": 0.08,
    "phi_cap": 0.30,
}

# ============================================================
#  BUILD RECTANGULAR CANTILEVER BEAM MESH
# ============================================================

def build_beam_mesh(L=1.0, H=0.2, lc=0.01, comm=MPI.COMM_WORLD):

    import gmsh

    rank = comm.rank

    if rank == 0:
        gmsh.initialize()
        gmsh.model.add("cantilever_beam")

        p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
        p2 = gmsh.model.geo.addPoint(L,   0.0, 0.0, lc)
        p3 = gmsh.model.geo.addPoint(L,   H,   0.0, lc)
        p4 = gmsh.model.geo.addPoint(0.0, H,   0.0, lc)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surface = gmsh.model.geo.addPlaneSurface([loop])

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface], 1)
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

mesh = build_beam_mesh(
    L=beam["L"],
    H=beam["H"],
    lc=beam["lc"],
    comm=MPI.COMM_WORLD
)

# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {

    "mesh": mesh,

    "shear_modulus": 100.0, # kPa

    # Clamp left edge of cantilever beam
    "disp_bc": lambda x: np.isclose(x[0], 0.0),

    "body_force": (0.0, 0.0),
    "traction_bcs": [],

    "load_cases": [
        {
            "name": "B_small",
            "B_app_mag": 25.0, # (mT)
            "B_app_dir": (0.0, 1.0),
            "tractions": {},
        },
        {
            "name": "B_medium",
            "B_app_mag": 100.0, # (mT)
            "B_app_dir": (0.0, 1.0),
            "tractions": {},
        },
        {
            "name": "B_large",
            "B_app_mag": 400.0, # (mT)
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

eval_config = {

    # Models to be evaluated
    "G_models": ["default", "guth", "mooney", "kerner", "hill", "LP", "LPA"],
    "hyperelastic_models": ["stVenant", "neoHookean1", "neoHookean2"],

    "output_dir": RESULTS_DIR,

    "write_bp": True,
    "write_csv": True,
    "csv_name": "beam_load_sweep_model_comparison.csv",
    "measurement_marker": lambda x: np.isclose(x[0], beam["L"]),

    "compute_compliance": False,
}

# ============================================================
#  DESIGN BUILDER (UNIFORM BEAM MAGNETIZATION)
# ============================================================

def build_beam_design(mesh):

    V = fem.functionspace(mesh, ("CG", 1))

    coords = V.tabulate_dof_coordinates()
    ndofs = coords.shape[0]

    # Standard density stays solid everywhere
    rho = np.ones(ndofs)

    # Uniform magnetic material density
    phi = beam["phi_cap"] * np.ones(ndofs)

    # Remanent magnetization points along the beam, +x direction
    theta = np.zeros(ndofs)

    return rho, phi, theta

design_source = {
    "type": "callable",
    "builder": build_beam_design,
}

# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    evaluate(fem_params, eval_config, design_source)