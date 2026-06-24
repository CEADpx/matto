import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fenitop.evaluate import evaluate

mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [100.0, 20.0]],
    [150, 30],
    cell_type=CellType.quadrilateral,
)

result_dir = Path(__file__).resolve().parent / "results_Cantilever_TractionDown_Bup_MooneyN1"

fem_params = {
    "mesh": mesh,
    "mesh_serial": None,

    "shear_modulus": 100.0,
    "poisson's ratio": 0.49,
    "hyperelastic": True,
    "hyperModel": "neoHookean1",
    "G_model": "mooney",

    "disp_bc": lambda x: np.isclose(x[0], 0.0),
    "body_force": (0.0, 0.0),
    "quadrature_degree": 2,

    "mu0": 1.256e3,
    "B_rem_mag": 60.0,
    "B_rem_dir": (1.0, 0.0),
    "B_app_mag": 0.0,
    "B_app_dir": (0.0, 1.0),

    "traction_bcs": [
        {
            "name": "out_right",
            "traction_max": (0.0, 0.0),
            "on_boundary": lambda x: np.isclose(x[0], 100.0),
        },
    ],

    "load_cases": [
        {
            "name": f"B_{B:g}",
            "weight": 1.0,
            "B_app_mag": float(B),
            "B_app_dir": (0.0, 1.0),
            "tractions": {
                "out_right": (0.0, -0.75),
            },
        }
        for B in [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]
    ],

    "load_steps": 50,

    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "snes_max_it": "500",
        "snes_error_if_not_converged": None,
    },
}

eval_config = {
    "G_models": ["mooney"],
    "hyperelastic_models": ["neoHookean1"],
    "output_dir": str(result_dir / "eval_B_sweepLarge"),
    "write_bp": True,
    "write_csv": True,
    "csv_name": "B_sweep.csv",
    "measurement_marker": lambda x: np.isclose(x[0], 100.0),
    "compute_compliance": True,
}

design_source = {
    "type": "files",
    "rho": str(result_dir / "final_rho_phys.npy"),
    "phi": str(result_dir / "final_phi_phys.npy"),
    "theta": str(result_dir / "final_theta_phys.npy"),
}

if __name__ == "__main__":
    evaluate(fem_params, eval_config, design_source)