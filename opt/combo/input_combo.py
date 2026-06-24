import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType

# Add FinalTop/ to Python's import path so this script can find fenitop/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fenitop.topopt import topopt

# ============================================================
#  MESH
# ============================================================

# Simple 2D cantilever: 100 × 30 rectangle (thicker beam)

# Simple 2D cantilever: 100 × 20 rectangle
mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [100.0, 20.0]],
    [150, 30],
    cell_type=CellType.quadrilateral,
)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF,
        [[0.0, 0.0], [100.0, 20.0]],
        [150, 30],
        CellType.quadrilateral
    )
else:
    mesh_serial = None


# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,

    # --- Mechanical model ---
    "shear_modulus": 100.0,      # base shear modulus G0
    "poisson's ratio": 0.49,      # only used for Kerner model if selected
    "hyperelastic": True,
    "hyperModel": "neoHookean1", # neoHookean2 ,stVenant

    # --- Shear modulus microstructure model ---
    # options: "default", "guth", "mooney", "kerner"
    "G_model": "mooney",

    # --- Boundary conditions ---
    # Fix left edge (x=0)
    "disp_bc": lambda x: np.isclose(x[0], 0.0),

    # --- Body force (none) ---
    "body_force": (0.0, 0.0),

    "quadrature_degree": 2,

    # ============================================================
    #  MAGNETIC PARAMETERS
    # ============================================================
    "mu0": 1.256e3,                 # magnetic permeability
    "B_rem_mag": 200.0,            # remanent field magnitude
    "B_rem_dir": (1.0, 0.0),        # direction of remanent field (x-direction)
    "B_app_mag": 00.0,
    "B_app_dir": (0.0, 1.0),


    # Mechanical traction applied on right edge downward
    "traction_bcs": [
        {
            "name": "out_right",
            "traction_max": (0.0, 0.0),   # overwritten
            "on_boundary": lambda x: np.isclose(x[0], 100.0),
        },
    ],


    "load_cases": [
        {
            "name": "traction_down_B_up_MooneyN1",
            "weight": 1.0,
            "B_app_mag": 25.0,
            "B_app_dir": (0.0, 1.0),
            "tractions": {
                "out_right": (0.0, -0.50),   # stronger downward traction
            },
        },
    ],

    # Load stepping
    "load_steps": 50,

    # PETSc solver
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

    # Volume fraction for density
    "vol_frac_rho": 0.50,

    # Magnetic material volume fraction
    "vol_frac_phi": 0.10,
    "phi_cap": 0.30,

    # Passive zones (none)
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone":  lambda x: np.full(x.shape[1], False),

    # Penalty
    "penalty": 3.0,
    "epsilon": 1e-6,

    # FILTERING
    "filter_radius": 1.5,
    "beta_interval": 25,
    "beta_max": 4.0,

    # Optimizer
    "use_oc": False,
    "move": 0.005,   # original 0.02

    # Stress constraint
    "stress_constraint": False,
    "stress_pnorm": 12,
    "sigma_max": 0.15,   

    # Strain-energy constraint
    "strain_constraint": False,

    # Base (fallback if ramping disabled)
    "U_max": 0.15,

    # Strain constraint ramping
    "strain_ramp": {
        "enabled": False,        # master toggle
        "U_start": 0.35,         # initial U_max
        "U_end": 0.15,           # final U_max (hard cap)
        "start_iter": 1,        # iteration to start ramping
        "end_iter": 100,         # iteration to finish ramping
        "schedule": "linear",   # "linear" or "exp"
    },


    # ------------------------------------------------------------
    # OBJECTIVE REGULARIZATION (NOT A CONSTRAINT)
    # ------------------------------------------------------------

    # Compliance constraint 
    "compliance_constraint": False,
    "compliance_ref": 0.7,  # start 0.255
    "compliance_gamma": 1.0,  

    # Reference compliance for OBJECTIVE scaling ONLY
    # Use iteration-1 compliance (~3.4e-03 from your logs)
    "compliance_ref": 3.4e-03,

    # Objective
    "objective_type": "compliance", # compliance, min_boundary_disp_norm, min_disp_norm, max_disp, disp_track
    "enforce_volume_equality": False,

    # Output
    "output_dir": str(Path(__file__).resolve().parent / "results_Cantilever_TractionDown_Bup_MooneyN1"),
    "sim_output_interval": 10,
    "sim_image_output_interval": 25,
}

# ============================================================
#  DESIGN VARIABLE TOGGLES
# ============================================================

design_variables = {
    "rho": {
        "active": True,
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