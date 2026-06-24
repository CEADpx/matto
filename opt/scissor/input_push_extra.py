import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fenitop.topopt import topopt

# ============================================================
#  MESH
# ============================================================

L = 10.0
H = 8.0

plate_width = 0.5
plate_x0 = L - plate_width

phi_buffer = 0.35
phi_void_x0 = plate_x0 - phi_buffer

mesh = create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [L, H]],
    [180, 54],
    cell_type=CellType.quadrilateral,
)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(
        MPI.COMM_SELF,
        [[0.0, 0.0], [L, H]],
        [180, 54],
        CellType.quadrilateral,
    )
else:
    mesh_serial = None


# ============================================================
#  GEOMETRIC HELPERS
# ============================================================

def left_clamp(x):
    return np.isclose(x[0], 0.0)

def right_edge(x):
    return np.isclose(x[0], L)

def output_plate_zone(x):
    return x[0] >= plate_x0

def phi_void_plate_buffer_zone(x):
    return x[0] >= phi_void_x0

def phi_void_zone(x):
    # Allow phi through the full mechanism, but not in/near the output plate.
    return phi_void_plate_buffer_zone(x)


def enforced_x_brace_zone(x):
    """
    Enforced rho=1 connectivity skeleton shaped like |<>|.

    Left center  -> upper/lower midpoint -> right pad center
    """
    y_mid = 0.5 * H
    y_amp = 0.38 * H          # how far the bars spread up/down at x=L/2
    bar_half_width = 0.025 * H  # thickness of each diagonal bar

    xcoord = x[0]
    ycoord = x[1]

    # Only enforce bars from left edge to beginning of output plate
    active_x = xcoord <= plate_x0

    # Centerline for upper/lower bars:
    # spread from y_mid at x=0 to y_mid +/- y_amp at x=L/2,
    # then converge back to y_mid at x=plate_x0.
    left_half = xcoord <= 0.5 * L

    s_left = xcoord / (0.5 * L)
    s_right = (xcoord - 0.5 * L) / (plate_x0 - 0.5 * L)

    y_upper_left = y_mid + y_amp * s_left
    y_lower_left = y_mid - y_amp * s_left

    y_upper_right = y_mid + y_amp * (1.0 - s_right)
    y_lower_right = y_mid - y_amp * (1.0 - s_right)

    y_upper = np.where(left_half, y_upper_left, y_upper_right)
    y_lower = np.where(left_half, y_lower_left, y_lower_right)

    upper_bar = np.abs(ycoord - y_upper) <= bar_half_width
    lower_bar = np.abs(ycoord - y_lower) <= bar_half_width

    return active_x & (upper_bar | lower_bar)

def prescribed_scissor_B_rem_func(x):
    """
    Prescribed remanence directions aligned with the four scissor arms.

    Direction switch occurs at the arm vertex x = L/2.
    """
    xcoord = x[0]
    ycoord = x[1]

    y_mid = 0.5 * H
    left_half = xcoord <= 0.5 * L
    top_half = ycoord >= y_mid

    c = 1.0 / np.sqrt(2.0)

    dirs = np.zeros((2, x.shape[1]), dtype=np.float64)

    # top-left arm: +45 deg
    mask = left_half & top_half
    dirs[0, mask] = c
    dirs[1, mask] = c

    # top-right arm: -45 deg
    mask = (~left_half) & top_half
    dirs[0, mask] = c
    dirs[1, mask] = -c

    # bottom-left arm: -45 deg
    mask = left_half & (~top_half)
    dirs[0, mask] = c
    dirs[1, mask] = -c

    # bottom-right arm: +45 deg
    mask = (~left_half) & (~top_half)
    dirs[0, mask] = c
    dirs[1, mask] = c

    return dirs


def false_zone(x):
    return np.full(x.shape[1], False)


# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,

    "shear_modulus": 100.0,
    "poisson's ratio": 0.49,
    "hyperelastic": True,
    "hyperModel": "neoHookean1",

    "G_model": "mooney",

    # Clamp left edge
    "disp_bc": left_clamp,

    "body_force": (0.0, 0.0),
    "quadrature_degree": 2,

    # Magnetic parameters
    "mu0": 1.256e3,
    "B_rem_mag": 200.0,
    "B_rem_dir": (1.0, 0.0),
    "B_rem_func": prescribed_scissor_B_rem_func,
    "theta_init_dir": (-1.0, 0.0),

    # overwritten by load case
    "B_app_mag": 0.0,
    "B_app_dir": (-1.0, 0.0),

    # Dummy traction marker only
    "traction_bcs": [
        {
            "name": "dummy_right",
            "traction_max": (0.0, 0.0),
            "on_boundary": right_edge,
        },
    ],

    "load_cases": [
        {
            "name": "plate_push_Bx",
            "weight": 1.0,
            "B_app_mag": 400.0,
            "B_app_dir": (1.0, 0.0),
            "tractions": {
                "dummy_right": (0.0, 0.0),
            },
        },
    ],

    "load_steps": 25,

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
    "max_iter": 300,
    "opt_tol": 1e-5,

    "vol_frac_rho": 0.50,

    "vol_frac_phi": 0.15,
    "phi_cap": 0.30,

    # Backward-compatible defaults
    "solid_zone": false_zone,
    "void_zone": false_zone,

    # Rho passive zones:
    # right output plate is mechanically solid
    "rho_solid_zone": lambda x: output_plate_zone(x) | enforced_x_brace_zone(x),
    "rho_void_zone": false_zone,

    # Phi passive zones:
    # no magnetic material in or near the output plate
    "phi_solid_zone": false_zone,
    "phi_void_zone": phi_void_zone,

    "penalty": 3.0,
    "epsilon": 1e-6,

    "filter_radius": 0.20,
    "beta_interval": 100,
    "beta_max": 4.0,

    "use_oc": False,
    "move": 0.01,

    "stress_constraint": False,
    "stress_pnorm": 12,
    "sigma_max": 1.0,

    "strain_constraint": False,
    "U_max": 1.0,
    "strain_ramp": {
        "enabled": False,
        "U_start": 1.0,
        "U_end": 1.0,
        "start_iter": 1,
        "end_iter": 100,
        "schedule": "linear",
    },

    "compliance_constraint": False,
    "compliance_ref": 1.0,
    "compliance_gamma": 1.0,

    "disp_constraint": False,

    # ============================================================
    #  OBJECTIVE: track right pad motion:
    #  moderate +x displacement, near-zero y displacement
    # ============================================================
    "objective_type": "disp_track",

    "disp_track": [
        {
            "point": (L - 0.25 * plate_width, 0.5 * H),
            "target": (2, 0.0),
            "sigma": 0.35,
            "weight": 1.0,
            "components": ("x", "y"),
        },
    ],

    "objective_bcs": [],

    "enforce_volume_equality": False,

    "output_dir": str(Path(__file__).resolve().parent / "results_TonguePush_Bleft_RhoPhiTheta"),
    "sim_output_interval": 2,
    "sim_image_output_interval": 5,
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
        "active": False,
        "type": "angle",
    },
}


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    topopt(fem_params, opt, design_variables=design_variables)