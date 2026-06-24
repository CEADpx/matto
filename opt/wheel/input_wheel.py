import os
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

# Add project root FinalTop/ to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fenitop.topopt import topopt


# ============================================================
#  WHEEL GEOMETRY PARAMETERS
#  Same geometry as eval wheel
# ============================================================

wheel = {
    "R": 10.0,
    "lc": 0.2,

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
        gmsh.model.add("wheel_spokes_opt")

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

    domain, cell_tags, facet_tags = model_to_mesh(
        gmsh.model,
        comm,
        0,
        gdim=2,
    )

    if rank == 0:
        gmsh.finalize()

    return domain


mesh = build_wheel_spokes_mesh(
    R=wheel["R"],
    lc=wheel["lc"],
    comm=MPI.COMM_WORLD,
)

if MPI.COMM_WORLD.rank == 0:
    mesh_serial = build_wheel_spokes_mesh(
        R=wheel["R"],
        lc=wheel["lc"],
        comm=MPI.COMM_SELF,
    )
else:
    mesh_serial = None


# ============================================================
#  FEM PARAMETERS
# ============================================================

fem_params = {
    "mesh": mesh,
    "mesh_serial": mesh_serial,

    # Mechanical model
    "shear_modulus": 100.0,
    "poisson's ratio": 0.49,
    "hyperelastic": True,
    "hyperModel": "stVenant",

    # G(phi) model used during optimization
    # options: default, guth, mooney, kerner
    "G_model": "kerner",

    # Clamp same square hub/inner-spoke boundary as input_eval.py
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

    # Magnetic loading
    "mu0": 1.256e3,
    "B_rem_mag": 100.0,
    "B_rem_dir": (1.0, 0.0),

    # Used as default applied field; load case overrides it too
    "B_app_mag": 100.0,
    "B_app_dir": (0.0, 1.0),

    "load_cases": [
        {
            "name": "B_up_rotation",
            "weight": 1.0,
            "B_app_mag": 100.0,
            "B_app_dir": (0.0, 1.0),
            "tractions": {},
        },
    ],

    "load_steps": 100,
    "quadrature_degree": 2,

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

    # rho inactive, but keep harmless defaults
    "vol_frac_rho": 1.0,

    # Magnetic material amount
    "vol_frac_phi": .30, 
    "phi_cap": wheel["phi_cap"],

    # No passive zones for now
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone":  lambda x: np.full(x.shape[1], False),

    # Rho penalty irrelevant when rho inactive,
    # but fem.py still expects these keys
    "penalty": 3.0,
    "epsilon": 1e-6,

    # Filtering
    # Larger radius helps prevent noisy theta checkerboarding
    "filter_radius": 1.0,
    "beta_interval": 100,
    "beta_max": 4.0,

    # Optimizer
    "use_oc": False,
    "move": 0.01,

    # Constraints off for first test
    "stress_constraint": False,
    "stress_pnorm": 12,
    "sigma_max": 0.15,

    "strain_constraint": False,
    "U_max": 0.15,

    "strain_ramp": {
        "enabled": False,
        "U_start": 0.35,
        "U_end": 0.15,
        "start_iter": 1,
        "end_iter": 100,
        "schedule": "linear",
    },

    "compliance_constraint": False,
    "compliance_ref": 1.0,
    "compliance_gamma": 1.0,

    # ========================================================
    #  ROTATION OBJECTIVE
    # ========================================================
    # Uses volumetric annular band near outer rim.
    # This avoids fragile rim boundary markers.
    "objective_type": "rotational_disp_band",

    # Center of rotation
    "rotation_center": (0.0, 0.0),

    # Measure rotation near outer rim
    "rotation_radius": 0.95 * wheel["R"],

    # Thickness of smooth objective band
    "rotation_band_sigma": 0.75,

    # +1 rewards CCW rotation, -1 rewards CW rotation
    # With B_app up and initial theta right, your eval wheel tended CCW.
    "rotation_sign": 1.0,

    "rotation_weight": 1.0,

    # Volume handling
    "enforce_volume_equality": False,

    # Output
    "output_dir": str(Path(__file__).resolve().parent / "results_Wheel_Rotation_PhiTheta"),
    "sim_output_interval": 2,
    "sim_image_output_interval": 25,
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