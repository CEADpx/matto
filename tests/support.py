"""Shared helpers for the hMSM restorative-beam tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dolfinx.mesh import CellType, create_box, create_rectangle
from mpi4py import MPI

from matto.design import volume_constraint
from matto.driver import OptimizationDriver
from matto.materials import HardMagneticSoftMaterial, MagnetoActiveElastomer

REPO_ROOT = Path(__file__).resolve().parents[1]

BEAM_LENGTH = 100.0
BEAM_HEIGHT = 20.0

# First printed objective of the committed 150 x 30 beam, load_steps=50.
FULL_BEAM_FIRST_OBJECTIVE = 7.151568e02


def _design_variable_specs():
    return {
        "rho": {
            "active": True,
            "initial": 0.50,
            "bounds": (0.05, 1.00),
            "prescribed_value": 1.00,
            "raw_space": ("DG", 0),
            "physical_space": ("CG", 1),
            "operators": [
                {"type": "density_filter", "radius": 1.0},
                {
                    "type": "heaviside",
                    "beta_initial": 1.0,
                    "beta_update_interval": 25,
                    "beta_max": 4.0,
                },
            ],
            "fixed_regions": [],
        },
        "phi": {
            "active": True,
            "initial": 0.10,
            "bounds": (0.00, 0.30),
            "prescribed_value": 0.00,
            "raw_space": ("DG", 0),
            "physical_space": ("CG", 1),
            "operators": [
                {"type": "density_filter", "radius": 1.0},
            ],
            "fixed_regions": [],
        },
        "theta": {
            "active": True,
            "initial": 0.0,
            "bounds": (-np.pi, np.pi),
            "prescribed_value": 0.0,
            "raw_space": ("DG", 0),
            "physical_space": ("CG", 1),
            "operators": [
                {"type": "density_filter", "radius": 1.0},
            ],
            "fixed_regions": [],
        },
    }


def build_beam_problem(
    comm=None,
    *,
    nx=12,
    ny=3,
    load_steps=5,
    max_iter=1,
):
    """Restorative beam with the committed physics and a chosen mesh."""
    if comm is None:
        comm = MPI.COMM_WORLD

    mesh = create_rectangle(
        comm,
        [[0.0, 0.0], [BEAM_LENGTH, BEAM_HEIGHT]],
        [nx, ny],
        cell_type=CellType.quadrilateral,
    )

    if comm.rank == 0:
        mesh_serial = create_rectangle(
            MPI.COMM_SELF,
            [[0.0, 0.0], [BEAM_LENGTH, BEAM_HEIGHT]],
            [nx, ny],
            cell_type=CellType.quadrilateral,
        )
    else:
        mesh_serial = None

    material_parameters = {
        "G0": 100.0,
        "p_rho": 3.0,
        "eps_rho": 1.0e-6,
        "mu0": 1.256e3,
        "B_rem_mag": 200.0,
    }
    material = HardMagneticSoftMaterial(**material_parameters)

    def build_objective(u_field, external_work, dx):
        return external_work

    def build_constraints(design_variables, dx):
        rho_phys = design_variables["rho"].phys
        phi_phys = design_variables["phi"].phys
        domain_volume = 1.0 * dx
        return {
            "rho_volume": {
                "form": rho_phys * dx,
                "normalize_by": domain_volume,
                "upper_bound": 0.50,
            },
            "phi_volume": {
                "form": phi_phys * dx,
                "normalize_by": domain_volume,
                "upper_bound": 0.10,
            },
        }

    def build_output_fields(design_variables):
        return {}

    return {
        "mesh": mesh,
        "mesh_serial": mesh_serial,
        "comm": comm,
        "material_parameters": material_parameters,
        "design_variables": _design_variable_specs(),
        "boundary_conditions": [
            {
                "name": "clamped_left",
                "on_boundary": lambda x: np.isclose(x[0], 0.0),
                "value": (0.0, 0.0),
            },
        ],
        "traction_boundaries": {
            "out_right": lambda x: np.isclose(x[0], BEAM_LENGTH),
        },
        "load_steps": load_steps,
        "load_cases": [
            {
                "name": "traction_down_B_up",
                "weight": 1.0,
                "body_force": (0.0, 0.0),
                "tractions": {"out_right": (0.0, -0.50)},
                "stimuli": {"B_app": (0.0, 25.0)},
            },
            {
                "name": "traction_up_B_down",
                "weight": 1.0,
                "body_force": (0.0, 0.0),
                "tractions": {"out_right": (0.0, 0.50)},
                "stimuli": {"B_app": (0.0, -25.0)},
            },
        ],
        "material": material,
        "build_objective": build_objective,
        "build_constraints": build_constraints,
        "build_output_fields": build_output_fields,
        "requested_output_fields": ["u"],
        "fem_options": {
            "quadrature_degree": 2,
            "solver_options": {
                "state": {
                    "atol": 1.0e-4,
                    "rtol": 1.0e-4,
                    "max_it": 50,
                    "petsc_options": {
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                    },
                },
                "adjoint": {
                    "rtol": 1.0e-8,
                    "atol": 1.0e-12,
                    "petsc_options": {
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                    },
                },
                "filter": {
                    "petsc_options": {
                        "ksp_type": "cg",
                        "pc_type": "gamg",
                    },
                },
            },
        },
        "optimization_options": {
            "max_iter": max_iter,
            "opt_tol": 1.0e-5,
            "move": 0.05,
        },
        "output_options": {
            "output_dir": str(REPO_ROOT / "tests" / "_unused_output"),
            "sim_output_interval": 10**9,
        },
    }


def build_mae_beam_problem(comm, nx, ny, load_steps, max_iter=1):
    """
    The hMSM cantilever with the isotropic MAE and one active field.

    rho is prescribed solid and phi, the magnetic fraction, is optimized
    under a downward traction with the field on. Built from the 2D hMSM
    problem so the two share every setting that is not material-specific.
    """

    problem = build_beam_problem(
        comm, nx=nx, ny=ny, load_steps=load_steps, max_iter=max_iter
    )

    mu0 = 4.0 * np.pi * 1.0e-7
    problem["material"] = MagnetoActiveElastomer(
        C1_Ga=108.0, C2_Ga=137.0, a1=4.97, a2=9.147 * mu0,
        b1=174.685, b2=13.024 * mu0, C1_sil=270.0, C2_sil=10.8,
        K=12000.0, mu0=mu0,
    )

    problem["design_variables"] = {
        "rho": {
            "active": False,
            "initial": 1.0,
            "bounds": (0.05, 1.0),
            "operators": [{"type": "density_filter", "radius": 1.0}],
        },
        "phi": {
            "active": True,
            "initial": 0.30,
            "bounds": (0.0, 1.0),
            "operators": [{"type": "density_filter", "radius": 1.0}],
        },
    }

    problem["load_cases"] = [
        {
            "name": "field_on",
            "weight": 1.0,
            "body_force": (0.0, 0.0),
            "tractions": {"out_right": (0.0, -0.50)},
            "stimuli": {"h": 0.45},
        },
    ]

    def build_constraints(design_variables, dx):
        phi_phys = design_variables["phi"].phys
        return {
            "magnetic_material_fraction": volume_constraint(phi_phys, 0.30, dx),
        }

    problem["build_constraints"] = build_constraints
    return problem


def build_hmsm_beam_3d_problem(comm, nx, ny, nz, load_steps, max_iter=1):
    """
    The hMSM beam extruded in z.

    Same fields, same material with dim=3, traction and applied field
    along y as in the plane-strain problem. The 2D problem supplies
    everything that is not dimension-specific.
    """

    problem = build_beam_problem(
        comm, nx=nx, ny=ny, load_steps=load_steps, max_iter=max_iter
    )

    depth = BEAM_HEIGHT
    problem["mesh"] = create_box(
        comm,
        [[0.0, 0.0, 0.0], [BEAM_LENGTH, BEAM_HEIGHT, depth]],
        [nx, ny, nz],
        CellType.hexahedron,
    )
    problem["mesh_serial"] = None

    # The finite-difference check passes only with the filter solved
    # directly: with CG/GAMG at the default tolerance the rho gradient is
    # 8 percent off the finite difference, and tightening the Newton
    # tolerance does not change it.
    problem["fem_options"]["solver_options"]["filter"]["petsc_options"] = {
        "ksp_type": "preonly", "pc_type": "lu",
    }

    parameters = dict(problem["material_parameters"], dim=3)
    problem["material_parameters"] = parameters
    problem["material"] = HardMagneticSoftMaterial(**parameters)

    problem["boundary_conditions"] = [
        {
            "name": "clamped_left",
            "on_boundary": lambda x: np.isclose(x[0], 0.0),
            "value": (0.0, 0.0, 0.0),
        },
    ]
    problem["load_cases"] = [
        {
            "name": "traction_down_B_up",
            "weight": 1.0,
            "body_force": (0.0, 0.0, 0.0),
            "tractions": {"out_right": (0.0, -0.50, 0.0)},
            "stimuli": {"B_app": (0.0, 25.0, 0.0)},
        },
        {
            "name": "traction_up_B_down",
            "weight": 1.0,
            "body_force": (0.0, 0.0, 0.0),
            "tractions": {"out_right": (0.0, 0.50, 0.0)},
            "stimuli": {"B_app": (0.0, -25.0, 0.0)},
        },
    ]
    return problem


class BeamSession:
    """Assemble the beam once and re-evaluate after raw-design changes."""

    def __init__(self, problem):
        self.driver = OptimizationDriver(problem)
        self.design_variables = self.driver.design_variables
        self.driver.forward()

    def raw_values(self, name):
        return self.design_variables[name].get_values()

    def set_raw_values(self, name, values):
        self.design_variables[name].set_values(values)
        self.driver.forward()

    def evaluate(self):
        result = self.driver.evaluate()
        return float(result["objective"]), result["objective_gradients"]
