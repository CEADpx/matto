"""Shared helpers for the hMSM restorative-beam tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from dolfinx.mesh import CellType, create_rectangle
from mpi4py import MPI
from petsc4py import PETSc

from matto.state import StateProblem
from matto.operators import DesignVariable
from matto.sensitivity import Sensitivity
from matto.topopt import (
    _owned_gradient,
    _owned_size,
    _owned_values,
    _set_constant,
    _zero_function,
)
from matto.utility import resolve_solver_options

REPO_ROOT = Path(__file__).resolve().parents[1]
HMSM_MATERIAL = REPO_ROOT / "examples" / "hMSM" / "material.py"

BEAM_LENGTH = 100.0
BEAM_HEIGHT = 20.0

# First printed objective of the committed 150 x 30 beam, load_steps=50.
FULL_BEAM_FIRST_OBJECTIVE = 7.151568e02


def load_hmsm_material():
    name = "matto_test_hmsm_material"
    module = sys.modules.get(name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(name, HMSM_MATERIAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


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
    material = load_hmsm_material()
    build_free_energy = material.make_build_free_energy(material_parameters)

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
        "build_free_energy": build_free_energy,
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
            "sim_image_output_interval": 10**9,
        },
    }


class BeamSession:
    """Assemble the beam once and re-evaluate after raw-design changes."""

    def __init__(self, problem):
        self.problem = problem
        self.mesh = problem["mesh"]
        self.comm = problem.get("comm", self.mesh.comm)
        self.load_steps = int(problem["load_steps"])
        self.load_cases = problem["load_cases"]

        solver_options = resolve_solver_options(
            problem.get("fem_options", {})
        )
        filter_petsc = solver_options["filter"].get("petsc_options", {})

        self.design_variables = {}
        for name, settings in problem["design_variables"].items():
            self.design_variables[name] = DesignVariable(
                name=name,
                mesh=self.mesh,
                settings=settings,
                petsc_options=filter_petsc,
            )

        self.state = StateProblem(problem, self.design_variables)
        self.sensitivity = Sensitivity(self.comm, self.state)
        self.fem_problem = self.state.nonlinear_problem
        self.u_field = self.state.u_field
        self.body_force = self.state.body_force
        self.traction_constants = self.state.traction_constants
        self.stimuli = self.state.stimuli

        for variable in self.design_variables.values():
            variable.forward(iteration=1)

    def raw_values(self, name):
        return _owned_values(self.design_variables[name].raw)

    def set_raw_values(self, name, values):
        variable = self.design_variables[name]
        size = _owned_size(variable.raw)
        variable.raw.x.array[:size] = values
        variable.raw.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD,
        )
        variable.forward(iteration=1)

    def _solve_load_case(self, load_case):
        body_force_target = load_case.get(
            "body_force",
            np.zeros_like(self.body_force.value),
        )
        traction_targets = load_case.get("tractions", {})
        stimulus_targets = load_case.get("stimuli", {})

        _zero_function(self.u_field)
        _set_constant(self.body_force, body_force_target, scale=0.0)
        for traction in self.traction_constants.values():
            _set_constant(traction, np.zeros_like(traction.value))
        for stimulus in self.stimuli.values():
            _set_constant(stimulus, np.zeros_like(stimulus.value))

        for step in range(1, self.load_steps + 1):
            load_fraction = step / self.load_steps
            _set_constant(
                self.body_force,
                body_force_target,
                scale=load_fraction,
            )
            for traction_name, traction in self.traction_constants.items():
                target = traction_targets.get(
                    traction_name,
                    np.zeros_like(traction.value),
                )
                _set_constant(traction, target, scale=load_fraction)
            for stimulus_name, stimulus in self.stimuli.items():
                target = stimulus_targets.get(
                    stimulus_name,
                    np.zeros_like(stimulus.value),
                )
                _set_constant(stimulus, target, scale=load_fraction)
            self.fem_problem.solve_fem()

    def evaluate(self):
        objective = 0.0
        gradients = {
            name: np.zeros(
                _owned_size(variable.raw),
                dtype=float,
            )
            for name, variable in self.design_variables.items()
            if variable.active
        }

        for load_case in self.load_cases:
            self._solve_load_case(load_case)
            function_values, physical_gradients = self.sensitivity.evaluate()
            weight = float(load_case.get("weight", 1.0))
            objective += weight * function_values["objective"]

            for name, variable in self.design_variables.items():
                if not variable.active:
                    continue
                raw_gradients = variable.backward(
                    [physical_gradients["objective"][name]]
                )
                gradients[name] += weight * _owned_gradient(
                    raw_gradients[0],
                    variable.raw,
                )

        return float(objective), gradients
