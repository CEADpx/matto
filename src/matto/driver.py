# Copyright (c) 2025-2026 Ian Galloway, Prashant K. Jha
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The optimization driver. Takes the problem dictionary from an input
file, builds the design variables, the state problem and the
sensitivity machinery, and runs the MMA loop over the load cases until
the design stops changing and every continuation schedule has run out.
Contains no material-specific equations.
"""

import os
import re
import time

import basix
import dolfinx.io
import numpy as np
from dolfinx import fem
from mpi4py import MPI

from .design import DesignVariable
from .mma import DEFAULT_MOVE, mma_optimizer
from .postprocess import PostProcessor
from .sensitivity import Sensitivity
from .state import StateProblem
from .utility import Communicator, resolve_solver_options


# ================================================================
# General helpers
# ================================================================

def _require_problem_entry(problem, name):
    if name not in problem:
        raise KeyError(
            f"Problem definition is missing '{name}'."
        )

    return problem[name]


def _safe_file_name(name):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(name),
    )


def _owned_size(function):
    index_map = function.function_space.dofmap.index_map
    block_size = function.function_space.dofmap.index_map_bs

    return index_map.size_local * block_size


def _owned_values(function):
    size = _owned_size(function)

    return function.x.array[:size].copy()


def _owned_gradient(values, function):
    if values is None:
        raise ValueError(
            f"Missing gradient for design variable "
            f"'{function.name}'."
        )

    size = _owned_size(function)

    array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if array.size < size:
        raise ValueError(
            f"Gradient for '{function.name}' has "
            f"{array.size} entries but requires {size}."
        )

    return array[:size].copy()


# ================================================================
# Output-field helpers
# ================================================================

def _create_output_function(mesh, output_name, expression):
    shape = expression.ufl_shape

    if shape == ():
        output_space = fem.functionspace(mesh, ("CG", 1))

    else:
        element = basix.ufl.element(
            "Lagrange",
            mesh.basix_cell(),
            1,
            shape=shape,
        )

        output_space = fem.functionspace(mesh, element)

    output_function = fem.Function(
        output_space,
        name=output_name,
    )

    interpolation_expression = fem.Expression(
        expression,
        output_space.element.interpolation_points(),
    )

    return output_function, interpolation_expression


def _prepare_output_fields(mesh, requested_output_fields, available_outputs):
    output_functions = {}
    interpolation_expressions = {}

    for output_name in requested_output_fields:
        if output_name not in available_outputs:
            raise KeyError(
                f"Requested output field '{output_name}' was "
                "not created by state.py or build_output_fields()."
            )

        output_expression = available_outputs[output_name]

        if isinstance(output_expression, fem.Function):
            output_expression.name = output_name
            output_functions[output_name] = output_expression
            interpolation_expressions[output_name] = None

        else:
            (
                output_function,
                interpolation_expression,
            ) = _create_output_function(
                mesh,
                output_name,
                output_expression,
            )

            output_functions[output_name] = output_function
            interpolation_expressions[output_name] = interpolation_expression

    return output_functions, interpolation_expressions


def _update_output_fields(output_functions, interpolation_expressions):
    for output_name, expression in interpolation_expressions.items():
        if expression is not None:
            output_functions[output_name].interpolate(expression)


# ================================================================
# Driver
# ================================================================

class OptimizationDriver:
    """
    Run a material and topology optimization problem.

    The input problem defines:

    - Mesh
    - Communicator (optional ``comm``, default ``mesh.comm``)
    - Design variables
    - Boundary conditions
    - Load cases
    - Free-energy density
    - Objective
    - Constraints
    - Requested output fields
    - FEM, MMA, and output options

    Construction reads and checks the problem and builds the design
    variables, the state problem and the sensitivity object. After
    that the object can be used three ways::

        driver.run()               optimize, save and report
        driver.evaluate()          solve every load case for the current
                                   design; objective, gradients,
                                   constraints, no optimization step
        driver.solve_load_case(c)  one forward analysis

    forward() pushes the raw design through the operator chains; call it
    after changing raw values by hand and before evaluate().
    """

    def __init__(self, problem):
        self.problem = problem

        self._read_problem()
        self._build_design_variables()
        self._build_state()
        self._build_objective_and_constraints()
        self._validate_load_cases()
        self._layout_design_vector()
        self._prepare_output_fields()
        self._read_optimization_options()

    # ============================================================
    # Problem definition
    # ============================================================

    def _read_problem(self):
        problem = self.problem

        self.mesh = _require_problem_entry(problem, "mesh")
        self.mesh_serial = problem.get("mesh_serial", None)
        self.design_settings = _require_problem_entry(
            problem,
            "design_variables",
        )
        self.load_cases = _require_problem_entry(problem, "load_cases")
        self.load_steps = int(_require_problem_entry(problem, "load_steps"))

        self.fem_options = problem.get("fem_options", {})
        self.optimization_options = problem.get("optimization_options", {})
        self.output_options = problem.get("output_options", {})
        self.requested_output_fields = problem.get(
            "requested_output_fields",
            [],
        )
        self.postprocessors = list(problem.get("postprocessors", []))

        for postprocessor in self.postprocessors:
            if not isinstance(postprocessor, PostProcessor):
                raise TypeError(
                    "problem['postprocessors'] must hold matto.postprocess."
                    f"PostProcessor objects, got {type(postprocessor).__name__}."
                )

        if not isinstance(self.design_settings, dict):
            raise TypeError(
                "problem['design_variables'] must be a dictionary."
            )

        if not isinstance(self.load_cases, list):
            raise TypeError(
                "problem['load_cases'] must be a list."
            )

        if len(self.load_cases) == 0:
            raise ValueError(
                "problem['load_cases'] must not be empty."
            )

        if self.load_steps < 1:
            raise ValueError(
                "problem['load_steps'] must be at least 1."
            )

        self.comm = problem.get("comm", self.mesh.comm)

    # ============================================================
    # Design-variable construction
    # ============================================================

    def _build_design_variables(self):
        solver_options = resolve_solver_options(self.fem_options)
        filter_petsc_options = solver_options["filter"].get(
            "petsc_options",
            {},
        )

        self.design_variables = {}
        kernels = {}

        for name, settings in self.design_settings.items():
            self.design_variables[name] = DesignVariable(
                name=name,
                mesh=self.mesh,
                settings=settings,
                petsc_options=filter_petsc_options,
                kernels=kernels,
            )

        self.active_design_variables = {
            name: variable
            for name, variable in self.design_variables.items()
            if variable.active
        }

        if len(self.active_design_variables) == 0:
            raise ValueError(
                "At least one design variable must be active."
            )

        self.active_names = list(self.active_design_variables)

        if self.comm.rank == 0:
            print("[matto] design variables:", flush=True)
            for variable in self.design_variables.values():
                print(f"  {variable.describe()}", flush=True)

    # ============================================================
    # FEM and sensitivity construction
    # ============================================================

    def _build_state(self):
        self.state = StateProblem(self.problem, self.design_variables)

    # ============================================================
    # Objective and constraints
    # ============================================================

    def _build_objective_and_constraints(self):
        state = self.state

        self.objective_form = self.problem["build_objective"](
            state.u_field,
            state.external_work_form,
            state.dx,
        )

        constraints = self.problem["build_constraints"](
            self.design_variables,
            state.dx,
        )

        if not isinstance(constraints, dict):
            raise TypeError("build_constraints() must return a dictionary.")

        required_keys = {"form", "normalize_by", "upper_bound"}

        for name, constraint in constraints.items():
            if not isinstance(constraint, dict):
                raise TypeError(f"Constraint '{name}' must be a dictionary.")

            missing_keys = required_keys - set(constraint)
            if missing_keys:
                raise KeyError(
                    f"Constraint '{name}' is missing: {sorted(missing_keys)}."
                )

            if float(constraint["upper_bound"]) <= 0.0:
                raise ValueError(
                    f"Constraint '{name}' must have a positive upper_bound."
                )

        self.constraints = constraints
        self.constraint_names = list(constraints)

        if len(self.constraint_names) == 0:
            raise ValueError(
                "At least one optimization constraint is required "
                "by the current MMA implementation."
            )

        self.sensitivity = Sensitivity(
            self.comm,
            state,
            self.objective_form,
            self.constraints,
        )

    # ============================================================
    # Validate load cases
    # ============================================================

    def _validate_load_cases(self):
        self.load_case_names = []

        for index, load_case in enumerate(self.load_cases):
            if not isinstance(load_case, dict):
                raise TypeError(
                    f"Load case {index} must be a dictionary."
                )

            load_case_name = load_case.get("name", f"load_case_{index}")

            if load_case_name in self.load_case_names:
                raise ValueError(
                    f"Duplicate load-case name: '{load_case_name}'."
                )

            self.load_case_names.append(load_case_name)

            unknown_tractions = set(
                load_case.get("tractions", {})
            ).difference(self.state.traction_constants)

            if unknown_tractions:
                raise KeyError(
                    f"Load case '{load_case_name}' references "
                    f"unknown traction boundaries: "
                    f"{sorted(unknown_tractions)}"
                )

            unknown_stimuli = set(
                load_case.get("stimuli", {})
            ).difference(self.state.stimuli)

            if unknown_stimuli:
                raise KeyError(
                    f"Load case '{load_case_name}' references "
                    f"unknown stimuli: "
                    f"{sorted(unknown_stimuli)}"
                )

    # ============================================================
    # Local MMA vector layout
    # ============================================================

    def _layout_design_vector(self):
        # Active variables are concatenated, in order, into one local
        # vector for MMA. Bounds are laid out the same way.
        self.design_slices = {}
        self.design_lower_bounds = {}
        self.design_upper_bounds = {}

        offset = 0

        for name, variable in self.active_design_variables.items():
            local_size = _owned_size(variable.raw)

            self.design_slices[name] = slice(offset, offset + local_size)
            self.design_lower_bounds[name] = variable.get_lower_bounds()
            self.design_upper_bounds[name] = variable.get_upper_bounds()

            offset += local_size

        self.design_vector_size = offset

        if self.design_vector_size == 0:
            raise ValueError(
                "The active design vector is empty."
            )

    def _reset_mma_history(self):
        self.design_vector_old_1 = np.zeros(self.design_vector_size, dtype=float)
        self.design_vector_old_2 = np.zeros(self.design_vector_size, dtype=float)
        self.lower_asymptotes = None
        self.upper_asymptotes = None

    # ============================================================
    # Output construction
    # ============================================================

    def _prepare_output_fields(self):
        self.output_dir = os.path.abspath(
            self.output_options.get("output_dir", "results")
        )

        available = {"u": self.state.u_field}

        for name, variable in self.design_variables.items():
            available[f"{name}_raw"] = variable.raw
            available[f"{name}_phys"] = variable.phys

        build_output_fields = self.problem.get("build_output_fields")

        if build_output_fields is not None:
            model_fields = build_output_fields(self.design_variables)

            if not isinstance(model_fields, dict):
                raise TypeError(
                    "build_output_fields() must return a dictionary."
                )

            duplicates = set(available) & set(model_fields)
            if duplicates:
                raise ValueError(
                    "build_output_fields() returned duplicate output names: "
                    f"{sorted(duplicates)}."
                )

            available.update(model_fields)

        (
            self.output_functions,
            self.interpolation_expressions,
        ) = _prepare_output_fields(
            self.mesh,
            self.requested_output_fields,
            available,
        )

        self.simulation_output_interval = int(
            self.output_options.get("sim_output_interval", 1)
        )

        if self.simulation_output_interval < 1:
            raise ValueError(
                "sim_output_interval must be at least 1."
            )

        # Writers are opened by run() and closed when it returns.
        self.output_writers = {}

    def _open_output(self):
        if self.comm.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        self.comm.barrier()

        if len(self.output_functions) > 0:
            fields_to_write = list(self.output_functions.values())

            for load_case_name in self.load_case_names:
                output_path = os.path.join(
                    self.output_dir,
                    "optimized_design_"
                    f"{_safe_file_name(load_case_name)}.bp",
                )

                self.output_writers[load_case_name] = dolfinx.io.VTXWriter(
                    self.comm,
                    output_path,
                    fields_to_write,
                    engine="BP4",
                )

    def _write_output(self, load_case_name, time_value):
        _update_output_fields(
            self.output_functions,
            self.interpolation_expressions,
        )

        writer = self.output_writers.get(load_case_name)

        if writer is not None:
            writer.write(float(time_value))

    # ============================================================
    # Optimization options
    # ============================================================

    def _read_optimization_options(self):
        options = self.optimization_options

        self.maximum_iterations = int(options.get("max_iter", 100))
        self.optimization_tolerance = float(options.get("opt_tol", 1.0e-5))
        self.move_limit = float(options.get("move", DEFAULT_MOVE))

        if self.maximum_iterations < 1:
            raise ValueError(
                "max_iter must be at least 1."
            )

        if self.optimization_tolerance <= 0.0:
            raise ValueError(
                "opt_tol must be positive."
            )

        if self.move_limit <= 0.0:
            raise ValueError(
                "move must be positive."
            )

    # ============================================================
    # Load-case solver
    # ============================================================

    def solve_load_case(self, load_case):
        """Solve one load case; returns its name and the max abs u."""

        load_case_name = load_case.get("name", "unnamed")
        max_displacement = self.state.solve(load_case, self.load_steps)

        return load_case_name, max_displacement

    # ============================================================
    # Optimization loop
    # ============================================================

    def continuation_complete(self):
        return all(
            variable.continuation_complete()
            for variable in self.active_design_variables.values()
        )

    def forward(self, iteration=0):
        """
        Push every raw design field through its operator chain.

        iteration is passed to the operators' continuation schedules;
        the default of 0 never triggers one. Returns True if any
        operator advanced its schedule.
        """

        updated = False

        for variable in self.design_variables.values():
            updated = variable.forward(iteration) or updated

        return updated

    def run(self):
        """
        Iterate to convergence, then re-solve, save and report the design.

        Returns a dictionary with the final objective, constraint values,
        maximum displacements, the number of iterations taken, and the
        output directory. Calling run() again continues from the current
        design and continuation state with a fresh MMA history.
        """

        self.optimization_iteration = 0
        self.change = 2.0 * self.optimization_tolerance
        self._reset_mma_history()

        self.last_objective_value = None
        self.last_constraint_values = None
        self.last_max_displacements = {}

        self._open_output()

        try:
            self._notify("on_start")

            try:
                while (
                    self.optimization_iteration < self.maximum_iterations
                    and (
                        self.change > self.optimization_tolerance
                        or not self.continuation_complete()
                    )
                ):
                    self._iterate()

            except RuntimeError as error:
                # A failed state solve raises on every rank.
                self._notify("on_failure", self.optimization_iteration, error)
                raise

            self._solve_final_design()
            self._save_final_arrays()
            self._write_final_report()

            result = {
                "objective": self.final_objective_value,
                "constraints": self.final_constraint_values,
                "max_displacements": self.final_max_displacements,
                "iterations": self.optimization_iteration,
                "output_dir": self.output_dir,
            }
            self._notify("on_finish", result)

        finally:
            self._close()

        return result

    def _notify(self, hook, *arguments):
        """
        Call one hook of every postprocessor.

        A postprocessor that raises on any rank is reported and dropped
        on all ranks, so the ranks keep making the same collective calls.
        """

        for postprocessor in list(self.postprocessors):
            message = None

            try:
                getattr(postprocessor, hook)(self, *arguments)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"

            failed = self.comm.allreduce(int(message is not None), op=MPI.MAX)

            if failed:
                self.postprocessors.remove(postprocessor)
                print(
                    f"[matto] rank {self.comm.rank}: postprocessor "
                    f"{type(postprocessor).__name__} dropped after {hook}"
                    + (f": {message}" if message else " failed on another rank"),
                    flush=True,
                )

    def _iterate(self):
        iteration_start = time.perf_counter()

        self.optimization_iteration += 1
        iteration = self.optimization_iteration

        operators_updated = self.forward(iteration)

        evaluation = self.evaluate(
            write_output=(iteration % self.simulation_output_interval == 0),
            report=True,
        )
        objective_value = evaluation["objective"]
        constraint_values = evaluation["constraints"]
        max_displacements = evaluation["max_displacements"]

        self._mma_update(
            evaluation["objective_gradients"],
            constraint_values,
            evaluation["constraint_gradients"],
        )

        # A continuation step moves the map itself, so the design has
        # not converged even if MMA barely moved it.
        if operators_updated:
            self.change = max(self.change, 2.0 * self.optimization_tolerance)

        # Iteration report
        iteration_time = time.perf_counter() - iteration_start

        if self.comm.rank == 0:
            constraint_report = ", ".join(
                f"{name}: {constraint_values[name]['value']:.4f}"
                for name in self.constraint_names
            )

            print(
                f"opt_iter: {iteration}, "
                f"opt_time: {iteration_time:.3g} s, "
                f"Obj: {objective_value:.6e}, "
                f"{constraint_report}, "
                f"change: {self.change:.3e}",
                flush=True,
            )

        self.last_objective_value = objective_value
        self.last_constraint_values = constraint_values
        self.last_max_displacements = max_displacements

        self._notify(
            "on_iteration",
            iteration,
            {
                "objective": objective_value,
                "constraints": constraint_values,
                "max_displacements": max_displacements,
                "change": self.change,
                "time": iteration_time,
            },
        )

    def evaluate(self, write_output=False, report=False):
        """
        Solve every load case and accumulate the objective and its gradients.

        The objective is the weighted sum over load cases and the
        gradients are in raw space. Constraints are design-dependent and
        load-independent, so their values and gradients are taken from
        the first load case only.

        Returns a dictionary::

            objective             weighted sum over load cases
            objective_gradients   {variable: owned raw-space array}
            constraints           {name: {"value", "residual"}}
            constraint_gradients  {name: {variable: owned array}}
            max_displacements     {load case: max abs u over all ranks}
        """

        active = self.active_design_variables
        constraint_names = self.constraint_names

        objective_value_total = 0.0

        objective_gradients_total = {
            name: np.zeros(_owned_size(variable.raw), dtype=float)
            for name, variable in active.items()
        }

        constraint_values = None
        constraint_gradients_raw = None

        max_displacements = {}

        for load_case_index, load_case in enumerate(self.load_cases):
            load_case_name, max_displacement = self.solve_load_case(load_case)
            max_displacements[load_case_name] = max_displacement

            if report and self.comm.rank == 0:
                print(
                    f"  [{load_case_name}] "
                    f"max abs displacement: "
                    f"{max_displacement:.4e}",
                    flush=True,
                )

            function_values, physical_gradients = self.sensitivity.evaluate()

            load_case_weight = float(load_case.get("weight", 1.0))

            objective_value_total += (
                load_case_weight * function_values["objective"]
            )

            use_constraints = load_case_index == 0

            if use_constraints:
                constraint_values = {
                    name: dict(values)
                    for name, values in function_values["constraints"].items()
                }

                constraint_gradients_raw = {
                    constraint_name: {}
                    for constraint_name in constraint_names
                }

            # physical -> raw gradients, objective first then constraints
            for variable_name, variable in active.items():
                physical_vectors = [
                    physical_gradients["objective"][variable_name]
                ]

                if use_constraints:
                    physical_vectors.extend(
                        physical_gradients["constraints"][constraint_name][
                            variable_name
                        ]
                        for constraint_name in constraint_names
                    )

                raw_gradients = variable.backward(physical_vectors)

                objective_gradients_total[variable_name] += (
                    load_case_weight
                    * _owned_gradient(raw_gradients[0], variable.raw)
                )

                if use_constraints:
                    for constraint_index, constraint_name in enumerate(
                        constraint_names
                    ):
                        constraint_gradients_raw[constraint_name][
                            variable_name
                        ] = _owned_gradient(
                            raw_gradients[constraint_index + 1],
                            variable.raw,
                        )

            if write_output:
                self._write_output(load_case_name, self.optimization_iteration)

        if constraint_values is None:
            raise RuntimeError(
                "Constraint values were not evaluated."
            )

        if constraint_gradients_raw is None:
            raise RuntimeError(
                "Constraint gradients were not evaluated."
            )

        return {
            "objective": objective_value_total,
            "objective_gradients": objective_gradients_total,
            "constraints": constraint_values,
            "constraint_gradients": constraint_gradients_raw,
            "max_displacements": max_displacements,
        }

    def _mma_update(self, objective_gradients, constraint_values,
                    constraint_gradients):
        """Assemble the local MMA vectors, take one MMA step, unpack."""

        active = self.active_design_variables
        active_names = self.active_names
        constraint_names = self.constraint_names

        design_vector = np.concatenate([
            _owned_values(variable.raw)
            for variable in active.values()
        ])

        minimum_vector = np.concatenate([
            self.design_lower_bounds[name] for name in active_names
        ])

        maximum_vector = np.concatenate([
            self.design_upper_bounds[name] for name in active_names
        ])

        objective_gradient_vector = np.concatenate([
            objective_gradients[name] for name in active_names
        ])

        constraint_residual_vector = np.asarray(
            [
                constraint_values[constraint_name]["residual"]
                for constraint_name in constraint_names
            ],
            dtype=float,
        )

        constraint_gradient_matrix = np.vstack([
            np.concatenate([
                constraint_gradients[constraint_name][variable_name]
                for variable_name in active_names
            ])
            for constraint_name in constraint_names
        ])

        (
            updated_design_vector,
            self.change,
            self.lower_asymptotes,
            self.upper_asymptotes,
        ) = mma_optimizer(
            len(constraint_names),
            self.design_vector_size,
            self.optimization_iteration,
            design_vector,
            minimum_vector,
            maximum_vector,
            self.design_vector_old_1,
            self.design_vector_old_2,
            objective_gradient_vector,
            constraint_residual_vector,
            constraint_gradient_matrix,
            self.lower_asymptotes,
            self.upper_asymptotes,
            comm=self.comm,
            move=self.move_limit,
        )

        self.design_vector_old_2 = self.design_vector_old_1.copy()
        self.design_vector_old_1 = design_vector.copy()

        for variable_name, variable in active.items():
            variable.set_values(
                updated_design_vector[self.design_slices[variable_name]]
            )

    # ============================================================
    # Final physical design
    # ============================================================

    def _solve_final_design(self):
        # Re-solve the final updated design so the output and the
        # reported objective correspond to the design saved below.
        self.forward()

        self.final_objective_value = 0.0
        self.final_constraint_values = None
        self.final_max_displacements = {}

        for load_case_index, load_case in enumerate(self.load_cases):
            load_case_name, max_displacement = self.solve_load_case(load_case)
            self.final_max_displacements[load_case_name] = max_displacement

            function_values, _ = self.sensitivity.evaluate()

            load_case_weight = float(load_case.get("weight", 1.0))

            self.final_objective_value += (
                load_case_weight * function_values["objective"]
            )

            if load_case_index == 0:
                self.final_constraint_values = {
                    name: dict(values)
                    for name, values in function_values["constraints"].items()
                }

            self._write_output(load_case_name, self.optimization_iteration + 1)

        if self.final_constraint_values is None:
            self.final_constraint_values = self.last_constraint_values

        if len(self.final_max_displacements) == 0:
            self.final_max_displacements = self.last_max_displacements

    # ============================================================
    # Save final design arrays
    # ============================================================

    def _save_final_arrays(self):
        comm = self.comm

        if self.mesh_serial is not None or comm.size > 1:
            for variable_name, variable in self.design_variables.items():
                raw_communicator = Communicator(
                    variable.raw.function_space,
                    self.mesh_serial,
                )

                physical_communicator = Communicator(
                    variable.phys.function_space,
                    self.mesh_serial,
                )

                raw_values = raw_communicator.gather(variable.raw)
                physical_values = physical_communicator.gather(variable.phys)

                if comm.rank == 0:
                    self._save_array(variable_name, "raw", raw_values)
                    self._save_array(variable_name, "phys", physical_values)

        else:
            # Serial fallback when no separate serial mesh was supplied.
            if comm.rank == 0:
                for variable_name, variable in self.design_variables.items():
                    self._save_array(
                        variable_name, "raw", variable.raw.x.array.copy()
                    )
                    self._save_array(
                        variable_name, "phys", variable.phys.x.array.copy()
                    )

    def _save_array(self, variable_name, kind, values):
        np.save(
            os.path.join(self.output_dir, f"final_{variable_name}_{kind}.npy"),
            values,
        )

    # ============================================================
    # Final report
    # ============================================================

    def _write_final_report(self):
        if self.comm.rank != 0:
            return

        report_lines = [
            f"FINAL objective value: {self.final_objective_value:.8e}",
            "",
            "FINAL maximum displacements:",
        ]

        for load_case_name, max_displacement in (
            self.final_max_displacements.items()
        ):
            report_lines.append(
                f"  {load_case_name}: {max_displacement:.8e}"
            )

        report_lines.extend([
            "",
            "FINAL constraints:",
        ])

        if self.final_constraint_values is not None:
            for constraint_name in self.constraint_names:
                values = self.final_constraint_values[constraint_name]

                report_lines.append(
                    f"  {constraint_name}: "
                    f"value={values['value']:.8e}, "
                    f"residual={values['residual']:.8e}"
                )

        final_report = "\n".join(report_lines)

        print(final_report, flush=True)

        with open(
            os.path.join(self.output_dir, "final_results.txt"),
            "w",
            encoding="utf-8",
        ) as report_file:
            report_file.write(final_report + "\n")

        print(
            f"Saved final results to: {self.output_dir}",
            flush=True,
        )

    # ============================================================
    # Close output files
    # ============================================================

    def _close(self):
        for writer in self.output_writers.values():
            writer.close()
