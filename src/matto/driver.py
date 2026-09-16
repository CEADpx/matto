"""
Original FEniTop authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7

Major modifications:
- Ian Galloway (ian.galloway@mines.sdsmt.edu)
- Prashant K. Jha (pjha.sci@gmail.com)

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
from petsc4py import PETSc

from .operators import DesignVariable
from .optimize import DEFAULT_MOVE, mma_optimizer
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


def _set_constant(constant, value, scale=1.0):
    target = np.asarray(
        value,
        dtype=PETSc.ScalarType,
    )

    try:
        constant.value[...] = scale * target

    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Load value with shape {target.shape} does not match "
            f"Constant shape {np.asarray(constant.value).shape}."
        ) from error


def _zero_function(function):
    with function.x.petsc_vec.localForm() as local:
        local.set(0.0)

    function.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD,
    )


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

    Construction sets everything up; run() does the iterations and
    writes the results.
    """

    def __init__(self, problem):
        self.problem = problem

        self._read_problem()
        self._build_design_variables()
        self._build_state()
        self._validate_load_cases()
        self._layout_design_vector()
        self._prepare_output()
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

        for name, settings in self.design_settings.items():
            self.design_variables[name] = DesignVariable(
                name=name,
                mesh=self.mesh,
                settings=settings,
                petsc_options=filter_petsc_options,
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
            print(
                "[topopt] Active design variables: "
                f"{self.active_names}",
                flush=True,
            )

    # ============================================================
    # FEM and sensitivity construction
    # ============================================================

    def _build_state(self):
        self.state = StateProblem(self.problem, self.design_variables)
        self.sensitivity = Sensitivity(self.comm, self.state)

        self.constraint_names = list(self.state.constraints)

        if len(self.constraint_names) == 0:
            raise ValueError(
                "At least one optimization constraint is required "
                "by the current MMA implementation."
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

        self.design_vector_old_1 = np.zeros(
            self.design_vector_size,
            dtype=float,
        )

        self.design_vector_old_2 = np.zeros(
            self.design_vector_size,
            dtype=float,
        )

        self.lower_asymptotes = None
        self.upper_asymptotes = None

    # ============================================================
    # Output construction
    # ============================================================

    def _prepare_output(self):
        self.output_dir = os.path.abspath(
            self.output_options.get("output_dir", "results")
        )

        if self.comm.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        self.comm.barrier()

        (
            self.output_functions,
            self.interpolation_expressions,
        ) = _prepare_output_fields(
            self.mesh,
            self.requested_output_fields,
            self.state.output_fields,
        )

        self.output_writers = {}

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

        self.simulation_output_interval = int(
            self.output_options.get("sim_output_interval", 1)
        )

        if self.simulation_output_interval < 1:
            raise ValueError(
                "sim_output_interval must be at least 1."
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
        """
        Solve one load case from the undeformed state.

        Body force, tractions and stimuli are ramped together over
        load_steps. Returns the load-case name and the maximum absolute
        displacement across all ranks.
        """

        state = self.state
        body_force = state.body_force
        traction_constants = state.traction_constants
        stimuli = state.stimuli

        load_case_name = load_case.get("name", "unnamed")

        body_force_target = load_case.get(
            "body_force",
            np.zeros_like(body_force.value),
        )
        traction_targets = load_case.get("tractions", {})
        stimulus_targets = load_case.get("stimuli", {})

        # Each load case begins from the undeformed state.
        _zero_function(state.u_field)

        # Begin every load at zero.
        _set_constant(body_force, body_force_target, scale=0.0)

        for traction in traction_constants.values():
            _set_constant(traction, np.zeros_like(traction.value))

        for stimulus in stimuli.values():
            _set_constant(stimulus, np.zeros_like(stimulus.value))

        # Ramp all body forces, tractions, and stimuli together.
        for step in range(1, self.load_steps + 1):
            load_fraction = step / self.load_steps

            _set_constant(body_force, body_force_target, scale=load_fraction)

            for traction_name, traction in traction_constants.items():
                target = traction_targets.get(
                    traction_name,
                    np.zeros_like(traction.value),
                )
                _set_constant(traction, target, scale=load_fraction)

            for stimulus_name, stimulus in stimuli.items():
                target = stimulus_targets.get(
                    stimulus_name,
                    np.zeros_like(stimulus.value),
                )
                _set_constant(stimulus, target, scale=load_fraction)

            state.nonlinear_problem.solve_fem()

        displacement_array = state.u_field.x.array

        if displacement_array.size > 0:
            local_max_displacement = float(np.max(np.abs(displacement_array)))
        else:
            local_max_displacement = 0.0

        max_displacement = self.comm.allreduce(
            local_max_displacement,
            op=MPI.MAX,
        )

        return load_case_name, max_displacement

    # ============================================================
    # Optimization loop
    # ============================================================

    def continuation_complete(self):
        return all(
            variable.continuation_complete()
            for variable in self.active_design_variables.values()
        )

    def run(self):
        """Iterate to convergence, then solve, save and report the final design."""

        self.optimization_iteration = 0
        self.change = 2.0 * self.optimization_tolerance

        self.last_objective_value = None
        self.last_constraint_values = None
        self.last_max_displacements = {}

        while (
            self.optimization_iteration < self.maximum_iterations
            and (
                self.change > self.optimization_tolerance
                or not self.continuation_complete()
            )
        ):
            self._iterate()

        self._solve_final_design()
        self._save_final_arrays()
        self._write_final_report()
        self._close()

    def _iterate(self):
        iteration_start = time.perf_counter()

        self.optimization_iteration += 1
        iteration = self.optimization_iteration

        # raw -> physical design variables
        operators_updated = False

        for variable in self.design_variables.values():
            operators_updated = variable.forward(iteration) or operators_updated

        (
            objective_value,
            objective_gradients,
            constraint_values,
            constraint_gradients,
            max_displacements,
        ) = self._evaluate_design(write_output=(
            iteration % self.simulation_output_interval == 0
        ))

        self._mma_update(
            objective_gradients,
            constraint_values,
            constraint_gradients,
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

    def _evaluate_design(self, write_output):
        """
        Solve every load case and accumulate the objective and its gradients.

        The objective is the weighted sum over load cases and the
        gradients are in raw space. Constraints are design-dependent and
        load-independent, so their values and gradients are taken from
        the first load case only.
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

            if self.comm.rank == 0:
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

        return (
            objective_value_total,
            objective_gradients_total,
            constraint_values,
            constraint_gradients_raw,
            max_displacements,
        )

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
        for variable in self.design_variables.values():
            variable.forward(iteration=0)

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
