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

Major additions to topopt.py:
- Material-model-independent optimization orchestration
- Generic design-variable handling
- Generic load cases and load stepping
- Generic objective and constraint handling
- Generic requested output fields
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

from .state import StateProblem
from .optimize import DEFAULT_MOVE, mma_optimizer
from .operators import DesignVariable
from .sensitivity import Sensitivity
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


def _expand_bound(bound, size, variable_name):
    values = np.asarray(
        bound,
        dtype=float,
    )

    if values.ndim == 0:
        return np.full(
            size,
            float(values),
            dtype=float,
        )

    values = values.reshape(-1)

    if values.size < size:
        raise ValueError(
            f"Bound array for design variable "
            f"'{variable_name}' has {values.size} entries "
            f"but requires {size}."
        )

    return values[:size].copy()


def _get_design_bounds(variable, settings):
    """
    Return local lower and upper MMA bounds.

    If DesignVariable provides get_bounds(), those bounds are used.
    Otherwise the bounds are created from settings["bounds"].
    """

    size = _owned_size(variable.raw)

    get_bounds = getattr(
        variable,
        "get_bounds",
        None,
    )

    if callable(get_bounds):
        lower_bound, upper_bound = get_bounds()

    elif (
        hasattr(variable, "lower_bounds")
        and hasattr(variable, "upper_bounds")
    ):
        lower_bound = variable.lower_bounds
        upper_bound = variable.upper_bounds

    else:
        if "bounds" not in settings:
            raise KeyError(
                f"Design variable '{variable.name}' "
                "is missing 'bounds'."
            )

        lower_bound, upper_bound = settings["bounds"]

    lower = _expand_bound(
        lower_bound,
        size,
        variable.name,
    )

    upper = _expand_bound(
        upper_bound,
        size,
        variable.name,
    )

    if np.any(lower > upper):
        raise ValueError(
            f"Design variable '{variable.name}' has "
            "lower bounds greater than upper bounds."
        )

    return lower, upper


def _assign_raw_values(variable, values):
    size = _owned_size(variable.raw)

    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if values.size != size:
        raise ValueError(
            f"Cannot assign {values.size} values to "
            f"design variable '{variable.name}', which "
            f"has {size} locally owned values."
        )

    variable.raw.x.array[:size] = values

    variable.raw.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD,
    )

    # DesignVariable may provide this method for fixed regions.
    enforce_fixed_regions = getattr(
        variable,
        "enforce_fixed_regions",
        None,
    )

    if callable(enforce_fixed_regions):
        enforce_fixed_regions()

        variable.raw.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD,
        )


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

def _create_output_function(
    mesh,
    output_name,
    expression,
):
    shape = expression.ufl_shape

    if shape == ():
        output_space = fem.functionspace(
            mesh,
            ("CG", 1),
        )

    else:
        element = basix.ufl.element(
            "Lagrange",
            mesh.basix_cell(),
            1,
            shape=shape,
        )

        output_space = fem.functionspace(
            mesh,
            element,
        )

    output_function = fem.Function(
        output_space,
        name=output_name,
    )

    interpolation_expression = fem.Expression(
        expression,
        output_space.element.interpolation_points(),
    )

    return output_function, interpolation_expression


def _prepare_output_fields(
    mesh,
    requested_output_fields,
    available_outputs,
):
    output_functions = {}
    interpolation_expressions = {}

    for output_name in requested_output_fields:
        if output_name not in available_outputs:
            raise KeyError(
                f"Requested output field '{output_name}' was "
                "not created by fem.py or build_output_fields()."
            )

        output_expression = available_outputs[
            output_name
        ]

        if isinstance(output_expression, fem.Function):
            output_expression.name = output_name

            output_functions[output_name] = (
                output_expression
            )

            interpolation_expressions[
                output_name
            ] = None

        else:
            (
                output_function,
                interpolation_expression,
            ) = _create_output_function(
                mesh,
                output_name,
                output_expression,
            )

            output_functions[output_name] = (
                output_function
            )

            interpolation_expressions[
                output_name
            ] = interpolation_expression

    return output_functions, interpolation_expressions


def _update_output_fields(
    output_functions,
    interpolation_expressions,
):
    for output_name, expression in (
        interpolation_expressions.items()
    ):
        if expression is not None:
            output_functions[
                output_name
            ].interpolate(expression)


# ================================================================
# Main optimization routine
# ================================================================

def topopt(problem):
    """
    Run a topology optimization problem.

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

    topopt.py does not contain material-specific equations.
    """

    # ============================================================
    # Problem definition
    # ============================================================

    mesh = _require_problem_entry(
        problem,
        "mesh",
    )

    mesh_serial = problem.get(
        "mesh_serial",
        None,
    )

    design_settings = _require_problem_entry(
        problem,
        "design_variables",
    )

    load_cases = _require_problem_entry(
        problem,
        "load_cases",
    )

    load_steps = int(
        _require_problem_entry(
            problem,
            "load_steps",
        )
    )

    fem_options = problem.get(
        "fem_options",
        {},
    )

    optimization_options = problem.get(
        "optimization_options",
        {},
    )

    output_options = problem.get(
        "output_options",
        {},
    )

    requested_output_fields = problem.get(
        "requested_output_fields",
        [],
    )

    if not isinstance(design_settings, dict):
        raise TypeError(
            "problem['design_variables'] must be a dictionary."
        )

    if not isinstance(load_cases, list):
        raise TypeError(
            "problem['load_cases'] must be a list."
        )

    if len(load_cases) == 0:
        raise ValueError(
            "problem['load_cases'] must not be empty."
        )

    if load_steps < 1:
        raise ValueError(
            "problem['load_steps'] must be at least 1."
        )

    comm = problem.get(
        "comm",
        mesh.comm,
    )

    # ============================================================
    # Design-variable construction
    # ============================================================

    solver_options = resolve_solver_options(fem_options)
    filter_petsc_options = solver_options["filter"].get(
        "petsc_options",
        {},
    )

    design_variables = {}

    for name, settings in design_settings.items():
        design_variables[name] = DesignVariable(
            name=name,
            mesh=mesh,
            settings=settings,
            petsc_options=filter_petsc_options,
        )

    active_design_variables = {
        name: variable
        for name, variable in design_variables.items()
        if variable.active
    }

    if len(active_design_variables) == 0:
        raise ValueError(
            "At least one design variable must be active."
        )

    active_names = list(
        active_design_variables.keys()
    )

    if comm.rank == 0:
        print(
            "[topopt] Active design variables: "
            f"{active_names}",
            flush=True,
        )

    # ============================================================
    # FEM and sensitivity construction
    # ============================================================

    state = StateProblem(
        problem,
        design_variables,
    )

    sensitivity = Sensitivity(
        comm,
        state,
    )

    fem_problem = state.nonlinear_problem
    u_field = state.u_field

    body_force = state.body_force
    traction_constants = state.traction_constants
    stimuli = state.stimuli

    constraint_names = list(state.constraints)

    if len(constraint_names) == 0:
        raise ValueError(
            "At least one optimization constraint is required "
            "by the current MMA implementation."
        )

    # ============================================================
    # Validate load cases
    # ============================================================

    load_case_names = []

    for index, load_case in enumerate(load_cases):
        if not isinstance(load_case, dict):
            raise TypeError(
                f"Load case {index} must be a dictionary."
            )

        load_case_name = load_case.get(
            "name",
            f"load_case_{index}",
        )

        if load_case_name in load_case_names:
            raise ValueError(
                f"Duplicate load-case name: "
                f"'{load_case_name}'."
            )

        load_case_names.append(
            load_case_name
        )

        unknown_tractions = set(
            load_case.get("tractions", {})
        ).difference(traction_constants)

        if unknown_tractions:
            raise KeyError(
                f"Load case '{load_case_name}' references "
                f"unknown traction boundaries: "
                f"{sorted(unknown_tractions)}"
            )

        unknown_stimuli = set(
            load_case.get("stimuli", {})
        ).difference(stimuli)

        if unknown_stimuli:
            raise KeyError(
                f"Load case '{load_case_name}' references "
                f"unknown stimuli: "
                f"{sorted(unknown_stimuli)}"
            )

    # ============================================================
    # Local MMA vector layout
    # ============================================================

    design_slices = {}
    design_lower_bounds = {}
    design_upper_bounds = {}

    offset = 0

    for name, variable in (
        active_design_variables.items()
    ):
        local_size = _owned_size(
            variable.raw
        )

        design_slices[name] = slice(
            offset,
            offset + local_size,
        )

        (
            design_lower_bounds[name],
            design_upper_bounds[name],
        ) = _get_design_bounds(
            variable,
            design_settings[name],
        )

        offset += local_size

    design_vector_size = offset

    if design_vector_size == 0:
        raise ValueError(
            "The active design vector is empty."
        )

    design_vector_old_1 = np.zeros(
        design_vector_size,
        dtype=float,
    )

    design_vector_old_2 = np.zeros(
        design_vector_size,
        dtype=float,
    )

    lower_asymptotes = None
    upper_asymptotes = None

    # ============================================================
    # Output construction
    # ============================================================

    output_dir = os.path.abspath(
        output_options.get(
            "output_dir",
            "results",
        )
    )

    if comm.rank == 0:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    comm.barrier()

    available_outputs = state.output_fields

    (
        output_functions,
        interpolation_expressions,
    ) = _prepare_output_fields(
        mesh,
        requested_output_fields,
        available_outputs,
    )

    output_writers = {}

    if len(output_functions) > 0:
        fields_to_write = list(
            output_functions.values()
        )

        for load_case_name in load_case_names:
            output_path = os.path.join(
                output_dir,
                "optimized_design_"
                f"{_safe_file_name(load_case_name)}.bp",
            )

            output_writers[
                load_case_name
            ] = dolfinx.io.VTXWriter(
                comm,
                output_path,
                fields_to_write,
                engine="BP4",
            )

    simulation_output_interval = int(
        output_options.get(
            "sim_output_interval",
            1,
        )
    )

    if simulation_output_interval < 1:
        raise ValueError(
            "sim_output_interval must be at least 1."
        )

    # ============================================================
    # Load-case solver
    # ============================================================

    def solve_load_case(load_case):
        load_case_name = load_case.get(
            "name",
            "unnamed",
        )

        body_force_target = load_case.get(
            "body_force",
            np.zeros_like(body_force.value),
        )

        traction_targets = load_case.get(
            "tractions",
            {},
        )

        stimulus_targets = load_case.get(
            "stimuli",
            {},
        )

        # Each load case begins from the undeformed state.
        _zero_function(u_field)

        # Begin every load at zero.
        _set_constant(
            body_force,
            body_force_target,
            scale=0.0,
        )

        for traction in traction_constants.values():
            _set_constant(
                traction,
                np.zeros_like(traction.value),
            )

        for stimulus in stimuli.values():
            _set_constant(
                stimulus,
                np.zeros_like(stimulus.value),
            )

        # Ramp all body forces, tractions, and stimuli together.
        for step in range(1, load_steps + 1):
            load_fraction = (
                step / load_steps
            )

            _set_constant(
                body_force,
                body_force_target,
                scale=load_fraction,
            )

            for traction_name, traction in (
                traction_constants.items()
            ):
                target = traction_targets.get(
                    traction_name,
                    np.zeros_like(traction.value),
                )

                _set_constant(
                    traction,
                    target,
                    scale=load_fraction,
                )

            for stimulus_name, stimulus in (
                stimuli.items()
            ):
                target = stimulus_targets.get(
                    stimulus_name,
                    np.zeros_like(stimulus.value),
                )

                _set_constant(
                    stimulus,
                    target,
                    scale=load_fraction,
                )

            fem_problem.solve_fem()

        displacement_array = u_field.x.array

        if displacement_array.size > 0:
            local_max_displacement = float(
                np.max(
                    np.abs(displacement_array)
                )
            )

        else:
            local_max_displacement = 0.0

        max_displacement = comm.allreduce(
            local_max_displacement,
            op=MPI.MAX,
        )

        return load_case_name, max_displacement

    # ============================================================
    # Optimization options
    # ============================================================

    maximum_iterations = int(
        optimization_options.get(
            "max_iter",
            100,
        )
    )

    optimization_tolerance = float(
        optimization_options.get(
            "opt_tol",
            1.0e-5,
        )
    )

    move_limit = float(
        optimization_options.get(
            "move",
            DEFAULT_MOVE,
        )
    )

    if maximum_iterations < 1:
        raise ValueError(
            "max_iter must be at least 1."
        )

    if optimization_tolerance <= 0.0:
        raise ValueError(
            "opt_tol must be positive."
        )

    if move_limit <= 0.0:
        raise ValueError(
            "move must be positive."
        )

    # ============================================================
    # Optimization loop
    # ============================================================

    def continuation_complete():
        return all(
            variable.continuation_complete()
            for variable in active_design_variables.values()
        )
        
    optimization_iteration = 0
    change = 2.0 * optimization_tolerance

    last_objective_value = None
    last_constraint_values = None
    last_max_displacements = {}

    while (
        optimization_iteration < maximum_iterations
        and (
            change > optimization_tolerance
            or not continuation_complete()
        )
    ):
        iteration_start = time.perf_counter()

        optimization_iteration += 1

        # --------------------------------------------------------
        # Raw -> physical design variables
        # --------------------------------------------------------

        beta_updated = False

        for variable in design_variables.values():
            beta_updated = (
                variable.forward(optimization_iteration)
                or beta_updated
            )

        # --------------------------------------------------------
        # Aggregated objective data
        # --------------------------------------------------------

        objective_value_total = 0.0

        objective_gradients_total = {
            name: np.zeros(
                _owned_size(variable.raw),
                dtype=float,
            )
            for name, variable in (
                active_design_variables.items()
            )
        }

        constraint_values = None
        constraint_gradients_raw = None

        max_displacements = {}

        # --------------------------------------------------------
        # Load cases
        # --------------------------------------------------------

        for load_case_index, load_case in enumerate(
            load_cases
        ):
            (
                load_case_name,
                max_displacement,
            ) = solve_load_case(
                load_case
            )

            max_displacements[
                load_case_name
            ] = max_displacement

            if comm.rank == 0:
                print(
                    f"  [{load_case_name}] "
                    f"max abs displacement: "
                    f"{max_displacement:.4e}",
                    flush=True,
                )

            (
                function_values,
                physical_gradients,
            ) = sensitivity.evaluate()

            load_case_weight = float(
                load_case.get(
                    "weight",
                    1.0,
                )
            )

            objective_value_total += (
                load_case_weight
                * function_values["objective"]
            )

            # Constraints are design-dependent and load-independent.
            # Their values and gradients are taken from the first case.
            use_constraints = (
                load_case_index == 0
            )

            if use_constraints:
                constraint_values = {
                    name: dict(values)
                    for name, values in (
                        function_values[
                            "constraints"
                        ].items()
                    )
                }

                constraint_gradients_raw = {
                    constraint_name: {}
                    for constraint_name in (
                        constraint_names
                    )
                }

            # ----------------------------------------------------
            # Physical -> raw gradients
            # ----------------------------------------------------

            for variable_name, variable in (
                active_design_variables.items()
            ):
                physical_vectors = [
                    physical_gradients[
                        "objective"
                    ][variable_name]
                ]

                if use_constraints:
                    physical_vectors.extend(
                        physical_gradients[
                            "constraints"
                        ][constraint_name][
                            variable_name
                        ]
                        for constraint_name in (
                            constraint_names
                        )
                    )

                raw_gradients = variable.backward(
                    physical_vectors
                )

                objective_gradient_case = (
                    _owned_gradient(
                        raw_gradients[0],
                        variable.raw,
                    )
                )

                objective_gradients_total[
                    variable_name
                ] += (
                    load_case_weight
                    * objective_gradient_case
                )

                if use_constraints:
                    for constraint_index, constraint_name in (
                        enumerate(constraint_names)
                    ):
                        constraint_gradients_raw[
                            constraint_name
                        ][variable_name] = (
                            _owned_gradient(
                                raw_gradients[
                                    constraint_index + 1
                                ],
                                variable.raw,
                            )
                        )

            # ----------------------------------------------------
            # Iteration output
            # ----------------------------------------------------

            if (
                optimization_iteration
                % simulation_output_interval
                == 0
            ):
                _update_output_fields(
                    output_functions,
                    interpolation_expressions,
                )

                writer = output_writers.get(
                    load_case_name
                )

                if writer is not None:
                    writer.write(
                        float(
                            optimization_iteration
                        )
                    )

        if constraint_values is None:
            raise RuntimeError(
                "Constraint values were not evaluated."
            )

        if constraint_gradients_raw is None:
            raise RuntimeError(
                "Constraint gradients were not evaluated."
            )

        # ========================================================
        # Assemble local MMA vectors
        # ========================================================

        design_vector = np.concatenate([
            _owned_values(variable.raw)
            for variable in (
                active_design_variables.values()
            )
        ])

        minimum_vector = np.concatenate([
            design_lower_bounds[name]
            for name in active_names
        ])

        maximum_vector = np.concatenate([
            design_upper_bounds[name]
            for name in active_names
        ])

        objective_gradient_vector = np.concatenate([
            objective_gradients_total[name]
            for name in active_names
        ])

        constraint_residual_vector = np.asarray(
            [
                constraint_values[
                    constraint_name
                ]["residual"]
                for constraint_name in (
                    constraint_names
                )
            ],
            dtype=float,
        )

        constraint_gradient_matrix = np.vstack([
            np.concatenate([
                constraint_gradients_raw[
                    constraint_name
                ][variable_name]
                for variable_name in active_names
            ])
            for constraint_name in constraint_names
        ])

        # ========================================================
        # MMA update
        # ========================================================

        (
            updated_design_vector,
            change,
            lower_asymptotes,
            upper_asymptotes,
        ) = mma_optimizer(
            len(constraint_names),
            design_vector_size,
            optimization_iteration,
            design_vector,
            minimum_vector,
            maximum_vector,
            design_vector_old_1,
            design_vector_old_2,
            objective_gradient_vector,
            constraint_residual_vector,
            constraint_gradient_matrix,
            lower_asymptotes,
            upper_asymptotes,
            comm=comm,
            move=move_limit,
        )

        if beta_updated:
            change = max(
                change,
                2.0 * optimization_tolerance,
            )

        design_vector_old_2 = (
            design_vector_old_1.copy()
        )

        design_vector_old_1 = (
            design_vector.copy()
        )

        # --------------------------------------------------------
        # Unpack updated raw design variables
        # --------------------------------------------------------

        for variable_name, variable in (
            active_design_variables.items()
        ):
            variable_slice = design_slices[
                variable_name
            ]

            _assign_raw_values(
                variable,
                updated_design_vector[
                    variable_slice
                ],
            )

        # ========================================================
        # Iteration report
        # ========================================================

        iteration_time = (
            time.perf_counter()
            - iteration_start
        )

        if comm.rank == 0:
            constraint_report = ", ".join(
                (
                    f"{name}: "
                    f"{constraint_values[name]['value']:.4f}"
                )
                for name in constraint_names
            )

            print(
                f"opt_iter: {optimization_iteration}, "
                f"opt_time: {iteration_time:.3g} s, "
                f"Obj: {objective_value_total:.6e}, "
                f"{constraint_report}, "
                f"change: {change:.3e}",
                flush=True,
            )

        last_objective_value = (
            objective_value_total
        )

        last_constraint_values = (
            constraint_values
        )

        last_max_displacements = (
            max_displacements
        )

    # ============================================================
    # Final physical design
    # ============================================================

    for variable in design_variables.values():
        variable.forward(iteration=0)

    # Re-solve the final updated design so final output and reported
    # objective correspond to the design saved below.
    final_objective_value = 0.0
    final_constraint_values = None
    final_max_displacements = {}

    for load_case_index, load_case in enumerate(
        load_cases
    ):
        (
            load_case_name,
            max_displacement,
        ) = solve_load_case(
            load_case
        )

        final_max_displacements[
            load_case_name
        ] = max_displacement

        (
            function_values,
            _,
        ) = sensitivity.evaluate()

        load_case_weight = float(
            load_case.get(
                "weight",
                1.0,
            )
        )

        final_objective_value += (
            load_case_weight
            * function_values["objective"]
        )

        if load_case_index == 0:
            final_constraint_values = {
                name: dict(values)
                for name, values in (
                    function_values[
                        "constraints"
                    ].items()
                )
            }

        _update_output_fields(
            output_functions,
            interpolation_expressions,
        )

        writer = output_writers.get(
            load_case_name
        )

        if writer is not None:
            writer.write(
                float(
                    optimization_iteration + 1
                )
            )

    if final_constraint_values is None:
        final_constraint_values = (
            last_constraint_values
        )

    if final_objective_value is None:
        final_objective_value = (
            last_objective_value
        )

    if len(final_max_displacements) == 0:
        final_max_displacements = (
            last_max_displacements
        )

    # ============================================================
    # Save final design arrays
    # ============================================================

    if mesh_serial is not None or comm.size > 1:
        for variable_name, variable in (
            design_variables.items()
        ):
            raw_communicator = Communicator(
                variable.raw.function_space,
                mesh_serial,
            )

            physical_communicator = Communicator(
                variable.phys.function_space,
                mesh_serial,
            )

            raw_values = raw_communicator.gather(
                variable.raw
            )

            physical_values = (
                physical_communicator.gather(
                    variable.phys
                )
            )

            if comm.rank == 0:
                np.save(
                    os.path.join(
                        output_dir,
                        f"final_{variable_name}_raw.npy",
                    ),
                    raw_values,
                )

                np.save(
                    os.path.join(
                        output_dir,
                        f"final_{variable_name}_phys.npy",
                    ),
                    physical_values,
                )

    else:
        # Serial fallback when no separate serial mesh was supplied.
        if comm.rank == 0:
            for variable_name, variable in (
                design_variables.items()
            ):
                np.save(
                    os.path.join(
                        output_dir,
                        f"final_{variable_name}_raw.npy",
                    ),
                    variable.raw.x.array.copy(),
                )

                np.save(
                    os.path.join(
                        output_dir,
                        f"final_{variable_name}_phys.npy",
                    ),
                    variable.phys.x.array.copy(),
                )

    # ============================================================
    # Final report
    # ============================================================

    if comm.rank == 0:
        report_lines = [
            (
                "FINAL objective value: "
                f"{final_objective_value:.8e}"
            ),
            "",
            "FINAL maximum displacements:",
        ]

        for load_case_name, max_displacement in (
            final_max_displacements.items()
        ):
            report_lines.append(
                f"  {load_case_name}: "
                f"{max_displacement:.8e}"
            )

        report_lines.extend([
            "",
            "FINAL constraints:",
        ])

        if final_constraint_values is not None:
            for constraint_name in constraint_names:
                values = final_constraint_values[
                    constraint_name
                ]

                report_lines.append(
                    f"  {constraint_name}: "
                    f"value={values['value']:.8e}, "
                    f"residual={values['residual']:.8e}"
                )

        final_report = "\n".join(
            report_lines
        )

        print(
            final_report,
            flush=True,
        )

        with open(
            os.path.join(
                output_dir,
                "final_results.txt",
            ),
            "w",
            encoding="utf-8",
        ) as report_file:
            report_file.write(
                final_report + "\n"
            )

        print(
            f"Saved final results to: "
            f"{output_dir}",
            flush=True,
        )

    # ============================================================
    # Close output files
    # ============================================================

    for writer in output_writers.values():
        writer.close()