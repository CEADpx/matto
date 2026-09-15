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

This module constructs a material-independent nonlinear finite-element
problem from functions and settings supplied by an input file.
"""

import numpy as np
import ufl
from petsc4py import PETSc

import basix
from dolfinx import fem
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.mesh import locate_entities_boundary, meshtags
from ufl import grad, inner

from .utility import WrapNonlinearProblem, resolve_solver_options


def form_fem(problem, design_variables):
    """
    Construct the nonlinear finite-element problem.

    Parameters
    ----------
    problem:
        Complete problem dictionary supplied by the input file.

    design_variables:
        Dictionary of runtime DesignVariable objects created by topopt.py.

    Returns
    -------
    fem_data:
        Dictionary containing the nonlinear problem, fields, load Constants,
        objective, constraints, sensitivity forms, and requested outputs.
    """

    # ============================================================
    #  MESH AND OPTIONS
    # ============================================================

    mesh = problem["mesh"]

    fem_options = problem.get(
        "fem_options",
        {},
    )
    solver_options = resolve_solver_options(fem_options)
    quadrature_degree = fem_options.get(
        "quadrature_degree",
        2,
    )

    dim = mesh.geometry.dim
    fdim = mesh.topology.dim - 1

    if dim != 2:
        raise ValueError(
            "The current multimaterial optimization framework "
            "supports only 2D problems."
        )

    # ============================================================
    #  DISPLACEMENT AND ADJOINT FIELDS
    # ============================================================

    displacement_element = basix.ufl.element(
        "Lagrange",
        mesh.basix_cell(),
        1,
        shape=(dim,),
    )

    V = fem.functionspace(
        mesh,
        displacement_element,
    )

    u_field = Function(
        V,
        name="u",
    )

    lambda_field = Function(
        V,
        name="lambda",
    )

    v = ufl.TestFunction(V)

    # ============================================================
    #  DIRICHLET BOUNDARY CONDITIONS
    # ============================================================

    bcs = []

    for bc_definition in problem["boundary_conditions"]:
        name = bc_definition["name"]
        boundary_function = bc_definition["on_boundary"]

        value = np.asarray(
            bc_definition["value"],
            dtype=PETSc.ScalarType,
        )

        if value.shape != (dim,):
            raise ValueError(
                f"Dirichlet boundary condition '{name}' must have "
                f"a value with shape ({dim},)."
            )

        location = bc_definition.get(
            "location",
            "boundary",
        )

        if location == "boundary":
            facets = locate_entities_boundary(
                mesh,
                fdim,
                boundary_function,
            )

            dofs = locate_dofs_topological(
                V,
                fdim,
                facets,
            )

        elif location in ("interior", "geometrical"):
            dofs = fem.locate_dofs_geometrical(
                V,
                boundary_function,
            )

        else:
            raise ValueError(
                f"Unknown location '{location}' for Dirichlet "
                f"boundary condition '{name}'."
            )

        bc_value = Constant(
            mesh,
            value,
        )

        bcs.append(
            dirichletbc(
                bc_value,
                dofs,
                V,
            )
        )

    # ============================================================
    #  TRACTION BOUNDARY MARKERS
    # ============================================================

    traction_boundaries = problem.get(
        "traction_boundaries",
        {},
    )

    traction_markers = {}

    all_facets = []
    all_markers = []

    for marker, (name, boundary_function) in enumerate(
        traction_boundaries.items()
    ):
        current_facets = locate_entities_boundary(
            mesh,
            fdim,
            boundary_function,
        )

        traction_markers[name] = marker

        all_facets.extend(current_facets)
        all_markers.extend(
            [marker] * len(current_facets)
        )

    metadata = {
        "quadrature_degree": quadrature_degree,
    }

    dx = ufl.Measure(
        "dx",
        domain=mesh,
        metadata=metadata,
    )

    if all_facets:
        all_facets = np.asarray(
            all_facets,
            dtype=np.int32,
        )

        all_markers = np.asarray(
            all_markers,
            dtype=np.int32,
        )

        unique_facets, facet_counts = np.unique(
            all_facets,
            return_counts=True,
        )

        if np.any(facet_counts > 1):
            raise ValueError(
                "Two traction boundaries contain the same mesh facet."
            )

        sort_indices = np.argsort(all_facets)

        facet_tags = meshtags(
            mesh,
            fdim,
            all_facets[sort_indices],
            all_markers[sort_indices],
        )

        ds = ufl.Measure(
            "ds",
            domain=mesh,
            metadata=metadata,
            subdomain_data=facet_tags,
        )

    else:
        facet_tags = None

        ds = ufl.Measure(
            "ds",
            domain=mesh,
            metadata=metadata,
        )

    traction_measures = {
        name: ds(marker)
        for name, marker in traction_markers.items()
    }

    # ============================================================
    #  LOAD CONSTANTS
    # ============================================================

    load_cases = problem["load_cases"]

    if not load_cases:
        raise ValueError(
            "problem['load_cases'] must contain at least one load case."
        )

    # Body force is updated by topopt.py for each load step.
    body_force = Constant(
        mesh,
        np.zeros(
            dim,
            dtype=PETSc.ScalarType,
        ),
    )

    # One traction Constant is created for each named traction boundary.
    traction_constants = {
        name: Constant(
            mesh,
            np.zeros(
                dim,
                dtype=PETSc.ScalarType,
            ),
        )
        for name in traction_boundaries
    }

    # Determine which generic stimuli exist and their scalar/vector shapes.
    stimulus_shapes = {}

    for load_case in load_cases:
        case_name = load_case.get(
            "name",
            "unnamed",
        )

        case_body_force = np.asarray(
            load_case.get(
                "body_force",
                np.zeros(dim),
            ),
            dtype=PETSc.ScalarType,
        )

        if case_body_force.shape != (dim,):
            raise ValueError(
                f"Body force in load case '{case_name}' must have "
                f"shape ({dim},)."
            )

        for traction_name, traction_value in load_case.get(
            "tractions",
            {},
        ).items():
            if traction_name not in traction_boundaries:
                raise KeyError(
                    f"Load case '{case_name}' refers to unknown "
                    f"traction boundary '{traction_name}'."
                )

            traction_value = np.asarray(
                traction_value,
                dtype=PETSc.ScalarType,
            )

            if traction_value.shape != (dim,):
                raise ValueError(
                    f"Traction '{traction_name}' in load case "
                    f"'{case_name}' must have shape ({dim},)."
                )

        for stimulus_name, stimulus_value in load_case.get(
            "stimuli",
            {},
        ).items():
            stimulus_value = np.asarray(
                stimulus_value,
                dtype=PETSc.ScalarType,
            )

            stimulus_shape = stimulus_value.shape

            if stimulus_name in stimulus_shapes:
                if stimulus_shapes[stimulus_name] != stimulus_shape:
                    raise ValueError(
                        f"Stimulus '{stimulus_name}' does not have "
                        "the same shape in every load case."
                    )
            else:
                stimulus_shapes[stimulus_name] = stimulus_shape

    stimuli = {}

    for stimulus_name, stimulus_shape in stimulus_shapes.items():
        if stimulus_shape == ():
            initial_value = PETSc.ScalarType(0.0)
        else:
            initial_value = np.zeros(
                stimulus_shape,
                dtype=PETSc.ScalarType,
            )

        stimuli[stimulus_name] = Constant(
            mesh,
            initial_value,
        )

    # ============================================================
    #  FREE-ENERGY DENSITY
    # ============================================================

    build_free_energy = problem["build_free_energy"]

    W, F = build_free_energy(
        u_field,
        design_variables,
        stimuli,
    )

    if W.ufl_shape != ():
        raise ValueError(
            "build_free_energy() must return a scalar energy density W."
        )

    if F.ufl_shape != grad(v).ufl_shape:
        raise ValueError(
            "The deformation-gradient variable returned by "
            "build_free_energy() must have the same shape as grad(v)."
        )

    # First Piola-Kirchhoff stress tensor
    P = ufl.diff(
        W,
        F,
    )

    # ============================================================
    #  RESIDUAL AND NONLINEAR PROBLEM
    # ============================================================

    # Internal virtual work
    a = inner(
        grad(v),
        P,
    ) * dx

    # External virtual work
    L = inner(
        v,
        body_force,
    ) * dx

    for name, traction in traction_constants.items():
        L += inner(
            v,
            traction,
        ) * traction_measures[name]

    # Preserve the existing residual sign convention.
    R = L - a

    fem_problem = WrapNonlinearProblem(
        u_field,
        R,
        bcs,
        solver_options["state"],
    )

    # ============================================================
    #  INTERNAL-FORCE FORM FOR SENSITIVITY ANALYSIS
    # ============================================================

    f_int = ufl.derivative(
        W * dx,
        u_field,
        v,
    )

    # ============================================================
    #  OBJECTIVE
    # ============================================================

    # External work evaluated using u rather than the test function v.
    external_work = inner(
        u_field,
        body_force,
    ) * dx

    for name, traction in traction_constants.items():
        external_work += inner(
            u_field,
            traction,
        ) * traction_measures[name]

    objective_form = problem["build_objective"](
        u_field,
        external_work,
        dx,
    )

    # ============================================================
    #  CONSTRAINTS
    # ============================================================

    constraints = problem["build_constraints"](
        design_variables,
        dx,
    )

    if not isinstance(constraints, dict):
        raise TypeError(
            "build_constraints() must return a dictionary."
        )

    required_constraint_keys = {
        "form",
        "normalize_by",
        "upper_bound",
    }

    for constraint_name, constraint in constraints.items():
        if not isinstance(constraint, dict):
            raise TypeError(
                f"Constraint '{constraint_name}' must be a dictionary."
            )

        missing_keys = (
            required_constraint_keys
            - set(constraint)
        )

        if missing_keys:
            raise KeyError(
                f"Constraint '{constraint_name}' is missing: "
                f"{sorted(missing_keys)}."
            )

        if float(constraint["upper_bound"]) <= 0.0:
            raise ValueError(
                f"Constraint '{constraint_name}' must have a "
                "positive upper_bound."
            )

    # ============================================================
    #  REQUESTED OUTPUT FIELDS
    # ============================================================

    available_output_fields = {
        "u": u_field,
    }

    for name, variable in design_variables.items():
        available_output_fields[f"{name}_raw"] = variable.raw
        available_output_fields[f"{name}_phys"] = variable.phys

    build_output_fields = problem.get(
        "build_output_fields",
    )

    if build_output_fields is not None:
        model_output_fields = build_output_fields(
            design_variables,
        )

        if not isinstance(model_output_fields, dict):
            raise TypeError(
                "build_output_fields() must return a dictionary."
            )

        duplicate_names = (
            set(available_output_fields)
            & set(model_output_fields)
        )

        if duplicate_names:
            raise ValueError(
                "build_output_fields() returned duplicate output names: "
                f"{sorted(duplicate_names)}."
            )

        available_output_fields.update(
            model_output_fields
        )

    requested_output_names = problem.get(
        "requested_output_fields",
        list(available_output_fields),
    )

    unknown_output_names = (
        set(requested_output_names)
        - set(available_output_fields)
    )

    if unknown_output_names:
        raise KeyError(
            "Unknown requested output fields: "
            f"{sorted(unknown_output_names)}."
        )

    output_fields = {
        name: available_output_fields[name]
        for name in requested_output_names
    }

    # ============================================================
    #  RETURN DATA NEEDED BY TOPOPT AND SENSITIVITY
    # ============================================================

    return {
        "fem_problem": fem_problem,

        "u_field": u_field,
        "lambda_field": lambda_field,
        "test_function": v,

        "design_variables": design_variables,

        "body_force": body_force,
        "traction_constants": traction_constants,
        "traction_markers": traction_markers,
        "stimuli": stimuli,

        "dx": dx,
        "ds": ds,

        "W": W,
        "F": F,
        "P": P,

        "residual_form": R,
        "internal_force_form": f_int,
        "external_work_form": external_work,
        "objective_form": objective_form,
        "constraints": constraints,

        "output_fields": output_fields,
        "solver_options": solver_options,
    }