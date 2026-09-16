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

The nonlinear state problem: displacement space, boundary conditions,
load constants, the free-energy density supplied by the material file,
and the residual, objective and constraint forms built from it. No
material-specific code is in this module.
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


class StateProblem:
    """
    Quasi-static equilibrium of a stimulus-responsive solid on a mesh.

    Built once from the input-file dictionary and the DesignVariable
    objects. Afterwards the driver changes the load constants
    (body_force, traction_constants, stimuli) and calls
    nonlinear_problem.solve_fem(); the sensitivity code reads the forms.

    Attributes:
        u_field, lambda_field  displacement and adjoint fields on V
        test_function          the test function the forms are built on
        dx, ds                 measures with the requested quadrature
        W, F, P                energy density, deformation gradient, PK1
        residual_form          L - a, the weak equilibrium statement
        internal_force_form    d(W dx)/du, used by the adjoint
        objective_form         scalar UFL form from build_objective
        constraints            dict of {form, normalize_by, upper_bound}
        output_fields          name -> Function requested for output
        solver_options         resolved {state, adjoint, filter} blocks
    """

    def __init__(self, problem, design_variables):
        self.problem = problem
        self.design_variables = design_variables

        self.mesh = problem["mesh"]
        fem_options = problem.get("fem_options", {})
        self.solver_options = resolve_solver_options(fem_options)
        self.quadrature_degree = fem_options.get("quadrature_degree", 2)

        self.dim = self.mesh.geometry.dim
        self.fdim = self.mesh.topology.dim - 1

        if self.dim != 2:
            raise ValueError(
                "The current multimaterial optimization framework "
                "supports only 2D problems."
            )

        self._build_displacement_space()
        self._build_dirichlet_conditions()
        self._build_measures()
        self._build_load_constants()
        self._build_free_energy()
        self._build_residual()
        self._build_objective()
        self._build_constraints()
        self._build_output_fields()

    # ============================================================
    #  DISPLACEMENT AND ADJOINT FIELDS
    # ============================================================

    def _build_displacement_space(self):
        element = basix.ufl.element(
            "Lagrange",
            self.mesh.basix_cell(),
            1,
            shape=(self.dim,),
        )

        self.V = fem.functionspace(self.mesh, element)
        self.u_field = Function(self.V, name="u")
        self.lambda_field = Function(self.V, name="lambda")
        self.test_function = ufl.TestFunction(self.V)

    # ============================================================
    #  DIRICHLET BOUNDARY CONDITIONS
    # ============================================================

    def _build_dirichlet_conditions(self):
        self.bcs = []

        for bc_definition in self.problem["boundary_conditions"]:
            name = bc_definition["name"]
            boundary_function = bc_definition["on_boundary"]

            value = np.asarray(
                bc_definition["value"],
                dtype=PETSc.ScalarType,
            )

            if value.shape != (self.dim,):
                raise ValueError(
                    f"Dirichlet boundary condition '{name}' must have "
                    f"a value with shape ({self.dim},)."
                )

            location = bc_definition.get("location", "boundary")

            if location == "boundary":
                facets = locate_entities_boundary(
                    self.mesh,
                    self.fdim,
                    boundary_function,
                )
                dofs = locate_dofs_topological(self.V, self.fdim, facets)

            elif location in ("interior", "geometrical"):
                dofs = fem.locate_dofs_geometrical(self.V, boundary_function)

            else:
                raise ValueError(
                    f"Unknown location '{location}' for Dirichlet "
                    f"boundary condition '{name}'."
                )

            bc_value = Constant(self.mesh, value)
            self.bcs.append(dirichletbc(bc_value, dofs, self.V))

    # ============================================================
    #  MEASURES AND TRACTION BOUNDARY MARKERS
    # ============================================================

    def _build_measures(self):
        traction_boundaries = self.problem.get("traction_boundaries", {})

        self.traction_markers = {}
        all_facets = []
        all_markers = []

        for marker, (name, boundary_function) in enumerate(
            traction_boundaries.items()
        ):
            current_facets = locate_entities_boundary(
                self.mesh,
                self.fdim,
                boundary_function,
            )

            self.traction_markers[name] = marker
            all_facets.extend(current_facets)
            all_markers.extend([marker] * len(current_facets))

        metadata = {"quadrature_degree": self.quadrature_degree}

        self.dx = ufl.Measure("dx", domain=self.mesh, metadata=metadata)

        if all_facets:
            all_facets = np.asarray(all_facets, dtype=np.int32)
            all_markers = np.asarray(all_markers, dtype=np.int32)

            _, facet_counts = np.unique(all_facets, return_counts=True)
            if np.any(facet_counts > 1):
                raise ValueError(
                    "Two traction boundaries contain the same mesh facet."
                )

            sort_indices = np.argsort(all_facets)
            self.facet_tags = meshtags(
                self.mesh,
                self.fdim,
                all_facets[sort_indices],
                all_markers[sort_indices],
            )

            self.ds = ufl.Measure(
                "ds",
                domain=self.mesh,
                metadata=metadata,
                subdomain_data=self.facet_tags,
            )

        else:
            self.facet_tags = None
            self.ds = ufl.Measure("ds", domain=self.mesh, metadata=metadata)

        self.traction_measures = {
            name: self.ds(marker)
            for name, marker in self.traction_markers.items()
        }

    # ============================================================
    #  LOAD CONSTANTS
    # ============================================================

    def _build_load_constants(self):
        load_cases = self.problem["load_cases"]
        traction_boundaries = self.problem.get("traction_boundaries", {})

        if not load_cases:
            raise ValueError(
                "problem['load_cases'] must contain at least one load case."
            )

        # The driver writes into these for every load step.
        self.body_force = Constant(
            self.mesh,
            np.zeros(self.dim, dtype=PETSc.ScalarType),
        )

        self.traction_constants = {
            name: Constant(
                self.mesh,
                np.zeros(self.dim, dtype=PETSc.ScalarType),
            )
            for name in traction_boundaries
        }

        # Work out which generic stimuli exist and whether each is a
        # scalar or a vector, checking the load cases agree.
        stimulus_shapes = {}

        for load_case in load_cases:
            case_name = load_case.get("name", "unnamed")

            case_body_force = np.asarray(
                load_case.get("body_force", np.zeros(self.dim)),
                dtype=PETSc.ScalarType,
            )

            if case_body_force.shape != (self.dim,):
                raise ValueError(
                    f"Body force in load case '{case_name}' must have "
                    f"shape ({self.dim},)."
                )

            for traction_name, traction_value in load_case.get(
                "tractions", {}
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

                if traction_value.shape != (self.dim,):
                    raise ValueError(
                        f"Traction '{traction_name}' in load case "
                        f"'{case_name}' must have shape ({self.dim},)."
                    )

            for stimulus_name, stimulus_value in load_case.get(
                "stimuli", {}
            ).items():
                stimulus_shape = np.asarray(
                    stimulus_value,
                    dtype=PETSc.ScalarType,
                ).shape

                if stimulus_name in stimulus_shapes:
                    if stimulus_shapes[stimulus_name] != stimulus_shape:
                        raise ValueError(
                            f"Stimulus '{stimulus_name}' does not have "
                            "the same shape in every load case."
                        )
                else:
                    stimulus_shapes[stimulus_name] = stimulus_shape

        self.stimuli = {}

        for stimulus_name, stimulus_shape in stimulus_shapes.items():
            if stimulus_shape == ():
                initial_value = PETSc.ScalarType(0.0)
            else:
                initial_value = np.zeros(
                    stimulus_shape,
                    dtype=PETSc.ScalarType,
                )

            self.stimuli[stimulus_name] = Constant(self.mesh, initial_value)

    # ============================================================
    #  FREE-ENERGY DENSITY
    # ============================================================

    def _build_free_energy(self):
        build_free_energy = self.problem["build_free_energy"]

        self.W, self.F = build_free_energy(
            self.u_field,
            self.design_variables,
            self.stimuli,
        )

        if self.W.ufl_shape != ():
            raise ValueError(
                "build_free_energy() must return a scalar energy density W."
            )

        if self.F.ufl_shape != grad(self.test_function).ufl_shape:
            raise ValueError(
                "The deformation-gradient variable returned by "
                "build_free_energy() must have the same shape as grad(v)."
            )

        # First Piola-Kirchhoff stress
        self.P = ufl.diff(self.W, self.F)

    # ============================================================
    #  RESIDUAL AND NONLINEAR PROBLEM
    # ============================================================

    def _build_residual(self):
        v = self.test_function

        # internal virtual work
        a = inner(grad(v), self.P) * self.dx

        # external virtual work
        L = inner(v, self.body_force) * self.dx
        for name, traction in self.traction_constants.items():
            L += inner(v, traction) * self.traction_measures[name]

        # Keep the existing residual sign convention.
        self.residual_form = L - a

        self.nonlinear_problem = WrapNonlinearProblem(
            self.u_field,
            self.residual_form,
            self.bcs,
            self.solver_options["state"],
        )

        # Internal force as the derivative of the stored energy; the
        # adjoint differentiates this rather than the residual.
        self.internal_force_form = ufl.derivative(
            self.W * self.dx,
            self.u_field,
            v,
        )

    # ============================================================
    #  OBJECTIVE
    # ============================================================

    def _build_objective(self):
        # External work evaluated with u rather than the test function.
        external_work = inner(self.u_field, self.body_force) * self.dx
        for name, traction in self.traction_constants.items():
            external_work += (
                inner(self.u_field, traction) * self.traction_measures[name]
            )

        self.external_work_form = external_work

        self.objective_form = self.problem["build_objective"](
            self.u_field,
            external_work,
            self.dx,
        )

    # ============================================================
    #  CONSTRAINTS
    # ============================================================

    def _build_constraints(self):
        constraints = self.problem["build_constraints"](
            self.design_variables,
            self.dx,
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

    # ============================================================
    #  REQUESTED OUTPUT FIELDS
    # ============================================================

    def _build_output_fields(self):
        available = {"u": self.u_field}

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

        requested = self.problem.get(
            "requested_output_fields",
            list(available),
        )

        unknown = set(requested) - set(available)
        if unknown:
            raise KeyError(
                f"Unknown requested output fields: {sorted(unknown)}."
            )

        self.output_fields = {name: available[name] for name in requested}
