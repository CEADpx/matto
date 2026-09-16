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

Major additions to sensitivity.py:
- Material-model-independent adjoint sensitivities
- Generic active design-variable handling
- Generic design-only constraint handling
"""

import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import assemble_scalar, form
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
    set_bc,
)

from .utility import apply_petsc_options


class Sensitivity:
    """
    Objective and constraint sensitivities with respect to the physical fields.

    Nothing here is specific to rho, phi, theta, or any material model.
    It works with whichever active design variables were created by
    operators.py; the forms come from state.py.

    Seen as one reverse sweep,

        raw -> operators -> phys -> state solve -> u -> objective -> J,

    this class is the transpose of the last two stages: it forms dJ/du,
    solves the adjoint for lambda, and returns dJ/dphys for every active
    field. DesignVariable.backward then carries that through the
    operator chain to dJ/draw.

    The gradient vectors returned by evaluate() are work vectors owned
    by this object and are overwritten by the next evaluate().
    """

    def __init__(self, comm, state, objective_form, constraints):
        self.comm = comm

        # ============================================================
        # FEM data
        # ============================================================

        self.bcs = state.bcs

        self.u_field = state.u_field
        self.lambda_field = state.lambda_field

        self.objective_ufl = objective_form
        self.internal_force_ufl = state.internal_force_form

        self.all_design_variables = state.design_variables

        self.active_design_variables = {
            name: variable
            for name, variable in self.all_design_variables.items()
            if variable.active
        }

        if len(self.active_design_variables) == 0:
            raise ValueError(
                "Sensitivity requires at least one active design variable."
            )

        # ============================================================
        # Objective value
        # ============================================================

        self.objective_form = form(self.objective_ufl)

        # ============================================================
        # Objective derivative with respect to displacement
        # ============================================================

        self.objective_depends_on_state = self._depends_on(
            self.objective_ufl,
            self.u_field,
        )

        if self.objective_depends_on_state:
            u_variation = ufl.TestFunction(
                self.u_field.function_space
            )

            self.dobjective_du_form = form(
                ufl.derivative(
                    self.objective_ufl,
                    self.u_field,
                    u_variation,
                )
            )

            self.dobjective_du_vector = create_vector(
                self.dobjective_du_form
            )

        else:
            self.dobjective_du_form = None
            self.dobjective_du_vector = None

        # ============================================================
        # State Jacobian
        # ============================================================

        if not self._depends_on(
            self.internal_force_ufl,
            self.u_field,
        ):
            raise ValueError(
                "The internal-force form does not depend on u_field."
            )

        u_trial = ufl.TrialFunction(
            self.u_field.function_space
        )

        self.dfdu_form = form(
            ufl.derivative(
                self.internal_force_ufl,
                self.u_field,
                u_trial,
            )
        )

        self.dfdu_matrix = create_matrix(
            self.dfdu_form
        )

        # ============================================================
        # Design-variable derivative forms
        # ============================================================

        self.objective_gradient_forms = {}
        self.objective_gradient_vectors = {}

        self.adjoint_contribution_forms = {}
        self.adjoint_contribution_vectors = {}

        # lambda^T f_int as a functional; its derivative with respect to
        # a design field is the vector (df_int/d design)^T lambda, so no
        # design Jacobian matrix is ever assembled.
        adjoint_work_ufl = ufl.action(
            self.internal_force_ufl,
            self.lambda_field,
        )

        for name, variable in self.active_design_variables.items():
            physical_field = variable.phys
            physical_space = physical_field.function_space

            # One work vector per field for the objective gradient,
            # whether or not the objective depends on it directly; the
            # adjoint contribution is added into the same vector.
            self.objective_gradient_vectors[name] = (
                physical_field.x.petsc_vec.copy()
            )

            # --------------------------------------------------------
            # Direct objective derivative
            # --------------------------------------------------------

            if self._depends_on(
                self.objective_ufl,
                physical_field,
            ):
                design_test = ufl.TestFunction(
                    physical_space
                )

                gradient_form = form(
                    ufl.derivative(
                        self.objective_ufl,
                        physical_field,
                        design_test,
                    )
                )

                self.objective_gradient_forms[name] = gradient_form

            else:
                self.objective_gradient_forms[name] = None

            # --------------------------------------------------------
            # Adjoint contribution, (df_int / d design)^T lambda
            # --------------------------------------------------------

            if self._depends_on(
                self.internal_force_ufl,
                physical_field,
            ):
                design_test = ufl.TestFunction(
                    physical_space
                )

                self.adjoint_contribution_forms[name] = form(
                    ufl.derivative(
                        adjoint_work_ufl,
                        physical_field,
                        design_test,
                    )
                )

                self.adjoint_contribution_vectors[name] = (
                    physical_field.x.petsc_vec.copy()
                )

            else:
                self.adjoint_contribution_forms[name] = None
                self.adjoint_contribution_vectors[name] = None

        # ============================================================
        # Adjoint solver
        # ============================================================

        adjoint_options = state.solver_options["adjoint"]

        self.adjoint_solver = PETSc.KSP().create(
            self.comm
        )

        self.adjoint_solver.setOperators(
            self.dfdu_matrix
        )

        self.adjoint_solver.setTolerances(
            rtol=float(adjoint_options.get("rtol", 1.0e-8)),
            atol=float(adjoint_options.get("atol", 1.0e-12)),
        )

        apply_petsc_options(
            self.adjoint_solver,
            adjoint_options.get("petsc_options", {}),
            prefix=f"adjoint_ksp_{id(self)}",
        )

        # ============================================================
        # Constraints
        # ============================================================

        self.constraints = {}

        for constraint_name, specification in constraints.items():
            self._initialize_constraint(
                constraint_name,
                specification,
            )

    # ================================================================
    # Initialization helpers
    # ================================================================

    @staticmethod
    def _depends_on(ufl_form, coefficient):
        """
        Check whether a UFL form explicitly depends on a Function.

        This prevents the creation of invalid zero derivative forms when,
        for example, the compliance objective has no direct dependence on
        a design variable.
        """

        try:
            return any(
                item is coefficient
                for item in ufl_form.coefficients()
            )

        except AttributeError:
            # The normal inputs to this class are UFL forms and should
            # provide coefficients(). This fallback allows UFL to handle
            # another valid form-like object if necessary.
            return True

    def _assemble_scalar_global(self, compiled_form):
        local_value = assemble_scalar(compiled_form)

        return self.comm.allreduce(
            local_value,
            op=MPI.SUM,
        )

    @staticmethod
    def _zero_vector(vector):
        with vector.localForm() as local:
            local.set(0.0)

    @staticmethod
    def _assemble_into_vector(vector, compiled_form):
        # Zero the ghost slots too: assembly adds into them and the
        # reverse scatter below sends them to their owners, so anything
        # left there from the last call would be counted again.
        with vector.localForm() as local:
            local.set(0.0)

        assemble_vector(
            vector,
            compiled_form,
        )

        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )

    def _initialize_constraint(
        self,
        constraint_name,
        specification,
    ):
        # The driver has already checked the keys and the bound.
        constraint_ufl = specification["form"]
        normalization_ufl = specification["normalize_by"]
        upper_bound = float(specification["upper_bound"])

        # State-dependent constraints would require their own adjoint
        # solves. They are intentionally not supported yet.
        if self._depends_on(
            constraint_ufl,
            self.u_field,
        ):
            raise NotImplementedError(
                f"Constraint '{constraint_name}' depends on the "
                "displacement field. State-dependent constraints "
                "are not supported yet."
            )

        if self._depends_on(
            normalization_ufl,
            self.u_field,
        ):
            raise ValueError(
                f"Constraint '{constraint_name}' has a "
                "state-dependent normalization."
            )

        for variable_name, variable in (
            self.all_design_variables.items()
        ):
            if self._depends_on(
                normalization_ufl,
                variable.phys,
            ):
                raise ValueError(
                    f"Constraint '{constraint_name}' normalization "
                    f"depends on design variable '{variable_name}'."
                )

        constraint_form = form(
            constraint_ufl
        )

        normalization_form = form(
            normalization_ufl
        )

        normalization_value = self._assemble_scalar_global(
            normalization_form
        )

        if normalization_value <= 0.0:
            raise ValueError(
                f"Constraint '{constraint_name}' has a "
                "nonpositive normalization value."
            )

        gradient_forms = {}
        gradient_vectors = {}

        for variable_name, variable in (
            self.active_design_variables.items()
        ):
            physical_field = variable.phys

            gradient_vectors[variable_name] = (
                physical_field.x.petsc_vec.copy()
            )

            if self._depends_on(
                constraint_ufl,
                physical_field,
            ):
                design_test = ufl.TestFunction(
                    physical_field.function_space
                )

                gradient_forms[variable_name] = form(
                    ufl.derivative(
                        constraint_ufl,
                        physical_field,
                        design_test,
                    )
                )

            else:
                gradient_forms[variable_name] = None

        self.constraints[constraint_name] = {
            "form": constraint_form,
            "normalization": normalization_value,
            "upper_bound": upper_bound,
            "gradient_forms": gradient_forms,
            "gradient_vectors": gradient_vectors,
        }

    # ================================================================
    # Objective sensitivity
    # ================================================================

    def _assemble_direct_objective_gradients(self):
        gradients = {}

        for name in self.active_design_variables:
            gradient_form = self.objective_gradient_forms[name]
            work_vector = self.objective_gradient_vectors[name]

            if gradient_form is None:
                self._zero_vector(work_vector)
            else:
                self._assemble_into_vector(
                    work_vector,
                    gradient_form,
                )

            gradients[name] = work_vector

        return gradients

    def _solve_adjoint(self):
        if not self.objective_depends_on_state:
            self.lambda_field.x.petsc_vec.zeroEntries()
            self.lambda_field.x.scatter_forward()
            return False

        # ------------------------------------------------------------
        # Assemble -dJ/du
        # ------------------------------------------------------------

        self._assemble_into_vector(
            self.dobjective_du_vector,
            self.dobjective_du_form,
        )

        # Adjoint values must be zero on displacement Dirichlet DOFs.
        set_bc(
            self.dobjective_du_vector,
            self.bcs,
            alpha=0.0,
        )

        self.dobjective_du_vector.scale(-1.0)

        # ------------------------------------------------------------
        # Assemble df_internal/du
        # ------------------------------------------------------------

        self.dfdu_matrix.zeroEntries()

        assemble_matrix(
            self.dfdu_matrix,
            self.dfdu_form,
            bcs=self.bcs,
        )

        self.dfdu_matrix.assemble()

        # ------------------------------------------------------------
        # Solve:
        #
        #     (df_internal/du)^T lambda = -dJ/du
        # ------------------------------------------------------------

        self.lambda_field.x.petsc_vec.zeroEntries()

        self.adjoint_solver.solveTranspose(
            self.dobjective_du_vector,
            self.lambda_field.x.petsc_vec,
        )

        self.lambda_field.x.scatter_forward()

        converged_reason = (
            self.adjoint_solver.getConvergedReason()
        )

        if converged_reason <= 0:
            raise RuntimeError(
                "The adjoint solver failed to converge. "
                f"PETSc reason: {converged_reason}"
            )

        return True

    def _add_adjoint_contributions(
        self,
        objective_gradients,
    ):
        for name in self.active_design_variables:
            contribution_form = self.adjoint_contribution_forms[name]

            if contribution_form is None:
                continue

            adjoint_vector = self.adjoint_contribution_vectors[name]

            self._assemble_into_vector(
                adjoint_vector,
                contribution_form,
            )

            objective_gradients[name].axpy(
                1.0,
                adjoint_vector,
            )

        return objective_gradients

    # ================================================================
    # Constraint sensitivities
    # ================================================================

    def _evaluate_constraints(self):
        constraint_values = {}
        constraint_gradients = {}

        for constraint_name, constraint in (
            self.constraints.items()
        ):
            integral_value = self._assemble_scalar_global(
                constraint["form"]
            )

            normalized_value = (
                integral_value
                / constraint["normalization"]
            )

            upper_bound = constraint["upper_bound"]

            # MMA constraint:
            #
            #     g = value / upper_bound - 1 <= 0

            residual_value = (
                normalized_value
                / upper_bound
                - 1.0
            )

            constraint_values[constraint_name] = {
                "value": float(normalized_value),
                "residual": float(residual_value),
            }

            variable_gradients = {}

            gradient_scale = (
                1.0
                / (
                    constraint["normalization"]
                    * upper_bound
                )
            )

            for variable_name, variable in (
                self.active_design_variables.items()
            ):
                gradient_form = (
                    constraint["gradient_forms"][
                        variable_name
                    ]
                )

                work_vector = (
                    constraint["gradient_vectors"][
                        variable_name
                    ]
                )

                if gradient_form is None:
                    self._zero_vector(work_vector)
                else:
                    self._assemble_into_vector(
                        work_vector,
                        gradient_form,
                    )
                    work_vector.scale(gradient_scale)

                variable_gradients[variable_name] = work_vector

            constraint_gradients[constraint_name] = (
                variable_gradients
            )

        return constraint_values, constraint_gradients

    # ================================================================
    # Public evaluation
    # ================================================================

    def evaluate(self):
        """
        Evaluate the current objective, constraints, and sensitivities.

        Returns
        -------
        function_values:
            {
                "objective": objective_value,

                "constraints": {
                    "constraint_name": {
                        "value": normalized physical value,
                        "residual": MMA constraint g <= 0,
                    },
                },
            }

        gradients:
            {
                "objective": {
                    "variable_name": dJ/d(variable.phys),
                },

                "constraints": {
                    "constraint_name": {
                        "variable_name": dg/d(variable.phys),
                    },
                },
            }
        """

        # ============================================================
        # Objective value
        # ============================================================

        objective_value = self._assemble_scalar_global(
            self.objective_form
        )

        # ============================================================
        # Objective gradient
        # ============================================================

        objective_gradients = (
            self._assemble_direct_objective_gradients()
        )

        adjoint_was_solved = self._solve_adjoint()

        if adjoint_was_solved:
            objective_gradients = (
                self._add_adjoint_contributions(
                    objective_gradients
                )
            )

        # ============================================================
        # Constraints
        # ============================================================

        (
            constraint_values,
            constraint_gradients,
        ) = self._evaluate_constraints()

        # ============================================================
        # Results
        # ============================================================

        function_values = {
            "objective": float(objective_value),
            "constraints": constraint_values,
        }

        gradients = {
            "objective": objective_gradients,
            "constraints": constraint_gradients,
        }

        return function_values, gradients