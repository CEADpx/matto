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
the residual built from it, and the load-stepped Newton solve. No
material-specific or optimization-specific code is in this module.
"""

import numpy as np
import ufl
from mpi4py import MPI
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


def set_constant(constant, value, scale=1.0):
    """Write scale * value into a dolfinx Constant, shape-checked."""
    target = np.asarray(value, dtype=PETSc.ScalarType)

    try:
        constant.value[...] = scale * target

    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Load value with shape {target.shape} does not match "
            f"Constant shape {np.asarray(constant.value).shape}."
        ) from error


def zero_function(function):
    with function.x.petsc_vec.localForm() as local:
        local.set(0.0)

    function.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD,
    )


class StateProblem:
    """
    Quasi-static equilibrium of a stimulus-responsive solid on a mesh.

    Built once from the input-file dictionary and the DesignVariable
    objects. solve(load_case, load_steps) then finds the equilibrium
    displacement for one load case; the sensitivity code reads the
    forms.

    Attributes:
        u_field, lambda_field  displacement and adjoint fields on V
        test_function          the test function the forms are built on
        dx, ds                 measures with the requested quadrature
        body_force, traction_constants, stimuli
                               the load Constants solve() ramps
        W, F, P                energy density, deformation gradient, PK1
        residual_form          L - a, the weak equilibrium statement
        internal_force_form    d(W dx)/du, used by the adjoint
        external_work_form     loads dotted with u, for work objectives
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
        self._build_external_work()

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
        self.material = self.problem.get("material")

        if self.material is not None:
            # A Material declares what it reads; check that before any
            # form is built.
            self.material.check_requirements(self.design_variables, self.stimuli)

            self.F = ufl.variable(ufl.Identity(self.dim) + grad(self.u_field))
            fields = {
                name: self.design_variables[name].phys
                for name in self.material.fields
            }
            stimuli = {name: self.stimuli[name] for name in self.material.stimuli}
            self.W = self.material.energy(self.F, fields, stimuli)

        elif "build_free_energy" in self.problem:
            # The callback form: the input file builds F itself and
            # returns (W, F).
            self.W, self.F = self.problem["build_free_energy"](
                self.u_field,
                self.design_variables,
                self.stimuli,
            )

        else:
            raise KeyError(
                "Problem definition needs a 'material' (a matto.materials."
                "Material) or a 'build_free_energy' callback."
            )

        if self.W.ufl_shape != ():
            raise ValueError(
                "The free-energy density W must be a scalar."
            )

        if self.F.ufl_shape != grad(self.test_function).ufl_shape:
            raise ValueError(
                "The deformation-gradient variable must have the same "
                "shape as grad(v)."
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
    #  EXTERNAL WORK
    # ============================================================

    def _build_external_work(self):
        # Loads dotted with u rather than the test function. An
        # objective built on the work done by the loads uses this.
        external_work = inner(self.u_field, self.body_force) * self.dx
        for name, traction in self.traction_constants.items():
            external_work += (
                inner(self.u_field, traction) * self.traction_measures[name]
            )

        self.external_work_form = external_work

    # ============================================================
    #  SOLVE
    # ============================================================

    def solve(self, load_case, load_steps):
        """
        Find the equilibrium displacement for one load case.

        Starts from the undeformed state and ramps the body force,
        tractions and stimuli of the load case together from zero in
        load_steps equal increments, Newton-solving at each. Loads the
        case does not mention are held at zero. Leaves u_field at the
        converged state and the load Constants at their full values.

        Returns the maximum absolute displacement over all ranks.
        """

        body_force = self.body_force
        traction_constants = self.traction_constants
        stimuli = self.stimuli

        body_force_target = load_case.get(
            "body_force",
            np.zeros_like(body_force.value),
        )
        traction_targets = load_case.get("tractions", {})
        stimulus_targets = load_case.get("stimuli", {})

        zero_function(self.u_field)

        set_constant(body_force, body_force_target, scale=0.0)

        for traction in traction_constants.values():
            set_constant(traction, np.zeros_like(traction.value))

        for stimulus in stimuli.values():
            set_constant(stimulus, np.zeros_like(stimulus.value))

        for step in range(1, load_steps + 1):
            load_fraction = step / load_steps

            set_constant(body_force, body_force_target, scale=load_fraction)

            for traction_name, traction in traction_constants.items():
                target = traction_targets.get(
                    traction_name,
                    np.zeros_like(traction.value),
                )
                set_constant(traction, target, scale=load_fraction)

            for stimulus_name, stimulus in stimuli.items():
                target = stimulus_targets.get(
                    stimulus_name,
                    np.zeros_like(stimulus.value),
                )
                set_constant(stimulus, target, scale=load_fraction)

            self.nonlinear_problem.solve_fem()

        displacement_array = self.u_field.x.array

        if displacement_array.size > 0:
            local_max = float(np.max(np.abs(displacement_array)))
        else:
            local_max = 0.0

        return self.mesh.comm.allreduce(local_max, op=MPI.MAX)
