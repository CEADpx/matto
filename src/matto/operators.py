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

Design-variable parameterization as a chain of operators.

Every design variable is stored twice: a raw field that the optimizer
controls and a physical field that enters the material model. The map
between them is a composition of operators,

    raw --T_1--> ... --T_k--> phys,

and the sensitivity of anything with respect to raw is obtained by
applying the adjoints in the opposite order,

    dJ/draw = T_1^T ... T_k^T dJ/dphys.

Each operator implements its forward map and its adjoint, and may
carry a continuation schedule (the Heaviside sharpness, for instance).
DesignVariable only walks the chain; nothing in it is specific to any
operator.
"""

from abc import ABC, abstractmethod

import numpy as np
import ufl
from dolfinx import la
from dolfinx.fem import Function, form, functionspace
from dolfinx.fem.petsc import create_matrix, assemble_matrix
from petsc4py import PETSc

from .utility import apply_petsc_options


class Operator(ABC):
    """
    One map in the parameterization chain, together with its adjoint.

    An operator reads ``self.input`` and writes ``self.output``. The two
    may be the same Function when the map acts in place.

    forward() must run before backward() within an iteration. Operators
    are free to cache the linearization at the current point during
    forward() and use it in backward().
    """

    def __init__(self, input_field, output_field):
        self.input = input_field
        self.output = output_field

    @abstractmethod
    def forward(self):
        """Apply the map: output <- T(input)."""

    @abstractmethod
    def backward(self, gradients):
        """
        Apply the adjoint to a list of PETSc vectors living on the output
        side and return the corresponding list on the input side.

        Entries may be None and are passed through untouched. The
        returned vectors may alias or overwrite the given ones.
        """

    def update(self, iteration):
        """
        Advance any continuation schedule at the start of an iteration.

        Returns True when the operator changed, so the caller can avoid
        declaring convergence on a step where the map itself moved.
        """
        return False

    @property
    def continuation_complete(self):
        """True once the continuation schedule, if any, has run out."""
        return True


class Identity(Operator):
    """Copy input to output. Used when a variable has no operators."""

    def __init__(self, input_field, output_field):
        super().__init__(input_field, output_field)

        if input_field.function_space != output_field.function_space:
            raise ValueError(
                "Identity requires the same function space on both sides."
            )

    def forward(self):
        self.output.x.petsc_vec.array[:] = self.input.x.petsc_vec.array
        self.output.x.scatter_forward()

    def backward(self, gradients):
        return gradients


class HelmholtzFilter(Operator):
    """
    PDE filter: solve -R^2 lap(phys) + phys = raw with natural boundary
    conditions. The raw and filtered fields may live in different spaces.

    Written as a linear map phys = Kf^{-1} T raw, the adjoint is
    T^T Kf^{-1} since Kf is symmetric.
    """

    def __init__(self, comm, input_field, output_field, radius,
                 petsc_options=None):
        super().__init__(input_field, output_field)

        if petsc_options is None:
            petsc_options = {}

        S0, S = input_field.function_space, output_field.function_space
        u0, u = ufl.TrialFunction(S0), ufl.TrialFunction(S)
        v, self.af = ufl.TestFunction(S), Function(S)

        self.output_wrap = la.create_petsc_vector_wrap(self.output.x)
        self.af_wrap = la.create_petsc_vector_wrap(self.af.x)
        self.vec_s0 = input_field.x.petsc_vec.copy()
        self.vec_s = output_field.x.petsc_vec.copy()

        # Kf and T from the weak form of the Helmholtz equation
        dx = ufl.Measure("dx", metadata={"quadrature_degree": 2})
        Kf_expr = (radius**2*ufl.dot(ufl.grad(u), ufl.grad(v)) + u*v)*dx
        T_expr = u0*v*dx
        Kf_form, T_form = form(Kf_expr), form(T_expr)
        Kf_mat, self.T_mat = create_matrix(Kf_form), create_matrix(T_form)

        self.solver = PETSc.KSP().create(comm)
        self.solver.setOperators(Kf_mat)
        prefix = apply_petsc_options(
            self.solver,
            petsc_options,
            prefix=f"filter_solver_{id(self)}",
        )
        Kf_mat.setOptionsPrefix(prefix)
        Kf_mat.setFromOptions()

        assemble_matrix(Kf_mat, Kf_form)
        Kf_mat.assemble()
        assemble_matrix(self.T_mat, T_form)
        self.T_mat.assemble()
        self.T_mat_transpose = self.T_mat.copy()
        self.T_mat_transpose.transpose()

    def forward(self):
        self.T_mat.mult(self.input.x.petsc_vec, self.vec_s)
        self.solver.solve(self.vec_s, self.output_wrap)
        self.output.x.scatter_forward()

    def backward(self, gradients):
        values = []
        for gradient in gradients:
            if gradient is None:
                values.append(None)
                continue

            self.solver.solve(gradient, self.af_wrap)
            self.af.x.scatter_forward()
            self.T_mat_transpose.mult(self.af.x.petsc_vec, self.vec_s0)
            values.append(self.vec_s0.copy())
        return values


class HeavisideProjection(Operator):
    """
    Smoothed Heaviside projection about the threshold eta.

        phys = [tanh(beta eta) + tanh(beta (x - eta))]
               / [tanh(beta eta) + tanh(beta (1 - eta))],

    with the sharpness beta doubled every ``update_interval`` iterations
    up to ``beta_max``. Acts pointwise, so input and output share a space
    and the adjoint is multiplication by dphys/dx at the current point.
    """

    def __init__(self, input_field, output_field, beta, beta_max,
                 update_interval, eta=0.5):
        super().__init__(input_field, output_field)

        if input_field.function_space != output_field.function_space:
            raise ValueError(
                "Heaviside projection requires the same function space "
                "on both sides."
            )

        self.beta = float(beta)
        self.beta_max = float(beta_max)
        self.update_interval = int(update_interval)
        self.eta = float(eta)
        self.dphys = None

        if self.beta <= 0.0:
            raise ValueError("Heaviside projection needs beta_initial > 0.")

        if self.beta_max < self.beta:
            raise ValueError(
                "Heaviside projection needs beta_max >= beta_initial."
            )

    def update(self, iteration):
        if (
            iteration > 0
            and self.update_interval > 0
            and iteration % self.update_interval == 0
            and self.beta < self.beta_max
        ):
            self.beta = min(2.0 * self.beta, self.beta_max)
            return True
        return False

    @property
    def continuation_complete(self):
        return self.beta >= self.beta_max

    def forward(self):
        beta, eta = self.beta, self.eta
        x = self.input.x.petsc_vec.array
        denominator = np.tanh(beta*eta) + np.tanh(beta*(1-eta))

        self.dphys = beta*(1-np.tanh(beta*(x-eta))**2) / denominator
        self.output.x.petsc_vec.array[:] = (
            np.tanh(beta*eta) + np.tanh(beta*(x-eta))) / denominator
        self.output.x.scatter_forward()

    def backward(self, gradients):
        for gradient in gradients:
            if gradient is not None:
                gradient.array *= self.dphys
        return gradients


def build_operator_chain(name, specs, raw, phys, comm, petsc_options):
    """
    Turn the ``operators`` list of an input file into Operator objects.

    The chain is threaded from raw to phys: each operator reads the
    field the previous one wrote, and the last one writes phys. In-place
    operators that sit before the end get their own intermediate field
    so raw is never overwritten.
    """

    if not specs:
        return [Identity(raw, phys)]

    operators = []
    current = raw

    for index, spec in enumerate(specs):
        operator_type = spec["type"]
        last = index == len(specs) - 1

        if operator_type == "density_filter":
            output = phys if last else Function(
                phys.function_space,
                name=f"{name}_filtered_{index}",
            )
            operator = HelmholtzFilter(
                comm,
                current,
                output,
                radius=float(spec["radius"]),
                petsc_options=petsc_options,
            )

        elif operator_type == "heaviside":
            output = phys if last else Function(
                current.function_space,
                name=f"{name}_projected_{index}",
            )
            beta = float(spec.get("beta_initial", 1.0))
            operator = HeavisideProjection(
                current,
                output,
                beta=beta,
                beta_max=float(spec.get("beta_max", beta)),
                update_interval=int(spec.get("beta_update_interval", 0)),
                eta=float(spec.get("eta", 0.5)),
            )

        else:
            raise ValueError(f"unknown operator '{operator_type}'.")

        operators.append(operator)
        current = operator.output

    return operators


class DesignVariable:
    """
    Manage one optimization design variable.

    The class owns:
        raw  - the variable controlled by MMA
        phys - the variable after the operator chain

    It walks the chain forward to update phys and backward to carry
    sensitivities from phys to raw.
    """

    def __init__(
        self,
        name,
        settings,
        mesh,
        petsc_options=None,
    ):
        self.name = name
        self.settings = settings
        self.mesh = mesh
        self.comm = mesh.comm

        if petsc_options is None:
            petsc_options = {}

        # ----------------------------------------------------
        # Basic settings
        # ----------------------------------------------------

        self.active = settings["active"]
        self.initial = settings["initial"]
        self.prescribed_value = settings["prescribed_value"]

        self.lower_bound = float(settings["bounds"][0])
        self.upper_bound = float(settings["bounds"][1])

        if self.lower_bound >= self.upper_bound:
            raise ValueError(
                f"Design variable '{name}' must have lower_bound < upper_bound."
            )

        # ----------------------------------------------------
        # Function spaces and fields
        # ----------------------------------------------------

        self.raw_space_spec = tuple(settings["raw_space"])
        self.physical_space_spec = tuple(settings["physical_space"])

        self.raw_space = functionspace(
            mesh,
            self.raw_space_spec,
        )

        self.physical_space = functionspace(
            mesh,
            self.physical_space_spec,
        )

        self.raw = Function(
            self.raw_space,
            name=f"{name}_raw",
        )

        self.phys = Function(
            self.physical_space,
            name=f"{name}_phys",
        )

        self.size = self.raw.x.petsc_vec.array.size

        # ----------------------------------------------------
        # Parameterization operators
        # ----------------------------------------------------

        try:
            self.operators = build_operator_chain(
                name,
                settings.get("operators", []),
                self.raw,
                self.phys,
                self.comm,
                petsc_options,
            )
        except ValueError as error:
            raise ValueError(
                f"Design variable '{name}': {error}"
            ) from error

        # ----------------------------------------------------
        # Fixed regions
        # ----------------------------------------------------

        self.fixed_regions = []

        coordinates = (
            self.raw_space.tabulate_dof_coordinates()[:self.size].T
        )

        for region in settings.get("fixed_regions", []):
            where = region["where"]
            value = float(region["value"])

            mask = np.asarray(
                where(coordinates),
                dtype=bool,
            )

            if mask.size != self.size:
                raise ValueError(
                    f"Fixed-region mask for design variable '{name}' "
                    "has the wrong size."
                )

            if value < self.lower_bound or value > self.upper_bound:
                raise ValueError(
                    f"Fixed-region value {value} for design variable "
                    f"'{name}' lies outside its bounds."
                )

            self.fixed_regions.append((mask, value))

        # ----------------------------------------------------
        # Initialization
        # ----------------------------------------------------

        if self.active:
            self._assign(self.raw, self.initial)
            self.apply_fixed_regions()
            self.forward(iteration=0)
        else:
            # For an inactive variable, prescribed_value represents
            # the physical field used directly by the material model.
            self._assign(self.raw, self.prescribed_value)
            self._assign(self.phys, self.prescribed_value)

    def _assign(self, field, value):
        """Assign a scalar or spatial callable to a Function."""

        if callable(value):
            field.interpolate(value)
            field.x.scatter_forward()
            return

        with field.x.petsc_vec.localForm() as local:
            local.set(float(value))

        field.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD,
        )

    def apply_fixed_regions(self):
        """Reapply prescribed raw values in fixed regions."""

        if not self.active:
            return

        values = self.raw.x.petsc_vec.array

        for mask, fixed_value in self.fixed_regions:
            values[mask] = fixed_value

        self.raw.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD,
        )

    def continuation_complete(self):
        """True once every operator has finished its continuation."""

        return all(
            operator.continuation_complete
            for operator in self.operators
        )

    def forward(self, iteration):
        """
        Map raw values to physical values.

        Returns True when any operator advanced its continuation this
        iteration.
        """

        if not self.active:
            self._assign(self.phys, self.prescribed_value)
            return False

        self.apply_fixed_regions()

        # raw -> T_1 -> ... -> T_k -> phys
        updated = False
        for operator in self.operators:
            updated = operator.update(iteration) or updated
            operator.forward()

        return updated

    def backward(self, physical_gradients):
        """
        Map sensitivity vectors from phys back to raw.

        physical_gradients must be a list of PETSc vectors.
        The returned values are NumPy arrays in raw-variable space.
        """

        if not isinstance(physical_gradients, (list, tuple)):
            raise TypeError(
                "physical_gradients must be a list or tuple."
            )

        if not self.active:
            return [
                (
                    np.zeros(self.size, dtype=float)
                    if gradient is not None
                    else None
                )
                for gradient in physical_gradients
            ]

        # Operators may work on the vectors in place, so pass them copies.
        gradients = [
            (
                gradient.copy()
                if gradient is not None
                else None
            )
            for gradient in physical_gradients
        ]

        # phys -> T_k^T -> ... -> T_1^T -> raw
        for operator in reversed(self.operators):
            gradients = operator.backward(gradients)

        return [
            (
                gradient.array.copy()
                if gradient is not None
                else None
            )
            for gradient in gradients
        ]

    def get_values(self):
        """Return the current raw MMA values."""

        return self.raw.x.petsc_vec.array.copy()

    def get_lower_bounds(self):
        """Return the raw lower-bound array."""

        return np.full(
            self.size,
            self.lower_bound,
            dtype=float,
        )

    def get_upper_bounds(self):
        """Return the raw upper-bound array."""

        return np.full(
            self.size,
            self.upper_bound,
            dtype=float,
        )

    def set_values(self, values):
        """Assign an updated MMA array to the raw field."""

        values = np.asarray(values, dtype=float)

        if values.size != self.size:
            raise ValueError(
                f"Updated values for design variable '{self.name}' "
                f"have size {values.size}; expected {self.size}."
            )

        self.raw.x.petsc_vec.array[:] = values
        self.apply_fixed_regions()

        self.raw.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD,
        )
