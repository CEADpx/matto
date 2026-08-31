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

Major additions to parameterize.py:
- Generic design-variable representation
- Configurable filtering and projection operators
- Active and prescribed design-variable handling
- Fixed-region support and sensitivity backpropagation
"""

import numpy as np
import ufl
from dolfinx import la
from dolfinx.fem import Function, form, functionspace
from dolfinx.fem.petsc import create_matrix, assemble_matrix
from petsc4py import PETSc


class DensityFilter():
    def __init__(self, comm, rho, rho_tilde, R=1.0, petsc_options={}):
        """Construct a PDE filter."""
        # Initialization
        S0, S = rho.function_space, rho_tilde.function_space
        u0, u = ufl.TrialFunction(S0), ufl.TrialFunction(S)
        v, self.af = ufl.TestFunction(S), Function(S)
        
        self.rho, self.rho_tilde = rho, rho_tilde
        self.rho_tilde_wrap = la.create_petsc_vector_wrap(self.rho_tilde.x)
        self.af_wrap = la.create_petsc_vector_wrap(self.af.x)
        self.vec_s0, self.vec_s = rho.x.petsc_vec.copy(), rho_tilde.x.petsc_vec.copy()
        
        # Construct Kf and T matrices based on the Helmholtz PDE
        dx = ufl.Measure("dx", metadata={"quadrature_degree": 2})
        Kf_expr = (R**2*ufl.dot(ufl.grad(u), ufl.grad(v)) + u*v)*dx
        T_expr = u0*v*dx
        Kf_form, T_form = form(Kf_expr), form(T_expr)
        Kf_mat, self.T_mat = create_matrix(Kf_form), create_matrix(T_form)
        
        # Construct a filtering solver
        self.solver = PETSc.KSP().create(comm)
        self.solver.setOperators(Kf_mat)
        prefix = f"filter_solver_{id(self)}"
        self.solver.setOptionsPrefix(prefix)
        
        # Apply PETSc options
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in petsc_options.items():
            opts[key] = value
        opts.prefixPop()
        self.solver.setFromOptions()
        Kf_mat.setOptionsPrefix(prefix)
        Kf_mat.setFromOptions()
        
        # Assemble Kf and T matrices
        assemble_matrix(Kf_mat, Kf_form)
        Kf_mat.assemble()
        assemble_matrix(self.T_mat, T_form)
        self.T_mat.assemble()
        self.T_mat_transpose = self.T_mat.copy()
        self.T_mat_transpose.transpose()

    def forward(self):
        """Compute the filtered variables."""
        self.T_mat.mult(self.rho.x.petsc_vec, self.vec_s)
        self.solver.solve(self.vec_s, self.rho_tilde_wrap)
        self.rho_tilde.x.scatter_forward()
        return self.rho_tilde
        
    def backward(self, sf_vectors):
        """Recover the sensitivities."""
        values = []
        for sf in sf_vectors:
            if sf is not None:
                self.solver.solve(sf, self.af_wrap)
                self.af.x.scatter_forward()
                self.T_mat_transpose.mult(self.af.x.petsc_vec, self.vec_s0)
                values.append(self.vec_s0.array.copy())
            else:
                values.append(None)
        return values

class Heaviside():
    def __init__(self, rho_phys):
        self.rho_phys = rho_phys

    def forward(self, beta, eta=0.5):
        denominator = np.tanh(beta*eta) + np.tanh(beta*(1-eta))
        self.drho = beta*(1-np.tanh(beta*(self.rho_phys.x.petsc_vec-eta))**2) / denominator
        self.rho_phys.x.petsc_vec.array = (
            np.tanh(beta*eta)+np.tanh(beta*(self.rho_phys.x.petsc_vec-eta))) / denominator
        self.rho_phys.x.scatter_forward()
    
    def backward(self, vectors):
        for vector in vectors:
            if vector is not None:
                vector.array *= self.drho

class DesignVariable:
    """
    Manage one optimization design variable.

    The class owns:
        raw  - the variable controlled by MMA
        phys - the variable after filtering/projection

    It also applies the configured operators forward and propagates
    sensitivities backward from phys to raw.
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

        self.density_filter = None
        self.heaviside = None

        self.beta = None
        self.beta_update_interval = None
        self.beta_max = None

        for operator in settings.get("operators", []):
            operator_type = operator["type"]

            if operator_type == "density_filter":
                if self.density_filter is not None:
                    raise ValueError(
                        f"Design variable '{name}' has more than one "
                        "density filter."
                    )

                if self.heaviside is not None:
                    raise ValueError(
                        f"Design variable '{name}' must apply its density "
                        "filter before its Heaviside projection."
                    )

                self.density_filter = DensityFilter(
                    self.comm,
                    self.raw,
                    self.phys,
                    R=float(operator["radius"]),
                    petsc_options=petsc_options,
                )

            elif operator_type == "heaviside":
                if self.heaviside is not None:
                    raise ValueError(
                        f"Design variable '{name}' has more than one "
                        "Heaviside projection."
                    )

                self.heaviside = Heaviside(self.phys)

                self.beta = float(
                    operator.get("beta_initial", 1.0)
                )

                self.beta_update_interval = int(
                    operator.get("beta_update_interval", 0)
                )

                self.beta_max = float(
                    operator.get("beta_max", self.beta)
                )

                if self.beta <= 0.0:
                    raise ValueError(
                        f"Design variable '{name}' must have beta_initial > 0."
                    )

                if self.beta_max < self.beta:
                    raise ValueError(
                        f"Design variable '{name}' must have "
                        "beta_max >= beta_initial."
                    )

            else:
                raise ValueError(
                    f"Unknown operator '{operator_type}' for "
                    f"design variable '{name}'."
                )

        # Without a density filter, raw and phys must use the same
        # function space so their values can be copied directly.
        if (
            self.density_filter is None
            and self.raw_space_spec != self.physical_space_spec
        ):
            raise ValueError(
                f"Design variable '{name}' has different raw and physical "
                "spaces but no density filter connecting them."
            )

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

    def forward(self, iteration):
        """
        Map raw values to physical values.

        Returns True when the Heaviside beta value is increased.
        """

        if not self.active:
            self._assign(self.phys, self.prescribed_value)
            return False

        self.apply_fixed_regions()

        # raw -> filtered phys
        if self.density_filter is not None:
            self.density_filter.forward()
        else:
            self.phys.x.petsc_vec.array[:] = (
                self.raw.x.petsc_vec.array
            )

            self.phys.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT,
                mode=PETSc.ScatterMode.FORWARD,
            )

        beta_updated = False

        # filtered phys -> Heaviside-projected phys
        if self.heaviside is not None:
            if (
                iteration > 0
                and self.beta_update_interval > 0
                and iteration % self.beta_update_interval == 0
                and self.beta < self.beta_max
            ):
                self.beta = min(
                    2.0 * self.beta,
                    self.beta_max,
                )
                beta_updated = True

            self.heaviside.forward(self.beta)

        return beta_updated

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

        # Copy the vectors because Heaviside.backward modifies them.
        working_gradients = [
            (
                gradient.copy()
                if gradient is not None
                else None
            )
            for gradient in physical_gradients
        ]

        # Reverse order:
        # phys -> before Heaviside -> raw
        if self.heaviside is not None:
            self.heaviside.backward(working_gradients)

        if self.density_filter is not None:
            return self.density_filter.backward(
                working_gradients
            )

        return [
            (
                gradient.array.copy()
                if gradient is not None
                else None
            )
            for gradient in working_gradients
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