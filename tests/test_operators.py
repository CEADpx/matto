"""
Adjoint checks for the parameterization operators on a small mesh.

For an operator T at the current point, the Jacobian action J d is
taken by central differences of forward() and the adjoint action J^T y
from backward(). The two must agree in the dot-product sense,

    <J d, y> = <d, J^T y>,

to roundoff. Helmholtz filtering is linear so the difference quotient
is exact; the Heaviside projection is smooth enough that the O(h^2)
error is far below the tolerance. Every operator, and every chain of
them, has to pass this.
"""

import numpy as np
import pytest
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, create_rectangle
from mpi4py import MPI

from matto.operators import (
    DesignVariable,
    HeavisideProjection,
    HelmholtzFilter,
    HelmholtzKernel,
    build_operator_chain,
)

COMM = MPI.COMM_WORLD
STEP = 1.0e-6
REL_TOL = 1.0e-7

# A direct solve, so the check is of the adjoint and not of a Krylov
# tolerance.
LU = {"ksp_type": "preonly", "pc_type": "lu"}


def _mesh():
    return create_rectangle(
        COMM,
        [[0.0, 0.0], [2.0, 1.0]],
        [8, 4],
        CellType.quadrilateral,
    )


def _rng():
    return np.random.default_rng(COMM.rank + 1)


def _dot(a, b):
    return COMM.allreduce(float(np.dot(a, b)), op=MPI.SUM)


def _set_owned(function, values):
    function.x.petsc_vec.array[:] = values
    function.x.scatter_forward()


def _check_adjoint(forward, backward, input_field, output_field, rng):
    """
    forward(): recompute output_field from input_field.
    backward(y): adjoint action on a PETSc vector on the output side,
                 returning owned values on the input side.
    """

    base = input_field.x.petsc_vec.array.copy()
    direction = rng.standard_normal(base.size)

    _set_owned(input_field, base + STEP * direction)
    forward()
    plus = output_field.x.petsc_vec.array.copy()

    _set_owned(input_field, base - STEP * direction)
    forward()
    minus = output_field.x.petsc_vec.array.copy()

    # Back to the base point so backward() linearizes there.
    _set_owned(input_field, base)
    forward()

    jacobian_action = (plus - minus) / (2.0 * STEP)
    cotangent = rng.standard_normal(jacobian_action.size)

    y = output_field.x.petsc_vec.copy()
    y.array[:] = cotangent
    adjoint_action = backward(y)

    lhs = _dot(jacobian_action, cotangent)
    rhs = _dot(direction, adjoint_action)

    assert lhs == pytest.approx(rhs, rel=REL_TOL), (
        f"<J d, y> = {lhs:.12e}, <d, J^T y> = {rhs:.12e}"
    )


def _check_operator(operator, rng):
    _check_adjoint(
        operator.forward,
        lambda y: operator.backward([y])[0].array,
        operator.input,
        operator.output,
        rng,
    )


def test_helmholtz_filter_adjoint():
    mesh = _mesh()
    rng = _rng()
    raw = Function(functionspace(mesh, ("DG", 0)))
    filtered = Function(functionspace(mesh, ("CG", 1)))
    _set_owned(raw, rng.uniform(0.2, 0.8, raw.x.petsc_vec.array.size))

    kernel = HelmholtzKernel(
        COMM, raw.function_space, filtered.function_space, 0.3, LU
    )
    operator = HelmholtzFilter(kernel, raw, filtered)
    _check_operator(operator, rng)


def test_heaviside_projection_adjoint():
    mesh = _mesh()
    rng = _rng()
    space = functionspace(mesh, ("CG", 1))
    before, after = Function(space), Function(space)
    _set_owned(before, rng.uniform(0.2, 0.8, before.x.petsc_vec.array.size))

    operator = HeavisideProjection(
        before, after, beta=4.0, beta_max=4.0, update_interval=0
    )
    _check_operator(operator, rng)


def _variable(mesh, operators):
    return DesignVariable(
        name="rho",
        mesh=mesh,
        settings={
            "active": True,
            "initial": 0.5,
            "bounds": (0.05, 1.0),
            "prescribed_value": 1.0,
            "raw_space": ("DG", 0),
            "physical_space": ("CG", 1),
            "operators": operators,
            "fixed_regions": [],
        },
        petsc_options=LU,
    )


def _check_variable(variable, rng):
    variable.set_values(rng.uniform(0.2, 0.8, variable.size))
    variable.forward(0)

    _check_adjoint(
        lambda: variable.forward(0),
        lambda y: variable.backward([y])[0],
        variable.raw,
        variable.phys,
        rng,
    )


def test_filter_then_projection_chain_adjoint():
    variable = _variable(_mesh(), [
        {"type": "density_filter", "radius": 0.3},
        {"type": "heaviside", "beta_initial": 4.0},
    ])
    _check_variable(variable, _rng())


def test_projection_then_filter_chain_adjoint():
    # The projection is not last, so it gets its own DG0 intermediate
    # and the filter maps that to CG1.
    variable = _variable(_mesh(), [
        {"type": "heaviside", "beta_initial": 4.0},
        {"type": "density_filter", "radius": 0.3},
    ])
    assert variable.operators[0].output is not variable.raw
    assert variable.operators[1].output is variable.phys
    _check_variable(variable, _rng())


def test_no_operators_is_identity():
    mesh = _mesh()
    variable = DesignVariable(
        name="phi",
        mesh=mesh,
        settings={
            "active": True,
            "initial": 0.3,
            "bounds": (0.0, 1.0),
            "prescribed_value": 0.0,
            "raw_space": ("CG", 1),
            "physical_space": ("CG", 1),
            "operators": [],
        },
    )
    _check_variable(variable, _rng())


def test_unknown_operator_type_names_the_known_ones():
    mesh = _mesh()
    raw = Function(functionspace(mesh, ("DG", 0)))
    phys = Function(functionspace(mesh, ("CG", 1)))

    with pytest.raises(ValueError, match="density_filter"):
        build_operator_chain(
            "rho", [{"type": "median"}], raw, phys, COMM, {}
        )
