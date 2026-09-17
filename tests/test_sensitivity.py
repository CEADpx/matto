"""
Checks on Sensitivity that hold for any number of ranks.

The gradients are assembled into ghosted vectors and scattered back to
their owners, so a slot that is not cleared between calls shows up as
a result that drifts from one evaluate() to the next. That only
happens with more than one rank, and the one-iteration example tests
cannot see it. Both tests here run the coarse beam; run them under
mpirun -n 2 as well as serially.
"""

import numpy as np
import pytest
from mpi4py import MPI

from tests.support import (
    BeamSession,
    build_beam_problem,
    use_direct_filter_solve,
)

COMM = MPI.COMM_WORLD
STEP = 1.0e-5
REL_TOL = 5.0e-2


def _session():
    problem = build_beam_problem(COMM, nx=12, ny=3, load_steps=5)
    return BeamSession(use_direct_filter_solve(problem))


def _all_close(a, b):
    # Relative: at four ranks the parallel factorization gives
    # differences near 1e-10 on gradients of order 10. The stale-ghost
    # error this test exists for is of order 1.
    local = bool(np.allclose(a, b, rtol=1.0e-9, atol=1.0e-12))
    return COMM.allreduce(local, op=MPI.LAND)


def test_repeated_evaluate_gives_the_same_gradients():
    session = _session()
    first = session.driver.evaluate()
    second = session.driver.evaluate()

    assert second["objective"] == pytest.approx(first["objective"], rel=1e-12)

    for name in first["objective_gradients"]:
        assert _all_close(
            first["objective_gradients"][name],
            second["objective_gradients"][name],
        ), f"objective gradient for {name} changed between calls"

    for constraint, gradients in first["constraint_gradients"].items():
        for name in gradients:
            assert _all_close(
                gradients[name],
                second["constraint_gradients"][constraint][name],
            ), f"{constraint} gradient for {name} changed between calls"


def test_constraint_gradients_match_finite_difference():
    session = _session()
    driver = session.driver
    base_result = driver.evaluate()

    rng = np.random.default_rng(0)

    for name in ("rho", "phi", "theta"):
        base = session.raw_values(name)
        direction = rng.standard_normal(base.size)
        norm = COMM.allreduce(float(direction @ direction), op=MPI.SUM) ** 0.5
        direction = direction / norm

        session.set_raw_values(name, base + STEP * direction)
        plus = driver.evaluate()["constraints"]
        session.set_raw_values(name, base - STEP * direction)
        minus = driver.evaluate()["constraints"]
        session.set_raw_values(name, base)

        for constraint in base_result["constraints"]:
            fd = (
                plus[constraint]["residual"] - minus[constraint]["residual"]
            ) / (2.0 * STEP)
            gradient = base_result["constraint_gradients"][constraint][name]
            adjoint = COMM.allreduce(float(gradient @ direction), op=MPI.SUM)

            scale = max(abs(adjoint), abs(fd), 1.0e-8)
            if scale <= 1.0e-8:
                continue  # this constraint does not see this field

            assert abs(adjoint - fd) / scale < REL_TOL, (
                f"{constraint}/{name}: gradient={adjoint:.6e}, fd={fd:.6e}"
            )
