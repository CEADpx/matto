"""
Finite-difference check of the adjoint on coarse beams.

The hMSM beam has three active fields; the isotropic MAE beam has one.
"""

import numpy as np
import pytest
from mpi4py import MPI

from tests.support import (
    BeamSession,
    build_beam_problem,
    build_mae_beam_problem,
)

COMM = MPI.COMM_WORLD
STEP = 1.0e-5
REL_TOL = 5.0e-2


def _directional_product(gradient, direction, comm):
    local = float(np.dot(gradient, direction))
    return comm.allreduce(local, op=MPI.SUM)


def _relative_error(adjoint_value, finite_difference):
    scale = max(abs(adjoint_value), abs(finite_difference), 1.0e-8)
    return abs(adjoint_value - finite_difference) / scale


@pytest.mark.parametrize(
    ("build", "active_fields"),
    [
        (build_beam_problem, ("rho", "phi", "theta")),
        (build_mae_beam_problem, ("phi",)),
    ],
    ids=["hmsm_three_fields", "mae_one_field"],
)
def test_coarse_beam_adjoint_matches_finite_difference(build, active_fields):
    problem = build(
        COMM,
        nx=12,
        ny=3,
        load_steps=5,
    )
    session = BeamSession(problem)
    objective, gradients = session.evaluate()
    assert np.isfinite(objective)

    rng = np.random.default_rng(0)
    for name in active_fields:
        base = session.raw_values(name)
        direction = rng.standard_normal(base.size)
        direction_norm = COMM.allreduce(
            float(np.dot(direction, direction)),
            op=MPI.SUM,
        ) ** 0.5
        assert direction_norm > 0.0
        direction = direction / direction_norm

        session.set_raw_values(name, base + STEP * direction)
        objective_plus, _ = session.evaluate()

        session.set_raw_values(name, base - STEP * direction)
        objective_minus, _ = session.evaluate()

        session.set_raw_values(name, base)

        finite_difference = (objective_plus - objective_minus) / (2.0 * STEP)
        adjoint_value = _directional_product(gradients[name], direction, COMM)
        error = _relative_error(adjoint_value, finite_difference)

        assert error < REL_TOL, (
            f"{name}: adjoint={adjoint_value:.6e}, "
            f"fd={finite_difference:.6e}, rel_err={error:.3e}"
        )
