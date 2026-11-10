"""Parallel agreement of the restorative-beam objective."""

import pytest
from mpi4py import MPI

from tests.support import BeamSession, build_beam_problem

COMM = MPI.COMM_WORLD

# First objective of the 12 x 3 beam with 5 load steps. Must match
# under mpirun -n 1, 2, and 4.
COARSE_BEAM_FIRST_OBJECTIVE = 1.0222916128e02


def test_coarse_beam_objective_is_rank_independent():
    problem = build_beam_problem(
        COMM,
        nx=12,
        ny=3,
        load_steps=5,
    )
    session = BeamSession(problem)
    objective, _ = session.evaluate()

    gathered = COMM.allgather(objective)
    assert all(abs(value - gathered[0]) < 1.0e-10 for value in gathered)
    assert objective == pytest.approx(
        COARSE_BEAM_FIRST_OBJECTIVE,
        rel=1.0e-5,
        abs=1.0e-3,
    )

