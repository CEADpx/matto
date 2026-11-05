"""
Finite-difference check of the adjoint on coarse beams.

The hMSM beam has three active fields; the isotropic MAE beam has one.
"""

import numpy as np
import pytest
from mpi4py import MPI

from matto.materials import HardMagneticSoftMaterial
from tests.support import (
    BeamSession,
    build_beam_problem,
    build_hmsm_beam_3d_problem,
    build_mae_beam_problem,
    use_direct_filter_solve,
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


def _build_3d(comm, nx, ny, load_steps):
    # The 3D box needs a gentler ramp than the plane-strain beam for
    # Newton to follow it on a mesh this coarse.
    return build_hmsm_beam_3d_problem(comm, nx, ny, ny, 2 * load_steps)


def _with_rotation_map(problem):
    parameters = dict(problem["material_parameters"], magnetization_map="R")
    problem["material_parameters"] = parameters
    problem["material"] = HardMagneticSoftMaterial(**parameters)
    return problem


def _build_rotation_map(comm, nx, ny, load_steps):
    return _with_rotation_map(build_beam_problem(
        comm, nx=nx, ny=ny, load_steps=load_steps
    ))


def _build_rotation_map_3d(comm, nx, ny, load_steps):
    return _with_rotation_map(_build_3d(comm, nx, ny, load_steps))


@pytest.mark.parametrize(
    ("build", "active_fields"),
    [
        (build_beam_problem, ("rho", "phi", "theta")),
        (build_mae_beam_problem, ("phi",)),
        (_build_3d, ("rho", "phi", "theta")),
        (_build_rotation_map, ("rho", "phi", "theta")),
        (_build_rotation_map_3d, ("rho", "phi", "theta")),
    ],
    ids=[
        "hmsm_three_fields",
        "mae_one_field",
        "hmsm_three_fields_3d",
        "hmsm_rotation_map",
        "hmsm_rotation_map_3d",
    ],
)
def test_coarse_beam_adjoint_matches_finite_difference(build, active_fields):
    problem = use_direct_filter_solve(build(
        COMM,
        nx=12,
        ny=3,
        load_steps=5,
    ))
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
