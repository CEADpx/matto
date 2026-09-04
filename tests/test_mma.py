"""MMA calling-convention checks. These do not assemble a finite-element problem."""

import inspect

import numpy as np
import pytest
from mpi4py import MPI

from matto.optimize import DEFAULT_MOVE, mma_optimizer


def _dummy_mma_args():
    n, m = 4, 1
    x = np.full(n, 0.5)
    xmin = np.zeros(n)
    xmax = np.ones(n)
    return (
        m,
        n,
        1,
        x,
        xmin,
        xmax,
        x.copy(),
        x.copy(),
        np.ones(n),
        np.array([-0.1]),
        np.ones((m, n)),
        xmin.copy(),
        xmax.copy(),
    )


def test_move_and_comm_are_keyword_only():
    signature = inspect.signature(mma_optimizer)
    assert signature.parameters["move"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["comm"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["a0"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["move"].default == DEFAULT_MOVE


def test_positional_move_is_a_type_error():
    with pytest.raises(TypeError, match="positional"):
        mma_optimizer(*_dummy_mma_args(), 0.05)


def test_missing_comm_is_a_type_error():
    with pytest.raises(TypeError, match="comm"):
        mma_optimizer(*_dummy_mma_args(), move=0.05)


def test_keyword_move_updates_the_design():
    x_new, change, _, _ = mma_optimizer(
        *_dummy_mma_args(),
        comm=MPI.COMM_WORLD,
        move=0.05,
    )
    assert x_new.shape == (4,)
    assert change > 0.0
