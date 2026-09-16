"""One-iteration run of every example. Use with mpirun -n 1, 2, and 4."""

from __future__ import annotations

import contextlib
import importlib
import io
import re
import runpy
import sys
import tempfile
from pathlib import Path

import pytest
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[1]

EXAMPLES = [
    ("examples/hMSM/input_beam.py", 7.151568e02),
    ("examples/hMSM/input_scissor.py", 1.420926e00),
    ("examples/hMSM/input_wheel.py", -2.450457e02),
    ("examples/Akbari2021_MAE/input_beam.py", 3.008104e01),
    ("examples/Akbari2021_MAE/input_bridge.py", 2.097988e00),
    ("examples/Garai2025_MAE/input_beam.py", 1.579665e01),
    ("examples/Garai2025_MAE/input_bridge.py", 1.053736e00),
    ("examples/Barrera2024_LCE/input_morphing_strip.py", 5.230148e-03),
    ("examples/Barrera2024_LCE/input_pusher.py", 4.892504e-03),
    ("examples/Barrera2024_LCE/input_vertical_extension.py", 2.850636e-03),
]

OBJECTIVE_PATTERN = re.compile(
    r"opt_iter:\s*1,.*?Obj:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def _shared_output_dir(comm):
    if comm.rank == 0:
        path = tempfile.mkdtemp(prefix="matto_example_")
    else:
        path = None
    return comm.bcast(path, root=0)


def _parse_first_objective(text):
    match = OBJECTIVE_PATTERN.search(text)
    if match is None:
        return None
    return float(match.group(1))


def _run_example(script, output_dir):
    driver_mod = importlib.import_module("matto.driver")
    original = driver_mod.OptimizationDriver
    captured = {"text": ""}

    class OneIteration(original):
        """Same driver, but one iteration into a temporary directory."""

        def __init__(self, problem):
            options = problem.setdefault("optimization_options", {})
            options["max_iter"] = 1
            output_options = problem.setdefault("output_options", {})
            output_options["output_dir"] = str(output_dir)
            output_options["sim_output_interval"] = 10**9
            super().__init__(problem)

        def run(self):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                super().run()
            captured["text"] = buffer.getvalue()

    driver_mod.OptimizationDriver = OneIteration
    import matto
    matto.OptimizationDriver = OneIteration

    script_path = (REPO_ROOT / script).resolve()
    sys.path.insert(0, str(script_path.parent))
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        driver_mod.OptimizationDriver = original
        matto.OptimizationDriver = original
        if sys.path and sys.path[0] == str(script_path.parent):
            sys.path.pop(0)
    
    return captured["text"]


@pytest.mark.examples
@pytest.mark.parametrize(
    ("script", "expected"),
    EXAMPLES,
    ids=[Path(script).stem for script, _ in EXAMPLES],
)
def test_example_reaches_first_iteration(script, expected):
    comm = MPI.COMM_WORLD
    output_dir = _shared_output_dir(comm)
    error = None
    text = ""
    try:
        text = _run_example(script, output_dir)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    errors = comm.allgather(error)
    if any(errors):
        pytest.fail("; ".join(item for item in errors if item))

    objective = _parse_first_objective(text) if comm.rank == 0 else None
    objective = comm.bcast(objective, root=0)

    assert objective is not None, f"{script} did not print opt_iter: 1"
    assert objective == pytest.approx(expected, rel=1.0e-3, abs=1.0e-3)
