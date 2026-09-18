"""Postprocessor hooks of the driver. Runs serially and under mpirun."""

import csv
import os
import tempfile

import numpy as np
import pytest
from dolfinx.mesh import CellType, create_box
from mpi4py import MPI

from matto import DesignSnapshots, HistoryWriter, OptimizationDriver, PostProcessor, SnapshotPlotter
from matto.postprocess import boundary_faces, default_views
from tests.support import BEAM_HEIGHT, BEAM_LENGTH, build_beam_problem, build_hmsm_beam_3d_problem

COMM = MPI.COMM_WORLD
NX, NY = 12, 3


def shared_directory():
    path = tempfile.mkdtemp(prefix="matto_postprocess_") if COMM.rank == 0 else None
    return COMM.bcast(path, root=0)


def beam_problem(postprocessors, max_iter=3):
    problem = build_beam_problem(COMM, nx=NX, ny=NY, load_steps=5, max_iter=max_iter)
    problem["output_options"] = {"output_dir": shared_directory(), "sim_output_interval": 10**9}
    problem["postprocessors"] = postprocessors
    return problem


def test_history_has_one_row_per_iteration():
    problem = beam_problem([HistoryWriter()])
    result = OptimizationDriver(problem).run()
    if COMM.rank != 0:
        return
    with open(os.path.join(result["output_dir"], "history.csv")) as stream:
        rows = list(csv.reader(stream))
    assert rows[0][:2] == ["iteration", "objective"]
    assert rows[0][-2:] == ["change", "time_s"]
    assert [int(row[0]) for row in rows[1:]] == [1, 2, 3]
    assert float(rows[-1][1]) > 0.0


class RaisesOnFirstCall(PostProcessor):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def on_iteration(self, driver, iteration, info):
        self.calls += 1
        raise ValueError("deliberate")


def test_failing_postprocessor_is_dropped_and_run_continues():
    failing, history = RaisesOnFirstCall(), HistoryWriter()
    driver = OptimizationDriver(beam_problem([failing, history]))
    result = driver.run()
    assert result["iterations"] == 3
    assert failing.calls == 1
    assert driver.postprocessors == [history]


def test_postprocessors_must_be_postprocessor_objects():
    with pytest.raises(TypeError, match="PostProcessor"):
        OptimizationDriver(beam_problem([object()]))


def test_snapshots_hold_every_design_field():
    problem = beam_problem([DesignSnapshots(every=2)])
    result = OptimizationDriver(problem).run()
    if COMM.rank != 0:
        return
    directory = os.path.join(result["output_dir"], "snapshots")
    assert sorted(os.listdir(directory)) == ["coordinates.npz", "iter_0001.npz", "iter_0002.npz"]
    nodes, cells = (NX + 1) * (NY + 1), NX * NY
    snapshot = np.load(os.path.join(directory, "iter_0002.npz"))
    assert snapshot["rho_raw"].shape == (cells,)
    assert snapshot["rho_phys"].shape == (nodes,)
    assert snapshot["u"].shape == (nodes, 2)
    assert int(snapshot["iteration"]) == 2
    coordinates = np.load(os.path.join(directory, "coordinates.npz"))
    assert coordinates["rho_phys"].shape == (nodes, 3)
    assert coordinates["rho_phys"][:, 0].max() == pytest.approx(BEAM_LENGTH)


class FailsAtSecondIteration(OptimizationDriver):
    def _iterate(self):
        if self.optimization_iteration >= 1:
            self.optimization_iteration += 1
            raise RuntimeError("forced state failure")
        super()._iterate()


def test_design_is_saved_when_the_state_solve_fails():
    problem = beam_problem([DesignSnapshots(every=100)])
    driver = FailsAtSecondIteration(problem)
    with pytest.raises(RuntimeError, match="forced"):
        driver.run()
    if COMM.rank != 0:
        return
    snapshot = np.load(os.path.join(driver.output_dir, "snapshots", "failure_iter_0002.npz"))
    assert "rho_raw" in snapshot.files
    assert "u" not in snapshot.files


def test_plotter_draws_a_2d_design():
    pytest.importorskip("matplotlib")
    plotter = SnapshotPlotter(every=2, fields=["rho", "phi"], direction="theta", weight="phi")
    result = OptimizationDriver(beam_problem([plotter])).run()
    if COMM.rank != 0:
        return
    picture = os.path.join(result["output_dir"], "snapshots", "iter_0002.png")
    assert os.path.getsize(picture) > 5000


def test_plotter_draws_a_3d_design():
    pytest.importorskip("matplotlib")
    nx, ny, nz = 12, 3, 3
    problem = build_hmsm_beam_3d_problem(COMM, nx, ny, nz, load_steps=10, max_iter=1)
    if COMM.rank == 0:
        problem["mesh_serial"] = create_box(
            MPI.COMM_SELF,
            [[0.0, 0.0, 0.0], [BEAM_LENGTH, BEAM_HEIGHT, BEAM_HEIGHT]],
            [nx, ny, nz],
            CellType.hexahedron,
        )
    problem["output_options"] = {"output_dir": shared_directory(), "sim_output_interval": 10**9}
    # the initial density is 0.5 everywhere; a level of 0.4 selects the whole beam
    plotter = SnapshotPlotter(every=1, direction="theta", weight="phi", threshold=("rho", 0.4))
    problem["postprocessors"] = [plotter]
    result = OptimizationDriver(problem).run()
    if COMM.rank != 0:
        return
    picture = os.path.join(result["output_dir"], "snapshots", "iter_0001.png")
    assert os.path.getsize(picture) > 5000


def test_boundary_faces_drop_the_shared_face():
    one = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    two = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 3, 9, 5, 10, 7, 11]])
    assert boundary_faces(one, "hexahedron")[0].shape == (6, 4)
    faces, owners = boundary_faces(two, "hexahedron")
    assert faces.shape == (10, 4)
    assert sorted(np.bincount(owners)) == [5, 5]
    assert boundary_faces(np.array([[0, 1, 2, 3]]), "tetrahedron")[0].shape == (4, 3)


def test_plate_like_body_gets_a_view_from_below():
    assert len(default_views(np.array([10.0, 8.0, 6.0]))) == 1
    views = default_views(np.array([24.0, 24.0, 3.0]))
    assert [view["elev"] > 0 for view in views] == [True, False]
