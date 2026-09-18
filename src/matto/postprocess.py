# Copyright (c) 2025-2026 Ian Galloway, Prashant K. Jha
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Postprocessors: objects the OptimizationDriver calls while it runs.

A problem lists them under ``problem["postprocessors"]``. The driver calls

    on_start(driver)                           before the first iteration
    on_iteration(driver, iteration, info)      after every iteration
    on_failure(driver, iteration, error)       when the state solve fails
    on_finish(driver, result)                  after the final report

on every rank. ``info`` holds the objective, the constraint values, the
maximum displacements, the design change and the iteration time. A
postprocessor that raises is reported once and not called again; the
optimization continues.

A postprocessor that gathers fields must do all its gathers before any
work that only rank 0 does. An error on rank 0 then cannot leave the
other ranks waiting inside a collective call.

Provided here: HistoryWriter (a CSV row per iteration), DesignSnapshots
(the design fields and the displacement as arrays, every few
iterations and at a failure) and SnapshotPlotter (the same, with a
picture drawn by matplotlib). To draw something else, subclass
SnapshotPlotter and override draw().
"""

import csv
import os

import numpy as np
from dolfinx.fem import functionspace

from .utility import Communicator


class PostProcessor:
    """Base class; override the hooks that are needed."""

    def __init__(self, every=1):
        self.every = int(every)
        if self.every < 1:
            raise ValueError("every must be at least 1.")

    def due(self, iteration):
        """True at iteration 1 and at every multiple of ``every``."""
        return iteration == 1 or iteration % self.every == 0

    def on_start(self, driver):
        pass

    def on_iteration(self, driver, iteration, info):
        pass

    def on_failure(self, driver, iteration, error):
        pass

    def on_finish(self, driver, result):
        pass


class HistoryWriter(PostProcessor):
    """One CSV row per iteration: objective, constraints, change, time."""

    def __init__(self, filename="history.csv"):
        super().__init__(every=1)
        self.filename = filename
        self.path = None

    def on_start(self, driver):
        if driver.comm.rank != 0:
            return
        self.path = os.path.join(driver.output_dir, self.filename)
        header = ["iteration", "objective"]
        header += list(driver.constraint_names)
        header += ["change", "time_s"]
        with open(self.path, "w", newline="") as stream:
            csv.writer(stream).writerow(header)

    def on_iteration(self, driver, iteration, info):
        if driver.comm.rank != 0:
            return
        row = [iteration, repr(float(info["objective"]))]
        row += [
            repr(float(info["constraints"][name]["value"]))
            for name in driver.constraint_names
        ]
        row += [repr(float(info["change"])), f"{info['time']:.3f}"]
        with open(self.path, "a", newline="") as stream:
            csv.writer(stream).writerow(row)


class _RootFields:
    """
    Gathers Functions to rank 0 in the dof order of the plotting mesh.

    The plotting mesh is problem["mesh_serial"] when given and the
    driver's mesh in a serial run without it. A parallel run needs
    mesh_serial, as the driver's final arrays do.
    """

    def __init__(self, driver):
        self.comm = driver.comm
        self.mesh_serial = driver.mesh_serial
        # mesh_serial exists on rank 0 only, so rank 0 decides for all.
        self.has_serial = self.comm.bcast(self.mesh_serial is not None, root=0)
        if not self.has_serial and self.comm.size > 1:
            raise ValueError(
                "Gathering fields in a parallel run needs problem['mesh_serial'] "
                "(a copy of the mesh on MPI.COMM_SELF, built on rank 0)."
            )
        self.plot_mesh = self.mesh_serial if self.has_serial else driver.mesh
        self._communicators = {}
        self._root_spaces = {}

    def values(self, function):
        """The function's values on rank 0, None elsewhere. Collective."""
        if not self.has_serial:
            return function.x.array.copy()
        space = function.function_space
        key = id(space)
        if key not in self._communicators:
            self._communicators[key] = Communicator(space, self.mesh_serial)
        return self._communicators[key].gather(function)

    def root_space(self, function):
        """The function's space on the plotting mesh. Rank 0 only."""
        space = function.function_space
        if not self.has_serial:
            return space
        key = id(space)
        if key not in self._root_spaces:
            self._root_spaces[key] = functionspace(self.plot_mesh, space.ufl_element())
        return self._root_spaces[key]

    def cell_values(self, function, values):
        """One value per cell: the mean of the cell's dof values. Rank 0 only."""
        dofs = self.root_space(function).dofmap.list
        return np.asarray(values)[dofs].mean(axis=1)


class DesignSnapshots(PostProcessor):
    """
    Saves the design every ``every`` iterations and when the state solve fails.

    Each snapshot is ``<output_dir>/<directory>/iter_NNNN.npz`` with the
    raw and physical values of every design variable, in the dof order
    of the plotting mesh, and the displacement when ``displacement`` is
    true. The dof coordinates are saved once as ``coordinates.npz``. A
    snapshot taken at a failure is named ``failure_iter_NNNN.npz`` and
    has no displacement.
    """

    def __init__(self, every=10, displacement=True, directory="snapshots"):
        super().__init__(every=every)
        self.displacement = bool(displacement)
        self.directory = directory
        self.path = None
        self.fields = None

    def on_start(self, driver):
        self.fields = _RootFields(driver)
        self.path = os.path.join(driver.output_dir, self.directory)
        if driver.comm.rank == 0:
            os.makedirs(self.path, exist_ok=True)
            coordinates = {}
            for name, variable in driver.design_variables.items():
                for kind, function in (("raw", variable.raw), ("phys", variable.phys)):
                    space = self.fields.root_space(function)
                    coordinates[f"{name}_{kind}"] = space.tabulate_dof_coordinates()
            space = self.fields.root_space(driver.state.u_field)
            coordinates["u"] = space.tabulate_dof_coordinates()
            np.savez(os.path.join(self.path, "coordinates.npz"), **coordinates)

    def gather(self, driver, with_displacement):
        """All arrays of one snapshot on rank 0, None elsewhere. Collective."""
        arrays = {}
        for name, variable in driver.design_variables.items():
            arrays[f"{name}_raw"] = self.fields.values(variable.raw)
            arrays[f"{name}_phys"] = self.fields.values(variable.phys)
        if with_displacement:
            arrays["u"] = self.fields.values(driver.state.u_field)
        if driver.comm.rank != 0:
            return None
        if with_displacement:
            block = driver.state.u_field.function_space.dofmap.index_map_bs
            arrays["u"] = np.asarray(arrays["u"]).reshape(-1, block)
        return arrays

    def on_iteration(self, driver, iteration, info):
        if not self.due(iteration):
            return
        arrays = self.gather(driver, self.displacement)
        if driver.comm.rank == 0:
            self.save(driver, f"iter_{iteration:04d}", iteration, info["objective"], arrays)

    def on_failure(self, driver, iteration, error):
        arrays = self.gather(driver, with_displacement=False)
        if driver.comm.rank == 0:
            self.save(driver, f"failure_iter_{iteration:04d}", iteration, float("nan"), arrays)

    def save(self, driver, tag, iteration, objective, arrays):
        """Write one snapshot. Rank 0 only."""
        np.savez(
            os.path.join(self.path, f"{tag}.npz"),
            iteration=iteration,
            objective=objective,
            **arrays,
        )


_FACES = {
    "tetrahedron": [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
    # dolfinx numbers hexahedron vertices as i + 2 j + 4 k
    "hexahedron": [
        (0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
        (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5),
    ],
}


def boundary_faces(cells, cell_type):
    """
    The faces that belong to exactly one of the given cells.

    cells is an (n, vertices per cell) array of vertex indices. Returns
    the faces as an (m, vertices per face) array and, for each face, the
    row of the cell it belongs to.
    """
    local_faces = np.asarray(_FACES[cell_type])
    faces = cells[:, local_faces].reshape(-1, local_faces.shape[1])
    owners = np.repeat(np.arange(cells.shape[0]), local_faces.shape[0])
    _, first, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    keep = first[counts == 1]
    return faces[keep], owners[keep]


def default_views(extent):
    """
    Camera angles for a 3D body with the given extents along x, y, z.

    One view from above. A plate-like body, smallest extent under a
    quarter of the largest, gets a second view from below, since a
    design under a solid top layer shows only from there.
    """
    views = [{"elev": 25.0, "azim": -60.0}]
    if min(extent) < 0.25 * max(extent):
        views.append({"elev": -30.0, "azim": -60.0})
    return views


class SnapshotPlotter(DesignSnapshots):
    """
    DesignSnapshots with a picture beside each array file.

    2D: one panel per field, coloured cell by cell. 3D: the cells where
    the ``threshold`` field exceeds its level, drawn as a body coloured
    by ``color``, one panel per camera view.

    fields      names of the design variables to draw; default all
    direction   name of an angle field; in 2D its direction (cos, sin) is
                drawn as arrows where ``weight`` exceeds ``arrow_cutoff``
                times its maximum. Arrows inside a 3D body would not be
                visible and are not drawn.
    weight      name of the field that scales the arrows' presence
    threshold   (field name, level) selecting the 3D body; default
                ("rho", 0.5) when the problem has a field rho
    color       field that colours the 3D body; default ``weight``, else
                the threshold field
    views       list of {"elev": degrees, "azim": degrees}; default from
                default_views()

    Needs matplotlib, imported here and not by the package. Pictures
    are for following a run: matplotlib sorts 3D faces by depth only
    approximately, and a mesh of more than about 200 000 cells makes
    each picture slow.
    """

    def __init__(self, every=10, fields=None, direction=None, weight=None,
                 threshold=None, color=None, views=None, arrow_cutoff=0.1,
                 displacement=True, directory="snapshots"):
        super().__init__(every=every, displacement=displacement, directory=directory)
        try:
            import matplotlib
        except ImportError as error:
            raise ImportError(
                "SnapshotPlotter needs matplotlib; install it or use "
                "DesignSnapshots, which saves the arrays only."
            ) from error
        matplotlib.use("Agg")
        self.field_names = fields
        self.direction = direction
        self.weight = weight
        self.threshold = threshold
        self.color = color
        self.views = views
        self.arrow_cutoff = float(arrow_cutoff)
        self.geometry = None

    def on_start(self, driver):
        super().on_start(driver)
        names = list(driver.design_variables)
        if self.field_names is None:
            self.field_names = [n for n in names if n != self.direction]
        for name in [*self.field_names, self.direction, self.weight, self.color]:
            if name is not None and name not in names:
                raise KeyError(f"'{name}' is not a design variable; known: {names}.")
        if self.threshold is None and "rho" in names:
            self.threshold = ("rho", 0.5)
        if driver.comm.rank == 0:
            mesh = self.fields.plot_mesh
            geometry_space = functionspace(mesh, ("Lagrange", 1))
            self.geometry = {
                "points": geometry_space.tabulate_dof_coordinates(),
                "cells": np.asarray(geometry_space.dofmap.list),
                "cell_type": mesh.topology.cell_name(),
                "dim": mesh.geometry.dim,
            }

    def save(self, driver, tag, iteration, objective, arrays):
        super().save(driver, tag, iteration, objective, arrays)
        import matplotlib.pyplot as plt

        needed = set(self.field_names)
        needed.update(n for n in (self.direction, self.weight, self.color) if n)
        if self.threshold is not None:
            needed.add(self.threshold[0])
        data = dict(self.geometry)
        data["iteration"] = iteration
        data["objective"] = objective
        data["u"] = arrays.get("u")
        data["u_points"] = self.fields.root_space(
            driver.state.u_field
        ).tabulate_dof_coordinates()
        data["fields"] = {
            name: self.fields.cell_values(
                driver.design_variables[name].phys, arrays[f"{name}_phys"]
            )
            for name in needed
        }
        data["bounds"] = {
            name: (driver.design_variables[name].lower_bound,
                   driver.design_variables[name].upper_bound)
            for name in needed
        }
        figure = plt.figure()
        try:
            self.draw(figure, data)
            figure.savefig(os.path.join(self.path, f"{tag}.png"), dpi=120, bbox_inches="tight")
        finally:
            plt.close(figure)

    # ------------------------------------------------------------------
    def draw(self, figure, data):
        """
        Draw one snapshot.

        Override for a different picture; draw_field() and draw_body()
        draw one panel each into an axis of your own.

        data: points (n, 3), cells (m, vertices per cell), cell_type, dim,
        iteration, objective, fields {name: value per cell}, bounds
        {name: (lower, upper)}, u (n, dim) or None with the coordinates
        of its rows in u_points.
        """
        if data["dim"] == 2:
            count = len(self.field_names)
            figure.set_size_inches(5.5 * count, 3.2)
            for index, name in enumerate(self.field_names):
                axis = figure.add_subplot(1, count, index + 1)
                image = self.draw_field(axis, data, name)
                figure.colorbar(image, ax=axis, fraction=0.03)
        else:
            extent = data["points"].max(axis=0) - data["points"].min(axis=0)
            views = self.views or default_views(extent)
            figure.set_size_inches(6.0 * len(views), 5.0)
            for index, camera in enumerate(views):
                axis = figure.add_subplot(1, len(views), index + 1, projection="3d")
                mappable = self.draw_body(axis, data, camera)
            figure.colorbar(mappable, ax=figure.axes, fraction=0.02, label=self.body_color())
        figure.suptitle(f"iteration {data['iteration']}   objective {data['objective']:.4e}")

    def body_color(self):
        """Name of the field that colours the 3D body."""
        if self.color or self.weight:
            return self.color or self.weight
        return self.threshold[0] if self.threshold else self.field_names[0]

    def _arrows(self, data):
        """Centroids and unit directions of the cells that carry arrows."""
        theta = data["fields"][self.direction]
        centres = data["points"][data["cells"]].mean(axis=1)
        if self.weight is not None:
            weight = data["fields"][self.weight]
            shown = weight > self.arrow_cutoff * max(weight.max(), 1.0e-30)
            theta, centres = theta[shown], centres[shown]
        stride = max(1, len(theta) // 200)
        return centres[::stride], np.cos(theta[::stride]), np.sin(theta[::stride])

    def draw_field(self, axis, data, name):
        """One 2D field, coloured cell by cell, into a 2D axis. Returns the image."""
        points, cells = data["points"], data["cells"]
        if data["cell_type"] == "quadrilateral":
            triangles = np.vstack([cells[:, [0, 1, 3]], cells[:, [0, 3, 2]]])
            repeat = 2
        else:
            triangles, repeat = cells, 1
        lower, upper = data["bounds"][name]
        image = axis.tripcolor(
            points[:, 0], points[:, 1], triangles,
            facecolors=np.tile(data["fields"][name], repeat),
            vmin=lower, vmax=upper, cmap="viridis",
        )
        if self.direction is not None and name == (self.weight or self.field_names[0]):
            centres, cx, sy = self._arrows(data)
            extent = points.max(axis=0) - points.min(axis=0)
            length = 0.025 * extent.max()
            axis.quiver(centres[:, 0], centres[:, 1], length * cx, length * sy,
                        angles="xy", scale_units="xy", scale=1.0, pivot="mid",
                        color="white", width=0.0025)
        axis.set_aspect("equal")
        axis.set_title(name)
        return image

    def draw_body(self, axis, data, camera):
        """
        The 3D body seen from one camera, into a 3D axis.

        camera is {"elev": degrees, "azim": degrees}. Returns the
        mappable of the body's colours, for a colorbar.
        """
        from matplotlib import cm, colors
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        points, cells = data["points"], data["cells"]
        if self.threshold is not None:
            body = data["fields"][self.threshold[0]] > self.threshold[1]
        else:
            body = np.ones(cells.shape[0], dtype=bool)
        color_name = self.body_color()
        norm = colors.Normalize(*data["bounds"][color_name])
        selected = np.flatnonzero(body)
        if selected.size > 0:
            faces, owners = boundary_faces(cells[selected], data["cell_type"])
            axis.add_collection3d(Poly3DCollection(
                points[faces],
                facecolors=cm.viridis(norm(data["fields"][color_name][selected][owners])),
                edgecolors=(0, 0, 0, 0.15), linewidths=0.2,
            ))
        low, high = points.min(axis=0), points.max(axis=0)
        extent = high - low
        axis.set_xlim(low[0], high[0])
        axis.set_ylim(low[1], high[1])
        axis.set_zlim(low[2], high[2])
        if extent[2] < 0.25 * extent.max():
            axis.set_zticks([low[2], high[2]])
        axis.set_box_aspect(np.maximum(extent, 1.0e-12))
        axis.view_init(elev=camera["elev"], azim=camera["azim"])
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(f"elev {camera['elev']:g}, azim {camera['azim']:g}")
        return cm.ScalarMappable(norm=norm, cmap="viridis")
