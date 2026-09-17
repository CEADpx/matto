# Copyright (c) 2025-2026 Ian Galloway, Prashant K. Jha
# Copyright (c) 2024 Yingqi Jia, Chao Wang, Xiaojia Shelly Zhang (FEniTop)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Communicator and compare_matrices are FEniTop's. See NOTICE.
"""
Solver options and their defaults, the Newton wrapper for the state
problem, and the gather of a parallel field into serial ordering for
the saved arrays.
"""

from dolfinx.fem.petsc import NonlinearProblem as DolfinxNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import numpy as np
from scipy.spatial import cKDTree
from petsc4py import PETSc
import dolfinx.fem
from dolfinx.fem import Function


DEFAULT_STATE_SOLVER = {
    "atol": 1.0e-4,
    "rtol": 1.0e-4,
    "convergence_criterion": "incremental",
    "petsc_options": {
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
}

DEFAULT_ADJOINT_SOLVER = {
    "rtol": 1.0e-8,
    "atol": 1.0e-12,
    "petsc_options": {
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
}

DEFAULT_FILTER_SOLVER = {
    "petsc_options": {},
}


def apply_petsc_options(petsc_object, petsc_options, prefix=None):
    """Apply a dict of PETSc options to one prefixed solver object."""
    if petsc_options is None:
        petsc_options = {}

    if prefix is None:
        prefix = f"matto_{id(petsc_object)}"

    petsc_object.setOptionsPrefix(prefix)

    opts = PETSc.Options()
    opts.prefixPush(prefix)
    for key, value in petsc_options.items():
        if value is None:
            continue
        opts[key] = value
    opts.prefixPop()
    petsc_object.setFromOptions()
    return prefix


def _merged_options(defaults, override):
    merged = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in defaults.items()
    }
    if not override:
        return merged
    if not isinstance(override, dict):
        raise TypeError("Each solver_options block must be a dictionary.")

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def resolve_solver_options(fem_options):
    """Return ``{state, adjoint, filter}`` solver settings with defaults.

    Accepts ``fem_options["solver_options"]``. If only the older
    ``fem_options["petsc_options"]`` key is present, that dict is used for
    both the state and filter solvers.
    """
    if fem_options is None:
        fem_options = {}
    if not isinstance(fem_options, dict):
        raise TypeError("fem_options must be a dictionary.")

    raw = fem_options.get("solver_options")
    if raw is None and "petsc_options" in fem_options:
        legacy = fem_options["petsc_options"]
        raw = {
            "state": {"petsc_options": legacy},
            "filter": {"petsc_options": legacy},
        }
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("solver_options must be a dictionary.")

    unknown = set(raw) - {"state", "adjoint", "filter"}
    if unknown:
        raise ValueError(
            "Unknown solver_options keys: "
            f"{sorted(unknown)}. Expected state, adjoint, filter."
        )

    return {
        "state": _merged_options(
            DEFAULT_STATE_SOLVER,
            raw.get("state"),
        ),
        "adjoint": _merged_options(
            DEFAULT_ADJOINT_SOLVER,
            raw.get("adjoint"),
        ),
        "filter": _merged_options(
            DEFAULT_FILTER_SOLVER,
            raw.get("filter"),
        ),
    }


class WrapNonlinearProblem:
    def __init__(self, u, R, bcs=None, solver_options=None):
        """Wrap the nonlinear residual and apply the state solver settings."""
        if bcs is None:
            bcs = []
        if solver_options is None:
            solver_options = {}

        self.u = u
        self.bcs = bcs
        self.problem = DolfinxNonlinearProblem(R, u, bcs)
        self.solver = NewtonSolver(
            u.function_space.mesh.comm,
            self.problem,
        )

        self.solver.atol = float(
            solver_options.get("atol", DEFAULT_STATE_SOLVER["atol"])
        )
        self.solver.rtol = float(
            solver_options.get("rtol", DEFAULT_STATE_SOLVER["rtol"])
        )
        self.solver.convergence_criterion = solver_options.get(
            "convergence_criterion",
            DEFAULT_STATE_SOLVER["convergence_criterion"],
        )
        if "max_it" in solver_options:
            self.solver.max_it = int(solver_options["max_it"])
        if "relaxation_parameter" in solver_options:
            self.solver.relaxation_parameter = float(
                solver_options["relaxation_parameter"]
            )
        if "error_on_nonconvergence" in solver_options:
            self.solver.error_on_nonconvergence = bool(
                solver_options["error_on_nonconvergence"]
            )
        if "report" in solver_options:
            self.solver.report = bool(solver_options["report"])

        apply_petsc_options(
            self.solver.krylov_solver,
            solver_options.get("petsc_options", {}),
            prefix=f"state_ksp_{id(self)}",
        )

    def solve_fem(self):
        """Run Newton iteration until R(u) is within the state tolerances."""
        num_its, converged = self.solver.solve(self.u)
        self.u.x.scatter_forward()

        assert converged, f"Newton solver did not converge in {num_its} iterations"

    def __del__(self):
        self.solver.krylov_solver.destroy()


class Communicator():
    """Communicate information among different processes."""

    def __init__(self, func_space, mesh_serial, size=1):
        self.size = size
        self.comm = func_space.mesh.comm
        idx_map = func_space.dofmap.index_map
        
        num_local_nodes = idx_map.size_local
        num_global_nodes = idx_map.size_global
        num_nodal_dofs = func_space.dofmap.index_map_bs
        self.num_global_dofs = num_global_nodes * num_nodal_dofs
        
        local_nodal_range = np.asarray(idx_map.local_range, dtype=np.int32) # [start, end]
        local_dof_range = local_nodal_range * num_nodal_dofs  # [start, end]
        local_nodes = func_space.tabulate_dof_coordinates()[:num_local_nodes]
        
        # Gather to Process 0
        local_nodal_range_gather = self.comm.gather(local_nodal_range, root=0)
        self.local_dof_range_gather = self.comm.gather(local_dof_range, root=0)
        local_nodes_gather = self.comm.gather(local_nodes, root=0)
        
        element = func_space.ufl_element()
        if self.comm.rank == 0:
            func_space_serial = dolfinx.fem.functionspace(mesh_serial, element)
            nodes_serial = func_space_serial.tabulate_dof_coordinates()

            nodes_collect = np.zeros((num_global_nodes, 3))
            for r, nodes in zip(local_nodal_range_gather, local_nodes_gather):
                nodes_collect[r[0]:r[1]] = nodes
            global_to_local_nodes = compare_matrices(nodes_serial, nodes_collect)
            local_to_global_nodes = compare_matrices(nodes_collect, nodes_serial)
            
            def node2dof(nodes, num_nodal_dofs):
                return (np.tile(nodes, (num_nodal_dofs, 1))*num_nodal_dofs
                        + np.arange(num_nodal_dofs).reshape(-1, 1)).ravel("F")

            global_to_local_dofs = node2dof(global_to_local_nodes, num_nodal_dofs)
            self.local_to_global_dofs = node2dof(local_to_global_nodes, num_nodal_dofs)
            self.local_to_global_dofs = (
                np.tile(self.local_to_global_dofs.reshape(-1, 1), (1, size))*size + np.arange(size)).ravel()
        else:
            global_to_local_dofs = None
        global_to_local_dofs = self.comm.bcast(global_to_local_dofs, root=0)
        self.idx = global_to_local_dofs[local_dof_range[0]:local_dof_range[1]]

    def bcast(self, func, global_values):
        """Broadcast data from Process 0 to all the other processes."""
        if func.vector.size != global_values.size:
            raise ValueError("Mismatched sizes.")
        func.x.array = global_values[self.idx]

    def gather(self, func):
        """Gather owned values to Process 0 from all the other processes."""
        if type(func) is Function:
            index_map = func.function_space.dofmap.index_map
            block_size = func.function_space.dofmap.index_map_bs
            owned = index_map.size_local * block_size
            values_gather = self.comm.gather(func.x.array[:owned], root=0)
        elif type(func) is PETSc.Vec:
            owned = func.getLocalSize()
            values_gather = self.comm.gather(func.array[:owned], root=0)
        elif type(func) is np.ndarray:
            values_gather = self.comm.gather(func, root=0)
        else:
            raise TypeError("Unsupported func.")
            
        if self.comm.rank == 0:
            values_collect = np.zeros(self.num_global_dofs*self.size)
            for r, local_values in zip(self.local_dof_range_gather, values_gather):
                values_collect[r[0]*self.size:r[1]*self.size] = local_values
            global_values = values_collect[self.local_to_global_dofs]
        else:
            global_values = None
        return global_values

def compare_matrices(array1, array2, precision=12, k=1):
    """Find the "args" such that array1[args] == array2."""
    kd_tree = cKDTree(array1.round(precision))
    return kd_tree.query(array2.round(precision), k=k)[1]
