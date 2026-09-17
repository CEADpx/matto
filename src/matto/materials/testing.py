"""
Consistency checks any material should pass, for use in its tests.

check_material() puts the model on a small mesh with an affine
displacement, so the deformation gradient is exactly what was asked
for, and checks three things at zero stimulus and at the given one:

  1. The reference configuration is stress-free: with u = 0 and no
     stimulus the internal-force vector vanishes.
  2. Frame indifference: W(Q F) = W(F) for a rotation Q, with any
     in-plane vector stimulus rotated along.
  3. A moderate stretch raises the stored energy at zero stimulus:
     W(F) >= W(I) for one stretch F.

Run it on a new material before running an optimization with it.
"""

import numpy as np
import ufl
from dolfinx.fem import Constant, Function, assemble_scalar, form, functionspace
from dolfinx.fem.petsc import assemble_vector
from dolfinx.mesh import CellType, create_box, create_rectangle
from mpi4py import MPI
from petsc4py import PETSc

DEFAULT_FIELD_VALUES = {"rho": 1.0, "phi": 0.5, "theta": 0.3}


def _affine_displacement(u, F0):
    dim = F0.shape[0]
    A = np.asarray(F0) - np.eye(dim)
    u.interpolate(lambda x: np.vstack([
        sum(A[i, j] * x[j] for j in range(dim)) for i in range(dim)
    ]))
    u.x.scatter_forward()


def _rotation(angle, dim):
    c, s = np.cos(angle), np.sin(angle)
    if dim == 2:
        return np.array([[c, -s], [s, c]])
    # about the z axis, then tilted about x so the rotation is not in-plane
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = 0.4
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])
    return Rx @ Rz


def check_material(material, field_values=None, stimulus_values=None,
                   comm=MPI.COMM_WORLD, stretch=None, rotation=0.7,
                   rel_tol=1.0e-8, dim=2, frame_indifference=True):
    """
    Raise AssertionError with a message naming the failed check.

    field_values: name -> constant value for each field the material
        reads; defaults cover rho, phi, theta.
    stimulus_values: name -> value; defaults to zero for every stimulus.
    dim: 2 or 3, the dimension the material is meant for.
    frame_indifference: skip check 2 for a small-strain model that is
        not objective by construction.
    """

    if dim == 2:
        mesh = create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [2, 2],
                                CellType.quadrilateral)
    else:
        mesh = create_box(comm, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [2, 2, 2],
                          CellType.hexahedron)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 2})

    V = functionspace(mesh, ("Lagrange", 1, (dim,)))
    S = functionspace(mesh, ("Lagrange", 1))
    u = Function(V)
    v = ufl.TestFunction(V)

    fields = {}
    for name in material.fields:
        value = (field_values or {}).get(name, DEFAULT_FIELD_VALUES.get(name, 0.5))
        fields[name] = Function(S)
        fields[name].x.array[:] = value

    stimuli = {}
    for name, shape in material.stimuli.items():
        value = (stimulus_values or {}).get(name, np.zeros(shape))
        stimuli[name] = Constant(mesh, np.asarray(value, dtype=PETSc.ScalarType))

    F = ufl.variable(ufl.Identity(dim) + ufl.grad(u))
    W = material.energy(F, fields, stimuli)
    P = ufl.diff(W, F)

    energy_form = form(W * dx)
    force_form = form(ufl.inner(ufl.grad(v), P) * dx)

    def total_energy():
        return comm.allreduce(assemble_scalar(energy_form), op=MPI.SUM)

    def force_norm():
        b = assemble_vector(force_form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return b.norm()

    if stretch is not None:
        F0 = np.asarray(stretch)
    elif dim == 2:
        F0 = np.array([[1.15, 0.10], [0.05, 0.92]])
    else:
        F0 = np.array([[1.15, 0.10, 0.02], [0.05, 0.92, -0.03], [0.01, 0.04, 1.05]])
    Q = _rotation(rotation, dim)

    def set_stimuli(rotated):
        for name, shape in material.stimuli.items():
            value = np.asarray((stimulus_values or {}).get(name, np.zeros(shape)), dtype=float)
            if rotated and tuple(shape) == (dim,):
                value = Q @ value
            stimuli[name].value[...] = value

    def zero_stimuli():
        for name, shape in material.stimuli.items():
            stimuli[name].value[...] = np.zeros(shape)

    name = type(material).__name__
    report = {}

    # 1. stress-free reference at zero stimulus
    zero_stimuli()
    _affine_displacement(u, np.eye(dim))
    reference_force = force_norm()
    W_reference = total_energy()
    _affine_displacement(u, F0)
    stretched_force = force_norm()
    W_stretched = total_energy()
    report["reference_force"] = reference_force
    assert reference_force <= rel_tol * stretched_force, (
        f"{name}: the reference configuration is not stress-free at zero "
        f"stimulus (|f| = {reference_force:.3e} vs {stretched_force:.3e} "
        "when stretched)."
    )

    # 3. a stretch raises the stored energy at zero stimulus
    report["W_reference"] = W_reference
    report["W_stretched"] = W_stretched
    assert W_stretched >= W_reference - rel_tol * abs(W_stretched), (
        f"{name}: stretching lowers the stored energy at zero stimulus "
        f"(W = {W_stretched:.6e} < W(I) = {W_reference:.6e})."
    )

    # 2. frame indifference, with the stimulus present
    if not frame_indifference:
        return report

    set_stimuli(rotated=False)
    _affine_displacement(u, F0)
    W_unrotated = total_energy()
    set_stimuli(rotated=True)
    _affine_displacement(u, Q @ F0)
    W_rotated = total_energy()
    report["W_unrotated"] = W_unrotated
    report["W_rotated"] = W_rotated
    scale = max(abs(W_unrotated), abs(W_rotated), 1.0e-30)
    assert abs(W_rotated - W_unrotated) <= rel_tol * scale, (
        f"{name}: not frame indifferent, W(F) = {W_unrotated:.12e} but "
        f"W(QF) = {W_rotated:.12e}."
    )

    return report
