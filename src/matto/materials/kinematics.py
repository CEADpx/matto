# Copyright (c) 2025-2026 Ian Galloway, Prashant K. Jha
# SPDX-License-Identifier: GPL-3.0-or-later
"""Kinematic quantities the material models share."""

from collections import namedtuple

import ufl

IsochoricInvariants = namedtuple(
    "IsochoricInvariants", ["J", "Cbar", "I1bar", "I2bar"]
)


def plane_strain_3d(F2):
    """Embed an in-plane 2x2 deformation gradient as 3x3 plane strain."""
    return ufl.as_tensor((
        (F2[0, 0], F2[0, 1], 0.0),
        (F2[1, 0], F2[1, 1], 0.0),
        (0.0,      0.0,      1.0),
    ))


def isochoric_invariants(F):
    """J, the isochoric right Cauchy-Green tensor and its two invariants."""
    J = ufl.det(F)
    C = F.T * F
    Cbar = J**(-2.0 / 3.0) * C
    I1bar = ufl.tr(Cbar)
    I2bar = 0.5 * (ufl.tr(Cbar)**2 - ufl.tr(Cbar * Cbar))
    return IsochoricInvariants(J, Cbar, I1bar, I2bar)


def green_lagrange(F):
    I = ufl.Identity(F.ufl_shape[0])
    return 0.5 * (F.T * F - I)


def polar_rotation(F, eps=1.0e-14):
    """Rotation R of the right polar decomposition F = R U, as a UFL tensor.

    Differentiable in F, so the automatic derivatives used for the
    tangent and the adjoint pass through it. Requires det(F) > 0.

    In 2D, R is the in-plane rotation by the angle of (F00 + F11, F10 - F01).
    In 3D, U^{-1} is the closed form of Hoger and Carlson, Q. Appl. Math.
    42 (1984) 113-117, written with the invariants of C = F^T F and the
    largest principal stretch. That form divides by terms that vanish when
    C has repeated eigenvalues, including F = I. Each square root is
    therefore taken of its argument plus eps, and the cosine argument is
    clamped just inside [-1, 1]. eps is absolute: F is taken to be of
    order one, as a deformation gradient is.
    """
    dim = F.ufl_shape[0]

    if dim == 2:
        a = F[0, 0] + F[1, 1]
        b = F[1, 0] - F[0, 1]
        return ufl.as_tensor(((a, -b), (b, a))) / ufl.sqrt(a * a + b * b)

    def root(x):
        return ufl.sqrt(x + eps)

    I = ufl.Identity(3)
    C = F.T * F
    I1C = ufl.tr(C)
    I2C = 0.5 * (I1C**2 - ufl.tr(C * C))
    I3C = ufl.det(C)

    # Largest principal stretch of U from the trigonometric solution of
    # the characteristic equation of C.
    s = I1C**2 - 3.0 * I2C
    arg = (2.0 * I1C**3 - 9.0 * I1C * I2C + 27.0 * I3C) / (
        2.0 * root(s)**3
    )
    # The derivative of acos is infinite at +-1, so clamp just inside.
    arg = ufl.max_value(ufl.min_value(arg, 1.0 - 1.0e-12), -1.0 + 1.0e-12)
    lam = root(I1C + 2.0 * root(s) * ufl.cos(ufl.acos(arg) / 3.0)) / ufl.sqrt(3.0)

    # Invariants of U.
    I3U = root(I3C)
    I2U = I3U / lam + root(I1C * lam**2 - lam**4 + 2.0 * I3U * lam)
    I1U = lam + root(I1C - lam**2 + 2.0 * I3U / lam)
    delta = I1U * I2U - I3U

    U_inv = (
        I1U * (C * C)
        - (I3U + I1U**3 - 2.0 * I1U * I2U) * C
        + (I1U * I2U**2 - I3U * I1U**2 - I3U * I2U) * I
    ) / (I3U * delta)

    return F * U_inv


def director(theta, dim=2):
    """Unit vector at angle theta from the first axis, in the plane."""
    if dim == 2:
        return ufl.as_vector((ufl.cos(theta), ufl.sin(theta)))
    return ufl.as_vector((ufl.cos(theta), ufl.sin(theta), 0.0))
