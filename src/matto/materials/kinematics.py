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


def director(theta, dim=2):
    """Unit vector at angle theta from the first axis, in the plane."""
    if dim == 2:
        return ufl.as_vector((ufl.cos(theta), ufl.sin(theta)))
    return ufl.as_vector((ufl.cos(theta), ufl.sin(theta), 0.0))
