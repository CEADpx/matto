"""Design-field interpolation the material models share."""


def simp(rho, p, eps):
    """Solid isotropic material with penalization, eps + (1 - eps) rho^p."""
    return eps + (1.0 - eps) * rho**p


def two_phase(W_active, W_matrix, phi_scale):
    """Mix an active phase into a matrix by the penalized fraction."""
    return phi_scale * W_active + (1.0 - phi_scale) * W_matrix
