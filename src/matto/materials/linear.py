"""Linear elasticity as a stored energy."""

import ufl

from .base import Material
from .interpolation import simp


class LinearElastic(Material):
    """
    Small-strain isotropic linear elasticity, plane strain in 2D.

    W = 1/2 lambda tr(e)^2 + mu e:e with e = sym(grad u), scaled by the
    SIMP density interpolation.

    Fields: rho. No stimuli.
    """

    fields = ("rho",)
    stimuli = {}
    parameters = {
        "E": None,
        "nu": None,
        "p_rho": 3.0,
        "eps_rho": 1.0e-6,
    }

    def energy(self, F, fields, stimuli):
        rho_phys = fields["rho"]

        dim = F.ufl_shape[0]
        I = ufl.Identity(dim)
        strain = ufl.sym(F - I)

        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        W = 0.5 * lam * ufl.tr(strain)**2 + mu * ufl.inner(strain, strain)

        return simp(rho_phys, self.p_rho, self.eps_rho) * W
