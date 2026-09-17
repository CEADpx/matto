# Copyright (c) 2025-2026 Ian Galloway, Prashant K. Jha
# SPDX-License-Identifier: GPL-3.0-or-later
"""Isotropic magneto-active elastomer in a silicone matrix, plane strain."""

import ufl

from .base import Material
from .interpolation import simp, two_phase
from .kinematics import isochoric_invariants, plane_strain_3d


class MagnetoActiveElastomer(Material):
    """
    Isotropic magneto-active elastomer in a silicone matrix.

    The second Mooney-Rivlin coefficient of the active phase stiffens
    with the applied field; the two phases are mixed by the design
    fraction phi.

    Fields: rho (density), phi (MAE fraction). Stimulus: h, the applied
    flux density mu0 |H| as a scalar.
    """

    fields = ("rho", "phi")
    stimuli = {"h": ()}
    parameters = {
        "C1_Ga": None, "C2_Ga": None,
        "a1": None, "a2": None, "b1": None, "b2": None,
        "C1_sil": None, "C2_sil": None,
        "K": None,
        "mu0": None,
        "p_rho": 3.0,
        "eps_rho": 1.0e-6,
        "p_phi": 1.0,
    }
    reference = (
        "Garai & Haldar, Int. J. Mech. Sci. 286 (2025) 109860, "
        "doi:10.1016/j.ijmecsci.2024.109860"
    )

    def energy(self, F, fields, stimuli):
        rho_phys = fields["rho"]
        phi_phys = fields["phi"]
        h = stimuli["h"]

        J, _, I1bar, I2bar = isochoric_invariants(plane_strain_3d(F))

        # h = mu0*|H| [T], while the Garai parameters use |H| [A/m].
        H_mag = h / self.mu0

        zeta_2 = self.b1 * ufl.ln(self.b2 * H_mag + 1.0)
        eta_2 = self.a1 * (ufl.exp(self.a2 * H_mag) - 1.0)
        C2_hat = zeta_2 * ufl.atan(eta_2 * H_mag)

        W_Ga = (
            0.5 * self.K * (J - 1.0)**2
            + self.C1_Ga * (I1bar - 3.0)
            + (self.C2_Ga + C2_hat) * (I2bar - 3.0)
        )

        W_sil = (
            0.5 * self.K * (J - 1.0)**2
            + self.C1_sil * (I1bar - 3.0)
            + self.C2_sil * (I2bar - 3.0)
        )

        rho_scale = simp(rho_phys, self.p_rho, self.eps_rho)
        phi_scale = phi_phys**self.p_phi

        return rho_scale * two_phase(W_Ga, W_sil, phi_scale)
