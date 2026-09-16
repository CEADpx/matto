"""Anisotropic magneto-active elastomer in a silicone matrix, plane strain."""

import numpy as np
import ufl

from .base import Material
from .interpolation import simp, two_phase
from .kinematics import director, isochoric_invariants, plane_strain_3d


class AnisotropicMagnetoActiveElastomer(Material):
    """
    Anisotropic magneto-active elastomer with particle chains, in silicone.

    The chains lie along the design direction theta. The magnetic
    stiffening saturates with the field and scales with the alignment
    of the chains with the fixed magnetization direction theta_M. The
    two phases are mixed by the design fraction phi.

    Fields: rho (density), phi (MAE fraction), theta (chain direction).
    Stimulus: h, applied field magnitude as a scalar.
    """

    fields = ("rho", "phi", "theta")
    stimuli = {"h": ()}
    parameters = {
        "A_Ak": None, "a_Ak": None, "b_Ak": None,
        "r": None, "s": None, "hs": None, "K_Ak": None,
        "A_sil": None, "a_sil": None, "b_sil": None, "K_sil": None,
        "theta_M": 0.0,
        "delta_theta": 1.0e-6,
        "p_rho": 3.0,
        "eps_rho": 1.0e-6,
        "p_phi": 1.0,
    }
    reference = (
        "Akbari & Khajehsaeid, Smart Mater. Struct. 30 (2021) 015008, "
        "doi:10.1088/1361-665X/abc72f"
    )

    def energy(self, F, fields, stimuli):
        rho_phys = fields["rho"]
        phi_phys = fields["phi"]
        theta_phys = fields["theta"]
        h = stimuli["h"]

        J, Cbar, I1bar, _ = isochoric_invariants(plane_strain_3d(F))

        Nhat = director(theta_phys, dim=3)
        Mhat = ufl.as_vector((np.cos(self.theta_M), np.sin(self.theta_M), 0.0))

        I4bar = ufl.dot(Nhat, Cbar * Nhat)

        # Smoothed |cos| of the angle between chains and magnetization.
        alignment_raw = ufl.dot(Nhat, Mhat)
        alignment = (
            ufl.sqrt(alignment_raw**2 + self.delta_theta**2) - self.delta_theta
        ) / (np.sqrt(1.0 + self.delta_theta**2) - self.delta_theta)

        Astar = self.s * (1.0 - ufl.exp(-(h / (2.0 * self.hs))**2)) * alignment

        def g(a, b):
            return (
                (1.0 / a) * ufl.exp(a * (I1bar - 3.0))
                + b * (I1bar - 2.0) * (1.0 - ufl.ln(I1bar - 2.0))
                - 1.0 / a
                - b
            )

        W_Ak = (
            0.5 * self.K_Ak * (J - 1.0)**2
            + (self.A_Ak + Astar) * g(self.a_Ak, self.b_Ak)
            + self.r * (I4bar - 1.0)**2
        )

        W_sil = 0.5 * self.K_sil * (J - 1.0)**2 + self.A_sil * g(self.a_sil, self.b_sil)

        rho_scale = simp(rho_phys, self.p_rho, self.eps_rho)
        phi_scale = phi_phys**self.p_phi

        return rho_scale * two_phase(W_Ak, W_sil, phi_scale)
