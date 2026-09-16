"""Hard-magnetic soft material, plane strain."""

import ufl

from .base import Material
from .interpolation import simp
from .kinematics import director


class HardMagneticSoftMaterial(Material):
    """
    Hard-magnetic soft material: a reinforced matrix with remanent magnetization.

    The magnetic energy is linear in the applied flux density,
    -(1/mu0) phi (F B_rem) . B_app, with B_rem of fixed magnitude along
    the design direction theta.

    Fields: rho (density), phi (particle fraction), theta (remanent
    direction). Stimulus: B_app, in-plane applied flux density.
    """

    fields = ("rho", "phi", "theta")
    stimuli = {"B_app": (2,)}
    parameters = {
        "G0": None,          # matrix shear modulus
        "p_rho": 3.0,
        "eps_rho": 1.0e-6,
        "mu0": None,         # vacuum permeability
        "B_rem_mag": None,   # remanent flux density magnitude
    }
    reference = (
        "Galloway & Jha, Model-informed joint material-structural "
        "optimization of hard-magnetic soft materials, arXiv:2607.14397"
    )

    def energy(self, F, fields, stimuli):
        rho_phys = fields["rho"]
        phi_phys = fields["phi"]
        theta_phys = fields["theta"]
        B_app = stimuli["B_app"]

        C = F.T * F
        I1 = ufl.tr(C)
        J = ufl.det(F)

        G_matrix = self.G0 * simp(rho_phys, self.p_rho, self.eps_rho)

        # Mooney particle-reinforcement model.
        reinforcement = ufl.exp(2.5 * phi_phys / (1.0 - 1.35 * phi_phys))
        mu = G_matrix * reinforcement

        # Bulk modulus is density-interpolated but independent of phi.
        K = 500.0 * G_matrix

        W_elastic = (
            (mu / 2.0) * (I1 - 3.0 - 2.0 * ufl.ln(J))
            + (K / 2.0) * (J - 1.0)**2
        )

        B_rem = self.B_rem_mag * director(theta_phys)
        phi_eff = rho_phys * phi_phys

        W_magnetic = -(1.0 / self.mu0) * phi_eff * ufl.inner(F * B_rem, B_app)

        return W_elastic + W_magnetic
