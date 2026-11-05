# Copyright (c) 2025-2026 Ian Galloway, Prashant K. Jha
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hard-magnetic soft material, in 2D plane strain or 3D."""

import ufl

from .base import Material
from .interpolation import simp
from .kinematics import director, polar_rotation


class HardMagneticSoftMaterial(Material):
    """
    Hard-magnetic soft material: a reinforced matrix with remanent magnetization.

    The magnetic energy is linear in the applied flux density,
    -(1/mu0) phi (F B_rem) . B_app, with B_rem of fixed magnitude along
    the design direction theta. With magnetization_map="R", F is
    replaced by the rotation R of its polar decomposition, so the
    remanent magnetization turns with the material but does not stretch.

    Fields: rho (density), phi (particle fraction), theta (remanent
    direction). Stimulus: B_app, the applied flux density vector.

    Works in 2D and 3D; pass dim=3 for a 3D problem. theta is one
    angle in both cases: in 3D the remanent magnetization lies in the
    x-y plane at that angle, and the applied field may point anywhere.
    """

    fields = ("rho", "phi", "theta")
    parameters = {
        "G0": None,          # matrix shear modulus
        "p_rho": 3.0,
        "eps_rho": 1.0e-6,
        "mu0": None,         # vacuum permeability
        "B_rem_mag": None,   # remanent flux density magnitude
        "dim": 2,
        "magnetization_map": "F",   # "F" or "R"; see the class docstring
    }

    def __init__(self, **parameters):
        super().__init__(**parameters)

        if self.dim not in (2, 3):
            raise ValueError("HardMagneticSoftMaterial needs dim=2 or dim=3.")

        if self.magnetization_map not in ("F", "R"):
            raise ValueError(
                "HardMagneticSoftMaterial needs magnetization_map='F' or "
                f"'R', got {self.magnetization_map!r}."
            )

        self.stimuli = {"B_app": (self.dim,)}
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

        B_rem = self.B_rem_mag * director(theta_phys, dim=self.dim)
        phi_eff = rho_phys * phi_phys

        if self.magnetization_map == "R":
            B_mapped = polar_rotation(F) * B_rem
        else:
            B_mapped = F * B_rem

        W_magnetic = -(1.0 / self.mu0) * phi_eff * ufl.inner(B_mapped, B_app)

        return W_elastic + W_magnetic
