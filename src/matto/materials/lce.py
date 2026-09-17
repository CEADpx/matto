# Copyright (c) 2025-2026 Ian Galloway, Prashant K. Jha
# SPDX-License-Identifier: GPL-3.0-or-later
"""Liquid crystal elastomer, plane strain, small-strain energy."""

import ufl

from .base import Material
from .interpolation import simp
from .kinematics import director, green_lagrange


class LiquidCrystalElastomer(Material):
    """
    Liquid crystal elastomer: Saint Venant-Kirchhoff plus a nematic coupling.

    The coupling is -1/2 beta dS (Q . E) with Q = 3 n n - I built on the
    programmed director angle theta. The stimulus is the activation,
    ramped from zero, which lowers the order parameter from S0:
    dS = S0 activation.

    Fields: rho (density), phi (LCE fraction), theta (director).
    Stimulus: activation, scalar in [0, 1].
    """

    fields = ("rho", "phi", "theta")
    stimuli = {"activation": ()}
    parameters = {
        "E_LCE": None,
        "nu": None,
        "beta": None,      # nematic coupling modulus
        "S0": None,        # initial order parameter
        "p_rho": 3.0,
        "eps_rho": 1.0e-6,
        "p_phi": 3.0,
    }
    reference = "Barrera et al. 2024 (LCE energy used in the examples)"

    def energy(self, F, fields, stimuli):
        rho_phys = fields["rho"]
        phi_phys = fields["phi"]
        theta_phys = fields["theta"]
        activation = stimuli["activation"]

        mu = self.E_LCE / (2.0 * (1.0 + self.nu))
        lam = self.E_LCE * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        I = ufl.Identity(2)
        E_GL = green_lagrange(F)

        n = director(theta_phys)
        Q_nem = 3.0 * ufl.outer(n, n) - I

        delta_S = self.S0 * activation

        W_passive = 0.5 * lam * ufl.tr(E_GL)**2 + mu * ufl.inner(E_GL, E_GL)
        W_coupling = -0.5 * self.beta * delta_S * ufl.inner(Q_nem, E_GL)

        rho_scale = simp(rho_phys, self.p_rho, self.eps_rho)
        phi_scale = phi_phys**self.p_phi

        return rho_scale * (W_passive + phi_scale * W_coupling)
