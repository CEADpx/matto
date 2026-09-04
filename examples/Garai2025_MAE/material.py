"""Plane-strain Garai2025 isotropic MAP energy shared by the examples in this directory."""

import ufl


def make_build_free_energy(material_parameters):
    """Return the ``build_free_energy`` callback expected by ``topopt``."""

    def build_free_energy(u_field, design_variables, stimuli):
        rho_phys = design_variables["rho"].phys
        phi_phys = design_variables["phi"].phys

        h = stimuli["h"]

        C1_Ga = material_parameters["C1_Ga"]
        C2_Ga = material_parameters["C2_Ga"]
        a1 = material_parameters["a1"]
        a2 = material_parameters["a2"]
        b1 = material_parameters["b1"]
        b2 = material_parameters["b2"]

        C1_sil = material_parameters["C1_sil"]
        C2_sil = material_parameters["C2_sil"]

        K = material_parameters["K"]
        mu0 = material_parameters["mu0"]

        p_rho = material_parameters["p_rho"]
        eps_rho = material_parameters["eps_rho"]
        p_phi = material_parameters["p_phi"]

        I2 = ufl.Identity(2)
        F2 = ufl.variable(I2 + ufl.grad(u_field))

        F = ufl.as_tensor((
            (F2[0, 0], F2[0, 1], 0.0),
            (F2[1, 0], F2[1, 1], 0.0),
            (0.0,      0.0,      1.0),
        ))

        J = ufl.det(F)
        C = F.T * F
        Cbar = J**(-2.0 / 3.0) * C

        I1bar = ufl.tr(Cbar)
        I2bar = 0.5 * (
            ufl.tr(Cbar)**2
            - ufl.tr(Cbar * Cbar)
        )

        # h = mu0*|H| [T], while the Garai parameters use |H| [A/m].
        H_mag = h / mu0

        zeta_2 = b1 * ufl.ln(b2 * H_mag + 1.0)
        eta_2 = a1 * (ufl.exp(a2 * H_mag) - 1.0)
        C2_hat = zeta_2 * ufl.atan(eta_2 * H_mag)

        W_Ga = (
            0.5 * K * (J - 1.0)**2
            + C1_Ga * (I1bar - 3.0)
            + (C2_Ga + C2_hat) * (I2bar - 3.0)
        )

        W_sil = (
            0.5 * K * (J - 1.0)**2
            + C1_sil * (I1bar - 3.0)
            + C2_sil * (I2bar - 3.0)
        )

        rho_scale = (
            eps_rho
            + (1.0 - eps_rho) * rho_phys**p_rho
        )
        phi_scale = phi_phys**p_phi

        W = rho_scale * (
            phi_scale * W_Ga
            + (1.0 - phi_scale) * W_sil
        )

        return W, F2

    return build_free_energy
