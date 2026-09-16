"""Plane-strain Akbari2021 MRE energy shared by the examples in this directory."""

import numpy as np
import ufl


def make_build_free_energy(material_parameters):
    """Return the ``build_free_energy`` callback expected by ``OptimizationDriver``."""

    def build_free_energy(u_field, design_variables, stimuli):
        rho_phys = design_variables["rho"].phys
        phi_phys = design_variables["phi"].phys
        theta_phys = design_variables["theta"].phys

        h = stimuli["h"]

        A_Ak = material_parameters["A_Ak"]
        a_Ak = material_parameters["a_Ak"]
        b_Ak = material_parameters["b_Ak"]
        r = material_parameters["r"]
        s = material_parameters["s"]
        hs = material_parameters["hs"]
        K_Ak = material_parameters["K_Ak"]

        A_sil = material_parameters["A_sil"]
        a_sil = material_parameters["a_sil"]
        b_sil = material_parameters["b_sil"]
        K_sil = material_parameters["K_sil"]

        theta_M = material_parameters["theta_M"]
        delta_theta = material_parameters["delta_theta"]

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

        Nhat = ufl.as_vector((
            ufl.cos(theta_phys),
            ufl.sin(theta_phys),
            0.0,
        ))

        Mhat = ufl.as_vector((
            np.cos(theta_M),
            np.sin(theta_M),
            0.0,
        ))

        I4bar = ufl.dot(Nhat, Cbar * Nhat)

        alignment_raw = ufl.dot(Nhat, Mhat)
        alignment = (
            ufl.sqrt(alignment_raw**2 + delta_theta**2)
            - delta_theta
        ) / (
            np.sqrt(1.0 + delta_theta**2)
            - delta_theta
        )

        Astar = (
            s
            * (1.0 - ufl.exp(-(h / (2.0 * hs))**2))
            * alignment
        )

        g_Ak = (
            (1.0 / a_Ak) * ufl.exp(a_Ak * (I1bar - 3.0))
            + b_Ak
            * (I1bar - 2.0)
            * (1.0 - ufl.ln(I1bar - 2.0))
            - 1.0 / a_Ak
            - b_Ak
        )

        W_Ak = (
            0.5 * K_Ak * (J - 1.0)**2
            + (A_Ak + Astar) * g_Ak
            + r * (I4bar - 1.0)**2
        )

        g_sil = (
            (1.0 / a_sil) * ufl.exp(a_sil * (I1bar - 3.0))
            + b_sil
            * (I1bar - 2.0)
            * (1.0 - ufl.ln(I1bar - 2.0))
            - 1.0 / a_sil
            - b_sil
        )

        W_sil = (
            0.5 * K_sil * (J - 1.0)**2
            + A_sil * g_sil
        )

        rho_scale = (
            eps_rho
            + (1.0 - eps_rho) * rho_phys**p_rho
        )
        phi_scale = phi_phys**p_phi

        W = rho_scale * (
            phi_scale * W_Ak
            + (1.0 - phi_scale) * W_sil
        )

        return W, F2

    return build_free_energy
