"""2D hMSM free-energy density shared by the examples in this directory."""

import ufl


def make_build_free_energy(material_parameters):
    """Return the ``build_free_energy`` callback expected by ``OptimizationDriver``."""

    def build_free_energy(u_field, design_variables, stimuli):
        rho_phys = design_variables["rho"].phys
        phi_phys = design_variables["phi"].phys
        theta_phys = design_variables["theta"].phys

        B_app = stimuli["B_app"]

        G0 = material_parameters["G0"]
        p_rho = material_parameters["p_rho"]
        eps_rho = material_parameters["eps_rho"]
        mu0 = material_parameters["mu0"]
        B_rem_mag = material_parameters["B_rem_mag"]

        I = ufl.Identity(2)
        F = ufl.variable(I + ufl.grad(u_field))

        C = F.T * F
        I1 = ufl.tr(C)
        J = ufl.det(F)

        rho_scale = (
            eps_rho
            + (1.0 - eps_rho) * rho_phys**p_rho
        )

        G_matrix = G0 * rho_scale

        # Mooney particle-reinforcement model.
        reinforcement = ufl.exp(
            2.5 * phi_phys
            / (1.0 - 1.35 * phi_phys)
        )

        mu = G_matrix * reinforcement

        # Bulk modulus is density-interpolated but independent of phi.
        K = 500.0 * G_matrix

        W_elastic = (
            (mu / 2.0)
            * (
                I1
                - 3.0
                - 2.0 * ufl.ln(J)
            )
            + (K / 2.0) * (J - 1.0)**2
        )

        B_rem = B_rem_mag * ufl.as_vector((
            ufl.cos(theta_phys),
            ufl.sin(theta_phys),
        ))

        phi_eff = rho_phys * phi_phys

        W_magnetic = (
            -(1.0 / mu0)
            * phi_eff
            * ufl.inner(F * B_rem, B_app)
        )

        W = W_elastic + W_magnetic

        return W, F

    return build_free_energy
