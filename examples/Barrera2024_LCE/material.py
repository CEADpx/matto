"""2D plane-strain Barrera2024 LCE energy shared by the examples in this directory."""

import ufl


def make_build_free_energy(material_parameters):
    """Return the ``build_free_energy`` callback expected by ``topopt``."""

    def build_free_energy(u_field, design_variables, stimuli):
        rho_phys = design_variables["rho"].phys
        phi_phys = design_variables["phi"].phys
        theta_phys = design_variables["theta"].phys

        activation = stimuli["activation"]

        E_LCE = material_parameters["E_LCE"]
        nu = material_parameters["nu"]
        beta = material_parameters["beta"]
        S0 = material_parameters["S0"]

        p_rho = material_parameters["p_rho"]
        eps_rho = material_parameters["eps_rho"]
        p_phi = material_parameters["p_phi"]

        mu = E_LCE / (2.0 * (1.0 + nu))
        lam = (
            E_LCE * nu
            / ((1.0 + nu) * (1.0 - 2.0 * nu))
        )

        I = ufl.Identity(2)
        F = ufl.variable(I + ufl.grad(u_field))
        C = F.T * F
        E_GL = 0.5 * (C - I)

        # Programmed in-plane mesogen director.
        director = ufl.as_vector((
            ufl.cos(theta_phys),
            ufl.sin(theta_phys),
        ))

        Q_nem = 3.0 * ufl.outer(director, director) - I

        # The solver ramps activation upward from zero. This corresponds to
        # decreasing the order parameter from S0 to zero:
        #     S = S0 * (1 - activation)
        #     delta_S = S0 - S = S0 * activation
        delta_S = S0 * activation

        W_passive = (
            0.5 * lam * ufl.tr(E_GL)**2
            + mu * ufl.inner(E_GL, E_GL)
        )

        W_coupling = (
            -0.5
            * beta
            * delta_S
            * ufl.inner(Q_nem, E_GL)
        )

        rho_scale = (
            eps_rho
            + (1.0 - eps_rho) * rho_phys**p_rho
        )
        phi_scale = phi_phys**p_phi

        W = rho_scale * (
            W_passive
            + phi_scale * W_coupling
        )

        return W, F

    return build_free_energy
