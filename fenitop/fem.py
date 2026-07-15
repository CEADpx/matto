"""
Original FEniTop authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7

Major modifications:
- Ian Galloway (ian.galloway@mines.sdsmt.edu)
- Prashant Jha (prashant.jha@sdsmt.edu)

Major additions to fem.py:
- Hyperelastic material models
- Magneto-mechanical coupling through remanent and applied magnetic fields
- Magnetic material fraction design variable (phi)
- Remanence-direction design variable (theta)
- Multi-load-case magnetic and traction loading support
- Compliance, displacement-tracking, and rotational objectives
"""

import numpy as np
import ufl
from petsc4py import PETSc

import basix
from dolfinx import fem
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.mesh import locate_entities_boundary, meshtags
from ufl import grad, inner

from fenitop.utility import WrapNonlinearProblem

def form_fem(fem_params, opt):
    """Form an FEA problem."""

    # ============================================================
    # Function Spaces and State / Design Fields
    # ============================================================
    mesh = fem_params["mesh"]
    element = basix.ufl.element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    V = fem.functionspace(mesh, element)
    S0 = fem.functionspace(mesh, ("DG", 0))
    S = fem.functionspace(mesh, ("CG", 1))
    v = ufl.TestFunction(V)
    u_field = Function(V, name="u") 
    lambda_field = Function(V, name="lambda")  

    # Initialize material density field
    rho_field = Function(S0, name="rho")  
    rho_phys_field = Function(S, name="rho_phys")  
    
    # Initialize magnetic density field
    phi_field = Function(S0, name="phi")  
    phi_phys_field = Function(S, name="phi_phys")  

    # Initialize theta (remanence direction angle) field
    # theta_field: design variable (DG0)
    # theta_phys_field: filtered physical field (CG1)
    theta_field = Function(S0, name="theta")
    theta_phys_field = Function(S, name="theta_phys")

    # Expose theta fields for optimizer and sensitivity modules
    opt["theta_field"] = theta_field
    opt["theta_phys_field"] = theta_phys_field

    # ============================================================
    # Inactive Design Variable Handling
    # ============================================================
    dv_cfg = opt.get("design_variables", {})

    # Apply rho inactive → rho_phys = 1 (CG1)
    if not dv_cfg.get("rho", {}).get("active", True):
        with rho_phys_field.x.petsc_vec.localForm() as loc:
            loc.set(1.0)
        rho_phys_field.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD
        )

    # Apply phi inactive → phi_phys = 0 (CG1)
    if not dv_cfg.get("phi", {}).get("active", True):
        with phi_phys_field.x.petsc_vec.localForm() as loc:
            loc.set(0.0)
        phi_phys_field.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD
        )

    theta_active = dv_cfg.get("theta", {}).get("active", False)
 
    # ============================================================
    # Material Interpolation
    # ============================================================
    G0 = fem_params["shear_modulus"]  # kPa = mN/mm^2

    # Rho penalization 
    p, eps = opt["penalty"], opt["epsilon"]
    rho_penalty = eps + (1 - eps) * rho_phys_field**p
    G0 = G0 * rho_penalty

    model = fem_params.get("G_model", "default")

    # --- Material shear modulus models (all use physical φ ∈ [0, 0.3]) ---
    if model == "default":
        mu = G0
    elif model == "guth":
        mu = G0 * (1 + 2.5*phi_phys_field + 14.1*phi_phys_field**2)
    elif model == "mooney":
        mu = G0 * ufl.exp(2.5*phi_phys_field / (1.0 - 1.35*phi_phys_field))
    elif model == "kerner":
        A = 2.5
        mu = G0 * (1 + (A*phi_phys_field) / (1.0 - phi_phys_field))
    elif model == "LP":
        mu = G0 / (1.0 - phi_phys_field)**2.5
    elif model == "LPA":
        g = 3.73
        chi = 1.0 + 0.67*g*phi_phys_field + 1.62*(g*phi_phys_field)**2
        mu = G0*(1.0 - phi_phys_field)*chi
    elif model == "hill":
        mu = G0 / (1.0 - 2.5*phi_phys_field)
    else:
        raise ValueError(f"Unknown G_model: {model}")

    # Penalized bulk modulus, independent of phi
    K = 500 * G0

    # Magnetic permeability in the mT-kPa unit system
    mu_0_val = fem_params.get("mu0", 1.256e3)  # mT^2/kPa
    mu0 = Constant(mesh, PETSc.ScalarType(mu_0_val))

    # ============================================================
    # Magnetic Fields
    # ============================================================
    B_rem_mag = float(fem_params.get("B_rem_mag", 50.0))

    if theta_active:
        # Spatial direction controlled by the filtered theta field
        B_rem = B_rem_mag * ufl.as_vector((
            ufl.cos(theta_phys_field),
            ufl.sin(theta_phys_field),
        ))

    else:
        # Prescribed remanence direction
        vector_element = basix.ufl.element(
            "DG",
            mesh.basix_cell(),
            0,
            shape=(mesh.geometry.dim,),
        )
        S0_vector = fem.functionspace(mesh, vector_element)
        B_rem_field = Function(S0_vector, name="B_rem")

        B_rem_func = fem_params.get("B_rem_func")

        if B_rem_func is not None:
            def mag_flux_density(x):
                directions = B_rem_func(x)
                values = np.zeros(
                    (mesh.geometry.dim, x.shape[1]),
                    dtype=np.float64,
                )

                norms = np.sqrt(
                    directions[0, :]**2 + directions[1, :]**2
                )
                nonzero = norms > 1.0e-14

                values[0, nonzero] = (
                    B_rem_mag * directions[0, nonzero] / norms[nonzero]
                )
                values[1, nonzero] = (
                    B_rem_mag * directions[1, nonzero] / norms[nonzero]
                )
                return values

        else:
            B_rem_dir = np.array(
                fem_params.get("B_rem_dir", (1.0, 0.0)),
                dtype=np.float64,
            )
            norm = np.linalg.norm(B_rem_dir)
            if norm > 0:
                B_rem_dir /= norm

            def mag_flux_density(x):
                values = np.zeros(
                    (mesh.geometry.dim, x.shape[1]),
                    dtype=np.float64,
                )
                values[0, :] = B_rem_mag * B_rem_dir[0]
                values[1, :] = B_rem_mag * B_rem_dir[1]
                return values

        B_rem_field.interpolate(mag_flux_density)
        B_rem = B_rem_field

    # --- Applied field (B_app) ---
    B_app_mag = float(fem_params.get("B_app_mag", 0.0))
    B_app_dir = np.array(fem_params.get("B_app_dir", (0.0, 0.0)), dtype=np.float64)

    # Safe normalization: allow zero direction when mag == 0
    nrm_app = np.linalg.norm(B_app_dir)
    if nrm_app > 0:
        B_app_dir /= nrm_app
    else:
        B_app_dir[:] = 0.0

    # Applied magnetic field (updated per load case)
    B_app = Constant(mesh, B_app_mag * B_app_dir)

    # Expose handle so topopt.py can update per load case
    opt["B_app"] = B_app

    # ============================================================
    # Boundary / Interior Displacement BC
    # ============================================================

    dim = mesh.topology.dim
    fdim = dim - 1

    interior_bc = fem_params.get("interior_BC", False)

    if interior_bc:
        # Clamp interior DOFs geometrically (for hub clamps etc.)
        disp_dofs = fem.locate_dofs_geometrical(V, fem_params["disp_bc"])

    else:
        # Standard boundary facet BC (original behavior)
        disp_facets = locate_entities_boundary(mesh, fdim, fem_params["disp_bc"])
        disp_dofs = locate_dofs_topological(V, fdim, disp_facets)

    bc = dirichletbc(Constant(mesh, np.full(dim, 0.0)), disp_dofs, V)

    # Tractions
    facets, markers, traction_constants = [], [], []

    for marker, bc_dict in enumerate(fem_params["traction_bcs"]):
        traction_max = np.array(bc_dict["traction_max"], dtype=float)
        traction_func = bc_dict["on_boundary"]    
        traction_const = Constant(mesh, np.zeros_like(traction_max))  # start zero, update externally
        traction_constants.append(traction_const)
        current_facets = locate_entities_boundary(mesh, fdim, traction_func)
        facets.extend(current_facets)
        markers.extend([marker,]*len(current_facets))

    facets = np.array(facets, dtype=np.int32)
    markers = np.array(markers, dtype=np.int32)
    _, unique_indices = np.unique(facets, return_index=True)
    facets, markers = facets[unique_indices], markers[unique_indices]
    sorted_indices = np.argsort(facets)
    facet_tags = meshtags(mesh, fdim, facets[sorted_indices], markers[sorted_indices])
    
    metadata = {"quadrature_degree": fem_params["quadrature_degree"]}
    dx = ufl.Measure("dx", metadata=metadata)
    ds = ufl.Measure("ds", domain=mesh, metadata=metadata, subdomain_data=facet_tags)
    
    b = Constant(mesh, np.array(fem_params["body_force"], dtype=float))

    # ============================================================
    # Kinematics and Energy Densities
    # ============================================================
    I = ufl.Identity(dim)          #Identity Matrix
    F = ufl.variable(ufl.Identity(dim) + ufl.grad(u_field)) # Deformation gradient
    C = F.T * F                    # Right Cauchy-Green tensor
    Ic = ufl.tr(C)                 # First invariant
    J = ufl.det(F)                 # Jacobian determinant

    # Stored strain energy density 
    if fem_params["hyperModel"] == "neoHookean1":
        W_elastic = (mu / 2) * (J**(-2/3)*Ic - 3) + (K/2) * (J - 1)**2
    elif fem_params["hyperModel"] == "neoHookean2":
        W_elastic = (mu / 2) * (Ic - 3 - 2*ufl.ln(J)) + (K/2) * (J - 1)**2
    elif fem_params["hyperModel"] == "stVenant":
        Egreen = (C - I) / 2
        Edev = Egreen - (ufl.tr(Egreen) / 3.0) * I
        W_elastic = mu * ufl.tr(Edev * Edev) + (K / 2.0) * (ufl.tr(Egreen))**2       

    # Magnetic energy density
    W_magnetic = -(1/mu0) * inner(F * B_rem, B_app)

    # Effective magnetic fraction
    phi_eff = phi_phys_field * rho_phys_field

    W_magnetic = phi_eff * W_magnetic

    # Total energy density
    W = W_elastic + W_magnetic

    P = ufl.diff(W, F)             # First Piola-Kirchhoff stress tensor

    # Assemble Residual
    a = inner(grad(v), P)*dx     # Semilinear form
    L = inner(v, b)*dx           # Linear form
    for marker, t in enumerate(traction_constants):   # Add tractions
        L += inner(v, t)*ds(marker)
    R = L - a 

    # Wrap nonlinear problem
    femProblem = WrapNonlinearProblem(u_field, R, [bc], fem_params["petsc_options"])

    # ============================================================
    # Objective
    # ============================================================

    # Derivative of the total internal energy with respect to displacement
    opt["f_int"] = ufl.derivative(W * dx, u_field, v)

    obj_type = opt.get("objective_type", "compliance")

    if obj_type == "compliance":
        J = inner(u_field, b) * dx
        for marker, traction in enumerate(traction_constants):
            J += inner(u_field, traction) * ds(marker)

    elif obj_type == "rotational_disp_band":
        center = opt.get("rotation_center")
        if center is None:
            raise ValueError(
                "rotational_disp_band requires opt['rotation_center']."
            )

        rotation_radius = float(opt["rotation_radius"])
        band_sigma = float(opt["rotation_band_sigma"])
        rotation_sign = float(opt.get("rotation_sign", 1.0))
        rotation_weight = float(opt.get("rotation_weight", 1.0))

        if rotation_radius <= 0:
            raise ValueError("rotation_radius must be positive.")
        if band_sigma <= 0:
            raise ValueError("rotation_band_sigma must be positive.")

        X = ufl.SpatialCoordinate(mesh)
        rx = X[0] - float(center[0])
        ry = X[1] - float(center[1])
        radius = ufl.sqrt(rx**2 + ry**2 + 1.0e-12)

        band_weight = ufl.exp(
            -((radius - rotation_radius) / band_sigma)**2
        )
        tangent = rotation_sign * ufl.as_vector((
            -ry / radius,
            rx / radius,
        ))

        J = (
            -rotation_weight
            * band_weight
            * inner(u_field, tangent)
            * dx
        )

    elif obj_type == "disp_track":
        X = ufl.SpatialCoordinate(mesh)
        J = 0

        for index, config in enumerate(opt["disp_track"]):
            target_point = config["point"]
            target_displacement = config["target"]
            sigma = float(config.get("sigma", 1.0))
            weight = float(config.get("weight", 1.0))
            components = config.get("components", ("x", "y"))

            if sigma <= 0:
                raise ValueError(
                    f"disp_track[{index}]['sigma'] must be positive."
                )
            if not components or any(
                component not in ("x", "y")
                for component in components
            ):
                raise ValueError(
                    f"disp_track[{index}]['components'] must contain "
                    "'x', 'y', or both."
                )

            ux_target = float(target_displacement[0])
            uy_target = float(target_displacement[1])

            distance_squared = (
                (X[0] - target_point[0])**2
                + (X[1] - target_point[1])**2
            )
            localization = ufl.exp(-distance_squared / sigma**2)

            tracking_error = 0
            if "x" in components:
                tracking_error += (u_field[0] - ux_target)**2
            if "y" in components:
                tracking_error += (u_field[1] - uy_target)**2

            J += weight * localization * tracking_error * dx

    else:
        raise ValueError(
            f"Unknown objective_type: {obj_type}. Supported objectives are "
            "'compliance', 'disp_track', and 'rotational_disp_band'."
        )

    opt["objective_form"] = J
    opt["dObj_du_form"] = ufl.derivative(J, u_field)
    opt["dObj_drho_form"] = ufl.derivative(J, rho_phys_field)
    opt["dObj_dphi_form"] = ufl.derivative(J, phi_phys_field)
    opt["dObj_dtheta_form"] = ufl.derivative(
        J,
        theta_phys_field,
    )

    # ============================================================
    # Cauchy stress field for output
    # ============================================================
    J_det = ufl.det(F)
    sigma_cauchy = (1.0 / J_det) * (P * ufl.transpose(F))
    sigma_dev = ufl.dev(ufl.sym(sigma_cauchy))
    sigma_vm = ufl.sqrt(
        1.5 * ufl.inner(sigma_dev, sigma_dev)
    )
    opt["sigma_vm_expr"] = sigma_vm

    # Volume-constraint forms
    opt["volume_phi"] = phi_phys_field * dx
    opt["volume_rho"] = rho_phys_field * dx
    opt["total_volume"] = Constant(mesh, 1.0) * dx
    
    phi_eff_field = Function(S, name="phi_eff")

    return (
        femProblem,
        u_field,
        lambda_field,
        rho_field,
        rho_phys_field,
        phi_field,
        phi_phys_field,
        phi_eff_field,
        theta_phys_field,
        traction_constants,
        ds,
    )
