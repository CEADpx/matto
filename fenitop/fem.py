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
- Additional objective and constraint formulations
"""

import os
import numpy as np
import time
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.linalg import solve
import dolfinx.io
from dolfinx import fem, mesh
from dolfinx.mesh import create_box, CellType, locate_entities_boundary, meshtags, create_rectangle
from dolfinx.fem import (Function, Constant, dirichletbc, locate_dofs_topological, 
                        form, assemble_scalar, functionspace)
from dolfinx import la
from dolfinx.fem.petsc import (create_vector, create_matrix, assemble_vector, assemble_matrix, set_bc)
import pyvista
pyvista.set_jupyter_backend('html')
import basix
from basix.ufl import element
from ufl import variable, inner, grad, det, tr, Identity, outer, dev, sym

from fenitop.utility import create_mechanism_vectors
from fenitop.utility import LinearProblem
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

    # ============================================================
    # Remanence Angle Initialization
    # ============================================================
    theta_active = dv_cfg.get("theta", {}).get("active", False)

    if not theta_active:
        # theta inactive → fixed direction from B_rem_dir
        _B_dir = np.array(fem_params.get("B_rem_dir", (1.0, 0.0)), dtype=np.float64)
    else:
        # Initial direction for theta (used as starting guess only)
        _B_dir = np.array(
            fem_params.get("theta_init_dir", (1.0, 0.0)),
            dtype=np.float64
        )

    _n = np.linalg.norm(_B_dir)
    if _n > 0:
        _B_dir /= _n

    theta0 = float(np.arctan2(_B_dir[1], _B_dir[0]))

    with theta_phys_field.x.petsc_vec.localForm() as loc:
        loc.set(theta0)
    theta_phys_field.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD
    )
 
    # ============================================================
    # Material Interpolation
    # ============================================================
    G0 = fem_params["shear_modulus"]
    nu = fem_params["poisson's ratio"]  # only needed for Kerner model

    # Rho penalization 
    p, eps = opt["penalty"], opt["epsilon"]
    rho_penalty = eps + (1 - eps) * rho_phys_field**p
    G0 = G0 * rho_penalty

    # WE-weighted void penalty infrastructure
    w_void = Function(S0, name="w_void")

    # Normalization constant P0 (set once in topopt.py when weights are first frozen)
    P0_void = Constant(mesh, PETSc.ScalarType(1.0))

    # Expose void-penalty fields for optimizer control
    opt["we_voidpen_weight_field"] = w_void
    opt["we_voidpen_P0_const"] = P0_void   

    model = fem_params.get("G_model", "default")

    # --- Material shear modulus models (all use physical φ ∈ [0, 0.3]) ---
    if model == "default":
        mu = G0
    elif model == "guth":
        mu = G0 * (1 + 2.5*phi_phys_field + 14.1*phi_phys_field**2)
    elif model == "mooney":
        mu = G0 * ufl.exp(2.5*phi_phys_field / (1.0 - 1.35*phi_phys_field))
    elif model == "kerner":
        A = 15*(1 - nu) / (8 - 10*nu)
        mu = G0 * (1 + (A*phi_phys_field) / (1.0 - phi_phys_field))
    else:
        raise ValueError(f"Unknown G_model: {model}")

    # Bulk modulus 
    K = 1000 * G0
    
    # Magnetic parameters (now prescribed by input file)
    mu_0_val = fem_params.get("mu0", 1.256e+3)  # mN/(kA)^2
    mu0 = Constant(mesh, PETSc.ScalarType(mu_0_val))

    # ============================================================
    # Magnetic Fields
    # ============================================================
    B_rem_mag = float(fem_params.get("B_rem_mag", 50.0))

    element2 = basix.ufl.element("DG", mesh.basix_cell(), 0, shape=(mesh.geometry.dim,))
    S0_vector = fem.functionspace(mesh, element2)
    B_rem_field = Function(S0_vector, name="B_rem")  # for output/diagnostics

    theta_active = dv_cfg.get("theta", {}).get("active", False)

    # Remanent magnetic field
    if theta_active:
        # UFL expression (updates automatically when theta_phys_field changes)
        B_rem = B_rem_mag * ufl.as_vector((ufl.cos(theta_phys_field), ufl.sin(theta_phys_field)))

        # Use current theta_phys_field value if present (it is already frozen when inactive; when active it starts at 0)
        def _mag_flux_density_init(x):
            f = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=np.float64)
            f[0, :] = B_rem_mag * 1.0
            f[1, :] = B_rem_mag * 0.0
            return f
        B_rem_field.interpolate(_mag_flux_density_init)

    else:
        # Prescribed spatial remanence field from input, if provided.
        B_rem_func = fem_params.get("B_rem_func", None)

        if B_rem_func is not None:
            def mag_flux_density(x):
                dirs = B_rem_func(x)

                f = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=np.float64)

                nrm = np.sqrt(dirs[0, :]**2 + dirs[1, :]**2)
                good = nrm > 1.0e-14

                f[0, good] = B_rem_mag * dirs[0, good] / nrm[good]
                f[1, good] = B_rem_mag * dirs[1, good] / nrm[good]

                return f

        else:
            # Prescribed uniform direction from input.
            B_rem_dir = np.array(fem_params.get("B_rem_dir", (1.0, 0.0)), dtype=np.float64)
            nrm = np.linalg.norm(B_rem_dir)
            if nrm > 0:
                B_rem_dir /= nrm

            def mag_flux_density(x):
                f = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=np.float64)
                f[0, :] = B_rem_mag * B_rem_dir[0]
                f[1, :] = B_rem_mag * B_rem_dir[1]
                return f

        B_rem_field.interpolate(mag_flux_density)
        B_rem = B_rem_field
        
    # Expose both for downstream use
    opt["B_rem_field"] = B_rem_field
    opt["B_rem_expr"] = B_rem

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
    facets, markers, traction_constants, tractions = [], [], [], [] 

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
    
    # ============================================================
    # Objective Boundary Markers
    # ============================================================
    objective_bcs = opt.get("objective_bcs", [])
    if objective_bcs is None:
        objective_bcs = []

    if len(objective_bcs) > 0:

        # If user provides "marker", use gmsh facet tags directly.
        marker_based = any(("marker" in bc) for bc in objective_bcs)

        if marker_based:
            facet_tags_gmsh = fem_params.get("facet_tags", None)
            if facet_tags_gmsh is None:
                raise RuntimeError(
                    "Marker-based objective_bcs requires fem_params['facet_tags'] "
                    "(gmsh facet tags) to be provided."
                )

            ds_obj = ufl.Measure("ds", domain=mesh, metadata=metadata, subdomain_data=facet_tags_gmsh)

            # Store for downstream use
            opt["ds_obj"] = ds_obj
            opt["facet_tags_obj"] = facet_tags_gmsh
            opt["objective_marker_map"] = {bc["name"]: int(bc["marker"]) for bc in objective_bcs}

        else:
            # Predicate-based objective boundaries (legacy path)
            obj_facets, obj_markers = [], []

            for obj_marker, bc_dict in enumerate(objective_bcs):
                obj_func = bc_dict["on_boundary"]
                current_facets = locate_entities_boundary(mesh, fdim, obj_func)
                obj_facets.extend(current_facets)
                obj_markers.extend([obj_marker] * len(current_facets))

            obj_facets = np.array(obj_facets, dtype=np.int32)
            obj_markers = np.array(obj_markers, dtype=np.int32)

            # Remove duplicates + keep consistent ordering
            _, unique_indices = np.unique(obj_facets, return_index=True)
            obj_facets, obj_markers = obj_facets[unique_indices], obj_markers[unique_indices]
            sorted_indices = np.argsort(obj_facets)

            facet_tags_obj = meshtags(mesh, fdim, obj_facets[sorted_indices], obj_markers[sorted_indices])
            ds_obj = ufl.Measure("ds", domain=mesh, metadata=metadata, subdomain_data=facet_tags_obj)

            opt["facet_tags_obj"] = facet_tags_obj
            opt["ds_obj"] = ds_obj
            opt["objective_marker_map"] = {bc["name"]: i for i, bc in enumerate(objective_bcs)}

    else:
        opt["ds_obj"] = None
        opt["objective_marker_map"] = {}

 
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
        W_elastic = (K/2) * (ufl.tr(Egreen))**2 + mu * ufl.tr(Egreen * Egreen)

    # Magnetic energy density
    W_magnetic = -(1/mu0) * inner(F * B_rem, B_app)

    # Effective magnetic fraction
    phi_eff = phi_phys_field * rho_phys_field

    # For post-processing: magnetization-like vector field
    # m(x) = phi_eff * [cos(theta_phys), sin(theta_phys)]
    opt["m_expr"] = phi_eff * ufl.as_vector((ufl.cos(theta_phys_field), ufl.sin(theta_phys_field)))

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
    # GENERIC OBJECTIVE FORMS
    # ============================================================

    # Internal force 
    opt["f_int"] = ufl.derivative(W * dx, u_field, v)

    # Determine objective type
    obj_type = opt.get("objective_type", "compliance")

    if obj_type == "compliance":
        J = inner(u_field, b) * dx
        for marker, t in enumerate(traction_constants):
            J += inner(u_field, t) * ds(marker)

    elif obj_type == "max_disp":
        # Maximize boundary displacement magnitude
        target_marker = 0
        J = - ufl.inner(u_field, ufl.as_vector((0.0, 1.0))) * ds(target_marker)

    elif obj_type == "max_disp_norm":
        # Maximize displacement magnitude on right boundary (direction-free)
        target_marker = 0
        u_norm = ufl.sqrt(u_field[0]**2 + u_field[1]**2)
        J = - u_norm * ds(target_marker)
        
    elif obj_type == "boundary_disp":
        # Directional boundary displacement objective (actuator/gripper style)
        ds_obj = opt.get("ds_obj", None)
        objective_bcs = opt.get("objective_bcs", [])

        if ds_obj is None or objective_bcs is None or len(objective_bcs) == 0:
            raise RuntimeError(
                "objective_type='boundary_disp' requires opt['objective_bcs'] (non-empty) "
                "so fem.py can build ds_obj markers."
            )

        J = 0
        for i, cfg in enumerate(objective_bcs):

            # Direction (normalize safely)
            d = np.array(cfg.get("direction", (0.0, 0.0)), dtype=float)
            nrm = np.linalg.norm(d)
            if nrm > 0:
                d /= nrm
            else:
                raise RuntimeError(f"objective_bcs[{i}] has zero direction vector.")

            w = float(cfg.get("weight", 1.0))

            d_vec = ufl.as_vector((float(d[0]), float(d[1])))

            # Accumulate directional boundary work-like term
            marker = int(cfg.get("marker", i))
            J += - w * ufl.inner(u_field, d_vec) * ds_obj(marker)

    elif obj_type == "rotational_disp":
        # Rotational displacement objective.
        #
        # This maximizes tangential displacement of an output boundary
        # around a user-defined rotation center.
        #
        # For rotation_sign = +1:
        #     t = [-ry, rx] / ||r||
        # rewards CCW rotation.
        #
        # For rotation_sign = -1:
        #     rewards CW rotation.

        ds_obj = opt.get("ds_obj", None)
        objective_bcs = opt.get("objective_bcs", [])

        if ds_obj is None or objective_bcs is None or len(objective_bcs) == 0:
            raise RuntimeError(
                "objective_type='rotational_disp' requires opt['objective_bcs'] "
                "so fem.py can build ds_obj markers."
            )

        X = ufl.SpatialCoordinate(mesh)

        J = 0
        for i, cfg in enumerate(objective_bcs):

            center = cfg.get("center", None)
            if center is None:
                raise RuntimeError(
                    f"objective_bcs[{i}] missing 'center' for rotational_disp."
                )

            xc = float(center[0])
            yc = float(center[1])

            marker = int(cfg.get("marker", i))
            w = float(cfg.get("weight", 1.0))
            rotation_sign = float(cfg.get("rotation_sign", 1.0))

            rx = X[0] - xc
            ry = X[1] - yc

            # Small epsilon prevents division by zero if a facet lies too close to center.
            rnorm = ufl.sqrt(rx**2 + ry**2 + 1.0e-12)

            # Unit tangent direction.
            # +1 gives CCW, -1 gives CW.
            t_vec = rotation_sign * ufl.as_vector((
                -ry / rnorm,
                 rx / rnorm
            ))

            # Minimize negative tangential displacement = maximize rotation.
            J += -w * ufl.inner(u_field, t_vec) * ds_obj(marker)
            
    elif obj_type == "rotational_disp_band":
        # Volumetric rotational displacement objective over a thin annular band.
        # This is for internal output regions where ds boundary markers are awkward.
        #
        # J = - ∫ w_band(x) u · t dx
        #
        # rotation_sign = +1 rewards CCW rotation.
        # rotation_sign = -1 rewards CW rotation.

        center = opt.get("rotation_center", None)
        if center is None:
            raise RuntimeError(
                "objective_type='rotational_disp_band' requires opt['rotation_center']."
            )

        xc = float(center[0])
        yc = float(center[1])

        R_obj = float(opt.get("rotation_radius", 1.0))
        sigma = float(opt.get("rotation_band_sigma", 0.10))
        rotation_sign = float(opt.get("rotation_sign", 1.0))
        w_obj = float(opt.get("rotation_weight", 1.0))

        X = ufl.SpatialCoordinate(mesh)

        rx = X[0] - xc
        ry = X[1] - yc

        r = ufl.sqrt(rx**2 + ry**2 + 1.0e-12)

        # Smooth annular band centered at r = R_obj
        w_band = ufl.exp(-((r - R_obj) / sigma)**2)

        # Unit tangent direction
        t_vec = rotation_sign * ufl.as_vector((
            -ry / r,
             rx / r
        ))

        J = -w_obj * w_band * ufl.inner(u_field, t_vec) * dx
 
    elif obj_type == "disp_track":
        # --------------------------------------------------------
        # Displacement tracking objective (multi-point, vector)
        #
        # Supports:
        #   opt["disp_track"] = dict  (single point, backward compatible)
        #   opt["disp_track"] = list of dicts (multi-point)
        #
        # Each point dict supports:
        #   "point":  (x, y)
        #   "target": (ux_t, uy_t)   <-- vector target (Option A)
        #   "sigma":  float          <-- Gaussian radius
        #   "weight": float          <-- optional per-point weight
        #
        # Objective:
        #   J = Σ_i weight_i ∫ w_i(x) ||u(x) - u_target_i||^2 dx
        # --------------------------------------------------------

        disp_cfg = opt["disp_track"]

        # Normalize to list for backward compatibility
        if isinstance(disp_cfg, dict):
            disp_cfg_list = [disp_cfg]
        else:
            disp_cfg_list = list(disp_cfg)

        # Spatial coordinate
        X = ufl.SpatialCoordinate(mesh)

        # Displacement vector field
        u_vec = ufl.as_vector((u_field[0], u_field[1]))

        # Build objective as sum over points
        J = 0

        # Store per-point diagnostics (optional, for printing in topopt.py)
        opt["track_terms"] = []

        for cfg in disp_cfg_list:
            # Required
            x_target = cfg["point"]  # (x, y)

            # Target vector: accept tuple/list/np array
            tgt = cfg["target"]
            ux_t = float(tgt[0])
            uy_t = float(tgt[1])
            u_tgt = ufl.as_vector((ux_t, uy_t))

            # Optional knobs
            sigma = float(cfg.get("sigma", 1.0))
            w_pt  = float(cfg.get("weight", 1.0))

            # Gaussian localization weight
            r2 = (X[0] - x_target[0])**2 + (X[1] - x_target[1])**2
            w  = ufl.exp(-r2 / sigma**2)

            # --------------------------------------------
            # Component-selective displacement tracking
            # --------------------------------------------
            components = cfg.get("components", ("x", "y"))

            term = 0

            if "x" in components:
                diff_x = u_field[0] - ux_t
                term += diff_x**2

            if "y" in components:
                diff_y = u_field[1] - uy_t
                term += diff_y**2

            J += w_pt * w * term * dx

            # Diagnostics:
            #   weighted-average displacement vector over the local region
            #   u_avg = (∫ u w dx) / (∫ w dx)
            opt["track_terms"].append(
                {
                    # scalar component integrals (UFL-safe)
                    "ux_form": u_field[0] * w * dx,
                    "uy_form": u_field[1] * w * dx,
                    "w_form":  w * dx,

                    "target": (ux_t, uy_t),
                    "point":  (float(x_target[0]), float(x_target[1])),
                    "sigma":  sigma,
                    "weight": w_pt,
                }
            )

    else:
        raise RuntimeError(f"Unknown objective_type: {obj_type}")

    # Store objective form
    opt["objective_form"] = J

    # Derivatives of objective wrt fields
    opt["dObj_du_form"]   = ufl.derivative(J, u_field)
    opt["dObj_drho_form"] = ufl.derivative(J, rho_phys_field)
    opt["dObj_dphi_form"] = ufl.derivative(J, phi_phys_field)
    opt["dObj_dtheta_form"] = ufl.derivative(J, theta_phys_field)

    # TIP DISPLACEMENT (for displacement constraint)
    target_marker = 0
    u_tip_form = ufl.inner(u_field, ufl.as_vector((0.0, 1.0))) * ds(target_marker)

    # Store scalar form
    opt["u_tip_form"] = u_tip_form

    # Partials for adjoint-based total derivatives
    opt["dUtip_du_form"]     = ufl.derivative(u_tip_form, u_field)
    opt["dUtip_drho_form"]   = ufl.derivative(u_tip_form, rho_phys_field)
    opt["dUtip_dphi_form"]   = ufl.derivative(u_tip_form, phi_phys_field)
    opt["dUtip_dtheta_form"] = ufl.derivative(u_tip_form, theta_phys_field)

    # Compliance constraint forms
    C_form = inner(u_field, b) * dx
    for marker, t in enumerate(traction_constants):
        C_form += inner(u_field, t) * ds(marker)

    opt["compliance_form"] = C_form

    # Partials needed for total derivative (adjoint-based)
    opt["dC_du_form"]   = ufl.derivative(C_form, u_field)
    opt["dC_drho_form"] = ufl.derivative(C_form, rho_phys_field)
    opt["dC_dphi_form"] = ufl.derivative(C_form, phi_phys_field)
    opt["dC_dtheta_form"] = ufl.derivative(C_form, theta_phys_field)

    # Stress constraint 
    # ---- material parameters ----
    pnorm = opt.get("stress_pnorm", 12)
    sigma_max = opt.get("sigma_max", 1e6)  # override in input script

    # ---- compute Cauchy stress ----
    P = ufl.diff(W, F)
    J_det = ufl.det(F)
    sigma_cauchy = (1.0 / J_det) * (P * ufl.transpose(F))

    # ---- von-Mises ----
    sigma_dev = ufl.dev(ufl.sym(sigma_cauchy))
    sigma_vm = ufl.sqrt(1.5 * ufl.inner(sigma_dev, sigma_dev))

    # ---- stress relaxation weight ----
    q_rho = float(opt.get("stress_q_rho", 1.0/3.0))
    eps = opt["epsilon"]
    w_sigma = eps + (1.0 - eps) * (rho_phys_field ** q_rho)

    # ---- p-norm integrand ----
    stress_integrand = (w_sigma * sigma_vm)**pnorm

    # store the p-power form ( ∫ f^p dx )
    opt["stress_power_form"] = stress_integrand * dx

    # actual p-norm G = ( ∫ f^p dx )^(1/p) will be computed in Sensitivity.py
    opt["stress_form"] = opt["stress_power_form"]

    # ---- derivatives wrt fields ----
    opt["dStress_du_form"]   = ufl.derivative(opt["stress_power_form"], u_field)
    opt["dStress_drho_form"] = ufl.derivative(opt["stress_power_form"], rho_phys_field)
    opt["dStress_dphi_form"] = ufl.derivative(opt["stress_power_form"], phi_phys_field)
    opt["dStress_dtheta_form"] = ufl.derivative(opt["stress_power_form"], theta_phys_field)

    # Store raw von-Mises expression for post-processing/output
    opt["sigma_vm_expr"] = sigma_vm

    # Strain-energy constraint
    # Store the elastic strain-energy integrand for constraint use
    opt["strain_energy_form"] = W_elastic * dx

    # Derivatives for adjoint computation (only used if enabled)
    opt["dU_du_form"]   = ufl.derivative(W_elastic * dx, u_field)
    opt["dU_drho_form"] = ufl.derivative(W_elastic * dx, rho_phys_field)
    opt["dU_dphi_form"] = ufl.derivative(W_elastic * dx, phi_phys_field)
    opt["dU_dtheta_form"] = ufl.derivative(W_elastic * dx, theta_phys_field)

    # For ParaView visualization (optional): local strain energy density
    opt["W_elastic_expr"] = W_elastic

    # Define global optimization-related variables 
    opt["volume_phi"] = phi_phys_field*dx
    opt["volume_rho"] = rho_phys_field * dx
    opt["total_volume"] = Constant(mesh, 1.0)*dx
    
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
