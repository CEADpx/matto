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
from fenitop.utility import wrapNonlinearProblem

def form_fem(fem_params, opt):
    """Form an FEA problem."""
    # Function spaces and functions
    mesh = fem_params["mesh"]
    element = basix.ufl.element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    V = fem.functionspace(mesh, element)
    S0 = fem.functionspace(mesh, ("DG", 0))
    S = fem.functionspace(mesh, ("CG", 1))
    #u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_field = Function(V, name="u")  # Displacement field
    lambda_field = Function(V, name="lambda")  # Adjoint variable field
    rho_field = Function(S0, name="rho")  # Density field
    rho_phys_field = Function(S, name="rho_phys")  # Physical density field

    # Material interpolation
    E0, nu = fem_params["young's modulus"], fem_params["poisson's ratio"]
    p, eps = opt["penalty"], opt["epsilon"]
    E = (eps + (1-eps)*rho_phys_field**p) * E0
    _lambda, mu = E*nu/(1+nu)/(1-2*nu), E/(2*(1+nu))  #lame constants
    if fem_params["hyperelastic"]:
        K = E/(3*(1-2*nu)) # bulk modulus
    else: 
        # Kinematics
        def epsilon(u):
            return ufl.sym(ufl.grad(u))
        def sigma(u):  # 3D or plane strain
            return 2*mu*epsilon(u) + _lambda*ufl.tr(epsilon(u))*ufl.Identity(len(u))
    # Boundary conditions
    dim = mesh.topology.dim
    fdim = dim - 1
    disp_facets = locate_entities_boundary(mesh, fdim, fem_params["disp_bc"])
    bc = dirichletbc(Constant(mesh, np.full(dim, 0.0)),
                     locate_dofs_topological(V, fdim, disp_facets), V)

    facets, markers, traction_constants, tractions = [], [], [], [] 
    if fem_params["hyperelastic"]:
        for marker, bc_dict in enumerate(fem_params["traction_bcs"]):
            traction_max = np.array(bc_dict["traction_max"], dtype=float)
            traction_func = bc_dict["on_boundary"]    
            traction_const = Constant(mesh, np.zeros_like(traction_max))  # start zero, update externally
            traction_constants.append(traction_const)
            current_facets = locate_entities_boundary(mesh, fdim, traction_func)
            facets.extend(current_facets)
            markers.extend([marker,]*len(current_facets))
    else: #linear case
        for marker, (traction, traction_bc) in enumerate(fem_params["traction_bcs"]):
            tractions.append(Constant(mesh, np.array(traction, dtype=float)))
            current_facets = locate_entities_boundary(mesh, fdim, traction_bc)
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

    # Optimizer
    if opt["opt_compliance"]:
        spring_vec = opt["l_vec"] = None
    else:
        spring_vec, opt["l_vec"] = create_mechanism_vectors(
            V, opt["in_spring"], opt["out_spring"])

    if fem_params["hyperelastic"]:
        # Kinematics
        I = ufl.Identity(dim)
        #F = variable(I + ufl.grad(u))  # Deformation gradient
        F = ufl.variable(ufl.Identity(dim) + ufl.grad(u_field))
        C = F.T * F                    # Right Cauchy-Green tensor
        Ic = ufl.tr(C)                 # First invariant
        J = ufl.det(F)                 # Jacobian determinant
        Egreen = (C - I)/2
    
        # Stored strain energy density (compressible neo-Hookean model)
        if fem_params["neoHookean"]:
            W = (mu / 2) * (J**(-2/3)*Ic - 3) + (K/2)*(J - 1)**2
        elif fem_params["StVenant"]:
            W = (_lambda/2)*(ufl.tr(Egreen))**2 + mu*ufl.tr(Egreen*Egreen)
            
        P = ufl.diff(W, F)             # First Piola-Kirchhoff stress tensor

         ## Assemble Residual
        a = inner(grad(v), P)*dx # semilinear form
        L = inner(v, b)*dx # linear form w/o tractions
        for marker, t in enumerate(traction_constants):   #add tractions
            L += inner(v, t)*ds(marker)
        R = L - a # assemble residual form
        opt["residual"] = R
        
        # Wrap nonlinear problem
        femProblem = WrapNonlinearProblem(u_field, R, [bc], fem_params["petsc_options"])

        # Define optimization-related variables(hyper)
        opt["f_int"] = ufl.derivative(W * dx, u_field, v) 
        #opt["strain_energy"] = W * dx
        epsilon = 1e-6
        def smooth_positive(x):
            return 0.5*(x + ufl.sqrt(x**2 + epsilon))

        #opt["strain_energy"] = 0.5 * inner(sigma(u), epsilon(u)) * dx
        opt["compliance"] = inner(u_field, b)*dx
        for marker, t in enumerate(traction_constants):
            #opt["compliance"] += inner(u_field, t)*ds(marker)
            opt["compliance"] += smooth_positive(inner(u_field, t))*ds(marker)
            
    else: #linear case
        # Establish the equilibrium and adjoint equations
        u = ufl.TrialFunction(V) 
        lhs = ufl.inner(sigma(u), epsilon(v))*dx
        rhs = ufl.dot(b, v)*dx
        for marker, t in enumerate(tractions):
            rhs += ufl.dot(t, v)*ds(marker)
        #wrap linear problem   
        femProblem = LinearProblem(u_field, lambda_field, lhs, rhs, opt["l_vec"],
                                       spring_vec, [bc], fem_params["petsc_options"])

        # Define optimization-related variables(linear)                               
        opt["f_int"] = ufl.inner(sigma(u_field), epsilon(v))*dx
        #opt["compliance"] = ufl.inner(sigma(u_field), epsilon(u_field))*dx #old version
        # Define compliance as work done by external loads
        opt["compliance"] = inner(u_field, b)*dx
        for marker, t in enumerate(tractions):
            opt["compliance"] += inner(u_field, t)*ds(marker)


    # Define optimization-related variables 
    opt["volume"] = rho_phys_field*dx
    opt["total_volume"] = Constant(mesh, 1.0)*dx
    
    return femProblem, u_field, lambda_field, rho_field, rho_phys_field, traction_constants, ds   