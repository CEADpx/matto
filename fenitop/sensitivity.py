"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""

"""
Modifications by Ian Galloway (ian.galloway@mines.sdsmt.edu) and Prashant Jha (prashant.jha@sdsmt.edu)

Edits to sensitivity.py:
- The overall structure of the Sensitivity class remains intact
- The computation of dCdrho and dUdrho has been modified
- Compliant mechanism support has been removed, dUdrho is now set to zero
- dCdrho has been reformulated using the adjoint method, solving a transposed linear system.
  The new formulation is based on differentiating the compliance and internal force,
  allowing accurate sensitivity computation for nonlinear problems.
  This replaces the simplified expression used in the original FEniTop.
"""

import ufl
from mpi4py import MPI
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import (create_vector, create_matrix, assemble_vector, assemble_matrix, set_bc)
from petsc4py import PETSc
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.linalg import solve
import numpy as np


class Sensitivity():
    def __init__(self, comm, opt, problem, u_field, lambda_field, rho_phys):
        self.opt_compliance = opt["opt_compliance"]
        self.comm = comm
        
        # Volume
        self.total_volume = comm.allreduce(assemble_scalar(form(opt["total_volume"])), op=MPI.SUM)
        self.V_form = form(opt["volume"])
        dVdrho_form = form(ufl.derivative(opt["volume"], rho_phys))
        self.dVdrho_vec = create_vector(dVdrho_form)
        assemble_vector(self.dVdrho_vec, dVdrho_form)
        self.dVdrho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.dVdrho_vec /= self.total_volume

        # Displacement holder
        self.dUdrho_vec = rho_phys.x.petsc_vec.copy()
        self.u_field, self.lambda_field = u_field, lambda_field
        self.problem = problem

        # Compliance (via adjoint)
        self.C_form = form(opt["compliance"])

        self.dCdrho_form = form(ufl.derivative(opt["compliance"], rho_phys))  
        self.dCdrho_vec = create_vector(self.dCdrho_form)

        self.dCdu_form = form(ufl.derivative(opt["compliance"], u_field))
        self.dCdu_vec = create_vector(self.dCdu_form)        

        self.dfdrho_form = form(ufl.derivative(opt["f_int"], rho_phys))
        self.dfdrho_mat = create_matrix(self.dfdrho_form)
        
        self.dfdu_form = form(ufl.derivative(opt["f_int"], u_field))
        self.dfdu_mat = create_matrix(self.dfdu_form)

        self.dAdjointTerm_vec = rho_phys.x.petsc_vec.copy() 

    def __del__(self):
        if not self.opt_compliance:
            self.prod_vec.destroy()

    def evaluate(self):
        # Volume
        actual_volume = self.comm.allreduce(assemble_scalar(self.V_form), op=MPI.SUM)
        V_value = actual_volume / self.total_volume
        self.dVdrho_vec_copy = self.dVdrho_vec.copy()

        # Compliance
        C_value = self.comm.allreduce(assemble_scalar(self.C_form), op=MPI.SUM)

        # Assemble direct derivative dCdrho
        with self.dCdrho_vec.localForm() as loc:
            loc.set(0)
        assemble_vector(self.dCdrho_vec, self.dCdrho_form)
        self.dCdrho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        self.dCdu_vec.zeroEntries()
        assemble_vector(self.dCdu_vec, self.dCdu_form)
        set_bc(self.dCdu_vec, self.problem.bcs)
        self.dCdu_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.dCdu_vec.scale(-1.0)

        self.dfdu_mat.zeroEntries()
        assemble_matrix(self.dfdu_mat, self.dfdu_form, bcs=self.problem.bcs)
        self.dfdu_mat.assemble()

        self.dfdrho_mat.zeroEntries()
        assemble_matrix(self.dfdrho_mat, self.dfdrho_form)
        self.dfdrho_mat.assemble()  
     
        # Solve (K)^T * lambda = (dC/du)^T
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(self.dfdu_mat)
        ksp.setTolerances(rtol=1e-8, atol=1e-12)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        self.lambda_field.x.petsc_vec.set(0.0) 
        ksp.solveTranspose(self.dCdu_vec, self.lambda_field.x.petsc_vec)
        self.lambda_field.x.scatter_forward()
       
        self.dfdrho_mat.multTranspose(self.lambda_field.x.petsc_vec, self.dAdjointTerm_vec)  # Assemble adjoint term
        self.dAdjointTerm_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.dCdrho_vec.axpy(1.0, self.dAdjointTerm_vec)   # Add adjoint term to direct dirivative 
        
        # Zero out Displacement
        U_value, self.dUdrho_vec = 0, None
        
        func_values = [C_value, V_value, U_value]
        sensitivities = [self.dCdrho_vec, self.dVdrho_vec_copy, self.dUdrho_vec]
        return func_values, sensitivities