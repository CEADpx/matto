# top_optim
**topoptim** is a research fork of [FEniTop](https://github.com/missionlab/fenitop), an open-source topology optimization program built on [FEniCSx](https://fenicsproject.org). This project extends the original framework of **FEniTop** to include non-linear models, including the St. Vernant and NeoHookean models. 

The program consists of an input file which prescribes the FEM setup and optimization settings, and six different modules which perform the optimization routine including: 
- **topopt.py** Runs the full optimization loop by calling on other modules and creates a results folder.
- **fem.py** Utilizes FEniCSx to form an FEA problem, including defining the material interpolation, boundary conditions, weak form equations, and optimization related variables.
- **parameterize.py** Defines **DensityFilter** class which applies a PDE-based Helmholtz filter to the raw density field, and **Heaviside** class which sharpens the filtered field via smooth projection.
- **sensitivity.py** Computes the full derivative of compliance with respect to the design variable, density, via the adjoint method.    
- **optimize.py** Employs the Optimality Criteria (OC) or Method of Moving Asymptotes (MMA) optimizers to update the design variables based off of the sensitivities.  
- **utility.py** Provides auxiliary functions, including plotters, communicators, and classes to wrap and solve linear and non-linear problems.

# Implementation of Non-Linear Models
The non-linear models were implemented into the **FEniTop** framework with only a few additions:
- Update program to use new version of FEniCSx 0.9.0
- Creation of **WrapNonlinearProblem** class to wrap a residual problem and solve iteratively.
- Changed definition of compliance from internal strain energy to external work.
- Implementation of incremental load stepping for nonlinear problems, where tractions are scaled progressively across a defined number of steps. At each step, the nonlinear problem is solved and updated via **WrapNonlinearProblem.solve_fem()**, improving stability and convergence under large deformations.
- New sensitivity class which can be applied to both linear and nonlinear problems, utilizing automatic differentiation and an adjoint sensitivity solve via a transposed linear system  to assemble the full derivative. 

# Optimization Response
To analyze the behavior of nonlinear models against the linear elastic model, a comparison test is performed. Each model solves the same problem with identical input parameters and settings. Three different problems are setup, each with increasing tractions to highlight how these models change with load magnitude when all other variables are held constant.

### Problem Setup
The problem consists of a 2D beam, fixed on the left edge and a traction applied in the downward direction over a small length on the right side. 
<img src="images/mechanism_2d.jpg">

### Results
<img src="images/topology_compare.png">

# How to run top_optim

### Installation and setup of top_optim
To copy the repository and set it up, simply copy these commands into your terminal: 
```
git clone https://github.com/CEADpx/top_optim
cd top_optim
```
Create and activate the environment:

```
conda env create -f environment.yml
conda activate top_optim_env
```

### Running an example
To run an example, simply execute using: `python3 Linear_input.py`

Note: **fenitop** was created to run the code in parallel. Such features do not work in **top_optim** for simplicity.  

## Authors and Refrence

### Authors 
- Ian Galloway (ian.galloway@mines.sdsmt.edu)
- Prashant Jha (prashant.jha@sdsmt.edu)

### Citation for fenitop
This project builds directly on the foundation provided by [FEniTop](https://github.com/missionlab/fenitop), which made it possible to develop and experiment with advanced topology optimization methods in FEniCSx. Huge thanks to the original authors for making their code public — top_optim would not exist without it.
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation for 2D and 3D topology optimization supporting parallel computing. Struct Multidisc Optim 67, 140 (2024). https://doi.org/10.1007/s00158-024-03818-7


