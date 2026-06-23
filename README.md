# top_optim

**top_optim** is a research fork of [FEniTop](https://github.com/missionlab/fenitop), an open-source topology optimization framework built on [FEniCSx](https://fenicsproject.org).

This version extends FEniTop for topology optimization of **hard-magnetic soft materials (hMSMs)**. The framework supports hyperelastic finite-element models, magneto-mechanical coupling, magnetic material distribution optimization, remanent magnetization-direction optimization, and model-comparison studies for different constitutive material models.

## Project Overview

The main goal of this code is to optimize magnetic soft material designs by controlling:

- **ρ (rho)**: mechanical material density
- **ϕ (phi)**: magnetic material fraction / magnetic density distribution
- **θ (theta)**: remanent magnetization direction

The current formulation supports different active design-variable choices. For example, a run may optimize only `phi` while keeping `rho = 1`, or may optimize combinations of `rho`, `phi`, and `theta`.

The code supports several objective types and is modular, so new objectives may easily be added. The code currently supports compliance minimization, maximized displacement, and displacement tracking objectives.   

## Motivation

The behavior of hard-magnetic soft materials is highly dependent on the constitutive model used to describe the elastomer matrix. Different model assumptions can produce substantially different deformation predictions under identical magnetic loading conditions.

This repository provides tools to:

- Evaluate constitutive-model sensitivity in hMSMs
- Compare magnetic actuator performance across constitutive models
- Investigate how deformation magnitude and magnetic-material placement influence model sensitivity
- Perform simultaneous optimization of material density (ρ), magnetic material distribution (ϕ), and magnetization direction (θ)

## Repository Structure

The repository is organized into three main folders:

### fenitop/

Core topology optimization framework and finite-element implementation.

Contains the modified FEniTop source code organized into seven primary modules:

- **topopt.py**  
  Runs the topology optimization loop, manages active design variables, applies filters, handles multi-load-case aggregation, calls sensitivities, updates the design with MMA, and writes output files.

- **fem.py**  
  Builds the finite-element problem in FEniCSx, including hyperelastic material models, magneto-mechanical energy terms, boundary conditions, load cases, objective forms, constraint forms, and derivative forms.

- **sensitivity.py**  
  Computes objective and constraint sensitivities using adjoint solves. Supports gradients with respect to `rho`, `phi`, and `theta`.

- **parameterize.py**  
  Provides the original FEniTop density filter and Heaviside projection tools.

- **optimize.py**  
  Provides the original FEniTop OC and MMA optimizers.

- **utility.py**  
  Provides plotting, communication, linear-problem utilities, and the added `WrapNonlinearProblem` class for nonlinear finite-element solves.

- **evaluate.py**  
  Evaluates a fixed design across different material models and exports displacement comparison data.

### eval/

Post-processing and model-comparison studies, calls upon **evaluate.py**.

Contains scripts used to evaluate designs under different constitutive models, magnetic-field conditions, or loading scenarios. This can be used to quantify differences in response under varying constitutive models, or to just perform FEA on a custom design. Current examples compare responses across multiple strain-energy and shear-modulus constitutive models. These include: 

- **beam_loadStep**
  Cantilever beam with constant remanent magnetization evaluated across increasing applied magnetic loads.
- **beam_tip**
  Cantilever beam with only 20% of the tip containing remanent magnetization.
- **gripper**
  Custom-designed gripper with two arms that close when a magnetic load is applied.
- **wheel** 
  A spoked wheel that rotates counter-clockwise when a magnetic load is applied.

Running a script in this folder does not perform optimization; it evaluates an existing design and saves performance metrics and displacement data.

### opt/

Optimization studies and example problems.

Contains input files used to define optimization problems, including:

- **input_scissor.py** 
  Optimize ϕ and θ to maximize outward displacement to produce a push-type actuator
- **input_wheel.py** 
  Using the same geometry for the wheel experiment in eval/, optimize ϕ and θ to maximize counter-clockwise rotation. 
- **input_combo.py**
  Optimize ρ, ϕ, and θ on a cantilever beam with a downward traction applied and an upward applied magnetic field, with the objective of compliance minimization. 

Running a script in this folder performs a topology optimization simulation and writes the optimized design to disk.


