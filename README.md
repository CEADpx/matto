# top_optim

**top_optim** is a research fork of [FEniTop](https://github.com/missionlab/fenitop), an open-source topology optimization framework built on [FEniCSx](https://fenicsproject.org).

This version extends FEniTop for the simulation and topology optimization of **hard-magnetic soft materials (hMSMs)**. The framework supports hyperelastic finite-element models, magneto-mechanical coupling, magnetic material distribution optimization, remanent magnetization-direction optimization, and comparisons among constitutive material models.

## Project overview

The design of an hMSM structure can be controlled through:

- **ρ (rho):** mechanical material density
- **ϕ (phi):** magnetic particle volume fraction
- **θ (theta):** remanent magnetization direction

Different combinations of these fields may be selected as active design variables. For example, a problem may optimize only `phi` while keeping `rho = 1`, or may jointly optimize `rho`, `phi`, and `theta`.

The code currently supports compliance minimization, displacement maximization, and displacement-tracking objectives. Its modular structure also allows additional objectives, constraints, and constitutive models to be implemented.

## Motivation

The predicted behavior of hMSMs depends strongly on the constitutive model used to describe the particle-filled elastomer. Different model assumptions can produce substantially different deformation predictions under otherwise identical magnetic loading conditions.

This repository provides tools to:

- Evaluate constitutive-model sensitivity in hMSMs
- Compare magnetic actuator performance across constitutive models
- Investigate how deformation magnitude and magnetic-material placement influence model sensitivity
- Optimize mechanical material density, magnetic particle distribution, and remanent magnetization direction

## Installation

The provided Conda environment is intended for Linux or WSL. From the repository root, run:

```bash
conda env create -f environment.yml
conda activate confenx
```

## Repository structure

### `fenitop/`

Core topology optimization and finite-element implementation:

- **`topopt.py`:** Runs the topology optimization loop, manages active design variables, applies filters, handles multiple load cases, computes sensitivities, updates the design with MMA, and writes output files.
- **`fem.py`:** Builds the nonlinear finite-element problem in FEniCSx, including the constitutive models, magneto-mechanical energy, boundary conditions, load cases, objectives, constraints, and derivative forms.
- **`sensitivity.py`:** Computes objective and constraint sensitivities using adjoint solves for `rho`, `phi`, and `theta`.
- **`parameterize.py`:** Provides density filtering and Heaviside projection tools derived from FEniTop.
- **`optimize.py`:** Provides the OC and MMA optimization algorithms derived from FEniTop.
- **`utility.py`:** Provides plotting, communication, linear-problem utilities, and the `WrapNonlinearProblem` class used for nonlinear finite-element solves.
- **`evaluate.py`:** Evaluates fixed designs across different material models and exports displacement-comparison data.

### `eval/`

Model-comparison and finite-element evaluation studies. These scripts evaluate existing designs without performing topology optimization.

- **`beam_loadStep/`:** Cantilever beam with uniform remanent magnetization, evaluated across increasing applied magnetic fields.
- **`beam_tip/`:** Cantilever beam with magnetic material restricted to 20% of the beam length near the free end.
- **`gripper/`:** Gripper with two arms that close under an applied magnetic field.
- **`wheel/`:** Spoked wheel that rotates counterclockwise under an applied magnetic field.
- **`Final_Results.xlsx`:** Summary of the model-comparison results.

### `opt/`

Topology optimization studies and example problems:

- **`restorative_beam/`:** Joint optimization of `rho`, `phi`, and `theta` for a magnetically restorative cantilever beam under mechanical loading.
- **`scissor/`:** Optimization of `phi` and `theta` for translational actuation.
- **`wheel/`:** Optimization of `phi` and `theta` to maximize counterclockwise wheel rotation.

## Running an example

Activate the environment, enter an example directory, and run its input script. For example:

```bash
conda activate confenx
cd eval/beam_loadStep
python input_beam_loadStep.py
```

Optimization studies are run in the same way from their corresponding directories. Simulations write their result files to the output locations defined in each input script.
