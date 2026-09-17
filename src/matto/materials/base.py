# Copyright (c) 2025-2026 Ian Galloway, Prashant K. Jha
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The material contract.

A material is a free-energy density W(F; fields, stimuli). The package
builds F = variable(I + grad u) once, calls energy(), and takes the
first Piola-Kirchhoff stress as diff(W, F). A material says up front
which design fields it reads, which stimuli it needs and what shape
they have, and which parameters it takes, so a mismatch with the input
file is reported at construction instead of surfacing as a KeyError
inside UFL.

Subclass Material, set the class attributes, write energy(). The class
can live in the package, next to the input scripts, or in the input
script itself.
"""

import numpy as np


class Material:
    #: design fields the energy reads, by name; the input file sets
    #: which of them are optimized and which are prescribed
    fields = ()

    #: stimulus name -> shape, () for a scalar and (dim,) for a vector;
    #: the load cases supply the values
    stimuli = {}

    #: parameter name -> default, or None when the caller must supply it
    parameters = {}

    #: history variables, name -> shape. Reserved for path-dependent
    #: models; check_requirements raises for a material that declares
    #: any, since history-dependent models are not supported.
    internal_variables = {}

    #: where the model comes from
    reference = ""

    def __init__(self, **parameters):
        unknown = sorted(set(parameters) - set(self.parameters))
        if unknown:
            raise TypeError(
                f"{type(self).__name__} does not take parameters {unknown}; "
                f"it takes {sorted(self.parameters)}."
            )

        missing = sorted(
            name for name, default in self.parameters.items()
            if default is None and name not in parameters
        )
        if missing:
            raise TypeError(
                f"{type(self).__name__} needs values for {missing}."
            )

        merged = dict(self.parameters)
        merged.update(parameters)
        self.__dict__["parameters"] = merged

    def __getattr__(self, name):
        # Parameters read as attributes: self.G0, self.p_rho, ...
        parameters = self.__dict__.get("parameters", {})
        if name in parameters:
            return parameters[name]
        raise AttributeError(name)

    def energy(self, F, fields, stimuli):
        """
        Free-energy density as a UFL scalar.

        F is ufl.variable(I + grad u), so the caller can differentiate
        with respect to it. fields maps each name in self.fields to its
        physical Function; stimuli maps each name in self.stimuli to its
        Constant.
        """
        raise NotImplementedError

    def check_requirements(self, design_variables, stimuli):
        """Raise if the problem does not provide what this material reads."""

        name = type(self).__name__

        missing_fields = [f for f in self.fields if f not in design_variables]
        if missing_fields:
            raise ValueError(
                f"{name} reads design fields {missing_fields}, which the "
                f"problem does not define; it defines "
                f"{sorted(design_variables)}."
            )

        missing_stimuli = [s for s in self.stimuli if s not in stimuli]
        if missing_stimuli:
            raise ValueError(
                f"{name} needs stimuli {missing_stimuli}, which no load "
                f"case supplies; the load cases supply {sorted(stimuli)}."
            )

        for stimulus, shape in self.stimuli.items():
            actual = tuple(np.shape(stimuli[stimulus].value))
            if actual != tuple(shape):
                raise ValueError(
                    f"{name} expects stimulus '{stimulus}' with shape "
                    f"{tuple(shape)}; the load cases give it shape {actual}."
                )

        if self.internal_variables:
            raise NotImplementedError(
                f"{name} declares internal variables "
                f"{sorted(self.internal_variables)}; history-dependent "
                "materials are not supported yet."
            )

    def __repr__(self):
        values = ", ".join(f"{k}={v!r}" for k, v in self.parameters.items())
        return f"{type(self).__name__}({values})"
