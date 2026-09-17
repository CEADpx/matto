"""
Material models.

A material is a free-energy density W(F; fields, stimuli) plus a
declaration of what it reads. See base.Material for the contract and
testing.check_material for what a model is expected to satisfy.
"""

from .base import Material
from .hmsm import HardMagneticSoftMaterial
from .lce import LiquidCrystalElastomer
from .linear import LinearElastic
from .mae import MagnetoActiveElastomer
from .mae_aniso import AnisotropicMagnetoActiveElastomer
from .interpolation import simp, two_phase
from .kinematics import director, green_lagrange, isochoric_invariants, plane_strain_3d
from .testing import check_material

__all__ = [
    "Material",
    "HardMagneticSoftMaterial",
    "LiquidCrystalElastomer",
    "LinearElastic",
    "MagnetoActiveElastomer",
    "AnisotropicMagnetoActiveElastomer",
    "simp", "two_phase",
    "director", "green_lagrange", "isochoric_invariants", "plane_strain_3d",
    "check_material",
]
