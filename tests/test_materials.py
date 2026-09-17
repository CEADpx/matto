"""Every shipped material passes the consistency checks in check_material."""

import numpy as np
import pytest

from matto.materials import (
    AnisotropicMagnetoActiveElastomer,
    HardMagneticSoftMaterial,
    LinearElastic,
    LiquidCrystalElastomer,
    MagnetoActiveElastomer,
    check_material,
)

MU0 = 4.0 * np.pi * 1.0e-7

CASES = [
    (
        HardMagneticSoftMaterial(G0=100.0, mu0=1.256e3, B_rem_mag=200.0),
        {"B_app": (0.0, 25.0)},
    ),
    (
        LiquidCrystalElastomer(E_LCE=9340.0, nu=0.48, beta=575.0, S0=0.40),
        {"activation": 0.5},
    ),
    (
        MagnetoActiveElastomer(
            C1_Ga=108.0, C2_Ga=137.0, a1=4.97, a2=9.147 * MU0,
            b1=174.685, b2=13.024 * MU0, C1_sil=270.0, C2_sil=10.8,
            K=12000.0, mu0=MU0,
        ),
        {"h": 0.3},
    ),
    (
        AnisotropicMagnetoActiveElastomer(
            A_Ak=326.75, a_Ak=2.785, b_Ak=3.40, r=64.02, s=318.16, hs=0.43,
            K_Ak=16000.0, A_sil=122.0, a_sil=0.28, b_sil=0.33, K_sil=6000.0,
        ),
        {"h": 0.3},
    ),
]


@pytest.mark.parametrize(
    ("material", "stimulus_values"),
    CASES,
    ids=[type(m).__name__ for m, _ in CASES],
)
def test_material_is_consistent(material, stimulus_values):
    check_material(material, stimulus_values=stimulus_values)


def test_linear_elastic_reference_is_stress_free_and_stable():
    # Small-strain model: no frame-indifference check by construction.
    material = LinearElastic(E=100.0, nu=0.25)
    check_material(material, dim=2, frame_indifference=False)
    check_material(material, dim=3, frame_indifference=False)


def test_hmsm_is_consistent_in_3d():
    material = HardMagneticSoftMaterial(G0=100.0, mu0=1.256e3, B_rem_mag=200.0, dim=3)
    check_material(material, stimulus_values={"B_app": (5.0, 0.0, 25.0)}, dim=3)


def test_material_refuses_unknown_and_missing_parameters():
    with pytest.raises(TypeError, match="needs values"):
        HardMagneticSoftMaterial(G0=100.0)

    with pytest.raises(TypeError, match="does not take"):
        HardMagneticSoftMaterial(G0=1.0, mu0=1.0, B_rem_mag=1.0, bogus=2.0)


def test_material_reports_a_missing_design_field():
    from mpi4py import MPI

    from tests.support import build_beam_problem
    from matto.driver import OptimizationDriver

    problem = build_beam_problem(MPI.COMM_WORLD, nx=4, ny=2, load_steps=1)
    problem["design_variables"].pop("theta")

    with pytest.raises(ValueError, match="theta"):
        OptimizationDriver(problem)
