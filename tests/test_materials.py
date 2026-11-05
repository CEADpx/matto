"""Every shipped material passes the consistency checks in check_material."""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from matto.materials import (
    AnisotropicMagnetoActiveElastomer,
    HardMagneticSoftMaterial,
    LinearElastic,
    LiquidCrystalElastomer,
    MagnetoActiveElastomer,
    check_material,
    polar_rotation,
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


def _hmsm(**extra):
    return HardMagneticSoftMaterial(
        G0=100.0, mu0=1.256e3, B_rem_mag=200.0, **extra
    )


def test_magnetization_map_defaults_to_F_and_rejects_other_values():
    assert _hmsm().magnetization_map == "F"
    assert _hmsm(magnetization_map="F").magnetization_map == "F"

    with pytest.raises(ValueError, match="magnetization_map"):
        _hmsm(magnetization_map="U")

    with pytest.raises(ValueError, match="magnetization_map"):
        _hmsm(magnetization_map="r")


@pytest.mark.parametrize("dim", [2, 3])
def test_hmsm_with_rotation_map_is_consistent(dim):
    material = _hmsm(dim=dim, magnetization_map="R")
    stimulus = (5.0, 25.0) if dim == 2 else (5.0, 0.0, 25.0)
    check_material(material, stimulus_values={"B_app": stimulus}, dim=dim)


def _unit_domain(dim):
    if dim == 2:
        return mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    return mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)


def _svd_rotation(F):
    left, _, right = np.linalg.svd(F)
    return left @ right


def _rotation_and_derivative(dim, F):
    """R(F) and d(sum(w * R))/dF at one F, through UFL."""
    domain = _unit_domain(dim)
    F_const = fem.Constant(domain, F)
    F_var = ufl.variable(F_const)
    weights = np.arange(1.0, dim * dim + 1.0).reshape(dim, dim)
    R = polar_rotation(F_var)
    dRdF = ufl.diff(ufl.inner(R, fem.Constant(domain, weights)), F_var)

    dx = ufl.dx(domain)
    rotation = np.array([
        [fem.assemble_scalar(fem.form(R[i, j] * dx)) for j in range(dim)]
        for i in range(dim)
    ])
    derivative = np.array([
        [fem.assemble_scalar(fem.form(dRdF[i, j] * dx)) for j in range(dim)]
        for i in range(dim)
    ])
    return rotation, derivative, weights


def _rotation_matrix(angle, dim):
    c, s = np.cos(angle), np.sin(angle)
    Q = np.eye(dim)
    Q[:2, :2] = [[c, -s], [s, c]]
    if dim == 3:
        Q = Q @ np.array([[1, 0, 0], [0, 0.9, -np.sqrt(0.19)],
                          [0, np.sqrt(0.19), 0.9]])
    return Q


@pytest.mark.parametrize("dim", [2, 3])
def test_polar_rotation_recovers_the_rotation_and_its_derivative(dim):
    rng = np.random.default_rng(1)
    sym = rng.standard_normal((dim, dim))
    stretch = np.eye(dim) + 0.2 * (sym + sym.T) / 2
    cases = [
        np.eye(dim),                                         # F = I
        1.2 * np.eye(dim),                                   # isotropic stretch
        np.diag([1.4] + [1.0] * (dim - 1)),                  # repeated stretch
        stretch,                                             # pure stretch
        _rotation_matrix(0.7, dim) @ stretch,                # F = Q U
        _rotation_matrix(2.5, dim) @ stretch,                # large rotation
        np.eye(dim) + 0.4 * rng.standard_normal((dim, dim)),
    ]

    step = 1.0e-6
    for F in cases:
        if np.linalg.det(F) <= 0.0:
            continue

        rotation, derivative, weights = _rotation_and_derivative(dim, F)
        np.testing.assert_allclose(rotation, _svd_rotation(F), atol=1.0e-10)

        expected = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                plus, minus = F.copy(), F.copy()
                plus[i, j] += step
                minus[i, j] -= step
                expected[i, j] = np.sum(
                    weights * (_svd_rotation(plus) - _svd_rotation(minus))
                ) / (2.0 * step)
        assert np.all(np.isfinite(derivative))
        np.testing.assert_allclose(derivative, expected, atol=1.0e-6)

    # Pure stretch gives the identity, F = Q U gives Q.
    np.testing.assert_allclose(
        _rotation_and_derivative(dim, stretch)[0], np.eye(dim), atol=1.0e-10
    )
    Q = _rotation_matrix(0.7, dim)
    np.testing.assert_allclose(
        _rotation_and_derivative(dim, Q @ stretch)[0], Q, atol=1.0e-10
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_omitted_and_explicit_F_map_give_the_same_energy(dim):
    domain = _unit_domain(dim)
    dx = ufl.dx(domain)
    rng = np.random.default_rng(2)
    F = fem.Constant(domain, np.eye(dim) + 0.2 * rng.standard_normal((dim, dim)))
    fields = {
        "rho": fem.Constant(domain, 0.7),
        "phi": fem.Constant(domain, 0.2),
        "theta": fem.Constant(domain, 0.4),
    }
    B_app = fem.Constant(domain, np.array([3.0, 25.0, 7.0][:dim]))

    values = []
    for extra in ({}, {"magnetization_map": "F"}):
        material = _hmsm(dim=dim, **extra)
        energy = material.energy(F, fields, {"B_app": B_app})
        values.append(fem.assemble_scalar(fem.form(energy * dx)))

    assert values[0] == values[1]
