"""Unit tests for the tensor RCWA block construction and eigen solver."""

import numpy as np
from numpy.testing import assert_allclose

from rcwa.core.adapters import EigensolverTensorAdapter, LayerTensorAdapter
from rcwa.model.layer import Layer
from rcwa.model.material import TensorMaterial
from rcwa.solve.source import Source


def _setup_tensor_layer(epsilon_tensor, mu_tensor, kx=0.0, ky=0.0):
    """Create a tensor-enabled layer configured for the given tensors."""

    source = Source(wavelength=1.0)
    tensor_mat = TensorMaterial(
        epsilon_tensor=np.array(epsilon_tensor, dtype=complex),
        mu_tensor=np.array(mu_tensor, dtype=complex),
        source=source,
    )

    layer = Layer(tensor_material=tensor_mat, thickness=0.5)
    layer.source = source
    layer.Kx = np.array([[kx]], dtype=complex)
    layer.Ky = np.array([[ky]], dtype=complex)
    layer.set_convolution_matrices(1)
    return layer


def test_tensor_blocks_match_isotropic_limit():
    """The tensor block formulation must reduce to the isotropic equations."""

    eps_val = 2.7 + 0.1j
    mu_val = 1.3 - 0.05j
    kx = 0.35
    ky = -0.12

    layer = _setup_tensor_layer(
        epsilon_tensor=np.diag([eps_val, eps_val, eps_val]),
        mu_tensor=np.diag([mu_val, mu_val, mu_val]),
        kx=kx,
        ky=ky,
    )

    blocks = LayerTensorAdapter._compute_tensor_blocks(layer, layer.Kx, layer.Ky)

    P_tensor = blocks['P']
    Q_tensor = blocks['Q']

    expected_P = np.array(
        [
            [kx * ky / eps_val, mu_val - (kx * kx) / eps_val],
            [(ky * ky) / eps_val - mu_val, -ky * kx / eps_val],
        ],
        dtype=complex,
    )
    expected_Q = np.array(
        [
            [kx * ky / mu_val, eps_val - (kx * kx) / mu_val],
            [(ky * ky) / mu_val - eps_val, -ky * kx / mu_val],
        ],
        dtype=complex,
    )

    assert_allclose(P_tensor, expected_P, atol=1e-12)
    assert_allclose(Q_tensor, expected_Q, atol=1e-12)
    assert_allclose(blocks['R'], np.zeros_like(blocks['R']), atol=1e-12)
    assert_allclose(blocks['S'], np.zeros_like(blocks['S']), atol=1e-12)


def test_tensor_modes_diagonal_anisotropy_normal_incidence():
    """Propagation constants for diagonal tensors at normal incidence are analytic."""

    eps_tensor = np.diag([2.0, 3.5, 1.8])
    mu_tensor = np.diag([1.1, 0.9, 1.0])

    layer = _setup_tensor_layer(eps_tensor, mu_tensor, kx=0.0, ky=0.0)

    blocks = LayerTensorAdapter._compute_tensor_blocks(layer, layer.Kx, layer.Ky)

    assert_allclose(blocks['R'], np.zeros_like(blocks['R']), atol=1e-12)
    assert_allclose(blocks['S'], np.zeros_like(blocks['S']), atol=1e-12)

    _, _, Lambda, _ = EigensolverTensorAdapter.solve_tensor_eigenproblem(
        layer, blocks['P'], blocks['Q']
    )

    lambda_diag = np.diag(Lambda)
    lambda_sq = np.sort(lambda_diag ** 2)

    expected_sq = np.sort(
        np.array(
            [
                -mu_tensor[0, 0] * eps_tensor[1, 1],
                -mu_tensor[1, 1] * eps_tensor[0, 0],
            ],
            dtype=complex,
        )
    )

    assert_allclose(lambda_sq, expected_sq, atol=1e-12)
