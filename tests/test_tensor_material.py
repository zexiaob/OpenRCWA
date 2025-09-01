"""
Tests for TensorMaterial class - anisotropic materials support
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rcwa.model.material import TensorMaterial
from rcwa.model.layer import Layer
from rcwa.solve.source import Source
from rcwa.utils import tabulated_dispersion


@pytest.fixture
def source():
    """Basic source for testing"""
    return Source(wavelength=1.0)


class TestTensorMaterial:
    """Test suite for TensorMaterial class"""
    
    def test_isotropic_tensor(self, source):
        """Test that isotropic case gives identity tensor"""
        mat = TensorMaterial(source=source)
        eps = mat.epsilon_tensor
        expected = np.eye(3, dtype=complex)
        assert_array_equal(eps, expected)
    
    def test_constant_diagonal_tensor(self, source):
        """Test diagonal tensor with constant values"""
        eps_diag = np.array([2.0, 3.0, 4.0])
        tensor = np.diag(eps_diag)
        mat = TensorMaterial(epsilon_tensor=tensor, source=source)
        
        eps = mat.epsilon_tensor
        assert_array_equal(eps, tensor)
        assert eps.shape == (3, 3)
    
    def test_from_diagonal_constructor(self, source):
        """Test the from_diagonal class method"""
        eps_xx, eps_yy, eps_zz = 2.0+1j, 3.0+0.5j, 4.0-0.2j
        mat = TensorMaterial.from_diagonal(eps_xx, eps_yy, eps_zz, source=source)
        
        expected = np.diag([eps_xx, eps_yy, eps_zz])
        assert_array_equal(mat.epsilon_tensor, expected)
    
    def test_uniaxial_material(self, source):
        """Test uniaxial material (eps_yy = eps_xx)"""
        eps_o, eps_e = 2.0, 3.0  # ordinary and extraordinary
        mat = TensorMaterial.from_diagonal(eps_o, eps_o, eps_e, source=source)
        
        expected = np.diag([eps_o, eps_o, eps_e])
        assert_array_equal(mat.epsilon_tensor, expected)
    
    def test_dispersive_diagonal_tensor(self, source):
        """Test dispersive diagonal tensor using functions"""
        def eps_xx_func(wl):
            return 2.0 + 0.1 * wl
        
        def eps_yy_func(wl):
            return 3.0 + 0.2 * wl
        
        def eps_zz_func(wl):
            return 4.0 + 0.3 * wl
        
        mat = TensorMaterial.from_diagonal(eps_xx_func, eps_yy_func, eps_zz_func, source=source)
        assert mat.dispersive
        
        # Test at specific wavelength
        source.wavelength = 2.0
        eps = mat.epsilon_tensor
        expected = np.diag([2.2, 3.4, 4.6])
        assert_allclose(eps, expected)
    
    def test_full_tensor_constant(self, source):
        """Test full 3x3 tensor with off-diagonal elements"""
        tensor = np.array([
            [2.0+1j, 0.1, 0.0],
            [0.1, 3.0, 0.2j],
            [0.0, 0.2j, 4.0-0.5j]
        ])
        
        mat = TensorMaterial(epsilon_tensor=tensor, source=source)
        assert_array_equal(mat.epsilon_tensor, tensor)
    
    def test_dispersive_full_tensor(self, source):
        """Test dispersive full tensor"""
        def tensor_func(wl):
            return np.array([
                [1.0 + wl, 0.1*wl, 0.0],
                [0.1*wl, 2.0 + wl, 0.0],
                [0.0, 0.0, 3.0 + wl]
            ])
        
        mat = TensorMaterial(epsilon_tensor=tensor_func, source=source)
        assert mat.dispersive
        
        # Test at wavelength = 1.5
        source.wavelength = 1.5
        eps = mat.epsilon_tensor
        expected = np.array([
            [2.5, 0.15, 0.0],
            [0.15, 3.5, 0.0],
            [0.0, 0.0, 4.5]
        ])
        assert_allclose(eps, expected)
    
    def test_rotation_constant_tensor(self, source):
        """Test rotation of constant tensor"""
        # Start with diagonal tensor
        original = np.diag([2.0, 3.0, 4.0])
        mat = TensorMaterial(epsilon_tensor=original, source=source)
        
        # Rotate 90 degrees around z-axis
        rotation = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=float)
        
        rotated_mat = mat.rotated(rotation)
        rotated_eps = rotated_mat.epsilon_tensor
        
        # After rotation: (2,3,4) -> (3,2,4)
        expected = np.diag([3.0, 2.0, 4.0])
        assert_allclose(rotated_eps, expected, atol=1e-14)
    
    def test_rotation_dispersive_tensor(self, source):
        """Test rotation of dispersive tensor"""
        def original_func(wl):
            return np.diag([1.0 + wl, 2.0 + wl, 3.0 + wl])
        
        mat = TensorMaterial(epsilon_tensor=original_func, source=source)
        
        # Identity rotation (should not change anything)
        rotation = np.eye(3)
        rotated_mat = mat.rotated(rotation)
        
        source.wavelength = 2.0
        original_eps = mat.epsilon_tensor
        rotated_eps = rotated_mat.epsilon_tensor
        
        assert_allclose(original_eps, rotated_eps)
    
    def test_mu_tensor(self, source):
        """Test permeability tensor functionality"""
        eps_tensor = np.diag([2.0, 3.0, 4.0])
        mu_tensor = np.diag([1.1, 1.2, 1.3])
        
        mat = TensorMaterial(epsilon_tensor=eps_tensor, mu_tensor=mu_tensor, source=source)
        
        assert_array_equal(mat.epsilon_tensor, eps_tensor)
        assert_array_equal(mat.mu_tensor, mu_tensor)
    
    def test_invalid_tensor_shape(self, source):
        """Test that invalid tensor shapes raise errors"""
        with pytest.raises(ValueError, match="Epsilon tensor must be 3x3"):
            TensorMaterial(epsilon_tensor=np.ones((2, 2)), source=source)
        
        with pytest.raises(ValueError, match="Mu tensor must be 3x3"):
            TensorMaterial(mu_tensor=np.ones((2, 3)), source=source)

    def test_n_tensor_constant(self, source):
        """Create material from refractive index tensor"""
        n_tensor = np.diag([1.5, 1.6, 1.7])
        mat = TensorMaterial(n_tensor=n_tensor, source=source)
        expected_eps = np.diag(np.square([1.5, 1.6, 1.7]))
        assert_array_equal(mat.epsilon_tensor, expected_eps)
        assert_array_equal(mat.n_tensor, n_tensor)

    def test_n_tensor_dispersive(self, source):
        """Dispersive refractive index tensor"""
        def n_func(wl):
            return np.diag([1.0 + wl, 1.5 + 0.5*wl, 2.0 + 0.2*wl])

        mat = TensorMaterial(n_tensor=n_func, source=source)
        source.wavelength = 1.0
        eps = mat.epsilon_tensor
        expected = np.diag([(2.0)**2, (2.0)**2, (2.2)**2])
        assert_allclose(eps, expected)

    def test_tabulated_dispersion_tensor(self, source):
        """Tabulated refractive index tensor with interpolation"""
        wavelengths = [1.0, 2.0]
        tensors = [
            np.diag([1.0, 1.5, 2.0]),
            np.diag([1.1, 1.6, 2.1]),
        ]
        n_func = tabulated_dispersion(wavelengths, tensors)
        mat = TensorMaterial(n_tensor=n_func, source=source)
        source.wavelength = 1.5
        expected = np.diag([1.05, 1.55, 2.05])
        assert_allclose(mat.n_tensor, expected)

        source.wavelength = 0.5
        with pytest.raises(ValueError):
            _ = mat.n_tensor

    def test_no_simplification_in_layer(self, source):
        """Ensure layer retains full tensor without simplification"""
        tensor = np.array([[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 4.0]], dtype=complex)
        mat = TensorMaterial(epsilon_tensor=tensor, source=source)
        layer = Layer(tensor_material=mat, thickness=1.0)
        layer.set_convolution_matrices(1)
        conv = layer._tensor_conv_matrices
        assert 'er_xy' in conv
        assert_allclose(conv['er_xy'], 0.5 * np.eye(1))
    
    def test_energy_conservation_property(self, source):
        """Test that tensor materials can maintain energy conservation"""
        # This is a basic property test - for lossless materials,
        # the imaginary part should be zero
        eps_real = np.diag([2.0, 3.0, 4.0])
        mat = TensorMaterial(epsilon_tensor=eps_real, source=source)
        
        eps = mat.epsilon_tensor
        assert np.allclose(np.imag(eps), 0), "Lossless material should have zero imaginary part"
