"""
Tests for layer rotation functionality
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from rcwa.model.layer import Layer
from rcwa.model.material import TensorMaterial
from rcwa.model.transforms import rotate_layer, euler_to_rotation_matrix, rotation_matrix_z
from rcwa.solve.source import Source


@pytest.fixture
def source():
    """Basic source for testing"""
    return Source(wavelength=1.0)


class TestLayerRotation:
    """Test suite for layer rotation functionality"""
    
    def test_euler_to_rotation_matrix_identity(self):
        """Test that zero angles give identity matrix"""
        R = euler_to_rotation_matrix(0, 0, 0)
        assert_allclose(R, np.eye(3), atol=1e-15)
    
    def test_euler_to_rotation_matrix_z_rotation(self):
        """Test pure z-rotation"""
        angle = np.pi / 4  # 45 degrees
        R = euler_to_rotation_matrix(angle, 0, 0)
        
        expected = rotation_matrix_z(angle)
        assert_allclose(R, expected, atol=1e-15)
    
    def test_rotation_matrix_z_90deg(self):
        """Test 90-degree rotation about z-axis"""
        R = rotation_matrix_z(np.pi/2)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        assert_allclose(R, expected, atol=1e-15)
    
    def test_rotate_isotropic_layer_identity(self, source):
        """Test that identity rotation doesn't change isotropic layer"""
        original_layer = Layer(er=2.0, ur=1.5, thickness=1.0)
        original_layer.source = source
        
        rotated_layer = rotate_layer(original_layer, (0, 0, 0))
        
        # After identity rotation, properties should be equivalent
        assert_allclose(rotated_layer.er, np.eye(3) * 2.0)
        assert_allclose(rotated_layer.ur, np.eye(3) * 1.5)
        assert rotated_layer.thickness == original_layer.thickness
    
    def test_rotate_isotropic_layer_90deg_z(self, source):
        """Test 90-degree z-rotation of isotropic layer"""
        original_layer = Layer(er=2.0, ur=1.5, thickness=1.0)
        original_layer.source = source
        
        rotated_layer = rotate_layer(original_layer, (np.pi/2, 0, 0))
        
        # For isotropic material, rotation shouldn't change the tensor values
        # (since it's diagonal with equal elements)
        expected_eps = np.eye(3) * 2.0
        expected_mu = np.eye(3) * 1.5
        assert_allclose(rotated_layer.er, expected_eps, atol=1e-15)
        assert_allclose(rotated_layer.ur, expected_mu, atol=1e-15)
    
    def test_rotate_anisotropic_layer_90deg_z(self, source):
        """Test 90-degree z-rotation of anisotropic layer"""
        # Create anisotropic material with different xx and yy components
        eps_tensor = np.diag([2.0, 3.0, 4.0])  # Different values for each axis
        tensor_material = TensorMaterial(epsilon_tensor=eps_tensor, source=source)
        
        original_layer = Layer(tensor_material=tensor_material, thickness=1.0)
        
        # Rotate 90 degrees about z-axis
        rotated_layer = rotate_layer(original_layer, (np.pi/2, 0, 0))
        
        # After 90-degree z-rotation: (2,3,4) -> (3,2,4)
        expected_eps = np.diag([3.0, 2.0, 4.0])
        assert_allclose(rotated_layer.er, expected_eps, atol=1e-14)
    
    def test_layer_rotated_method(self, source):
        """Test the Layer.rotated() convenience method"""
        eps_tensor = np.diag([2.0, 3.0, 4.0])
        tensor_material = TensorMaterial(epsilon_tensor=eps_tensor, source=source)
        original_layer = Layer(tensor_material=tensor_material, thickness=1.0)
        
        # Use the convenience method
        rotated_layer = original_layer.rotated((np.pi/2, 0, 0))
        
        # Should give same result as rotate_layer function
        expected_layer = rotate_layer(original_layer, (np.pi/2, 0, 0))
        assert_allclose(rotated_layer.er, expected_layer.er)
    
    def test_rotate_dispersive_tensor_layer(self, source):
        """Test rotation of layer with dispersive tensor material"""
        def eps_func(wl):
            return np.diag([1.0 + wl, 2.0 + wl, 3.0 + wl])
        
        tensor_material = TensorMaterial(epsilon_tensor=eps_func, source=source)
        original_layer = Layer(tensor_material=tensor_material, thickness=1.0)
        
        # Rotate 90 degrees about z-axis
        rotated_layer = rotate_layer(original_layer, (np.pi/2, 0, 0))
        
        # Test at specific wavelength
        source.wavelength = 2.0
        rotated_eps = rotated_layer.er
        
        # After rotation: (3, 4, 5) -> (4, 3, 5)
        expected_eps = np.diag([4.0, 3.0, 5.0])
        assert_allclose(rotated_eps, expected_eps, atol=1e-14)
    
    def test_patterned_layer_rotation_restriction(self, source):
        """Test that patterned layers only allow in-plane rotation"""
        from rcwa import Crystal
        
        # Create a simple 1D crystal for testing
        crystal = Crystal(np.array([1, 1]), np.array([1, 0]))
        patterned_layer = Layer(crystal=crystal, thickness=1.0)
        patterned_layer.source = source
        
        # In-plane rotation (about z-axis) should be allowed (but not implemented yet)
        with pytest.raises(NotImplementedError, match="Crystal layer rotation not yet implemented"):
            rotate_layer(patterned_layer, (np.pi/4, 0, 0))
        
        # Out-of-plane rotation should be explicitly forbidden
        with pytest.raises(NotImplementedError, match="Only in-plane rotation"):
            rotate_layer(patterned_layer, (np.pi/4, np.pi/6, 0))
    
    def test_rotation_preserves_thickness(self, source):
        """Test that rotation preserves layer thickness"""
        original_layer = Layer(er=2.0, thickness=1.5)
        original_layer.source = source
        
        rotated_layer = rotate_layer(original_layer, (np.pi/3, 0, 0))
        
        assert rotated_layer.thickness == original_layer.thickness
    
    def test_double_rotation(self, source):
        """Test that double rotation works as expected"""
        eps_tensor = np.diag([2.0, 3.0, 4.0])
        tensor_material = TensorMaterial(epsilon_tensor=eps_tensor, source=source)
        original_layer = Layer(tensor_material=tensor_material, thickness=1.0)
        
        # Rotate twice by 45 degrees (should equal 90 degrees)
        layer1 = rotate_layer(original_layer, (np.pi/4, 0, 0))
        layer2 = rotate_layer(layer1, (np.pi/4, 0, 0))
        
        # Compare with single 90-degree rotation
        layer_90 = rotate_layer(original_layer, (np.pi/2, 0, 0))
        
        assert_allclose(layer2.er, layer_90.er, atol=1e-14)
    
    def test_rotation_tensor_properties(self, source):
        """Test that rotation matrix has proper orthogonal properties"""
        alpha, beta, gamma = 0.3, 0.4, 0.5
        R = euler_to_rotation_matrix(alpha, beta, gamma)
        
        # Test orthogonality: R * R^T = I
        assert_allclose(R @ R.T, np.eye(3), atol=1e-15)
        
        # Test determinant = 1 (proper rotation)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-15)
    
    def test_invalid_rotation_convention(self, source):
        """Test that invalid rotation conventions raise errors"""
        with pytest.raises(NotImplementedError, match="Only ZYX convention"):
            euler_to_rotation_matrix(0, 0, 0, convention="XYZ")
    
    def test_invalid_rotation_center(self, source):
        """Test that invalid rotation centers raise errors"""
        layer = Layer(er=2.0)
        layer.source = source
        
        with pytest.raises(NotImplementedError, match="Only rotation about center"):
            rotate_layer(layer, (0, 0, 0), about="origin")
