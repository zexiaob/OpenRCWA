"""
Test Pydantic validation enhancements for TensorMaterial.

This test module validates the ROADMAP Task 1.1 requirements:
- Strong Pydantic validation for tensor shapes, frequency ranges, thickness
- SI unit consistency with nm()/deg() helpers
- Clear error messages for invalid inputs
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from rcwa.model.material import TensorMaterial, nm, um, mm, deg


class TestTensorMaterialValidation:
    """Test enhanced validation features of TensorMaterial."""
    
    def test_si_unit_helpers(self):
        """Test SI unit conversion helpers."""
        # Test length conversions
        assert_allclose(nm(500), 500e-9, rtol=1e-15)  # 500nm to meters
        assert_allclose(um(1.5), 1.5e-6, rtol=1e-15)  # 1.5um to meters
        assert_allclose(mm(2.0), 2.0e-3, rtol=1e-15)  # 2mm to meters
        
        # Test angle conversion
        assert_allclose(deg(90), np.pi/2)
        assert_allclose(deg(180), np.pi)
    
    def test_wavelength_range_validation(self):
        """Test wavelength range validation enforces SI units."""
        eps = np.eye(3) * 2.25  # Silicon-like
        
        # Valid wavelength range in SI units (meters)
        mat = TensorMaterial(
            epsilon_tensor=eps,
            wavelength_range=(nm(400), nm(700))  # 400-700nm
        )
        assert_allclose(mat.wavelength_range[0], 400e-9, rtol=1e-15)
        assert_allclose(mat.wavelength_range[1], 700e-9, rtol=1e-15)
        
        # Invalid: wavelength too large (likely wrong units)
        with pytest.raises(ValueError, match="too large.*SI units"):
            TensorMaterial(
                epsilon_tensor=eps,
                wavelength_range=(400, 700)  # Forgot nm() - wrong!
            )
        
        # Invalid: wavelength too small
        with pytest.raises(ValueError, match="too small.*SI units"):
            TensorMaterial(
                epsilon_tensor=eps,
                wavelength_range=(1e-12, 2e-12)
            )
        
        # Invalid: min >= max
        with pytest.raises(ValueError, match="Min wavelength must be"):
            TensorMaterial(
                epsilon_tensor=eps,
                wavelength_range=(nm(700), nm(400))
            )
    
    def test_thickness_range_validation(self):
        """Test thickness range validation enforces SI units."""
        eps = np.eye(3) * 2.25
        
        # Valid thickness range in SI units (meters)
        mat = TensorMaterial(
            epsilon_tensor=eps,
            thickness_range=(nm(10), um(10))  # 10nm to 10um
        )
        assert_allclose(mat.thickness_range[0], 10e-9, rtol=1e-15)
        assert_allclose(mat.thickness_range[1], 10e-6, rtol=1e-15)
        
        # Invalid: thickness too large (likely wrong units)
        with pytest.raises(ValueError, match="too large.*SI units"):
            TensorMaterial(
                epsilon_tensor=eps,
                thickness_range=(10, 100)  # Forgot unit conversion!
            )
        
        # Invalid: thickness too small
        with pytest.raises(ValueError, match="too small.*SI units"):
            TensorMaterial(
                epsilon_tensor=eps,
                thickness_range=(1e-15, 2e-15)
            )
    
    def test_tensor_shape_validation(self):
        """Test that tensor shapes are strictly validated."""
        # Valid 3x3 tensor
        eps_valid = np.eye(3) * 2.25
        mat = TensorMaterial(epsilon_tensor=eps_valid)
        
        # Invalid shapes
        with pytest.raises(ValueError, match="must be 3x3"):
            TensorMaterial(epsilon_tensor=np.eye(2))  # 2x2
        
        with pytest.raises(ValueError, match="must be 3x3"):
            TensorMaterial(epsilon_tensor=np.ones((3, 4)))  # 3x4
        
        with pytest.raises(ValueError, match="must be 3x3"):
            TensorMaterial(mu_tensor=np.eye(4))  # Wrong mu shape
    
    def test_dispersive_source_warning(self):
        """Test that dispersive materials without source issue warning."""
        def eps_func(wl):
            return np.eye(3) * (2.25 + 0.01/wl)
        
        # Should warn about missing source
        with pytest.warns(UserWarning, match="should have an associated source"):
            mat = TensorMaterial(epsilon_tensor=eps_func)
        
        # Create a dummy source-like object for testing
        class DummySource:
            def __init__(self, wavelength):
                self.wavelength = wavelength
        
        # No warning when source is provided
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mat = TensorMaterial(epsilon_tensor=eps_func, source=DummySource(500e-9))
            # Check that no source-related warnings were raised
            source_warnings = [warning for warning in w if "source" in str(warning.message)]
            assert len(source_warnings) == 0
    
    def test_error_message_clarity(self):
        """Test that error messages provide clear guidance."""
        eps = np.eye(3) * 2.25
        
        # Wrong wavelength units should mention helpers
        with pytest.raises(ValueError, match="nm\\(\\)/um\\(\\) helpers"):
            TensorMaterial(
                epsilon_tensor=eps,
                wavelength_range=(500, 600)  # Missing nm()
            )
        
        # Wrong thickness units should mention helpers  
        with pytest.raises(ValueError, match="nm\\(\\)/um\\(\\) helpers"):
            TensorMaterial(
                epsilon_tensor=eps,
                thickness_range=(1, 10)  # Missing unit conversion
            )
    
    def test_backward_compatibility(self):
        """Test that existing TensorMaterial usage still works."""
        # Simple constant tensor (existing usage)
        eps = np.array([
            [2.25, 0, 0],
            [0, 2.25, 0], 
            [0, 0, 2.25]
        ])
        mat = TensorMaterial(epsilon_tensor=eps)
        # For constant tensors, epsilon_tensor is a property, not callable
        assert_allclose(mat.epsilon_tensor, eps)
        
        # Function-based tensor (existing usage)
        def eps_func(wl):
            return np.eye(3) * 2.25
        
        # Create a dummy source for dispersive materials
        class DummySource:
            def __init__(self, wavelength):
                self.wavelength = wavelength
        
        mat = TensorMaterial(epsilon_tensor=eps_func, source=DummySource(500e-9))
        # For dispersive tensors, epsilon_tensor is callable via the property
        assert_allclose(mat.epsilon_tensor, np.eye(3) * 2.25)
    
    def test_from_diagonal_validation(self):
        """Test validation works with from_diagonal constructor."""
        # Valid diagonal values
        mat = TensorMaterial.from_diagonal(2.25, 2.25, 2.25)
        
        # Should still validate additional parameters
        with pytest.raises(ValueError, match="too large.*SI units"):
            TensorMaterial.from_diagonal(
                2.25, 2.25, 2.25,
                wavelength_range=(400, 700)  # Missing nm()
            )
    
    def test_validation_with_complex_tensors(self):
        """Test validation works with complex-valued tensors."""
        # Complex constant tensor
        eps_complex = np.eye(3) * (2.25 + 0.1j)
        mat = TensorMaterial(
            epsilon_tensor=eps_complex,
            wavelength_range=(nm(400), nm(800))
        )
        
        # For constant tensors, epsilon_tensor is a property
        assert_allclose(mat.epsilon_tensor, eps_complex)
        
        # Complex dispersive tensor
        def eps_complex_func(wl):
            n = 1.5 + 0.01j  # Complex refractive index
            return np.eye(3) * n**2
        
        # Create a dummy source for dispersive materials
        class DummySource:
            def __init__(self, wavelength):
                self.wavelength = wavelength
        
        mat = TensorMaterial(
            epsilon_tensor=eps_complex_func,
            thickness_range=(nm(50), um(5)),
            source=DummySource(500e-9)
        )
        
        expected = np.eye(3) * (1.5 + 0.01j)**2
        assert_allclose(mat.epsilon_tensor, expected)
