"""
Unit tests for complex amplitude completeness verification (Task 1.3).

Tests that all simulation results contain complete complex amplitude information,
intensity quantities are derived from complex amplitudes, and Results class
provides unified access interfaces.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rcwa import TensorMaterial, Source, Layer, LayerStack, Solver
from rcwa.solve.results import Results


class TestComplexAmplitudeCompleteness:
    """Test complete complex amplitude information preservation."""
    
    @pytest.fixture
    def simple_simulation_results(self):
        """Create a simple simulation and return results for testing."""
        source = Source(wavelength=1.0, theta=0)
        eps_iso = 2.25  # n = 1.5
        tensor_mat = TensorMaterial.from_diagonal(eps_iso, eps_iso, eps_iso, source=source)
        
        incident_layer = Layer()  # Air
        tensor_layer = Layer(tensor_material=tensor_mat, thickness=0.5)
        transmission_layer = Layer()  # Air
        
        stack = LayerStack(tensor_layer,
                          incident_layer=incident_layer,
                          transmission_layer=transmission_layer)
        
        solver = Solver(stack, source, n_harmonics=1)
        results = solver.solve()
        
        return results
    
    def test_complex_amplitudes_present(self, simple_simulation_results):
        """Test that all 6 complex amplitude components are present."""
        results = simple_simulation_results
        
        required_fields = ['rx', 'ry', 'rz', 'tx', 'ty', 'tz']
        
        for field in required_fields:
            assert hasattr(results, field), f"Missing complex amplitude field: {field}"
            assert field in results.keys(), f"Complex amplitude {field} not in results dictionary"
            
            # Verify the values are complex numbers or arrays
            value = getattr(results, field)
            assert np.iscomplexobj(value), f"Complex amplitude {field} is not complex type"
    
    def test_intensity_quantities_present(self, simple_simulation_results):
        """Test that intensity quantities are present and derived correctly."""
        results = simple_simulation_results
        
        # Check that R and T are present
        assert hasattr(results, 'R'), "Reflectance R not accessible"
        assert hasattr(results, 'T'), "Transmittance T not accessible"
        assert hasattr(results, 'RTot'), "Total reflectance RTot not accessible"
        assert hasattr(results, 'TTot'), "Total transmittance TTot not accessible"
        
        # Check that values are real and non-negative
        assert np.all(np.real(results.R) >= 0), "Reflectance should be non-negative"
        assert np.all(np.real(results.T) >= 0), "Transmittance should be non-negative"
        assert results.RTot >= 0, "Total reflectance should be non-negative"
        assert results.TTot >= 0, "Total transmittance should be non-negative"
    
    def test_energy_conservation(self, simple_simulation_results):
        """Test energy conservation."""
        results = simple_simulation_results
        
        # Test energy conservation property
        conservation = results.conservation
        assert_allclose(conservation, 1.0, atol=1e-6, 
                       err_msg=f"Energy not conserved: R+T = {conservation}")
        
        # Test verification method
        assert results.verify_energy_conservation(), "Energy conservation verification failed"
        assert results.verify_energy_conservation(tolerance=1e-6), "Energy conservation within tolerance failed"
    
    def test_complex_amplitude_access_methods(self, simple_simulation_results):
        """Test methods for accessing complex amplitude information."""
        results = simple_simulation_results
        
        # Test get_complex_amplitudes method
        r_complex, t_complex = results.get_complex_amplitudes()
        
        assert r_complex.shape == (3,), "Reflection complex amplitudes should have shape (3,)"
        assert t_complex.shape == (3,), "Transmission complex amplitudes should have shape (3,)"
        
        # Verify values match individual accessors
        assert_array_equal(r_complex, [results.rx, results.ry, results.rz])
        assert_array_equal(t_complex, [results.tx, results.ty, results.tz])
    
    def test_phase_extraction(self, simple_simulation_results):
        """Test phase information extraction."""
        results = simple_simulation_results
        
        r_phases, t_phases = results.get_phases()
        
        assert r_phases.shape == (3,), "Reflection phases should have shape (3,)"
        assert t_phases.shape == (3,), "Transmission phases should have shape (3,)"
        
        # Verify phases are real
        assert np.all(np.isreal(r_phases)), "Reflection phases should be real"
        assert np.all(np.isreal(t_phases)), "Transmission phases should be real"
        
        # Verify phases match manual calculation
        expected_r_phases = np.angle([results.rx, results.ry, results.rz])
        expected_t_phases = np.angle([results.tx, results.ty, results.tz])
        
        assert_array_equal(r_phases, expected_r_phases)
        assert_array_equal(t_phases, expected_t_phases)
    
    def test_intensity_extraction(self, simple_simulation_results):
        """Test intensity information extraction."""
        results = simple_simulation_results
        
        r_intensities, t_intensities = results.get_intensities()
        
        assert r_intensities.shape == (3,), "Reflection intensities should have shape (3,)"
        assert t_intensities.shape == (3,), "Transmission intensities should have shape (3,)"
        
        # Verify intensities are real and non-negative
        assert np.all(r_intensities >= 0), "Reflection intensities should be non-negative"
        assert np.all(t_intensities >= 0), "Transmission intensities should be non-negative"
        
        # Verify intensities match manual calculation
        expected_r_intensities = np.abs([results.rx, results.ry, results.rz])**2
        expected_t_intensities = np.abs([results.tx, results.ty, results.tz])**2
        
        assert_allclose(r_intensities, expected_r_intensities, rtol=1e-15)
        assert_allclose(t_intensities, expected_t_intensities, rtol=1e-15)
    
    def test_complex_intensity_consistency(self, simple_simulation_results):
        """Test that intensity quantities are consistent with complex amplitudes."""
        results = simple_simulation_results
        
        # Test the verification method
        assert results.verify_complex_consistency(), "Complex amplitude and intensity consistency check failed"
        
        # Manual verification
        r_intensities, t_intensities = results.get_intensities()
        
        # For single-mode case, total intensities should match R and T
        r_total = np.sum(r_intensities)
        t_total = np.sum(t_intensities)
        
        R_total = np.sum(np.atleast_1d(results.R))
        T_total = np.sum(np.atleast_1d(results.T))
        
        assert_allclose(r_total, R_total, rtol=1e-12, 
                       err_msg="Reflection intensity inconsistent with complex amplitudes")
        assert_allclose(t_total, T_total, rtol=1e-12,
                       err_msg="Transmission intensity inconsistent with complex amplitudes")
    
    def test_absorption_calculation(self, simple_simulation_results):
        """Test absorption calculation from R and T."""
        results = simple_simulation_results
        
        assert hasattr(results, 'A'), "Absorption A not accessible"
        
        # For lossless materials, absorption should be minimal
        absorption = results.A
        
        # Check that A = 1 - R - T
        expected_absorption = 1.0 - results.R - results.T
        
        if hasattr(absorption, '__iter__'):
            assert_allclose(absorption, expected_absorption, rtol=1e-15)
        else:
            assert_allclose(absorption, expected_absorption, rtol=1e-15)
    
    def test_backward_compatibility(self, simple_simulation_results):
        """Test that existing code interfaces still work."""
        results = simple_simulation_results
        
        # Test dictionary-style access
        assert results['rx'] is not None
        assert results['R'] is not None
        assert results['RTot'] is not None
        
        # Test that all required keys are accessible both ways
        for key in ['rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'R', 'T']:
            dict_access = results[key]
            if hasattr(results, key):
                prop_access = getattr(results, key)
                # Values should be identical (same object)
                if hasattr(dict_access, '__iter__') and hasattr(prop_access, '__iter__'):
                    assert_array_equal(dict_access, prop_access)
                else:
                    assert dict_access == prop_access


class TestComplexAmplitudeValidation:
    """Test Results class validation and error handling."""
    
    def test_missing_complex_amplitudes_error(self):
        """Test that missing complex amplitude fields issue warnings for backward compatibility."""
        incomplete_results = {
            'rx': 0.1+0.05j,
            'ry': 0.0+0.0j,
            'R': 0.01,
            'T': 0.99
        }
        
        with pytest.warns(UserWarning, match="Results missing complex amplitude fields"):
            Results(incomplete_results)
    
    def test_complete_results_validation_passes(self):
        """Test that complete results pass validation."""
        complete_results = {
            'rx': 0.1+0.05j, 'ry': 0.0+0.0j, 'rz': 0.0+0.0j,
            'tx': 0.9+0.1j, 'ty': 0.0+0.0j, 'tz': 0.0+0.0j,
            'R': np.array([0.01]), 'T': np.array([0.99]),
            'RTot': 0.01, 'TTot': 0.99, 'conservation': 1.0
        }
        
        # Should not raise any exception
        results = Results(complete_results)
        assert results is not None


class TestMultiWavelengthComplexAmplitudes:
    """Test complex amplitude handling for multi-wavelength simulations."""
    
    def test_wavelength_sweep_complex_amplitudes(self):
        """Test that wavelength sweeps preserve complete complex amplitude information."""
        source = Source(wavelength=1.0, theta=0)
        eps_iso = 2.25
        tensor_mat = TensorMaterial.from_diagonal(eps_iso, eps_iso, eps_iso, source=source)
        
        tensor_layer = Layer(tensor_material=tensor_mat, thickness=0.5)
        stack = LayerStack(tensor_layer)
        
        solver = Solver(stack, source, n_harmonics=1)
        
        # Perform wavelength sweep
        results = solver.solve(wavelength=[0.8, 1.0, 1.2])
        
        # Check that results contain lists of complex amplitudes
        assert isinstance(results.rx, list), "Multi-wavelength rx should be a list"
        assert len(results.rx) == 3, "Should have 3 wavelength points"
        
        # Each element should be complex
        for i in range(3):
            assert np.iscomplexobj(results.rx[i]), f"rx[{i}] should be complex"
            assert np.iscomplexobj(results.tx[i]), f"tx[{i}] should be complex"
        
        # Energy should be conserved at each wavelength
        for i in range(3):
            conservation = results['RTot'][i] + results['TTot'][i]
            assert_allclose(conservation, 1.0, atol=1e-6,
                           err_msg=f"Energy not conserved at wavelength {i}: {conservation}")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
