"""
Test for tensor material integration with core solver
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from rcwa.model.material import TensorMaterial
    from rcwa.model.layer import Layer, LayerStack
    from rcwa.solve.source import Source
    from rcwa.core.solver import Solver
    from rcwa.core.adapters import TensorToConvolutionAdapter, LayerTensorAdapter
except ImportError as e:
    print(f"Import error: {e}")
    print("Testing with fallback imports...")
    from rcwa import TensorMaterial, Layer, LayerStack, Source, Solver


class TestTensorSolverIntegration:
    """Test tensor material integration with the RCWA solver"""
    
    def test_adapter_convolution_matrices(self):
        """Test convolution matrix adapter for tensor materials"""
        # Create a simple diagonal tensor
        eps_tensor = np.diag([2.0, 3.0, 4.0])
        mu_tensor = np.eye(3, dtype=complex)
        
        # Convert to convolution matrices
        conv_matrices = TensorToConvolutionAdapter.tensor_to_convolution_matrices(
            eps_tensor, mu_tensor, n_harmonics=1)
        
        # Check that all required components are present
        expected_eps_components = ['eps_xx', 'eps_xy', 'eps_xz', 'eps_yx', 'eps_yy', 
                                   'eps_yz', 'eps_zx', 'eps_zy', 'eps_zz']
        expected_mu_components = ['mu_xx', 'mu_xy', 'mu_xz', 'mu_yx', 'mu_yy', 
                                  'mu_yz', 'mu_zx', 'mu_zy', 'mu_zz']
        
        for comp in expected_eps_components + expected_mu_components:
            assert comp in conv_matrices
        
        # Check diagonal values
        assert_allclose(conv_matrices['eps_xx'], 2.0)
        assert_allclose(conv_matrices['eps_yy'], 3.0)
        assert_allclose(conv_matrices['eps_zz'], 4.0)
        assert_allclose(conv_matrices['mu_xx'], 1.0)
    
    def test_effective_properties_extraction(self):
        """Test extraction of effective properties from tensor"""
        eps_tensor = np.diag([2.0+0.1j, 3.0+0.2j, 4.0+0.3j])
        mu_tensor = np.diag([1.1, 1.2, 1.3])
        
        # Extract effective properties along z-direction
        eps_eff, mu_eff = TensorToConvolutionAdapter.extract_effective_properties(
            eps_tensor, mu_tensor, 'z')
        
        assert_allclose(eps_eff, 4.0+0.3j)
        assert_allclose(mu_eff, 1.3)
        
        # Test x-direction
        eps_eff_x, mu_eff_x = TensorToConvolutionAdapter.extract_effective_properties(
            eps_tensor, mu_tensor, 'x')
        
        assert_allclose(eps_eff_x, 2.0+0.1j)
        assert_allclose(mu_eff_x, 1.1)
    
    def test_tensor_layer_creation(self):
        """Test creating a layer with tensor material"""
        source = Source(wavelength=1.0)
        
        # Create tensor material
        eps_tensor = np.diag([2.0, 3.0, 4.0])
        tensor_mat = TensorMaterial(epsilon_tensor=eps_tensor, source=source)
        
        # Create layer with tensor material
        layer = Layer(tensor_material=tensor_mat, thickness=1.0)
        
        assert layer.is_anisotropic
        assert_allclose(layer.er, eps_tensor)
        assert layer.thickness == 1.0
    
    def test_tensor_layer_convolution_setup(self):
        """Test convolution matrix setup for tensor layers"""
        source = Source(wavelength=1.0)
        
        # Create uniaxial material
        eps_o, eps_e = 2.25, 4.0  # ordinary and extraordinary permittivities
        eps_tensor = np.diag([eps_o, eps_o, eps_e])
        tensor_mat = TensorMaterial(epsilon_tensor=eps_tensor, source=source)
        
        # Create layer
        layer = Layer(tensor_material=tensor_mat, thickness=0.5)
        layer.source = source
        
        # Set convolution matrices
        layer.set_convolution_matrices(1)  # Single harmonic
        
        # Check that effective properties were extracted properly
        assert hasattr(layer, 'er')
        assert hasattr(layer, 'ur')
        
        # For z-direction propagation, should use eps_zz = eps_e
        expected_eps_eff = eps_e
        assert_allclose(complex(layer.er), expected_eps_eff, rtol=1e-10)
    
    def test_tensor_matrix_calculations(self):
        """Test that tensor materials produce valid P and Q matrices"""
        source = Source(wavelength=1.0)
        
        # Create biaxial tensor material
        eps_tensor = np.diag([2.0, 3.0, 4.0])
        mu_tensor = np.diag([1.1, 1.2, 1.3])
        tensor_mat = TensorMaterial(epsilon_tensor=eps_tensor, mu_tensor=mu_tensor, source=source)
        
        # Create layer
        layer = Layer(tensor_material=tensor_mat, thickness=1.0)
        layer.source = source
        
        # Set up k-vectors for homogeneous case
        layer.Kx = 0.0
        layer.Ky = 0.0
        
        # Calculate P and Q matrices
        P = layer.P_matrix()
        Q = layer.Q_matrix()
        
        # Check that matrices have correct dimensions
        assert P.shape == (2, 2)
        assert Q.shape == (2, 2)
        
        # Check that matrices are not all zeros
        assert not np.allclose(P, np.zeros((2, 2)))
        assert not np.allclose(Q, np.zeros((2, 2)))
    
    def test_simple_tensor_simulation_setup(self):
        """Test setting up a simple simulation with tensor materials"""
        source = Source(wavelength=1.0, theta=0.0)
        
        # Create a simple uniaxial material
        eps_o, eps_e = 2.25, 4.0
        eps_tensor = np.diag([eps_o, eps_o, eps_e])
        tensor_mat = TensorMaterial(epsilon_tensor=eps_tensor, source=source)
        
        # Create layer stack
        air = Layer()  # Air (isotropic)
        tensor_layer = Layer(tensor_material=tensor_mat, thickness=0.5)
        
        stack = LayerStack(tensor_layer, 
                          incident_layer=air,
                          transmission_layer=air)
        
        # Create solver
        solver = Solver(stack, source, n_harmonics=1)
        
        # Check that the solver was created successfully
        assert solver.layer_stack == stack
        assert solver.source == source
        assert solver.n_harmonics == 1
        
        # Check that tensor layer maintains its properties
        assert tensor_layer.is_anisotropic
        assert_allclose(tensor_layer.er, eps_tensor)
    
    @pytest.mark.skip(reason="Full tensor solver integration still being implemented")
    def test_tensor_energy_conservation(self):
        """Test energy conservation for tensor materials (currently skipped)"""
        source = Source(wavelength=1.0, theta=0.0)
        
        # Lossless uniaxial material
        eps_o, eps_e = 2.25, 4.0
        eps_tensor = np.diag([eps_o, eps_o, eps_e])
        tensor_mat = TensorMaterial(epsilon_tensor=eps_tensor, source=source)
        
        # Create stack
        air = Layer()
        tensor_layer = Layer(tensor_material=tensor_mat, thickness=0.5)
        
        stack = LayerStack(tensor_layer,
                          incident_layer=air,
                          transmission_layer=air)
        
        solver = Solver(stack, source, n_harmonics=1)
        
        # This would test the full solver integration
        result = solver.solve()
        R_total = result.RTot if hasattr(result, 'RTot') else result['RTot']
        T_total = result.TTot if hasattr(result, 'TTot') else result['TTot']
        conservation = R_total + T_total
        
        assert_allclose(conservation, 1.0, atol=1e-6, 
                       err_msg=f"Energy not conserved: R+T = {conservation}")


if __name__ == "__main__":
    # Run the tests
    test_suite = TestTensorSolverIntegration()
    
    try:
        test_suite.test_adapter_convolution_matrices()
        print("✓ Convolution matrix adapter test passed")
    except Exception as e:
        print(f"✗ Convolution matrix adapter test failed: {e}")
    
    try:
        test_suite.test_effective_properties_extraction()
        print("✓ Effective properties extraction test passed")
    except Exception as e:
        print(f"✗ Effective properties extraction test failed: {e}")
    
    try:
        test_suite.test_tensor_layer_creation()
        print("✓ Tensor layer creation test passed")
    except Exception as e:
        print(f"✗ Tensor layer creation test failed: {e}")
    
    try:
        test_suite.test_tensor_layer_convolution_setup()
        print("✓ Tensor layer convolution setup test passed")
    except Exception as e:
        print(f"✗ Tensor layer convolution setup test failed: {e}")
    
    try:
        test_suite.test_tensor_matrix_calculations()
        print("✓ Tensor matrix calculations test passed")
    except Exception as e:
        print(f"✗ Tensor matrix calculations test failed: {e}")
    
    try:
        test_suite.test_simple_tensor_simulation_setup()
        print("✓ Simple tensor simulation setup test passed")
    except Exception as e:
        print(f"✗ Simple tensor simulation setup test failed: {e}")
    
    print("\nTensor solver integration tests completed!")
    print("Note: Full energy conservation test is currently skipped as per ROADMAP status.")
