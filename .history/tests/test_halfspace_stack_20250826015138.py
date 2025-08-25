"""
Tests for HalfSpace and simplified Stack construction.
"""

import pytest
import numpy as np
from rcwa import Material, TensorMaterial, Layer, HalfSpace, Air, Substrate, Stack, LayerStack


def test_halfspace_basic_creation():
    """Test basic HalfSpace creation."""
    # Default air halfspace
    air_halfspace = HalfSpace()
    assert air_halfspace.er == 1.0
    assert air_halfspace.ur == 1.0
    assert air_halfspace.n == 1.0
    assert air_halfspace.thickness == 0.0
    assert air_halfspace.homogenous == True
    assert air_halfspace.is_anisotropic == False
    
    # Halfspace with custom material
    si_material = Material(er=12.0, ur=1.0)
    si_halfspace = HalfSpace(material=si_material)
    assert si_halfspace.er == 12.0
    assert si_halfspace.ur == 1.0
    assert si_halfspace.thickness == 0.0


def test_halfspace_tensor_material():
    """Test HalfSpace with tensor materials."""
    # Create uniaxial tensor material
    epsilon_tensor = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], 
        [0.0, 0.0, 4.0]
    ], dtype=complex)
    
    tensor_material = TensorMaterial(epsilon_tensor=epsilon_tensor)
    tensor_halfspace = HalfSpace(tensor_material=tensor_material)
    
    assert tensor_halfspace.is_anisotropic == True
    assert np.allclose(tensor_halfspace.er, epsilon_tensor)
    assert tensor_halfspace.thickness == 0.0
    
    # Test that n property raises error for anisotropic
    with pytest.raises(ValueError, match="not well-defined for anisotropic"):
        _ = tensor_halfspace.n


def test_halfspace_to_layer_conversion():
    """Test conversion from HalfSpace to Layer."""
    # Isotropic case
    si_material = Material(er=12.0, ur=1.0)
    si_halfspace = HalfSpace(material=si_material)
    si_layer = si_halfspace.to_layer()
    
    assert isinstance(si_layer, Layer)
    assert si_layer.er == 12.0
    assert si_layer.ur == 1.0
    assert si_layer.thickness == 0.0
    
    # Anisotropic case
    epsilon_tensor = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], 
        [0.0, 0.0, 4.0]
    ], dtype=complex)
    
    tensor_material = TensorMaterial(epsilon_tensor=epsilon_tensor)
    tensor_halfspace = HalfSpace(tensor_material=tensor_material)
    tensor_layer = tensor_halfspace.to_layer()
    
    assert isinstance(tensor_layer, Layer)
    assert tensor_layer.is_anisotropic == True
    assert tensor_layer.thickness == 0.0
    assert np.allclose(tensor_layer.er, epsilon_tensor)


def test_air_factory():
    """Test Air constant."""
    air = Air
    assert isinstance(air, HalfSpace)
    assert air.er == 1.0
    assert air.ur == 1.0
    assert air.n == 1.0
    assert air.is_anisotropic == False


def test_substrate_factory():
    """Test Substrate() factory function."""
    # With isotropic material
    si_material = Material(er=12.0, ur=1.0)
    si_substrate = Substrate(si_material)
    
    assert isinstance(si_substrate, HalfSpace)
    assert si_substrate.er == 12.0
    assert si_substrate.ur == 1.0
    assert si_substrate.is_anisotropic == False
    
    # With anisotropic material
    epsilon_tensor = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], 
        [0.0, 0.0, 4.0]
    ], dtype=complex)
    
    tensor_material = TensorMaterial(epsilon_tensor=epsilon_tensor)
    tensor_substrate = Substrate(tensor_material)
    
    assert isinstance(tensor_substrate, HalfSpace)
    assert tensor_substrate.is_anisotropic == True
    assert np.allclose(tensor_substrate.er, epsilon_tensor)


def test_stack_with_halfspaces():
    """Test new Stack API with HalfSpace objects."""
    # Create a test layer
    test_layer = Layer(er=4.0, ur=1.0, thickness=1.0)
    
    # Create HalfSpace objects
    air_superstrate = Air
    si_material = Material(er=12.0, ur=1.0)
    si_substrate = Substrate(si_material)
    
    # Test new Stack API
    stack = Stack(test_layer, 
                  incident_layer=air_superstrate, 
                  transmission_layer=si_substrate)
    
    # Verify that the stack was created correctly
    assert len(stack.internal_layers) == 1
    assert stack.internal_layers[0] == test_layer
    
    # Verify that HalfSpace objects were converted to Layers
    assert isinstance(stack.incident_layer, Layer)
    assert isinstance(stack.transmission_layer, Layer)
    assert stack.incident_layer.er == 1.0
    assert stack.transmission_layer.er == 12.0
    assert stack.incident_layer.thickness == 0.0
    assert stack.transmission_layer.thickness == 0.0


def test_backward_compatibility_stack():
    """Test that existing LayerStack usage still works."""
    # Traditional usage should still work
    test_layer = Layer(er=4.0, ur=1.0, thickness=1.0)
    air_layer = Layer(er=1.0, ur=1.0, thickness=0.0)
    si_layer = Layer(er=12.0, ur=1.0, thickness=0.0)
    
    # Traditional LayerStack construction
    stack = LayerStack(test_layer,
                      incident_layer=air_layer,
                      transmission_layer=si_layer)
    
    assert len(stack.internal_layers) == 1
    assert stack.incident_layer.er == 1.0
    assert stack.transmission_layer.er == 12.0


def test_stack_alias():
    """Test that Stack is correctly aliased to LayerStack."""
    assert Stack is LayerStack


def test_stack_simplified_syntax():
    """Test the simplified syntax mentioned in ROADMAP."""
    # Mimic the example from the ROADMAP
    si_material = Material(er=12.0, ur=1.0)
    patterned_layer = Layer(er=4.0, ur=1.0, thickness=1.0)  # Placeholder for PatternedLayer
    
    # This is the syntax we want to support
    stack = Stack(patterned_layer,
                  incident_layer=Air,
                  transmission_layer=Substrate(si_material))
    
    # Verify the construction worked
    assert len(stack.internal_layers) == 1
    assert stack.incident_layer.er == 1.0
    assert stack.transmission_layer.er == 12.0
    assert stack.incident_layer.thickness == 0.0
    assert stack.transmission_layer.thickness == 0.0


def test_stack_default_parameters():
    """Test Stack with default parameters."""
    test_layer = Layer(er=4.0, ur=1.0, thickness=1.0)
    
    # Stack with default incident and transmission layers (air)
    stack = Stack(test_layer)
    
    assert stack.incident_layer.er == 1.0
    assert stack.incident_layer.ur == 1.0
    assert stack.transmission_layer.er == 1.0
    assert stack.transmission_layer.ur == 1.0


def test_halfspace_string_representation():
    """Test string representation of HalfSpace objects."""
    # Isotropic halfspace
    air = Air
    air_str = str(air)
    assert "HalfSpace" in air_str
    assert "er=1" in air_str
    assert "ur=1" in air_str
    
    # Anisotropic halfspace
    epsilon_tensor = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], 
        [0.0, 0.0, 4.0]
    ], dtype=complex)
    
    tensor_material = TensorMaterial(epsilon_tensor=epsilon_tensor)
    tensor_halfspace = HalfSpace(tensor_material=tensor_material)
    tensor_str = str(tensor_halfspace)
    assert "HalfSpace" in tensor_str
    assert "tensor_material" in tensor_str


if __name__ == "__main__":
    pytest.main([__file__])
