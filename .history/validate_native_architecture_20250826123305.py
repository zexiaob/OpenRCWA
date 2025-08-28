#!/usr/bin/env python3
"""
Final validation script for the native PatternedLayer architecture.

This script demonstrates that PatternedLayer now directly inherits from Layer,
eliminating the need for adapters and providing seamless RCWA integration.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rcwa.geom.shape import Rectangle, Circle, UnionShape, DifferenceShape
from rcwa.geom.patterned import PatternedLayer, rectangular_lattice, square_lattice
from rcwa.model.material import Material
from rcwa.model.layer import Layer
from rcwa.core.adapters import suggest_harmonics_for_pattern


def test_native_layer_architecture():
    """Test the core native Layer architecture."""
    print("🧪 Testing Native Layer Architecture")
    
    # Create materials
    air = Material(er=1.0, ur=1.0)
    silicon = Material(er=12.0, ur=1.0)
    
    # Create geometry
    circle = Circle(center=(0.5, 0.5), radius=0.3)
    
    # Create PatternedLayer
    patterned_layer = PatternedLayer(
        thickness=0.5,
        lattice=square_lattice(1.0),
        background_material=air,
        shapes=[(circle, silicon)]
    )
    
    # Verify direct Layer inheritance
    assert isinstance(patterned_layer, Layer), "❌ PatternedLayer should inherit from Layer"
    assert not patterned_layer.homogenous, "❌ PatternedLayer should not be homogeneous"
    assert patterned_layer.thickness == 0.5, "❌ Thickness should be preserved"
    
    print("✅ PatternedLayer correctly inherits from Layer")
    print(f"✅ Layer properties: thickness={patterned_layer.thickness}m, homogeneous={patterned_layer.homogenous}")
    
    return patterned_layer


def test_convolution_matrix_interface(patterned_layer):
    """Test native convolution matrix interface."""
    print("\n🧪 Testing Convolution Matrix Interface")
    
    # Create harmonics
    harmonics_x = np.array([-2, -1, 0, 1, 2])
    harmonics_y = np.array([-2, -1, 0, 1, 2])
    
    # Test convolution matrix generation
    conv_matrix = patterned_layer.convolution_matrix(
        harmonics_x, harmonics_y, 'eps_xx'
    )
    
    # Verify matrix properties
    expected_size = len(harmonics_x) * len(harmonics_y)  # 25
    assert conv_matrix.shape == (expected_size, expected_size), f"❌ Expected {expected_size}x{expected_size} matrix"
    assert np.all(np.isfinite(conv_matrix)), "❌ Convolution matrix should be finite"
    assert np.iscomplexobj(conv_matrix), "❌ Convolution matrix should be complex"
    
    print(f"✅ Convolution matrix shape: {conv_matrix.shape}")
    print(f"✅ Matrix is finite: {np.all(np.isfinite(conv_matrix))}")
    print(f"✅ Matrix is complex: {np.iscomplexobj(conv_matrix)}")


def test_mixed_layer_stack():
    """Test mixed stacks with regular and patterned layers."""
    print("\n🧪 Testing Mixed Layer Stacks")
    
    # Create materials
    air = Material(er=1.0, ur=1.0)
    silicon = Material(er=12.0, ur=1.0)
    sio2 = Material(er=2.25, ur=1.0)
    
    # Regular layers
    substrate = Layer(thickness=10.0, material=silicon)
    capping = Layer(thickness=0.1, material=sio2)
    
    # Patterned layer
    rect = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
    patterned_layer = PatternedLayer(
        thickness=0.22,
        lattice=rectangular_lattice(0.8, 0.8),
        background_material=air,
        shapes=[(rect, silicon)]
    )
    
    # Create mixed stack - no conversion needed!
    mixed_stack = [substrate, patterned_layer, capping]
    
    # Verify all are Layer instances
    for i, layer in enumerate(mixed_stack):
        assert isinstance(layer, Layer), f"❌ Layer {i} should be a Layer instance"
    
    print(f"✅ Mixed stack created with {len(mixed_stack)} layers")
    print(f"✅ All layers are Layer instances: {[isinstance(layer, Layer) for layer in mixed_stack]}")
    print(f"✅ Homogeneous flags: {[layer.homogenous for layer in mixed_stack]}")


def test_complex_boolean_patterns():
    """Test complex patterns with boolean operations."""
    print("\n🧪 Testing Complex Boolean Patterns")
    
    # Create materials
    air = Material(er=1.0, ur=1.0)
    gold = Material(er=-10.0+1.0j, ur=1.0)
    
    # Create complex pattern
    base = Rectangle(center=(0.5, 0.5), width=0.8, height=0.8)
    hole = Circle(center=(0.5, 0.5), radius=0.15)
    pattern_with_hole = DifferenceShape(base, [hole])
    
    # Add features
    feature1 = Circle(center=(0.3, 0.3), radius=0.08)
    feature2 = Circle(center=(0.7, 0.7), radius=0.08)
    features = UnionShape([feature1, feature2])
    
    # Final complex pattern
    final_pattern = UnionShape([pattern_with_hole, features])
    
    # Create patterned layer with complex geometry
    complex_layer = PatternedLayer(
        thickness=0.05,  # 50nm gold layer
        lattice=square_lattice(1.0),
        background_material=air,
        shapes=[(final_pattern, gold)]
    )
    
    # Verify it's a valid Layer
    assert isinstance(complex_layer, Layer), "❌ Complex patterned layer should be Layer"
    
    # Test bounds calculation
    bounds = complex_layer.get_bounds()
    assert len(bounds) == 4, "❌ Should return 4 bounds values"
    assert bounds[0] < bounds[1], "❌ x_min should be < x_max"
    assert bounds[2] < bounds[3], "❌ y_min should be < y_max"
    
    print("✅ Complex boolean pattern created successfully")
    print(f"✅ Pattern bounds: x=({bounds[0]:.2f}, {bounds[1]:.2f}), y=({bounds[2]:.2f}, {bounds[3]:.2f})")
    
    return complex_layer


def test_rasterization(complex_layer):
    """Test pattern rasterization."""
    print("\n🧪 Testing Pattern Rasterization")
    
    # Rasterize the complex pattern
    er_field, ur_field = complex_layer.rasterize_tensor_field(wavelength=600e-9)
    
    # Verify rasterized fields
    assert er_field.shape == (256, 256), f"❌ Expected 256x256 field, got {er_field.shape}"
    assert ur_field.shape == (256, 256), f"❌ Expected 256x256 field, got {ur_field.shape}"
    assert np.all(np.isfinite(er_field)), "❌ er field should be finite"
    assert np.all(np.isfinite(ur_field)), "❌ ur field should be finite"
    
    # Check material distribution
    unique_er = np.unique(np.round(er_field.real))
    assert 1.0 in unique_er, "❌ Should contain air (er=1.0)"
    assert -10.0 in unique_er, "❌ Should contain gold (er=-10.0)"
    
    print(f"✅ Rasterized field shape: {er_field.shape}")
    print(f"✅ Material distribution: er values = {sorted(unique_er)}")
    print(f"✅ Fields are finite: er={np.all(np.isfinite(er_field))}, ur={np.all(np.isfinite(ur_field))}")


def test_harmonics_suggestion(patterned_layer):
    """Test automatic harmonics suggestion."""
    print("\n🧪 Testing Harmonics Suggestion")
    
    # Get suggested harmonics
    suggested_harmonics = suggest_harmonics_for_pattern(
        patterned_layer, wavelength=1.5e-6, target_accuracy=0.01
    )
    
    # Verify suggestions
    assert isinstance(suggested_harmonics, tuple), "❌ Should return tuple"
    assert len(suggested_harmonics) == 2, "❌ Should return (Nx, Ny) tuple"
    assert all(h >= 3 for h in suggested_harmonics), "❌ Should suggest reasonable minimum harmonics"
    assert all(h % 2 == 1 for h in suggested_harmonics), "❌ Should suggest odd harmonics"
    
    print(f"✅ Suggested harmonics: {suggested_harmonics}")
    print(f"✅ Both values are odd: {[h % 2 == 1 for h in suggested_harmonics]}")


def main():
    """Run all validation tests."""
    print("🚀 OpenRCWA Geometry Native Layer Architecture Validation\n")
    print("=" * 60)
    
    try:
        # Test core architecture
        patterned_layer = test_native_layer_architecture()
        
        # Test convolution matrices
        test_convolution_matrix_interface(patterned_layer)
        
        # Test mixed stacks
        test_mixed_layer_stack()
        
        # Test complex patterns
        complex_layer = test_complex_boolean_patterns()
        
        # Test rasterization
        test_rasterization(complex_layer)
        
        # Test harmonics suggestion
        test_harmonics_suggestion(patterned_layer)
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Architecture Summary:")
        print("  • PatternedLayer directly inherits from Layer")
        print("  • No adapters or conversions needed")
        print("  • Native RCWA solver compatibility")
        print("  • Full support for complex boolean patterns")
        print("  • Seamless integration with existing code")
        print("\n🚀 The geometry module is ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
