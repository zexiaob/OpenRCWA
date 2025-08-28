"""
Integration test demonstrating Phase 2.1 and 2.1+ functionality.

This test shows the complete workflow from geometry creation to solver integration.
"""

import numpy as np
import pytest
from rcwa.geom.shape import Rectangle, Circle, UnionShape, DifferenceShape
from rcwa.geom.patterned import PatternedLayer
from rcwa.core.adapters import PatternedLayerAdapter, GeometryStackAdapter
from rcwa.model.material import Material
from rcwa.model.layer import Layer


def test_geometry_workflow_integration():
    """Test complete geometry workflow from shapes to solver-ready layers."""
    
    # Step 1: Create materials
    substrate = Material(er=12.0, ur=1.0)  # Silicon-like
    air = Material(er=1.0, ur=1.0)
    
    # Step 2: Create complex geometry with boolean operations
    # Base rectangular region
    base = Rectangle(center=(0.5, 0.5), width=0.8, height=0.8)
    
    # Circular hole
    hole = Circle(center=(0.5, 0.5), radius=0.15)
    
    # Small features for scattering
    feature1 = Circle(center=(0.3, 0.3), radius=0.08)
    feature2 = Circle(center=(0.7, 0.7), radius=0.08)
    features = UnionShape([feature1, feature2])
    
    # Create final pattern: base with hole but add small features
    pattern_with_hole = DifferenceShape(base, [hole])
    final_pattern = UnionShape([pattern_with_hole, features])
    
    # Step 3: Create PatternedLayer
    patterned_layer = PatternedLayer(
        thickness=0.22,  # 220nm
        lattice=((0.8, 0.0), (0.0, 0.8)),  # 800nm period
        background_material=air,
        shapes=[(final_pattern, substrate)]
    )
    
    # Step 4: Test pattern properties
    bounds = patterned_layer.get_bounds()
    assert len(bounds) == 4
    assert bounds[0] < bounds[1]  # x_min < x_max
    assert bounds[2] < bounds[3]  # y_min < y_max
    
    # Step 5: Rasterize pattern
    er_field, ur_field = patterned_layer.rasterize_tensor_field(wavelength=1.5e-6)
    
    assert er_field.shape == (256, 256)
    assert ur_field.shape == (256, 256)
    
    # Verify that both materials are present
    assert np.any(np.isclose(er_field, 1.0))   # Air background
    assert np.any(np.isclose(er_field, 12.0))  # Substrate features
    
    # Step 6: Convert to solver-compatible Layer
    harmonics = (7, 7)  # 7x7 harmonics for good convergence
    converted_layer = PatternedLayerAdapter.patterned_to_layer(
        patterned_layer, harmonics, wavelength=1.5e-6
    )
    
    assert converted_layer.thickness == 0.22
    assert hasattr(converted_layer, '_source_patterned_layer')
    
    # Step 7: Validate conversion
    validation = PatternedLayerAdapter.validate_pattern_conversion(
        patterned_layer, converted_layer, wavelength=1.5e-6
    )
    
    assert validation['thickness_match'] == True
    assert validation['pattern_bounds_reasonable'] == True
    assert validation['convolution_matrices_finite'] == True
    
    # Step 8: Test mixed stack with regular layers
    regular_layer1 = Layer(er=2.25, ur=1.0, thickness=0.1)  # SiO2-like
    regular_layer2 = Layer(er=16.0, ur=1.0, thickness=0.05)  # High-index
    
    mixed_stack = [regular_layer1, patterned_layer, regular_layer2]
    
    # Step 9: Convert entire stack
    converted_stack = GeometryStackAdapter.convert_geometry_stack(
        mixed_stack, harmonics=harmonics, wavelength=1.5e-6
    )
    
    assert len(converted_stack) == 3
    assert converted_stack[0] is regular_layer1  # Unchanged
    assert converted_stack[1] is not patterned_layer  # Converted
    assert converted_stack[2] is regular_layer2  # Unchanged
    assert converted_stack[1].thickness == 0.22
    
    # Step 10: Test automatic harmonics suggestion
    suggested_harmonics = GeometryStackAdapter.suggest_harmonics_for_pattern(
        patterned_layer, wavelength=1.5e-6, target_accuracy=0.01
    )
    
    assert isinstance(suggested_harmonics, tuple)
    assert len(suggested_harmonics) == 2
    assert suggested_harmonics[0] >= 3  # Reasonable minimum
    assert suggested_harmonics[1] >= 3
    assert suggested_harmonics[0] % 2 == 1  # Odd numbers
    assert suggested_harmonics[1] % 2 == 1


def test_parametric_geometry_integration():
    """Test parametric geometry with parameter sweeps."""
    
    # Create parametric geometry
    air = Material(er=1.0, ur=1.0)
    silicon = Material(er=12.0, ur=1.0)
    
    def radius_func(params):
        return 0.1 + 0.2 * params.get('fill_factor', 0.5)
    
    def width_func(params):
        fill = params.get('fill_factor', 0.5)
        return 0.3 + 0.4 * fill
    
    # Parametric shapes
    circle = Circle(center=(0.25, 0.25), radius=radius_func)
    rect = Rectangle(center=(0.75, 0.75), width=width_func, height=0.3)
    
    # Create parametric layer
    base_layer = PatternedLayer(
        thickness=0.2,
        lattice=((1.0, 0.0), (0.0, 1.0)),
        background_material=air,
        shapes=[(circle, silicon), (rect, silicon)]
    )
    
    # Parameter sweep
    fill_factors = [0.2, 0.5, 0.8]
    layers = []
    
    for ff in fill_factors:
        params = {'fill_factor': ff}
        layer_instance = base_layer.with_params(params)
        layers.append(layer_instance)
    
    assert len(layers) == 3
    
    # Verify parameter changes
    for i, layer in enumerate(layers):
        expected_radius = 0.1 + 0.2 * fill_factors[i]
        expected_width = 0.3 + 0.4 * fill_factors[i]
        
        # Extract shapes
        circle_shape = layer.shapes[0][0]
        rect_shape = layer.shapes[1][0]
        
        assert np.isclose(circle_shape.radius, expected_radius)
        assert np.isclose(rect_shape.width, expected_width)
    
    # Rasterize each and verify different fill factors
    fields = []
    for layer in layers:
        er_field, _ = layer.rasterize_tensor_field(wavelength=1.5e-6)
        silicon_fraction = np.sum(np.isclose(er_field, 12.0)) / er_field.size
        fields.append(silicon_fraction)
    
    # Higher fill factor should mean more silicon
    assert fields[0] < fields[1] < fields[2]


if __name__ == '__main__':
    test_geometry_workflow_integration()
    test_parametric_geometry_integration()
    print("✅ All integration tests passed!")
