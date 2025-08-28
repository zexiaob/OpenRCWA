"""
Tests for the geometry layer (Phase 2.1 and 2.1+).

This module tests the declarative geometry layer and parametric/boolean shape support
including Shape classes, PatternedLayer, and integration with core adapters.
"""

import pytest
import numpy as np
from rcwa.geom.shape import (
    Shape, Rectangle, Circle, Ellipse, Polygon, RegularPolygon,
    ComplexShape, UnionShape, IntersectionShape, DifferenceShape
)
from rcwa.geom.patterned import PatternedLayer
from rcwa.core.adapters import PatternedLayerAdapter, GeometryStackAdapter
from rcwa.model.material import Material


class TestShapeBasics:
    """Test basic Shape class functionality."""
    
    def test_rectangle_creation(self):
        """Test Rectangle shape creation and basic properties."""
        rect = Rectangle(center=(0, 0), width=2.0, height=1.0)
        
        assert rect.center.x == 0 and rect.center.y == 0
        assert rect.width == 2.0
        assert rect.height == 1.0
        assert rect.rotation == 0.0
    
    def test_rectangle_contains(self):
        """Test Rectangle containment logic."""
        rect = Rectangle(center=(0, 0), width=2.0, height=1.0)
        
        # Points inside
        assert rect.contains(0, 0) == True
        assert rect.contains(0.5, 0.25) == True
        assert rect.contains(-0.5, -0.25) == True
        
        # Points outside
        assert rect.contains(1.5, 0) == False
        assert rect.contains(0, 0.75) == False
        assert rect.contains(2, 2) == False
    
    def test_circle_creation_and_contains(self):
        """Test Circle shape creation and containment."""
        circle = Circle(center=(1, 1), radius=0.5)
        
        assert circle.center.x == 1 and circle.center.y == 1
        assert circle.radius == 0.5
        
        # Points inside
        assert circle.contains(1, 1) == True
        assert circle.contains(1.3, 1) == True
        assert circle.contains(1, 1.3) == True
        
        # Points outside
        assert circle.contains(2, 1) == False
        assert circle.contains(1, 2) == False
        assert circle.contains(0, 0) == False
    
    def test_ellipse_creation_and_contains(self):
        """Test Ellipse shape creation and containment."""
        ellipse = Ellipse(center=(0, 0), semi_major=2.0, semi_minor=1.0)
        
        # Points inside
        assert ellipse.contains(0, 0) == True
        assert ellipse.contains(1.5, 0) == True
        assert ellipse.contains(0, 0.8) == True
        
        # Points outside
        assert ellipse.contains(2.5, 0) == False
        assert ellipse.contains(0, 1.5) == False
    
    def test_polygon_creation_and_contains(self):
        """Test Polygon shape creation and containment."""
        # Simple triangle
        vertices = [(0, 0), (1, 0), (0.5, 1)]
        triangle = Polygon(vertices)
        
        # Points inside
        assert triangle.contains(0.5, 0.3) == True
        assert triangle.contains(0.25, 0.1) == True
        
        # Points outside
        assert triangle.contains(-0.5, 0.5) == False
        assert triangle.contains(1.5, 0.5) == False
        assert triangle.contains(0.5, 1.5) == False
    
    def test_regular_polygon_creation(self):
        """Test RegularPolygon shape creation."""
        # Regular hexagon
        hexagon = RegularPolygon(center=(0, 0), radius=1.0, num_sides=6)
        
        assert hexagon.center.x == 0 and hexagon.center.y == 0
        assert hexagon.radius == 1.0
        assert hexagon.num_sides == 6
        assert len(hexagon.vertices) == 6
        
        # Check that vertices form a regular hexagon
        vertices = np.array(hexagon.vertices)
        distances = np.sqrt(np.sum(vertices**2, axis=1))
        assert np.allclose(distances, 1.0)
    
    def test_shape_bounds(self):
        """Test shape bounding box calculations."""
        rect = Rectangle(center=(1, 2), width=4, height=2)
        x_min, x_max, y_min, y_max = rect.get_bounds()
        
        assert x_min == -1
        assert x_max == 3
        assert y_min == 1
        assert y_max == 3
        
        circle = Circle(center=(0, 0), radius=1)
        x_min, x_max, y_min, y_max = circle.get_bounds()
        
        assert x_min == -1
        assert x_max == 1
        assert y_min == -1
        assert y_max == 1


class TestBooleanShapes:
    """Test boolean shape operations."""
    
    def test_union_shape(self):
        """Test UnionShape boolean operation."""
        rect1 = Rectangle(center=(0, 0), width=2, height=1)
        rect2 = Rectangle(center=(1, 0), width=2, height=1)
        union = UnionShape([rect1, rect2])
        
        # Points in either shape should be in union
        assert union.contains(-0.5, 0) == True  # In rect1 only
        assert union.contains(1.5, 0) == True   # In rect2 only
        assert union.contains(0.5, 0) == True   # In both
        
        # Points in neither should not be in union
        assert union.contains(-2, 0) == False
        assert union.contains(3, 0) == False
    
    def test_intersection_shape(self):
        """Test IntersectionShape boolean operation."""
        rect1 = Rectangle(center=(0, 0), width=2, height=2)
        rect2 = Rectangle(center=(0.5, 0), width=2, height=2)
        intersection = IntersectionShape([rect1, rect2])
        
        # Points in both shapes should be in intersection
        assert intersection.contains(0.25, 0) == True
        
        # Points in only one shape should not be in intersection
        assert intersection.contains(-0.75, 0) == False  # In rect1 only
        assert intersection.contains(1.25, 0) == False   # In rect2 only
        
        # Points in neither should not be in intersection
        assert intersection.contains(2, 0) == False
    
    def test_difference_shape(self):
        """Test DifferenceShape boolean operation."""
        outer = Rectangle(center=(0, 0), width=4, height=4)
        inner = Circle(center=(0, 0), radius=1)
        difference = DifferenceShape(outer, [inner])
        
        # Points in outer but not inner should be in difference
        assert difference.contains(1.5, 0) == True
        
        # Points in inner should not be in difference
        assert difference.contains(0, 0) == False
        assert difference.contains(0.5, 0) == False
        
        # Points outside outer should not be in difference
        assert difference.contains(3, 0) == False
    
    def test_complex_shape_nesting(self):
        """Test nested boolean operations."""
        # Create a shape with a hole using nested operations
        base = Rectangle(center=(0, 0), width=4, height=4)
        hole1 = Circle(center=(-1, 0), radius=0.5)
        hole2 = Circle(center=(1, 0), radius=0.5)
        
        union_holes = UnionShape([hole1, hole2])
        complex_shape = DifferenceShape(base, [union_holes])
        
        # Points in base but not in holes
        assert complex_shape.contains(0, 1) == True
        
        # Points in holes should not be present
        assert complex_shape.contains(-1, 0) == False
        assert complex_shape.contains(1, 0) == False


class TestParametricShapes:
    """Test parametric shape support."""
    
    def test_parametric_rectangle(self):
        """Test parametric Rectangle with sweep parameters."""
        def width_func(params):
            return params.get('scale', 1.0) * 2.0
        
        def height_func(params):
            return params.get('scale', 1.0) * 1.0
        
        rect = Rectangle(
            center=(0, 0), 
            width=width_func,
            height=height_func
        )
        
        # Test with different parameter values
        params1 = {'scale': 1.0}
        rect_1 = rect.with_params(params1)
        assert rect_1.width == 2.0
        assert rect_1.height == 1.0
        
        params2 = {'scale': 2.0}
        rect_2 = rect.with_params(params2)
        assert rect_2.width == 4.0
        assert rect_2.height == 2.0
    
    def test_parametric_circle(self):
        """Test parametric Circle with sweep parameters."""
        def radius_func(params):
            return 0.5 + 0.3 * params.get('t', 0)
        
        circle = Circle(center=(0, 0), radius=radius_func)
        
        # Test parameter sweep
        for t in [0, 0.5, 1.0]:
            params = {'t': t}
            circle_t = circle.with_params(params)
            expected_radius = 0.5 + 0.3 * t
            assert circle_t.radius == expected_radius
    
    def test_parametric_validation(self):
        """Test validation of parametric shapes."""
        def invalid_width(params):
            return -1.0  # Invalid negative width
        
        rect = Rectangle(center=(0, 0), width=invalid_width, height=1.0)
        
        with pytest.raises(ValueError, match="Width must be positive"):
            rect.with_params({})


class TestPatternedLayer:
    """Test PatternedLayer functionality."""
    
    def test_patterned_layer_creation(self):
        """Test basic PatternedLayer creation."""
        material_a = Material(er=4.0, ur=1.0)
        material_b = Material(er=9.0, ur=1.0)
        
        rect = Rectangle(center=(0.25, 0.25), width=0.5, height=0.5)
        
        layer = PatternedLayer(
            thickness=0.5,
            lattice=((1.0, 0.0), (0.0, 1.0)),
            background_material=material_a,
            shapes=[
                (rect, material_b)
            ]
        )
        
        assert layer.thickness == 0.5
        assert layer.lattice == ((1.0, 0.0), (0.0, 1.0))
        assert len(layer.shapes) == 1
    
    def test_patterned_layer_rasterization(self):
        """Test PatternedLayer rasterization to tensor fields."""
        material_a = Material(er=2.0, ur=1.0)
        material_b = Material(er=8.0, ur=1.0)
        
        # Simple square in center
        rect = Rectangle(center=(0.5, 0.5), width=0.5, height=0.5)
        
        layer = PatternedLayer(
            thickness=0.5,
            lattice=((1.0, 0.0), (0.0, 1.0)),
            background_material=material_a,
            shapes=[(rect, material_b)]
        )
        
        # Rasterize to a coarse grid  
        er_field, ur_field = layer.rasterize_tensor_field(wavelength=1.0)
        
        assert er_field.shape == (64, 64)  # Default resolution
        assert ur_field.shape == (64, 64)
        
        # Check that background and pattern materials appear
        assert np.any(np.isclose(er_field, 2.0))  # Background
        assert np.any(np.isclose(er_field, 8.0))  # Pattern
    
    def test_convolution_matrix_generation(self):
        """Test convolution matrix generation from patterns."""
        material_a = Material(er=1.0, ur=1.0)
        material_b = Material(er=4.0, ur=1.0)
        
        rect = Rectangle(center=(0.5, 0.5), width=0.8, height=0.8)
        
        layer = PatternedLayer(
            thickness=0.5,
            lattice=((1.0, 0.0), (0.0, 1.0)),
            background_material=material_a,
            shapes=[(rect, material_b)]
        )
        
        # Generate convolution matrices
        harmonics = (5, 5)
        conv_matrices = layer.to_convolution_matrices(harmonics, wavelength=1.0)
        
        assert 'er_xx' in conv_matrices
        assert 'er_yy' in conv_matrices
        assert 'er_zz' in conv_matrices
        assert 'ur_xx' in conv_matrices
        assert 'ur_yy' in conv_matrices
        assert 'ur_zz' in conv_matrices
        
        # Check matrix dimensions
        total_harmonics = harmonics[0] * harmonics[1]
        for key, matrix in conv_matrices.items():
            assert matrix.shape == (total_harmonics, total_harmonics)
            assert np.all(np.isfinite(matrix))
    
    def test_patterned_layer_validation(self):
        """Test PatternedLayer validation."""
        material = Material(er=4.0, ur=1.0)
        
        # Invalid thickness
        with pytest.raises(ValueError, match="Layer thickness must be positive"):
            PatternedLayer(
                thickness=-0.1,
                lattice=((1.0, 0.0), (0.0, 1.0)),
                background_material=material,
                shapes=[]
            )
        
        # Invalid lattice (not invertible)
        with pytest.raises(ValueError, match="Lattice matrix must be invertible"):
            PatternedLayer(
                thickness=0.5,
                lattice=((0.0, 0.0), (0.0, 1.0)),  # Singular matrix
                background_material=material,
                shapes=[]
            )
    
    def test_patterned_layer_bounds(self):
        """Test PatternedLayer bounds calculation."""
        material_a = Material(er=2.0, ur=1.0)
        material_b = Material(er=8.0, ur=1.0)
        
        rect1 = Rectangle(center=(0.2, 0.3), width=0.2, height=0.3)
        rect2 = Rectangle(center=(0.7, 0.8), width=0.1, height=0.2)
        
        layer = PatternedLayer(
            thickness=0.5,
            lattice=((1.0, 0.0), (0.0, 1.0)),
            background_material=material_a,
            shapes=[(rect1, material_b), (rect2, material_b)]
        )
        
        x_min, x_max, y_min, y_max = layer.get_bounds()
        
        # Should encompass both rectangles
        assert x_min <= 0.1  # rect1 left edge
        assert x_max >= 0.75  # rect2 right edge
        assert y_min <= 0.15  # rect1 bottom edge
        assert y_max >= 0.9   # rect2 top edge


class TestPatternedLayerParametric:
    """Test parametric functionality in PatternedLayer."""
    
    def test_parametric_patterned_layer(self):
        """Test PatternedLayer with parametric shapes."""
        material_a = Material(er=1.0, ur=1.0)
        material_b = Material(er=4.0, ur=1.0)
        
        def width_func(params):
            return 0.2 + 0.3 * params.get('fill_factor', 0.5)
        
        rect = Rectangle(
            center=(0.5, 0.5),
            width=width_func,
            height=0.5
        )
        
        layer = PatternedLayer(
            thickness=0.5,
            lattice=((1.0, 0.0), (0.0, 1.0)),
            background_material=material_a,
            shapes=[(rect, material_b)]
        )
        
        # Test different fill factors
        params1 = {'fill_factor': 0.2}
        layer1 = layer.with_params(params1)
        
        params2 = {'fill_factor': 0.8}
        layer2 = layer.with_params(params2)
        
        # Generate fields for comparison
        field1, _ = layer1.rasterize_tensor_field((10, 10), 1.0)
        field2, _ = layer2.rasterize_tensor_field((10, 10), 1.0)
        
        # Higher fill factor should have more material_b
        count1 = np.sum(np.isclose(field1, 4.0))
        count2 = np.sum(np.isclose(field2, 4.0))
        assert count2 > count1
    
    def test_parametric_sweep_generation(self):
        """Test automatic parameter sweep generation."""
        material_a = Material(er=1.0, ur=1.0)
        material_b = Material(er=4.0, ur=1.0)
        
        def radius_func(params):
            return 0.1 + 0.3 * params.get('r', 0.5)
        
        circle = Circle(center=(0.5, 0.5), radius=radius_func)
        
        layer = PatternedLayer(
            thickness=0.5,
            lattice=((1.0, 0.0), (0.0, 1.0)),
            background_material=material_a,
            shapes=[(circle, material_b)]
        )
        
        # Generate parameter sweep
        r_values = np.linspace(0, 1, 5)
        layers = []
        
        for r in r_values:
            params = {'r': r}
            layers.append(layer.with_parameters(params))
        
        assert len(layers) == 5
        
        # Check that radius changes
        for i, layer_instance in enumerate(layers):
            expected_radius = 0.1 + 0.3 * r_values[i]
            actual_shape = layer_instance.shapes[0][0]  # (shape, material)
            assert np.isclose(actual_shape.radius, expected_radius)


class TestAdapterIntegration:
    """Test integration between PatternedLayer and core adapters."""
    
    def test_patterned_layer_adapter(self):
        """Test PatternedLayerAdapter conversion."""
        material_a = Material(er=1.0, ur=1.0)
        material_b = Material(er=4.0, ur=1.0)
        
        rect = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
        
        patterned_layer = PatternedLayer(
            thickness=0.3,
            lattice=((1.0, 0.0), (0.0, 1.0)),
            background_material=material_a,
            shapes=[(rect, material_b)]
        )
        
        # Convert to regular Layer
        harmonics = (3, 3)
        converted_layer = PatternedLayerAdapter.patterned_to_layer(
            patterned_layer, harmonics, wavelength=1.0
        )
        
        # Check basic properties
        assert converted_layer.thickness == 0.3
        assert hasattr(converted_layer, '_source_patterned_layer')
        assert converted_layer._source_patterned_layer is patterned_layer
    
    def test_validation(self):
        """Test adapter validation."""
        material_a = Material(er=1.0, ur=1.0)
        material_b = Material(er=4.0, ur=1.0)
        
        rect = Rectangle(center=(0.5, 0.5), width=0.4, height=0.4)
        
        patterned_layer = PatternedLayer(
            thickness=0.2,
            lattice=((1.0, 0.0), (0.0, 1.0)),
            background_material=material_a,
            shapes=[(rect, material_b)]
        )
        
        converted_layer = PatternedLayerAdapter.patterned_to_layer(
            patterned_layer, (3, 3), wavelength=1.0
        )
        
        # Validate conversion
        validation_results = PatternedLayerAdapter.validate_pattern_conversion(
            patterned_layer, converted_layer, wavelength=1.0
        )
        
        assert validation_results['thickness_match'] == True
        assert validation_results['pattern_bounds_reasonable'] == True
        assert validation_results['convolution_matrices_finite'] == True
    
    def test_geometry_stack_conversion(self):
        """Test mixed stack conversion."""
        from rcwa.model.layer import Layer
        
        # Regular layer
        regular_layer = Layer(er=2.0, ur=1.0, thickness=0.1)
        
        # Patterned layer
        material_a = Material(er=1.0, ur=1.0)
        material_b = Material(er=9.0, ur=1.0)
        circle = Circle(center=(0.5, 0.5), radius=0.3)
        patterned_layer = PatternedLayer(
            thickness=0.2,
            lattice=((1.0, 0.0), (0.0, 1.0)),
            background_material=material_a,
            shapes=[(circle, material_b)]
        )
        
        # Mixed stack
        mixed_stack = [regular_layer, patterned_layer]
        
        # Convert to all regular layers
        converted_stack = GeometryStackAdapter.convert_geometry_stack(
            mixed_stack, harmonics=(3, 3), wavelength=1.0
        )
        
        assert len(converted_stack) == 2
        assert converted_stack[0] is regular_layer  # Unchanged
        assert converted_stack[1] is not patterned_layer  # Converted
        assert converted_stack[1].thickness == 0.2
    
    def test_harmonics_suggestion(self):
        """Test automatic harmonics suggestion."""
        material_a = Material(er=1.0, ur=1.0)
        material_b = Material(er=4.0, ur=1.0)
        
        # Small features that need high harmonics
        small_circle = Circle(center=(0.5, 0.5), radius=0.05)
        
        patterned_layer = PatternedLayer(
            thickness=0.1,
            lattice=((1.0, 0.0), (0.0, 1.0)),
            background_material=material_a,
            shapes=[(small_circle, material_b)]
        )
        
        # Get suggested harmonics
        suggested_harmonics = GeometryStackAdapter.suggest_harmonics_for_pattern(
            patterned_layer, wavelength=1.0, target_accuracy=0.01
        )
        
        assert isinstance(suggested_harmonics, tuple)
        assert len(suggested_harmonics) == 2
        assert suggested_harmonics[0] >= 3  # Minimum harmonics
        assert suggested_harmonics[1] >= 3
        assert suggested_harmonics[0] <= 21  # Maximum cap
        assert suggested_harmonics[1] <= 21
        assert suggested_harmonics[0] % 2 == 1  # Odd for symmetry
        assert suggested_harmonics[1] % 2 == 1


class TestTensorFieldRasterization:
    """Test advanced tensor field rasterization."""
    
    def test_anisotropic_tensor_rasterization(self):
        """Test rasterization with anisotropic tensor materials."""
        # Background isotropic
        background = Material(er=1.0, ur=1.0)
        
        # Pattern anisotropic (simulated with different x,y components)
        from rcwa.model.material import TensorMaterial
        
        pattern_tensor = TensorMaterial(
            er_tensor=np.array([[4.0, 0.1], [0.1, 9.0]]),
            ur_tensor=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        
        rect = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
        
        layer = PatternedLayer(
            thickness=0.5,
            lattice=((1.0, 0.0), (0.0, 1.0)),
            background_material=background,
            shapes=[(rect, pattern_tensor)]
        )
        
        # Rasterize tensor fields
        er_field, ur_field = layer.rasterize_tensor_field((8, 8), wavelength=1.0)
        
        assert er_field.shape == (8, 8)
        assert ur_field.shape == (8, 8)
        
        # Check that both background and pattern values exist
        assert np.any(np.isclose(er_field, 1.0))  # Background
        assert np.any(er_field > 3.0)  # Pattern (should be 4.0 or 9.0)
    
    def test_high_resolution_rasterization(self):
        """Test high-resolution rasterization performance and accuracy."""
        material_a = Material(er=1.0, ur=1.0)
        material_b = Material(er=16.0, ur=1.0)
        
        # Complex shape with small features
        outer = Rectangle(center=(0.5, 0.5), width=0.8, height=0.8)
        inner = Circle(center=(0.5, 0.5), radius=0.1)
        complex_shape = DifferenceShape(outer, [inner])
        
        layer = PatternedLayer(
            thickness=0.5,
            lattice=((1.0, 0.0), (0.0, 1.0)),
            background_material=material_a,
            shapes=[(complex_shape, material_b)]
        )
        
        # High-resolution rasterization
        er_field, _ = layer.rasterize_tensor_field((64, 64), wavelength=1.0)
        
        assert er_field.shape == (64, 64)
        
        # Check that the hole is represented
        center_idx = 32  # Center of 64x64 grid
        hole_region = er_field[center_idx-2:center_idx+3, center_idx-2:center_idx+3]
        pattern_region = er_field[5:10, 5:10]  # Corner should be pattern
        
        # Hole should be background material
        assert np.all(np.isclose(hole_region, 1.0))
        
        # Pattern region should be pattern material
        assert np.all(np.isclose(pattern_region, 16.0))


if __name__ == '__main__':
    pytest.main([__file__])
