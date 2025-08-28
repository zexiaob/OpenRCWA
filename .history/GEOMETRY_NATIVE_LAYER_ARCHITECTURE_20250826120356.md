# Native Layer Architecture for Geometry Module

This document describes the updated architecture where **PatternedLayer directly inherits from Layer**, eliminating the need for adapters and providing seamless RCWA integration.

## Key Architectural Decision

**PatternedLayer IS a Layer** - no conversion required!

```python
from rcwa.geom.patterned import PatternedLayer
from rcwa.model.layer import Layer

# PatternedLayer directly inherits from Layer
patterned_layer = PatternedLayer(...)
assert isinstance(patterned_layer, Layer)  # True!

# Works immediately with solvers - no conversion needed
layer_stack = [substrate_layer, patterned_layer, superstrate_layer]
```

## Architecture Benefits

### 1. Simplified Workflow

**Before** (with adapters):
```python
# Old workflow - required conversion
patterned_layer = PatternedLayer(...)
converted_layer = PatternedLayerAdapter.patterned_to_layer(
    patterned_layer, harmonics=(7, 7), wavelength=1.5e-6
)
solver.add_layer(converted_layer)
```

**After** (native Layer):
```python
# New workflow - direct usage
patterned_layer = PatternedLayer(...)
solver.add_layer(patterned_layer)  # Works directly!
```

### 2. Native RCWA Interface

PatternedLayer implements the standard Layer interface:

```python
# Standard Layer methods work directly
conv_matrix = patterned_layer.convolution_matrix(harmonics_x, harmonics_y, 'eps_xx')
thickness = patterned_layer.thickness
is_homogeneous = patterned_layer.homogenous  # False for patterned layers
```

### 3. Seamless Stack Integration

```python
# Mix regular and patterned layers naturally
regular_layer = Layer(thickness=10.0, material=substrate)
patterned_layer = PatternedLayer(
    thickness=0.5,
    lattice=((1.0, 0.0), (0.0, 1.0)),
    background_material=air,
    shapes=[(circle, silicon)]
)

# All are Layer instances - no conversion needed
mixed_stack = [regular_layer, patterned_layer]
for layer in mixed_stack:
    assert isinstance(layer, Layer)  # All True!
```

## Implementation Details

### PatternedLayer Constructor

```python
class PatternedLayer(Layer):
    def __init__(self, thickness, lattice, background_material, shapes, **params):
        # Initialize as Layer with background material
        super().__init__(
            thickness=thickness,
            material=background_material if isinstance(background_material, Material) else None,
            tensor_material=background_material if isinstance(background_material, TensorMaterial) else None
        )
        
        # Override homogeneous flag - we are periodic/patterned
        self.homogenous = False
        
        # Store pattern-specific attributes
        self.lattice = lattice
        self.shapes = shapes
        # ... rest of initialization
```

### Native Convolution Matrix Interface

```python
def convolution_matrix(self, harmonics_x, harmonics_y, tensor_component='xx'):
    """Compute convolution matrix for RCWA."""
    # Convert tensor component names (eps_xx -> er_xx for compatibility)
    if tensor_component.startswith('eps_'):
        normalized_component = tensor_component.replace('eps_', 'er_')
    elif tensor_component.startswith('mu_'):
        normalized_component = tensor_component.replace('mu_', 'ur_')
    else:
        normalized_component = f'er_{tensor_component}'
    
    # Generate convolution matrices and return requested component
    conv_matrices = self.to_convolution_matrices(
        harmonics=(len(harmonics_x), len(harmonics_y)), 
        wavelength=1.0
    )
    return conv_matrices[normalized_component]
```

## Usage Examples

### Basic Pattern Creation

```python
from rcwa.geom.shape import Rectangle, Circle
from rcwa.geom.patterned import PatternedLayer
from rcwa.model.material import Material

# Materials
air = Material(er=1.0, ur=1.0)
silicon = Material(er=12.0, ur=1.0)

# Shapes with materials
rect = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
circle = Circle(center=(0.5, 0.5), radius=0.2)

# Create patterned layer (directly usable as Layer)
layer = PatternedLayer(
    thickness=0.22,
    lattice=((0.8, 0.0), (0.0, 0.8)),
    background_material=air,
    shapes=[(rect, silicon), (circle, Material(er=16.0, ur=1.0))]
)

# Use directly in solver
assert isinstance(layer, Layer)
assert not layer.homogenous  # Patterned layers are inhomogeneous
```

### Complex Boolean Patterns

```python
from rcwa.geom.shape import UnionShape, DifferenceShape

# Create complex pattern with boolean operations
base = Rectangle(center=(0.5, 0.5), width=0.8, height=0.8)
hole = Circle(center=(0.5, 0.5), radius=0.15)
pattern_with_hole = DifferenceShape(base, [hole])

# Add small scattering features
feature1 = Circle(center=(0.3, 0.3), radius=0.08)
feature2 = Circle(center=(0.7, 0.7), radius=0.08)
features = UnionShape([feature1, feature2])

# Combine into final pattern
final_pattern = UnionShape([pattern_with_hole, features])

# Single PatternedLayer with complex geometry
complex_layer = PatternedLayer(
    thickness=0.22,
    lattice=((0.8, 0.0), (0.0, 0.8)),
    background_material=air,
    shapes=[(final_pattern, silicon)]
)
```

### Direct Solver Integration

```python
# Create multi-layer structure
substrate = Layer(thickness=10.0, material=Material(er=12.0, ur=1.0))
active_layer = PatternedLayer(...)  # Complex patterned layer
capping_layer = Layer(thickness=0.1, material=Material(er=2.25, ur=1.0))

# Direct usage - no adapters or conversion
layer_stack = [substrate, active_layer, capping_layer]

# All layers are Layer instances
for layer in layer_stack:
    assert isinstance(layer, Layer)

# Pass directly to solver
solver = Solver()
for layer in layer_stack:
    solver.add_layer(layer)
```

### Convolution Matrix Generation

```python
# Generate harmonics arrays
harmonics_x = np.array([-3, -2, -1, 0, 1, 2, 3])
harmonics_y = np.array([-3, -2, -1, 0, 1, 2, 3])

# Get convolution matrices directly from PatternedLayer
eps_xx = layer.convolution_matrix(harmonics_x, harmonics_y, 'eps_xx')
eps_yy = layer.convolution_matrix(harmonics_x, harmonics_y, 'eps_yy')
eps_zz = layer.convolution_matrix(harmonics_x, harmonics_y, 'eps_zz')

# Matrix shape: (N_total, N_total) where N_total = len(harmonics_x) * len(harmonics_y)
assert eps_xx.shape == (49, 49)  # 7*7 = 49 total harmonics
assert np.all(np.isfinite(eps_xx))
```

## Utility Functions

Since PatternedLayer is now native, the adapter functions are simplified:

```python
from rcwa.core.adapters import suggest_harmonics_for_pattern

# Direct harmonics suggestion for patterned layers
suggested_harmonics = suggest_harmonics_for_pattern(
    patterned_layer, wavelength=1.5e-6, target_accuracy=0.01
)
# Returns (Nx, Ny) tuple
```

## Testing and Validation

The native architecture simplifies testing:

```python
def test_patterned_layer_is_layer():
    """Test that PatternedLayer is directly a Layer."""
    patterned_layer = PatternedLayer(...)
    
    # Direct Layer compatibility
    assert isinstance(patterned_layer, Layer)
    assert patterned_layer.thickness > 0
    assert not patterned_layer.homogenous
    
    # Native convolution matrix interface
    conv_matrix = patterned_layer.convolution_matrix(
        harmonics_x, harmonics_y, 'eps_xx'
    )
    assert isinstance(conv_matrix, np.ndarray)
    assert np.all(np.isfinite(conv_matrix))
```

## Migration from Adapter Architecture

If migrating from the previous adapter-based architecture:

**Old code:**
```python
from rcwa.core.adapters import PatternedLayerAdapter

patterned_layer = PatternedLayer(...)
converted_layer = PatternedLayerAdapter.patterned_to_layer(
    patterned_layer, harmonics, wavelength
)
solver.add_layer(converted_layer)
```

**New code:**
```python
# No adapter needed!
patterned_layer = PatternedLayer(...)
solver.add_layer(patterned_layer)  # Direct usage
```

## Performance Benefits

1. **No Conversion Overhead**: Direct Layer usage eliminates conversion time
2. **Memory Efficiency**: No duplicate layer objects
3. **Simpler Caching**: Single object with integrated caching
4. **Fewer Error Points**: Eliminates adapter-related failure modes

## Future Extensions

The native Layer architecture enables:

1. **Direct Solver Optimization**: Solvers can optimize specifically for patterned layers
2. **Advanced Caching**: Layer-level caching of computed properties
3. **Streaming Computation**: Direct integration with memory-efficient algorithms
4. **GPU Acceleration**: Native Layer interface can be GPU-accelerated

## Summary

The native Layer architecture provides:

- ✅ **Simplified API**: PatternedLayer works exactly like Layer
- ✅ **Better Performance**: No conversion overhead
- ✅ **Cleaner Code**: Eliminates adapter complexity  
- ✅ **Future Proof**: Direct solver integration enables optimizations
- ✅ **Backward Compatible**: Existing Layer-based code works unchanged

This architecture change makes geometry patterns first-class citizens in OpenRCWA while maintaining full compatibility with existing infrastructure.
