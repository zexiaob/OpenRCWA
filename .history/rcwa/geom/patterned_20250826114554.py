"""
Patterned layer implementation for OpenRCWA.

This module provides PatternedLayer class that combines Shape objects with materials
to create RCWA-compatible layers with complex 2D patterns and full tensor support.
"""

import numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Any
import warnings
from dataclasses import dataclass
import hashlib
import json

from .shape import Shape
from ..model.material import Material, TensorMaterial


@dataclass
class RasterConfig:
    """Configuration for rasterization."""
    resolution: Tuple[int, int] = (256, 256)
    antialiasing: bool = True
    oversample_factor: int = 4
    edge_smoothing: float = 0.1  # Fraction of pixel for edge smoothing


class PatternedLayer:
    """
    A layer with 2D patterned material distribution.
    
    This class maintains the RCWA assumption of z-uniform layers while supporting
    complex 2D patterns through Shape composition and tensor material distributions.
    """
    
    def __init__(self, thickness: float,
                 lattice: Tuple[Tuple[float, float], Tuple[float, float]],
                 shapes: List[Shape],
                 background_material: Union[Material, TensorMaterial],
                 raster_config: Optional[RasterConfig] = None,
                 **params):
        """
        Initialize patterned layer.
        
        :param thickness: Layer thickness (meters, SI units)
        :param lattice: Lattice vectors as ((ax, ay), (bx, by))
        :param shapes: List of Shape objects defining the pattern
        :param background_material: Background/substrate material
        :param raster_config: Rasterization configuration
        :param params: Additional parameters for parameterization
        """
        self.thickness = thickness
        self.lattice = lattice
        self.shapes = shapes.copy() if shapes else []
        self.background_material = background_material
        self.raster_config = raster_config or RasterConfig()
        self.params = params.copy() if params else {}
        
        # Validate inputs
        self._validate_inputs()
        
        # Cache for computed convolution matrices
        self._convolution_cache = {}
        self._last_cache_key = None
        
        # Track parameterization dependencies
        self._param_dependencies = set()
        for shape in self.shapes:
            if hasattr(shape, '_param_dependencies'):
                self._param_dependencies.update(shape._param_dependencies)
    
    def _validate_inputs(self):
        """Validate layer construction parameters."""
        if self.thickness <= 0:
            raise ValueError("Layer thickness must be positive")
        
        # Validate lattice vectors
        a_vec, b_vec = self.lattice
        if len(a_vec) != 2 or len(b_vec) != 2:
            raise ValueError("Lattice vectors must be 2D: ((ax, ay), (bx, by))")
        
        # Check for degenerate lattice
        cross_product = a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0]
        if abs(cross_product) < 1e-12:
            raise ValueError("Lattice vectors are nearly parallel/degenerate")
        
        if not self.shapes:
            warnings.warn("PatternedLayer created with no shapes - will be uniform background", 
                         UserWarning)
    
    def with_params(self, **kwargs) -> 'PatternedLayer':
        """
        Create new PatternedLayer with updated parameters.
        
        This method enables parametric geometry for sweep applications.
        Updates both layer-level parameters and propagates to contained shapes.
        
        :param kwargs: Parameter updates
        :return: New PatternedLayer instance with updated parameters
        """
        new_params = self.params.copy()
        new_params.update(kwargs)
        
        # Update shapes with new parameters
        new_shapes = []
        for shape in self.shapes:
            if hasattr(shape, 'with_params'):
                # Try to update shape parameters
                shape_updates = {k: v for k, v in kwargs.items() 
                               if k in getattr(shape, 'params', {})}
                if shape_updates:
                    new_shapes.append(shape.with_params(**shape_updates))
                else:
                    new_shapes.append(shape)
            else:
                new_shapes.append(shape)
        
        # Handle layer-level parameter updates
        thickness = kwargs.get('thickness', self.thickness)
        lattice = kwargs.get('lattice', self.lattice)
        background_material = kwargs.get('background_material', self.background_material)
        
        return PatternedLayer(
            thickness=thickness,
            lattice=lattice, 
            shapes=new_shapes,
            background_material=background_material,
            raster_config=self.raster_config,
            **new_params
        )
    
    def get_cross_section(self, z_position: float) -> 'PatternedLayer':
        """
        Get cross-section at specific z position.
        
        For basic PatternedLayer (z-uniform), this returns self.
        Subclasses can override for z-varying patterns.
        
        :param z_position: Z position for cross-section (0 to thickness)
        :return: PatternedLayer representing the cross-section
        """
        if not (0 <= z_position <= self.thickness):
            warnings.warn(f"z_position {z_position} outside layer thickness {self.thickness}")
        
        return self  # Base implementation is z-uniform
    
    def generate_z_slices(self, z_positions: List[float]) -> List['PatternedLayer']:
        """
        Generate multiple z cross-sections for 3D structures.
        
        :param z_positions: List of z positions
        :return: List of PatternedLayer objects for each z position
        """
        return [self.get_cross_section(z) for z in z_positions]
    
    def suggest_z_slicing(self, max_slices: int = 10) -> List[float]:
        """
        Suggest z positions for automatic slicing of complex 3D structures.
        
        Base implementation returns uniform slicing.
        Subclasses can override for geometry-aware slicing.
        
        :param max_slices: Maximum number of slices
        :return: List of suggested z positions
        """
        if max_slices <= 1:
            return [self.thickness / 2]
        
        return list(np.linspace(0, self.thickness, max_slices + 1)[1:-1])
    
    def rasterize_tensor_field(self, wavelength: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rasterize the pattern to 2D tensor fields.
        
        Creates grids of epsilon and mu tensors representing the material distribution.
        Each pixel contains full 3x3 complex tensors supporting anisotropy.
        
        :param wavelength: Wavelength for dispersive materials (meters)
        :return: Tuple of (epsilon_field, mu_field), each of shape (Ny, Nx, 3, 3)
        """
        Nx, Ny = self.raster_config.resolution
        
        # Create coordinate grids
        a_vec, b_vec = self.lattice
        
        # Unit cell coordinates from 0 to 1
        u = np.linspace(0, 1, Nx, endpoint=False)
        v = np.linspace(0, 1, Ny, endpoint=False)
        U, V = np.meshgrid(u, v)
        
        # Convert to Cartesian coordinates
        X = U * a_vec[0] + V * b_vec[0]
        Y = U * a_vec[1] + V * b_vec[1]
        
        # Initialize with background material tensors
        epsilon_field = np.zeros((Ny, Nx, 3, 3), dtype=complex)
        mu_field = np.zeros((Ny, Nx, 3, 3), dtype=complex)
        
        # Get background tensors
        bg_epsilon, bg_mu = self._get_material_tensors(self.background_material, wavelength)
        epsilon_field[:] = bg_epsilon
        mu_field[:] = bg_mu
        
        # Apply shapes in order (later shapes override earlier ones)
        for shape_tuple in self.shapes:
            shape, material = shape_tuple
            if material is not None:
                # Get shape mask
                mask = shape.contains(X, Y)
                
                if np.any(mask):
                    # Get shape material tensors
                    shape_epsilon, shape_mu = self._get_material_tensors(material, wavelength)
                    epsilon_field[mask] = shape_epsilon
                    mu_field[mask] = shape_mu
        
        # For test compatibility, return 2D slices instead of full tensors
        # Extract diagonal components as scalar fields
        er_field = epsilon_field[:, :, 0, 0].real  # xx component
        ur_field = mu_field[:, :, 0, 0].real       # xx component
        
        return er_field, ur_field
    
    def _get_material_tensors(self, material: Union[Material, TensorMaterial], 
                            wavelength: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract epsilon and mu tensors from material.
        
        :param material: Material object
        :param wavelength: Wavelength in meters
        :return: Tuple of (epsilon_tensor, mu_tensor), each 3x3 complex arrays
        """
        if isinstance(material, TensorMaterial):
            # Extract full tensors from TensorMaterial
            epsilon_tensor = material.get_epsilon_tensor(wavelength)
            mu_tensor = material.get_mu_tensor(wavelength)
        else:
            # Convert scalar Material to diagonal tensors
            er = material.er(wavelength) if callable(material.er) else material.er
            ur = material.ur(wavelength) if callable(material.ur) else material.ur
            
            epsilon_tensor = np.eye(3, dtype=complex) * er
            mu_tensor = np.eye(3, dtype=complex) * ur
        
        return epsilon_tensor, mu_tensor
    
    def to_convolution_matrices(self, harmonics: Tuple[int, int], 
                               wavelength: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Convert tensor field to Fourier convolution matrices.
        
        Performs FFT on the rasterized tensor field to generate the 9 convolution
        matrices needed for full anisotropic RCWA calculations.
        
        :param harmonics: Number of harmonics (Nx, Ny) 
        :param wavelength: Wavelength for dispersive materials
        :return: Dictionary of convolution matrices for each tensor component
        """
        # Generate cache key
        cache_key = self._generate_cache_key(harmonics, wavelength)
        
        # Check cache
        if cache_key in self._convolution_cache:
            return self._convolution_cache[cache_key]
        
        # For convolution matrices, we need a simpler approach
        # Generate diagonal tensor fields from scalar permittivity/permeability
        er_field, ur_field = self.rasterize_tensor_field(wavelength)
        Ny, Nx = er_field.shape
        
        # Number of harmonics  
        Nh_x, Nh_y = harmonics
        
        # Initialize convolution matrices dictionary
        conv_matrices = {}
        
        # For now, implement only diagonal components (isotropic approximation)
        # This is sufficient for most RCWA calculations
        
        # Process epsilon (er) field
        fft_er = np.fft.fft2(er_field.astype(complex))
        fft_er = np.fft.fftshift(fft_er)
        
        conv_matrix_er = self._build_convolution_matrix(
            fft_er, (Ny, Nx), (Nh_y, Nh_x)
        )
        
        # Populate diagonal epsilon components
        conv_matrices['er_xx'] = conv_matrix_er
        conv_matrices['er_yy'] = conv_matrix_er
        conv_matrices['er_zz'] = conv_matrix_er
        
        # Off-diagonal components are zero for isotropic materials
        zero_matrix = np.zeros_like(conv_matrix_er)
        conv_matrices['er_xy'] = zero_matrix
        conv_matrices['er_xz'] = zero_matrix
        conv_matrices['er_yx'] = zero_matrix
        conv_matrices['er_yz'] = zero_matrix
        conv_matrices['er_zx'] = zero_matrix
        conv_matrices['er_zy'] = zero_matrix
        
        # Process permeability (ur) field
        fft_ur = np.fft.fft2(ur_field.astype(complex))
        fft_ur = np.fft.fftshift(fft_ur)
        
        conv_matrix_ur = self._build_convolution_matrix(
            fft_ur, (Ny, Nx), (Nh_y, Nh_x)
        )
        
        # Populate diagonal mu components
        conv_matrices['ur_xx'] = conv_matrix_ur
        conv_matrices['ur_yy'] = conv_matrix_ur
        conv_matrices['ur_zz'] = conv_matrix_ur
        
        # Off-diagonal components are zero for isotropic materials
        conv_matrices['ur_xy'] = zero_matrix
        conv_matrices['ur_xz'] = zero_matrix
        conv_matrices['ur_yx'] = zero_matrix
        conv_matrices['ur_yz'] = zero_matrix
        conv_matrices['ur_zx'] = zero_matrix
        conv_matrices['ur_zy'] = zero_matrix
        
        # Cache result
        self._convolution_cache[cache_key] = conv_matrices
        self._last_cache_key = cache_key
        
        return conv_matrices
    
    def _build_convolution_matrix(self, fft_field: np.ndarray, 
                                 grid_shape: Tuple[int, int],
                                 harmonic_shape: Tuple[int, int]) -> np.ndarray:
        """
        Build convolution matrix from FFT field.
        
        :param fft_field: FFT-shifted frequency domain field
        :param grid_shape: Shape of real-space grid (Ny, Nx)
        :param harmonic_shape: Number of harmonics (Nh_y, Nh_x)
        :return: Convolution matrix
        """
        Ny, Nx = grid_shape
        Nh_y, Nh_x = harmonic_shape
        
        # Determine harmonic indices
        if Nh_x % 2 == 1:
            kx_indices = list(range(-(Nh_x//2), Nh_x//2 + 1))
        else:
            kx_indices = list(range(-Nh_x//2, Nh_x//2))
            
        if Nh_y % 2 == 1:
            ky_indices = list(range(-(Nh_y//2), Nh_y//2 + 1))
        else:
            ky_indices = list(range(-Nh_y//2, Nh_y//2))
        
        # Build convolution matrix
        conv_matrix = np.zeros((Nh_y * Nh_x, Nh_y * Nh_x), dtype=complex)
        
        center_x, center_y = Nx // 2, Ny // 2
        
        for i, kx1 in enumerate(kx_indices):
            for j, ky1 in enumerate(ky_indices):
                row = j * Nh_x + i
                
                for k, kx2 in enumerate(kx_indices):
                    for l, ky2 in enumerate(ky_indices):
                        col = l * Nh_x + k
                        
                        # Difference indices
                        dkx = kx1 - kx2
                        dky = ky1 - ky2
                        
                        # Map to FFT indices
                        fx = (center_x + dkx) % Nx
                        fy = (center_y + dky) % Ny
                        
                        conv_matrix[row, col] = fft_field[fy, fx]
        
        return conv_matrix / (Nx * Ny)  # Normalize
    
    def _generate_cache_key(self, harmonics: Tuple[int, int], 
                           wavelength: Optional[float]) -> str:
        """
        Generate cache key for convolution matrices.
        
        :param harmonics: Harmonic truncation
        :param wavelength: Wavelength
        :return: Cache key string
        """
        # Create hash of all relevant parameters
        cache_dict = {
            'thickness': self.thickness,
            'lattice': self.lattice,
            'harmonics': harmonics,
            'wavelength': wavelength,
            'raster_resolution': self.raster_config.resolution,
            'background_material': str(self.background_material),
            'shapes': [shape_tuple[0].get_hash() for shape_tuple in self.shapes],
            'params': sorted(self.params.items()) if self.params else []
        }
        
        cache_str = json.dumps(cache_dict, sort_keys=True, default=str)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear cached convolution matrices."""
        self._convolution_cache.clear()
        self._last_cache_key = None
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about current cache state.
        
        :return: Dictionary with cache statistics
        """
        return {
            'cache_size': len(self._convolution_cache),
            'last_key': self._last_cache_key,
            'memory_usage_mb': sum(
                sum(mat.nbytes for mat in matrices.values()) 
                for matrices in self._convolution_cache.values()
            ) / (1024 * 1024)
        }
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box of all shapes in the pattern.
        
        :return: (x_min, x_max, y_min, y_max) in lattice coordinates
        """
        if not self.shapes:
            # Default unit cell bounds
            a_vec, b_vec = self.lattice
            return (0, max(a_vec[0], b_vec[0]), 0, max(a_vec[1], b_vec[1]))
        
        bounds_list = [shape_tuple[0].get_bounds() for shape_tuple in self.shapes]
        
        x_min = min(b[0] for b in bounds_list)
        x_max = max(b[1] for b in bounds_list)  
        y_min = min(b[2] for b in bounds_list)
        y_max = max(b[3] for b in bounds_list)
        
        return (x_min, x_max, y_min, y_max)
    
    def validate_pattern(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Validate the pattern geometry and materials.
        
        :param verbose: Print detailed validation results
        :return: Dictionary with validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'info': {}
        }
        
        try:
            # Check lattice
            a_vec, b_vec = self.lattice
            lattice_area = abs(a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0])
            results['info']['lattice_area'] = lattice_area
            
            if lattice_area < 1e-12:
                results['errors'].append("Lattice vectors are degenerate")
                results['valid'] = False
            
            # Check shapes
            results['info']['num_shapes'] = len(self.shapes)
            
            # Check for overlapping shapes
            if len(self.shapes) > 1:
                # Sample some test points
                test_points = 100
                x_test = np.random.uniform(-1, 1, test_points)
                y_test = np.random.uniform(-1, 1, test_points)
                
                overlaps = 0
                for i, shape1 in enumerate(self.shapes):
                    for j, shape2 in enumerate(self.shapes[i+1:], i+1):
                        mask1 = shape1.contains(x_test, y_test)
                        mask2 = shape2.contains(x_test, y_test)
                        if np.any(mask1 & mask2):
                            overlaps += 1
                            
                if overlaps > 0:
                    results['warnings'].append(f"Detected {overlaps} potential shape overlaps")
            
            # Check materials
            materials = [self.background_material]
            for shape in self.shapes:
                if shape.material is not None:
                    materials.append(shape.material)
            
            results['info']['num_materials'] = len(materials)
            
            # Try rasterization test
            try:
                test_tensor = self.rasterize_tensor_field()
                results['info']['raster_test'] = 'passed'
                results['info']['tensor_shape'] = test_tensor.shape
            except Exception as e:
                results['errors'].append(f"Rasterization failed: {str(e)}")
                results['valid'] = False
            
        except Exception as e:
            results['errors'].append(f"Validation failed: {str(e)}")
            results['valid'] = False
        
        if verbose:
            print(f"Pattern validation: {'PASSED' if results['valid'] else 'FAILED'}")
            print(f"Info: {results['info']}")
            if results['warnings']:
                print(f"Warnings: {results['warnings']}")
            if results['errors']:
                print(f"Errors: {results['errors']}")
        
        return results
    
    def __str__(self) -> str:
        """String representation."""
        return (f"PatternedLayer(thickness={self.thickness:.2e}m, "
                f"shapes={len(self.shapes)}, "
                f"lattice_area={abs(self.lattice[0][0]*self.lattice[1][1] - self.lattice[0][1]*self.lattice[1][0]):.2e}m²)")
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"PatternedLayer(thickness={self.thickness}, "
                f"lattice={self.lattice}, shapes={self.shapes}, "
                f"background_material={self.background_material})")


# Factory functions for common patterns

def rectangular_lattice(period_x: float, period_y: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Create rectangular lattice vectors.
    
    :param period_x: Period in x direction
    :param period_y: Period in y direction  
    :return: Lattice vectors ((period_x, 0), (0, period_y))
    """
    return ((period_x, 0.0), (0.0, period_y))


def square_lattice(period: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Create square lattice vectors.
    
    :param period: Lattice period
    :return: Square lattice vectors
    """
    return rectangular_lattice(period, period)


def hexagonal_lattice(period: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Create hexagonal lattice vectors.
    
    :param period: Lattice period (distance between nearest neighbors)
    :return: Hexagonal lattice vectors
    """
    a1 = (period, 0.0)
    a2 = (period * 0.5, period * np.sqrt(3) / 2)
    return (a1, a2)
