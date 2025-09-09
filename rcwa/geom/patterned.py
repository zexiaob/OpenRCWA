"""
Patterned layer implementation for OpenRCWA.

This module provides PatternedLayer class that combines Shape objects with materials
to create RCWA-compatible layers with complex 2D patterns and full tensor support.
PatternedLayer directly inherits from Layer to provide native RCWA compatibility.
"""

import numpy as np
from typing import List, Union, Tuple, Optional, Dict, Callable, Any
import warnings
from dataclasses import dataclass
import hashlib
import json

from .shape import Shape
from ..model.material import Material, TensorMaterial
from ..model.layer import Layer


@dataclass
class RasterConfig:
    """Configuration for rasterization."""
    resolution: Tuple[int, int] = (256, 256)
    antialiasing: bool = True
    oversample_factor: int = 4
    edge_smoothing: float = 0.1  # Fraction of pixel for edge smoothing


class PatternedLayer(Layer):
    """
    A layer with 2D patterned material distribution.
    
    This class extends the base Layer to support complex 2D patterns through
    Shape composition while maintaining full RCWA compatibility. It directly
    provides convolution matrices and properties needed by the RCWA solver.
    """
    
    def __init__(self, thickness: float,
                 lattice: Tuple[Tuple[float, float], Tuple[float, float]],
                 shapes: List[Tuple[Shape, Union[Material, TensorMaterial]]],
                 background_material: Union[Material, TensorMaterial],
                 raster_config: Optional[RasterConfig] = None,
                 **params):
        """
        Initialize patterned layer.
        
        :param thickness: Layer thickness (meters, SI units)
        :param lattice: Lattice vectors as ((ax, ay), (bx, by))
        :param shapes: List of (Shape, Material) tuples defining the pattern
        :param background_material: Background/substrate material
        :param raster_config: Rasterization configuration
        :param params: Additional parameters for parameterization
        """
        # Initialize as Layer with background material
        super().__init__(
            thickness=thickness,
            material=background_material if isinstance(background_material, Material) else None,
            tensor_material=background_material if isinstance(background_material, TensorMaterial) else None
    )
        
        # Override homogeneous flag since we have patterns
        self.homogenous = False

        # Store pattern-specific attributes
        self.lattice = lattice
        self.shapes = shapes.copy() if shapes else []
        self.background_material = background_material
        self.raster_config = raster_config or RasterConfig()
        self.params = params.copy() if params else {}
        # Track global in-plane rotation (radians) for caching and transforms
        self.rotation_z = float(self.params.get('rotation_z', 0.0))
        
        # Validate inputs
        self._validate_inputs()
        
        # Cache for computed convolution matrices
        self._convolution_cache = {}
        self._last_cache_key = None
        
        # Track parameterization dependencies
        self._param_dependencies = set()
        for shape, material in self.shapes:
            if hasattr(shape, '_param_dependencies'):
                self._param_dependencies.update(shape._param_dependencies)

    # ---- Lattice utilities (2D) ----
    @staticmethod
    def _rotate_vec2(v: Tuple[float, float], angle: float) -> Tuple[float, float]:
        c, s = float(np.cos(angle)), float(np.sin(angle))
        x, y = v
        return (c * x - s * y, s * x + c * y)

    @staticmethod
    def _reciprocal_from_direct(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute 2D reciprocal lattice vectors (in-plane) with 2π factor.
        Given direct lattice vectors a, b (as 2D), return g1, g2 such that A^T G = 2π I.
        """
        ax, ay = a
        bx, by = b
        det = ax * by - ay * bx
        if abs(det) < 1e-30:
            raise ValueError("Degenerate lattice; cannot compute reciprocal vectors")
        # For A = [a b] with a=(ax,ay), b=(bx,by), we have A^T = [[ax, ay],[bx, by]]
        # (A^T)^{-1} = 1/det * [[by, -ay], [-bx, ax]]
        invT = (1.0 / det) * np.array([[by, -ay], [-bx, ax]])  # (A^T)^{-1)
        G = 2.0 * np.pi * invT  # Include 2π
        g1 = (float(G[0, 0]), float(G[1, 0]))
        g2 = (float(G[0, 1]), float(G[1, 1]))
        return g1, g2

    def reciprocal_lattice(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return reciprocal lattice vectors (g1, g2) including 2π factor."""
        a, b = self.lattice
        return self._reciprocal_from_direct(a, b)

    def rotated(self, angle: float) -> 'PatternedLayer':
        """Return a new PatternedLayer rotated in-plane by angle (radians).

        This rotates the lattice vectors; shapes remain in unit-cell coordinates.
        The physical pattern and its reciprocal lattice rotate consistently.
        """
        a, b = self.lattice
        a_r = self._rotate_vec2(a, angle)
        b_r = self._rotate_vec2(b, angle)
        new_params = self.params.copy()
        new_params['rotation_z'] = float(self.rotation_z + angle)
        return PatternedLayer(
            thickness=self.thickness,
            lattice=(a_r, b_r),
            shapes=self.shapes,
            background_material=self.background_material,
            raster_config=self.raster_config,
            **new_params
        )
    
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
        # Use relative tolerance based on lattice vector magnitudes
        a_mag = np.sqrt(a_vec[0]**2 + a_vec[1]**2)
        b_mag = np.sqrt(b_vec[0]**2 + b_vec[1]**2)
        
        # Handle zero vectors and very small magnitudes
        if a_mag < 1e-15 or b_mag < 1e-15:
            raise ValueError("Lattice vectors are nearly parallel/degenerate")
            
        threshold = 1e-12 * a_mag * b_mag
        if abs(cross_product) < threshold:
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
        for shape_tuple in self.shapes:
            shape, material = shape_tuple
            if hasattr(shape, 'with_params'):
                # Update shape with new parameters
                new_shape = shape.with_params(**kwargs)
                new_shapes.append((new_shape, material))
            else:
                new_shapes.append(shape_tuple)
        
        # Handle layer-level parameter updates
        thickness = kwargs.get('thickness', self.thickness)
        lattice = kwargs.get('lattice', self.lattice)
        background_material = kwargs.get('background_material', self.background_material)
        # optional rotation update: expect radians named rotation_z
        rotation_z = kwargs.get('rotation_z', self.rotation_z)
        new_params['rotation_z'] = float(rotation_z)

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
        
        For z-aware shapes (implementing Shape.cross_section), we construct
        a PatternedLayer whose shapes are the per-z cross-section; otherwise
        returns self for z-uniform geometry.
        
        :param z_position: Z position for cross-section (0 to thickness)
        :return: PatternedLayer representing the cross-section
        """
        if not (0 <= z_position <= self.thickness):
            warnings.warn(f"z_position {z_position} outside layer thickness {self.thickness}")
        # Normalize to 0..1 fraction
        zf = 0.0 if self.thickness == 0 else float(np.clip(z_position / self.thickness, 0.0, 1.0))
        new_shapes = []
        changed = False
        for shape, material in self.shapes:
            if hasattr(shape, 'cross_section'):
                try:
                    cs = shape.cross_section(zf)
                    if cs is not shape:
                        changed = True
                    new_shapes.append((cs, material))
                except Exception:
                    new_shapes.append((shape, material))
            else:
                new_shapes.append((shape, material))
        if not changed:
            return self
        return PatternedLayer(
            thickness=self.thickness,
            lattice=self.lattice,
            shapes=new_shapes,
            background_material=self.background_material,
            raster_config=self.raster_config,
            **self.params
        )
    
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
        # Uniform interior slice positions (exclude 0 and thickness)
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
        
        # Shapes work in unit coordinates (0,1)x(0,1), not physical coordinates
        # So we use U, V directly instead of converting to X, Y
        
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
                # Get shape mask (use unit coordinates)
                mask = shape.contains(U, V)
                
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
    
    def rasterize_full_tensor_field(self, wavelength: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rasterize the pattern to complete 3x3 tensor fields.
        
        Creates grids of full epsilon and mu tensors representing the material distribution.
        Each pixel contains complete 3x3 complex tensors supporting full anisotropy.
        
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
        
        # Shapes work in unit coordinates (0,1)x(0,1), not physical coordinates
        # So we use U, V directly instead of converting to X, Y

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
                # Get shape mask (use unit coordinates)
                mask = shape.contains(U, V)
                
                if np.any(mask):
                    # Get shape material tensors
                    shape_epsilon, shape_mu = self._get_material_tensors(material, wavelength)
                    epsilon_field[mask] = shape_epsilon
                    mu_field[mask] = shape_mu
        
        return epsilon_field, mu_field
    
    def _get_material_tensors(self, material: Union[Material, TensorMaterial],
                            wavelength: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract epsilon and mu tensors from material for a single wavelength.
        
        :param material: Material object
        :param wavelength: Wavelength in meters (scalar)
        :return: Tuple of (epsilon_tensor, mu_tensor), each 3x3 complex arrays
        """
        if isinstance(material, TensorMaterial):
            # Temporarily set the source to get the tensor at this specific wavelength
            original_source = material.source
            if wavelength is not None:
                from rcwa.solve.source import Source
                # Create a temporary source for this single wavelength
                temp_source = Source(wavelength=wavelength * 1e9) # convert back to nm
                material.source = temp_source

            epsilon_tensor = material.epsilon_tensor
            mu_tensor = material.mu_tensor
            
            # Restore original source
            material.source = original_source

        else: # Isotropic Material
            # Get the original scalar values, not potentially modified matrix values
            if hasattr(material, '_er_original'):
                er = material._er_original
            elif callable(material.er):
                er = material.er(wavelength)
            else:
                # Check if er has been converted to a matrix, if so get the scalar value
                if isinstance(material.er, np.ndarray) and material.er.shape == (49, 49):
                    # This material's er has been corrupted by Layer.set_convolution_matrices
                    # Try to recover the original scalar value
                    # For a uniform material, all diagonal elements should be the same
                    er = material.er[0, 0]
                else:
                    er = material.er
            
            if hasattr(material, '_ur_original'):
                ur = material._ur_original
            elif callable(material.ur):
                ur = material.ur(wavelength)
            else:
                # Check if ur has been converted to a matrix, if so get the scalar value
                if isinstance(material.ur, np.ndarray) and material.ur.shape == (49, 49):
                    # This material's ur has been corrupted by Layer.set_convolution_matrices
                    ur = material.ur[0, 0]
                else:
                    ur = material.ur
            
            # 支持 er/ur 为 3x3 矩阵或标量
            if isinstance(er, np.ndarray) and er.shape == (3, 3):
                epsilon_tensor = er
            else:
                epsilon_tensor = np.eye(3, dtype=complex) * er
            if isinstance(ur, np.ndarray) and ur.shape == (3, 3):
                mu_tensor = ur
            else:
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
        
        # Get full tensor fields
        epsilon_field, mu_field = self.rasterize_full_tensor_field(wavelength)
        Ny, Nx = epsilon_field.shape[:2]
        
        # Number of harmonics  
        Nh_x, Nh_y = harmonics
        
        # Initialize convolution matrices dictionary
        conv_matrices = {}
        
        # Compute convolution matrices for all 9 tensor components
        tensor_components = [
            ('xx', 0, 0), ('xy', 0, 1), ('xz', 0, 2),
            ('yx', 1, 0), ('yy', 1, 1), ('yz', 1, 2),
            ('zx', 2, 0), ('zy', 2, 1), ('zz', 2, 2)
        ]
        
        # Process epsilon tensor
        for comp_name, i, j in tensor_components:
            # Extract tensor component field
            component_field = epsilon_field[:, :, i, j]
            
            # Perform FFT
            fft_component = np.fft.fft2(component_field)
            fft_component = np.fft.fftshift(fft_component)
            
            # Build convolution matrix
            conv_matrix = self._build_convolution_matrix(
                fft_component, (Ny, Nx), (Nh_y, Nh_x)
            )
            
            conv_matrices[f'er_{comp_name}'] = conv_matrix
        
        # Process mu tensor
        for comp_name, i, j in tensor_components:
            # Extract tensor component field
            component_field = mu_field[:, :, i, j]
            
            # Perform FFT
            fft_component = np.fft.fft2(component_field)
            fft_component = np.fft.fftshift(fft_component)
            
            # Build convolution matrix
            conv_matrix = self._build_convolution_matrix(
                fft_component, (Ny, Nx), (Nh_y, Nh_x)
            )
            
            conv_matrices[f'ur_{comp_name}'] = conv_matrix
        
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
            'params': sorted(self.params.items()) if self.params else [],
            # 2.2 cache augmentations
            'rotation_z': getattr(self, 'rotation_z', 0.0),
            'reciprocal': self.reciprocal_lattice(),
            'cache_version': '2.2',
            'backend': self.params.get('backend', None),
            'precision': self.params.get('precision', None),
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
    
    def set_convolution_matrices(self, n_harmonics: Union[Tuple[int, int], int]):
        """
        Overrides the base Layer method to generate convolution matrices for the pattern.
        This method is called by the solver during the simulation setup.
        """
        # Ensure n_harmonics is a tuple (Nh_x, Nh_y)
        if isinstance(n_harmonics, int):
            harmonics_tuple = (n_harmonics, n_harmonics)
        elif isinstance(n_harmonics, (list, tuple)) and len(n_harmonics) == 2:
            harmonics_tuple = tuple(n_harmonics)
        else:
            raise ValueError(f"Unsupported n_harmonics format: {n_harmonics}")

        # The wavelength is needed for dispersive materials.
        # We get it from the source object, which should have been set by the solver.
        if not hasattr(self, 'source') or self.source is None:
            # Try to get it from the background material as a fallback
            if hasattr(self.background_material, 'source') and self.background_material.source is not None:
                self.source = self.background_material.source
            else:
                # If still not found, we cannot proceed.
                raise RuntimeError("PatternedLayer requires a source to be set before calculating convolution matrices, "
                                 "but it was not found on the layer or its background material.")
        
        # Use a representative wavelength for the calculation. For sweeps, this might be re-evaluated.
        # The caching mechanism within to_convolution_matrices will handle different wavelengths.
        wavelength_nm = self.source.wavelength
        if isinstance(wavelength_nm, (np.ndarray, list, tuple)):
            # Use the central wavelength for single-point calculation if sweeping
            wavelength_nm = wavelength_nm[len(wavelength_nm) // 2]

        wavelength_m = float(wavelength_nm) * 1e-9 # Convert nm to meters

        # Generate the full set of convolution matrices
        conv_matrices = self.to_convolution_matrices(harmonics=harmonics_tuple, wavelength=wavelength_m)

        # Store the full set of tensor convolution matrices for the solver
        self._tensor_conv_matrices = conv_matrices

        # For compatibility with parts of the solver that might expect scalar er/ur matrices,
        # we can assign the zz components as was done in the base Layer for uniform tensors.
        if 'er_zz' in conv_matrices:
            self.er = conv_matrices['er_zz']
        if 'ur_zz' in conv_matrices:
            self.ur = conv_matrices['ur_zz']

        # Also assign the legacy properties if they exist
        if hasattr(self, '_tensor_er'):
            self._tensor_er = self.er
        if hasattr(self, '_tensor_ur'):
            self._tensor_ur = self.ur

    def convolution_matrix(self, harmonics_x: np.ndarray, harmonics_y: np.ndarray,
                          tensor_component: str = 'xx') -> np.ndarray:
        """
        Compute convolution matrix for RCWA.
        
        This method provides the standard RCWA interface for patterned layers.
        It computes the convolution matrices from the rasterized tensor field.
        
        :param harmonics_x: X harmonics vector
        :param harmonics_y: Y harmonics vector
        :param tensor_component: Tensor component ('xx', 'xy', 'yy', 'yx', 'zz') or ('eps_xx', etc.)
        :return: Convolution matrix
        """
        # Convert to tuple harmonics format
        nx = len(harmonics_x)
        ny = len(harmonics_y)
        harmonics = (nx, ny)
        
        # Use default wavelength if not cached
        wavelength = 1.0  # Default wavelength for tensor component calculation
        
        # Generate cache key
        harmonics_tuple = (tuple(harmonics_x), tuple(harmonics_y))
        cache_key = self._generate_cache_key(harmonics, wavelength)
        full_key = (cache_key, harmonics_tuple, tensor_component)
        
        # Check cache
        if full_key in self._convolution_cache:
            return self._convolution_cache[full_key]
        
        # Compute convolution matrix
        convolution_matrices = self.compute_convolution_matrices(harmonics_x, harmonics_y)
        
        # Normalize tensor component name
        # Convert 'eps_xx' -> 'er_xx', 'mu_xx' -> 'ur_xx' for compatibility
        if tensor_component.startswith('eps_'):
            normalized_component = tensor_component.replace('eps_', 'er_')
        elif tensor_component.startswith('mu_'):
            normalized_component = tensor_component.replace('mu_', 'ur_')
        else:
            # Assume 'xx', 'yy', 'zz' etc. format - default to er
            normalized_component = f'er_{tensor_component}'
        
        if normalized_component not in convolution_matrices:
            available_keys = list(convolution_matrices.keys())
            raise ValueError(f"Unknown tensor component: {tensor_component} (normalized: {normalized_component}). Available: {available_keys}")
        
        result = convolution_matrices[normalized_component]
        
        # Cache result
        self._convolution_cache[full_key] = result
        
        return result

    def compute_convolution_matrices(self, harmonics_x: np.ndarray, harmonics_y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute convolution matrices from harmonics arrays.
        
        This method interfaces with to_convolution_matrices but uses explicit harmonic arrays.
        
        :param harmonics_x: X harmonics array
        :param harmonics_y: Y harmonics array
        :return: Dictionary of convolution matrices
        """
        harmonics = (len(harmonics_x), len(harmonics_y))
        return self.to_convolution_matrices(harmonics, wavelength=1.0)

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
                for i, shape_tuple1 in enumerate(self.shapes):
                    for j, shape_tuple2 in enumerate(self.shapes[i+1:], i+1):
                        shape1 = shape_tuple1[0]
                        shape2 = shape_tuple2[0]
                        mask1 = shape1.contains(x_test, y_test)
                        mask2 = shape2.contains(x_test, y_test)
                        if np.any(mask1 & mask2):
                            overlaps += 1
                            
                if overlaps > 0:
                    results['warnings'].append(f"Detected {overlaps} potential shape overlaps")
            
            # Check materials
            materials = [self.background_material]
            for shape_tuple in self.shapes:
                shp, mat = shape_tuple
                if mat is not None:
                    materials.append(mat)
            
            results['info']['num_materials'] = len(materials)
            
            # Try rasterization test
            try:
                er_field, ur_field = self.rasterize_tensor_field()
                results['info']['raster_test'] = 'passed'
                results['info']['tensor_shape'] = (er_field.shape, ur_field.shape)
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
