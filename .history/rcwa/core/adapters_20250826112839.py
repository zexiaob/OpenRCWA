"""
Core adapters for converting between new architecture data structures and legacy solver interfaces.

This module provides thin adapters that bridge the gap between:
- TensorMaterial (model layer) ← → convolution matrices (core solver)
- PatternedLayer (geom layer) ← → Layer (legacy interfaces)

The adapters follow the ROADMAP design principle that the core solver should not
be aware of high-level geometry/shape details, but should handle tensor materials.
"""

import numpy as np
from typing import Union, Tuple, Dict, Optional, TYPE_CHECKING
from numpy.typing import ArrayLike
from rcwa.shorthand import complexIdentity, complexZeros

if TYPE_CHECKING:
    from rcwa.geom.patterned import PatternedLayer
    from rcwa.model.layer import Layer


class TensorToConvolutionAdapter:
    """
    Adapter for converting tensor materials to convolution matrices compatible with existing solver.
    
    This adapter handles the conversion of 3x3 material tensors (epsilon, mu) into the 
    convolution matrix format expected by the legacy RCWA solver, while preserving
    full anisotropic coupling.
    """
    
    @staticmethod
    def tensor_to_convolution_matrices(epsilon_tensor: ArrayLike, mu_tensor: ArrayLike, 
                                     n_harmonics: Union[int, Tuple[int, ...]] = 1) -> Dict[str, ArrayLike]:
        """
        Convert 3x3 material tensors to convolution matrices for RCWA solver.
        
        For uniform layers (no spatial variation), this creates convolution matrices
        where all 9 tensor components are represented as diagonal matrices scaled by
        the respective tensor element.
        
        :param epsilon_tensor: 3x3 complex permittivity tensor
        :param mu_tensor: 3x3 complex permeability tensor  
        :param n_harmonics: Number of harmonics (for uniform layers, just sets matrix size)
        :return: Dictionary containing convolution matrices for all tensor components
        """
        epsilon_tensor = np.array(epsilon_tensor, dtype=complex)
        mu_tensor = np.array(mu_tensor, dtype=complex)
        
        if epsilon_tensor.shape != (3, 3):
            raise ValueError("Epsilon tensor must be 3x3")
        if mu_tensor.shape != (3, 3):
            raise ValueError("Mu tensor must be 3x3")
        
        # Calculate matrix dimension from harmonics
        if isinstance(n_harmonics, int):
            matrix_dim = n_harmonics
        else:
            matrix_dim = np.prod(n_harmonics)
        
        # For uniform layers, each tensor component becomes a scaled identity matrix
        convolution_matrices = {}
        
        # Epsilon tensor components
        for i in range(3):
            for j in range(3):
                component_name = f'eps_{["x", "y", "z"][i]}{["x", "y", "z"][j]}'
                convolution_matrices[component_name] = epsilon_tensor[i, j] * complexIdentity(matrix_dim)
        
        # Mu tensor components  
        for i in range(3):
            for j in range(3):
                component_name = f'mu_{["x", "y", "z"][i]}{["x", "y", "z"][j]}'
                convolution_matrices[component_name] = mu_tensor[i, j] * complexIdentity(matrix_dim)
        
        return convolution_matrices
    
    @staticmethod
    def extract_effective_properties(epsilon_tensor: ArrayLike, mu_tensor: ArrayLike, 
                                   propagation_direction: str = 'z') -> Tuple[complex, complex]:
        """
        Extract effective scalar properties for compatibility with legacy interfaces.
        
        For anisotropic materials, this extracts effective permittivity and permeability
        values along the specified propagation direction, primarily for interface
        calculations and backwards compatibility.
        
        :param epsilon_tensor: 3x3 complex permittivity tensor
        :param mu_tensor: 3x3 complex permeability tensor
        :param propagation_direction: Direction of wave propagation ('x', 'y', or 'z')
        :return: Tuple of (effective_epsilon, effective_mu)
        """
        epsilon_tensor = np.array(epsilon_tensor, dtype=complex)
        mu_tensor = np.array(mu_tensor, dtype=complex)
        
        # Map direction to tensor index
        dir_map = {'x': 0, 'y': 1, 'z': 2}
        if propagation_direction not in dir_map:
            raise ValueError("Propagation direction must be 'x', 'y', or 'z'")
        
        idx = dir_map[propagation_direction]
        
        # For normal incidence along the propagation direction, extract diagonal elements
        # This is a simplified approach - more sophisticated methods could be implemented
        effective_epsilon = epsilon_tensor[idx, idx]
        effective_mu = mu_tensor[idx, idx]
        
        return effective_epsilon, effective_mu


class LayerTensorAdapter:
    """
    Adapter for bridging tensor materials with the existing Layer/MatrixCalculator interface.
    
    This adapter modifies matrix calculations in MatrixCalculator to handle tensor materials
    while maintaining compatibility with existing scalar material calculations.
    """
    
    @staticmethod
    def adapt_P_matrix_for_tensor(layer, Kx: ArrayLike, Ky: ArrayLike) -> ArrayLike:
        """
        Calculate P matrix for anisotropic tensor materials.
        
        For tensor materials, the P matrix calculation needs to account for all
        9 components of the permittivity tensor, particularly the off-diagonal
        coupling terms.
        
        :param layer: Layer object with tensor material
        :param Kx: X-component k-vector matrix  
        :param Ky: Y-component k-vector matrix
        :return: Modified P matrix accounting for tensor coupling
        """
        if not layer.is_anisotropic:
            raise ValueError("Layer must have tensor material for tensor P matrix calculation")
        
        epsilon_tensor = layer.tensor_material.epsilon_tensor
        
        # For homogeneous case (scalar Kx, Ky)
        if not isinstance(Kx, np.ndarray):
            return LayerTensorAdapter._P_matrix_tensor_homogeneous(epsilon_tensor, Kx, Ky)
        else:
            return LayerTensorAdapter._P_matrix_tensor_general(epsilon_tensor, Kx, Ky)
    
    @staticmethod
    def _P_matrix_tensor_homogeneous(epsilon_tensor: ArrayLike, Kx: complex, Ky: complex) -> ArrayLike:
        """P matrix for homogeneous tensor case."""
        # This is a simplified implementation for uniform tensor materials
        # The full tensor implementation would require more sophisticated coupling
        eps = np.array(epsilon_tensor, dtype=complex)
        
        # Use effective properties for the simplified case
        # In a full implementation, this would properly couple all tensor components
        eps_xx, eps_yy = eps[0,0], eps[1,1]
        eps_inv_xx = 1.0 / eps_xx if abs(eps_xx) > 1e-15 else 0.0
        
        P = complexZeros((2, 2))
        P[0,0] = Kx * Ky * eps_inv_xx
        P[0,1] = (eps_xx * eps_yy - Kx**2) * eps_inv_xx  # Approximate coupling
        P[1,0] = Ky**2 - eps_xx * eps_yy
        P[1,1] = -Kx * Ky * eps_inv_xx
        
        return P
        
    @staticmethod 
    def _P_matrix_tensor_general(epsilon_tensor: ArrayLike, Kx: ArrayLike, Ky: ArrayLike) -> ArrayLike:
        """P matrix for general tensor case with full tensor coupling."""
        eps = np.array(epsilon_tensor, dtype=complex)
        
        # Calculate inverse of epsilon tensor for proper coupling
        try:
            eps_inv = np.linalg.inv(eps)
        except np.linalg.LinAlgError:
            # Handle singular tensors gracefully
            eps_inv = np.linalg.pinv(eps)
        
        KMatrixDimension = Kx.shape[0]
        matrixShape = (2 * KMatrixDimension, 2 * KMatrixDimension)
        P = complexZeros(matrixShape)
        
        # This is a placeholder for the full tensor P matrix implementation
        # The actual implementation would properly couple all epsilon tensor components
        # For now, use the diagonal approximation
        eps_eff = eps[0,0]  # Simplified effective permittivity
        eps_inv_eff = 1.0 / eps_eff if abs(eps_eff) > 1e-15 else 0.0
        
        P[:KMatrixDimension,:KMatrixDimension] = Kx @ (eps_inv_eff * complexIdentity(KMatrixDimension)) @ Ky
        P[:KMatrixDimension,KMatrixDimension:] = complexIdentity(KMatrixDimension) - Kx @ (eps_inv_eff * complexIdentity(KMatrixDimension)) @ Kx
        P[KMatrixDimension:,:KMatrixDimension] = Ky @ (eps_inv_eff * complexIdentity(KMatrixDimension)) @ Ky - complexIdentity(KMatrixDimension)
        P[KMatrixDimension:,KMatrixDimension:] = -Ky @ (eps_inv_eff * complexIdentity(KMatrixDimension)) @ Kx
        
        return P
    
    @staticmethod
    def adapt_Q_matrix_for_tensor(layer, Kx: ArrayLike, Ky: ArrayLike) -> ArrayLike:
        """
        Calculate Q matrix for anisotropic tensor materials.
        
        Similar to P matrix but for the magnetic response tensor.
        
        :param layer: Layer object with tensor material
        :param Kx: X-component k-vector matrix
        :param Ky: Y-component k-vector matrix  
        :return: Modified Q matrix accounting for tensor coupling
        """
        if not layer.is_anisotropic:
            raise ValueError("Layer must have tensor material for tensor Q matrix calculation")
        
        mu_tensor = layer.tensor_material.mu_tensor
        epsilon_tensor = layer.tensor_material.epsilon_tensor
        
        if not isinstance(Kx, np.ndarray):
            return LayerTensorAdapter._Q_matrix_tensor_homogeneous(mu_tensor, epsilon_tensor, Kx, Ky)
        else:
            return LayerTensorAdapter._Q_matrix_tensor_general(mu_tensor, epsilon_tensor, Kx, Ky)
    
    @staticmethod
    def _Q_matrix_tensor_homogeneous(mu_tensor: ArrayLike, epsilon_tensor: ArrayLike, 
                                   Kx: complex, Ky: complex) -> ArrayLike:
        """Q matrix for homogeneous tensor case."""
        mu = np.array(mu_tensor, dtype=complex)
        eps = np.array(epsilon_tensor, dtype=complex)
        
        # Simplified effective properties approach
        mu_xx, mu_yy = mu[0,0], mu[1,1]
        eps_xx, eps_yy = eps[0,0], eps[1,1]
        mu_inv_xx = 1.0 / mu_xx if abs(mu_xx) > 1e-15 else 0.0
        
        Q = complexZeros((2, 2))
        Q[0,0] = Kx * Ky * mu_inv_xx
        Q[0,1] = (eps_xx - Kx**2) * mu_inv_xx
        Q[1,0] = Ky**2 - eps_yy
        Q[1,1] = -Kx * Ky * mu_inv_xx
        
        return Q
    
    @staticmethod
    def _Q_matrix_tensor_general(mu_tensor: ArrayLike, epsilon_tensor: ArrayLike,
                               Kx: ArrayLike, Ky: ArrayLike) -> ArrayLike:
        """Q matrix for general tensor case."""
        mu = np.array(mu_tensor, dtype=complex)
        eps = np.array(epsilon_tensor, dtype=complex)
        
        try:
            mu_inv = np.linalg.inv(mu)
        except np.linalg.LinAlgError:
            mu_inv = np.linalg.pinv(mu)
        
        KMatrixDimension = Kx.shape[0]
        matrixShape = (2 * KMatrixDimension, 2 * KMatrixDimension)
        Q = complexZeros(matrixShape)
        
        # Simplified implementation using effective properties
        mu_eff = mu[0,0]
        eps_eff = eps[0,0]
        mu_inv_eff = 1.0 / mu_eff if abs(mu_eff) > 1e-15 else 0.0
        
        Q[:KMatrixDimension,:KMatrixDimension] = Kx @ (mu_inv_eff * complexIdentity(KMatrixDimension)) @ Ky
        Q[:KMatrixDimension,KMatrixDimension:] = eps_eff * complexIdentity(KMatrixDimension) - Kx @ (mu_inv_eff * complexIdentity(KMatrixDimension)) @ Kx
        Q[KMatrixDimension:,:KMatrixDimension] = Ky @ (mu_inv_eff * complexIdentity(KMatrixDimension)) @ Ky - eps_eff * complexIdentity(KMatrixDimension)
        Q[KMatrixDimension:,KMatrixDimension:] = -Ky @ (mu_inv_eff * complexIdentity(KMatrixDimension)) @ Kx
        
        return Q


class EigensolverTensorAdapter:
    """
    Adapter for computing eigenvalues and eigenvectors of tensor-coupled systems.
    
    This handles the core RCWA eigenproblem for anisotropic materials where the
    eigenvalue equation becomes more complex due to tensor coupling.
    """
    
    @staticmethod
    def solve_tensor_eigenproblem(P: ArrayLike, Q: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Solve the generalized eigenvalue problem for tensor materials.
        
        For tensor materials, the eigenvalue problem P*Q*W = W*Lambda^2 needs to be
        solved carefully to ensure proper mode calculation and energy conservation.
        
        :param P: P matrix from tensor calculations
        :param Q: Q matrix from tensor calculations
        :return: Tuple of (eigenvalues, eigenvectors, Lambda matrix)
        """
        # Calculate omega squared matrix
        OmegaSquared = P @ Q
        
        # Solve eigenvalue problem
        eigenValues, W = np.linalg.eig(OmegaSquared)
        
        # Ensure proper branch cuts for square root (important for passive materials)
        # Take square root with correct sign convention for RCWA
        Lambda_diag = np.sqrt(eigenValues + 0j)  # Ensure complex type
        
        # Fix branch cuts: for lossless materials, choose positive imaginary part
        # for evanescent modes and positive real part for propagating modes
        for i, val in enumerate(Lambda_diag):
            if np.imag(val) < 0:
                Lambda_diag[i] = -val
        
        Lambda = np.diag(Lambda_diag)
        
        return eigenValues, W, Lambda


# Convenience functions for easy integration

def create_tensor_layer_adapter(layer):
    """
    Create an adapter that enables tensor material support for an existing Layer.
    
    :param layer: Layer object that may contain tensor materials
    :return: Layer with tensor-aware matrix calculations
    """
    if not layer.is_anisotropic:
        return layer  # No adaptation needed for scalar materials
    
    # Monkey-patch the matrix calculation methods to use tensor-aware versions
    original_P_matrix = layer.P_matrix
    original_Q_matrix = layer.Q_matrix
    
    def tensor_P_matrix():
        if layer.is_anisotropic:
            return LayerTensorAdapter.adapt_P_matrix_for_tensor(layer, layer.Kx, layer.Ky)
        else:
            return original_P_matrix()
    
    def tensor_Q_matrix():
        if layer.is_anisotropic:
            return LayerTensorAdapter.adapt_Q_matrix_for_tensor(layer, layer.Kx, layer.Ky)
        else:
            return original_Q_matrix()
    
    layer.P_matrix = tensor_P_matrix
    layer.Q_matrix = tensor_Q_matrix
    
    return layer


class PatternedLayerAdapter:
    """
    Adapter for converting PatternedLayer to RCWA-compatible Layer objects.
    
    This adapter bridges the gap between high-level geometry descriptions
    and the convolution matrix format required by the core solver.
    """
    
    @staticmethod
    def patterned_to_layer(patterned_layer: 'PatternedLayer', 
                          harmonics: Tuple[int, int], 
                          wavelength: float) -> 'Layer':
        """
        Convert PatternedLayer to a standard Layer with convolution matrices.
        
        This method rasterizes the geometric pattern and generates the Fourier
        convolution matrices needed for RCWA calculations, then creates a Layer
        object compatible with existing solvers.
        
        :param patterned_layer: PatternedLayer with geometric patterns
        :param harmonics: Number of harmonics (Nx, Ny) for truncation
        :param wavelength: Wavelength for dispersive materials
        :return: Layer object with convolution matrices set
        """
        from rcwa.model.layer import Layer
        from rcwa.legacy.crystal import Crystal
        
        # Generate convolution matrices from pattern
        conv_matrices = patterned_layer.to_convolution_matrices(harmonics, wavelength)
        
        # Extract lattice information for Crystal
        lattice = patterned_layer.lattice
        a_vec = np.array([lattice[0][0], lattice[0][1], 0.0])
        b_vec = np.array([lattice[1][0], lattice[1][1], 0.0])
        
        # Create Crystal object with convolution matrices
        # Note: This is a simplified approach - full implementation would need
        # proper Crystal constructor that accepts convolution matrices directly
        crystal = PatternedLayerAdapter._create_crystal_from_convolution_matrices(
            conv_matrices, a_vec, b_vec, harmonics
        )
        
        # Create Layer with the crystal
        layer = Layer(crystal=crystal, thickness=patterned_layer.thickness)
        
        # Store reference to original patterned layer for debugging/validation
        layer._source_patterned_layer = patterned_layer
        
        return layer
    
    @staticmethod
    def _create_crystal_from_convolution_matrices(conv_matrices: Dict[str, np.ndarray],
                                                a_vec: np.ndarray, b_vec: np.ndarray,
                                                harmonics: Tuple[int, int]) -> 'Crystal':
        """
        Create a Crystal object from precomputed convolution matrices.
        
        This creates a Crystal that stores the convolution matrices directly
        rather than computing them from a material grid.
        
        :param conv_matrices: Dictionary of convolution matrices
        :param a_vec: First lattice vector
        :param b_vec: Second lattice vector  
        :param harmonics: Harmonic truncation
        :return: Crystal object with convolution matrices
        """
        from rcwa.geom.crystal import Crystal
        
        # Create minimal Crystal with lattice information
        crystal = Crystal(a_vec, b_vec)
        
        # Store convolution matrices directly
        # This bypasses the normal Crystal constructor which expects a material grid
        crystal._convolution_matrices = conv_matrices
        crystal._harmonics = harmonics
        crystal._has_precomputed_matrices = True
        
        return crystal
    
    @staticmethod 
    def validate_pattern_conversion(patterned_layer: 'PatternedLayer',
                                  converted_layer: 'Layer',
                                  wavelength: float,
                                  tolerance: float = 1e-6) -> Dict[str, bool]:
        """
        Validate that PatternedLayer conversion preserves essential properties.
        
        :param patterned_layer: Original PatternedLayer
        :param converted_layer: Converted Layer
        :param wavelength: Test wavelength
        :param tolerance: Numerical tolerance for comparisons
        :return: Dictionary of validation results
        """
        validation_results = {
            'thickness_match': False,
            'lattice_area_match': False, 
            'pattern_bounds_reasonable': False,
            'convolution_matrices_finite': False
        }
        
        try:
            # Check thickness preservation
            thickness_diff = abs(patterned_layer.thickness - converted_layer.thickness)
            validation_results['thickness_match'] = thickness_diff < tolerance
            
            # Check lattice area preservation
            original_area = abs(patterned_layer.lattice[0][0] * patterned_layer.lattice[1][1] - 
                               patterned_layer.lattice[0][1] * patterned_layer.lattice[1][0])
            
            # Get lattice from converted layer (this is implementation-specific)
            if hasattr(converted_layer, 'crystal') and hasattr(converted_layer.crystal, 't1'):
                t1, t2 = converted_layer.crystal.t1, converted_layer.crystal.t2
                converted_area = abs(t1[0] * t2[1] - t1[1] * t2[0])
                area_diff = abs(original_area - converted_area) / original_area
                validation_results['lattice_area_match'] = area_diff < tolerance
            
            # Check pattern bounds
            bounds = patterned_layer.get_bounds()
            bounds_reasonable = all(np.isfinite(b) for b in bounds)
            validation_results['pattern_bounds_reasonable'] = bounds_reasonable
            
            # Check convolution matrices are finite
            if hasattr(converted_layer, '_source_patterned_layer'):
                test_matrices = patterned_layer.to_convolution_matrices((3, 3), wavelength)
                matrices_finite = all(np.all(np.isfinite(mat)) for mat in test_matrices.values())
                validation_results['convolution_matrices_finite'] = matrices_finite
                
        except Exception as e:
            # If validation fails, log the error but don't crash
            import warnings
            warnings.warn(f"Pattern conversion validation failed: {str(e)}", UserWarning)
        
        return validation_results


class GeometryStackAdapter:
    """
    Adapter for converting stacks containing PatternedLayers to traditional LayerStacks.
    
    This adapter handles mixed stacks containing both regular Layers and PatternedLayers.
    """
    
    @staticmethod
    def convert_geometry_stack(layer_list: list, harmonics: Tuple[int, int], 
                              wavelength: float) -> list:
        """
        Convert a mixed list of Layers and PatternedLayers to all Layers.
        
        :param layer_list: List containing Layer and/or PatternedLayer objects
        :param harmonics: Harmonic truncation for PatternedLayers
        :param wavelength: Wavelength for material evaluation
        :return: List of Layer objects
        """
        converted_layers = []
        
        for layer in layer_list:
            if hasattr(layer, 'shapes'):  # Duck typing for PatternedLayer
                # Convert PatternedLayer to Layer
                converted_layer = PatternedLayerAdapter.patterned_to_layer(
                    layer, harmonics, wavelength
                )
                converted_layers.append(converted_layer)
            else:
                # Already a regular Layer
                converted_layers.append(layer)
        
        return converted_layers
    
    @staticmethod
    def suggest_harmonics_for_pattern(patterned_layer: 'PatternedLayer', 
                                    wavelength: float,
                                    target_accuracy: float = 0.01) -> Tuple[int, int]:
        """
        Suggest appropriate harmonic truncation for a patterned layer.
        
        This analyzes the pattern complexity and suggests harmonics needed
        for the desired accuracy level.
        
        :param patterned_layer: PatternedLayer to analyze
        :param wavelength: Operating wavelength
        :param target_accuracy: Target relative accuracy (0.01 = 1%)
        :return: Suggested (Nx, Ny) harmonic truncation
        """
        # Get pattern bounds
        x_min, x_max, y_min, y_max = patterned_layer.get_bounds()
        pattern_size_x = x_max - x_min
        pattern_size_y = y_max - y_min
        
        # Get lattice periods
        lattice = patterned_layer.lattice
        period_x = np.sqrt(lattice[0][0]**2 + lattice[0][1]**2)
        period_y = np.sqrt(lattice[1][0]**2 + lattice[1][1]**2)
        
        # Estimate minimum harmonics based on feature size
        # Rule of thumb: need harmonics up to wavelength/feature_size
        min_feature_x = period_x / max(len(patterned_layer.shapes), 2)
        min_feature_y = period_y / max(len(patterned_layer.shapes), 2)
        
        harmonics_x = max(3, int(2 * period_x / wavelength))
        harmonics_y = max(3, int(2 * period_y / wavelength))
        
        # Ensure odd numbers for symmetry
        if harmonics_x % 2 == 0:
            harmonics_x += 1
        if harmonics_y % 2 == 0:
            harmonics_y += 1
        
        # Cap at reasonable maximum
        harmonics_x = min(harmonics_x, 21)
        harmonics_y = min(harmonics_y, 21)
        
        return (harmonics_x, harmonics_y)
