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
    from rcwa.legacy.crystal import Crystal
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
        the respective tensor element. The returned dictionary uses the project-wide
        `er_ij`/`ur_ij` naming convention for electric and magnetic tensor components.
        
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
        
        # Permittivity tensor components (use er_ prefix for project-wide consistency)
        axes = ["x", "y", "z"]
        for i in range(3):
            for j in range(3):
                component_name = f'er_{axes[i]}{axes[j]}'
                convolution_matrices[component_name] = (
                    epsilon_tensor[i, j] * complexIdentity(matrix_dim)
                )

        # Permeability tensor components (use ur_ prefix)
        for i in range(3):
            for j in range(3):
                component_name = f'ur_{axes[i]}{axes[j]}'
                convolution_matrices[component_name] = (
                    mu_tensor[i, j] * complexIdentity(matrix_dim)
                )
        
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

        blocks = LayerTensorAdapter._compute_tensor_blocks(layer, Kx, Ky)
        layer._tensor_blocks = blocks
        return blocks['P']
    
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

        blocks = getattr(layer, '_tensor_blocks', None)
        if blocks is None:
            blocks = LayerTensorAdapter._compute_tensor_blocks(layer, Kx, Ky)
            layer._tensor_blocks = blocks
        return blocks['Q']

    @staticmethod
    def _compute_tensor_blocks(layer, Kx: ArrayLike, Ky: ArrayLike) -> Dict[str, ArrayLike]:
        """Compute the first-order system blocks for tensor materials."""

        conv = getattr(layer, '_tensor_conv_matrices', None)
        if conv is None:
            if getattr(layer, 'tensor_material', None) is None:
                raise ValueError("Tensor material data unavailable for anisotropic layer")
            from rcwa.core.adapters import TensorToConvolutionAdapter  # type: ignore
            conv = TensorToConvolutionAdapter.tensor_to_convolution_matrices(
                layer.tensor_material.epsilon_tensor,
                layer.tensor_material.mu_tensor,
                1,
            )

        def _component(prefix: str, comp: str) -> ArrayLike:
            key = f'{prefix}_{comp}'
            mat = conv.get(key)
            if mat is None:
                # Default: identity for diagonal terms, zero otherwise
                axes = comp[0] == comp[1]
                size = 1
                if isinstance(Kx, np.ndarray):
                    size = Kx.shape[0]
                if axes:
                    return np.identity(size, dtype=complex)
                return np.zeros((size, size), dtype=complex)
            mat = np.array(mat, dtype=complex)
            if mat.ndim == 0:
                size = 1
                if isinstance(Kx, np.ndarray):
                    size = Kx.shape[0]
                if size == 1:
                    return np.array([[mat]], dtype=complex)
                return mat * np.identity(size, dtype=complex)
            return mat

        if not isinstance(Kx, np.ndarray):
            Kx_mat = np.array([[Kx]], dtype=complex)
        else:
            Kx_mat = np.array(Kx, dtype=complex)
        if not isinstance(Ky, np.ndarray):
            Ky_mat = np.array([[Ky]], dtype=complex)
        else:
            Ky_mat = np.array(Ky, dtype=complex)

        exx = _component('er', 'xx')
        exy = _component('er', 'xy')
        exz = _component('er', 'xz')
        eyx = _component('er', 'yx')
        eyy = _component('er', 'yy')
        eyz = _component('er', 'yz')
        ezx = _component('er', 'zx')
        ezy = _component('er', 'zy')
        ezz = _component('er', 'zz')

        mxx = _component('ur', 'xx')
        mxy = _component('ur', 'xy')
        mxz = _component('ur', 'xz')
        myx = _component('ur', 'yx')
        myy = _component('ur', 'yy')
        myz = _component('ur', 'yz')
        mzx = _component('ur', 'zx')
        mzy = _component('ur', 'zy')
        mzz = _component('ur', 'zz')

        eps_tt = np.block([[exx, exy], [eyx, eyy]])
        eps_tz = np.vstack((exz, eyz))
        eps_zt = np.hstack((ezx, ezy))
        eps_zz = ezz if isinstance(ezz, np.ndarray) and ezz.ndim == 2 else np.array([[ezz]])

        mu_tt = np.block([[mxx, mxy], [myx, myy]])
        mu_tz = np.vstack((mxz, myz))
        mu_zt = np.hstack((mzx, mzy))
        mu_zz = mzz if isinstance(mzz, np.ndarray) and mzz.ndim == 2 else np.array([[mzz]])

        try:
            eps_zz_inv = np.linalg.inv(eps_zz)
        except np.linalg.LinAlgError:
            eps_zz_inv = np.linalg.pinv(eps_zz)

        try:
            mu_zz_inv = np.linalg.inv(mu_zz)
        except np.linalg.LinAlgError:
            mu_zz_inv = np.linalg.pinv(mu_zz)

        C = np.hstack((-Ky_mat, Kx_mat))
        S = np.vstack((Kx_mat, Ky_mat))

        size_n = Kx_mat.shape[0]
        identity = np.identity(size_n, dtype=complex)
        J = np.block([
            [np.zeros((size_n, size_n), dtype=complex), identity],
            [-identity, np.zeros((size_n, size_n), dtype=complex)]
        ])

        mu_eff = mu_tt - mu_tz @ mu_zz_inv @ mu_zt
        eps_eff = eps_tt - eps_tz @ eps_zz_inv @ eps_zt

        P = -S @ eps_zz_inv @ C + mu_eff @ J
        R = -S @ eps_zz_inv @ eps_zt - mu_tz @ mu_zz_inv @ C
        Q = -S @ mu_zz_inv @ C + eps_eff @ J
        S_mat = -S @ mu_zz_inv @ mu_zt + eps_tz @ eps_zz_inv @ C

        return {
            'P': P,
            'Q': Q,
            'R': R,
            'S': S_mat,
            'eps_zz_inv': eps_zz_inv,
            'mu_zz_inv': mu_zz_inv,
        }

    @staticmethod
    def _extract_epsilon_mu(layer) -> Tuple[ArrayLike, ArrayLike]:
        """Extract effective epsilon and mu tensors from a layer.

        Supports both uniform tensor-material layers and patterned layers that
        store anisotropy in ``_tensor_conv_matrices``. For patterned layers the
        zero-order Fourier coefficient of each convolution matrix is used as an
        effective tensor element. Missing tensor components default to unity on
        the diagonal and zero elsewhere.
        """

        if getattr(layer, 'tensor_material', None) is not None:
            eps = layer.tensor_material.epsilon_tensor
            mu = layer.tensor_material.mu_tensor
            return np.array(eps, dtype=complex), np.array(mu, dtype=complex)

        conv = getattr(layer, '_tensor_conv_matrices', {}) or {}
        axes = ['x', 'y', 'z']
        eps = np.zeros((3, 3), dtype=complex)
        mu = np.zeros((3, 3), dtype=complex)

        for i, ai in enumerate(axes):
            for j, aj in enumerate(axes):
                e_key = f'er_{ai}{aj}'
                m_key = f'ur_{ai}{aj}'
                e_mat = conv.get(e_key)
                m_mat = conv.get(m_key)
                if e_mat is not None:
                    eps[i, j] = e_mat[0, 0] if isinstance(e_mat, np.ndarray) else e_mat
                elif i == j:
                    eps[i, j] = 1.0
                if m_mat is not None:
                    mu[i, j] = m_mat[0, 0] if isinstance(m_mat, np.ndarray) else m_mat
                elif i == j:
                    mu[i, j] = 1.0

        return eps, mu
    
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
    def solve_tensor_eigenproblem(layer, P: ArrayLike, Q: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """
        Solve the generalized eigenvalue problem for tensor materials.
        
        For tensor materials, the eigenvalue problem P*Q*W = W*Lambda^2 needs to be
        solved carefully to ensure proper mode calculation and energy conservation.
        
        :param P: P matrix from tensor calculations
        :param Q: Q matrix from tensor calculations
        :return: Tuple of (eigenvalues, eigenvectors, Lambda matrix)
        """
        blocks = getattr(layer, '_tensor_blocks', None)
        if blocks is None:
            blocks = LayerTensorAdapter._compute_tensor_blocks(layer, layer.Kx, layer.Ky)
            layer._tensor_blocks = blocks

        R = blocks['R']
        S = blocks['S']

        top = np.hstack((R, P))
        bottom = np.hstack((Q, S))
        system_matrix = np.vstack((top, bottom))

        eigenValues, eigenVectors = np.linalg.eig(system_matrix)

        # Arrange eigenvectors into electric and magnetic partitions
        n = P.shape[0]
        W = eigenVectors[:n, :]
        V = eigenVectors[n:, :]

        Lambda_diag = np.array(eigenValues, dtype=complex)
        for i, val in enumerate(Lambda_diag):
            if np.imag(val) < 0:
                Lambda_diag[i] = -val
        Lambda = np.diag(Lambda_diag)

        return eigenValues, W, Lambda, V


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


# Note: PatternedLayerAdapter is no longer needed since PatternedLayer
# now inherits directly from Layer and provides native RCWA compatibility.


# Geometry utility functions for PatternedLayer support
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
