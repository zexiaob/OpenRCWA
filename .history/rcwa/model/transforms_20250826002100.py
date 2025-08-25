"""
Geometric transformations for RCWA layers and structures.

This module provides functions for rotating layers and handling coordinate transformations
while maintaining physical consistency.
"""
import numpy as np
from typing import Tuple, Union
from .layer import Layer
from .material import TensorMaterial


def euler_to_rotation_matrix(alpha: float, beta: float, gamma: float, convention: str = "ZYX") -> np.ndarray:
    """
    Convert Euler angles to rotation matrix using specified convention.
    
    :param alpha: First rotation angle (radians)
    :param beta: Second rotation angle (radians) 
    :param gamma: Third rotation angle (radians)
    :param convention: Rotation convention, default "ZYX" (Tait-Bryan angles)
    :return: 3x3 rotation matrix
    """
    if convention != "ZYX":
        raise NotImplementedError("Only ZYX convention is currently implemented")
    
    # ZYX convention: R = Rz(alpha) * Ry(beta) * Rx(gamma)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    
    # Individual rotation matrices
    Rz = np.array([
        [ca, -sa, 0],
        [sa, ca, 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [cb, 0, sb],
        [0, 1, 0],
        [-sb, 0, cb]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, cg, -sg],
        [0, sg, cg]
    ])
    
    # Combined rotation: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


def rotate_layer(layer: Layer, 
                euler_angles: Tuple[float, float, float], 
                about: str = "center",
                convention: str = "ZYX") -> Layer:
    """
    Create a new Layer with rotated material tensors.
    
    This function applies passive rotations to the material tensors according to:
    ε' = R * ε * R^T
    μ' = R * μ * R^T
    
    :param layer: Original Layer object
    :param euler_angles: Tuple of (alpha, beta, gamma) Euler angles in radians
    :param about: Rotation center, currently only "center" is supported
    :param convention: Euler angle convention, default "ZYX"
    :return: New Layer object with rotated material properties
    """
    if about != "center":
        raise NotImplementedError("Only rotation about center is currently implemented")
    
    alpha, beta, gamma = euler_angles
    rotation_matrix = euler_to_rotation_matrix(alpha, beta, gamma, convention)
    
    if layer.crystal is not None:
        # For patterned layers, only in-plane rotation (about z-axis) is allowed
        if not np.allclose([beta, gamma], [0, 0], atol=1e-10):
            raise NotImplementedError(
                "Only in-plane rotation (about z-axis) is supported for patterned layers. "
                "Set beta=0 and gamma=0 for pure z-rotation."
            )
        # TODO: Implement crystal/lattice rotation
        raise NotImplementedError("Crystal layer rotation not yet implemented")
    
    # Handle homogeneous layers
    if layer.is_anisotropic:
        # Rotate the tensor material
        rotated_tensor_material = layer.tensor_material.rotated(rotation_matrix)
        return Layer(
            thickness=layer.thickness,
            tensor_material=rotated_tensor_material
        )
    else:
        # Convert isotropic material to anisotropic and rotate
        # Create tensor material from scalar values
        eps_tensor = np.eye(3, dtype=complex) * layer.er
        mu_tensor = np.eye(3, dtype=complex) * layer.ur
        
        tensor_material = TensorMaterial(
            epsilon_tensor=eps_tensor,
            mu_tensor=mu_tensor,
            source=layer.source,
            name=f"{getattr(layer.material, 'name', 'material')}_rotated"
        )
        
        rotated_tensor_material = tensor_material.rotated(rotation_matrix)
        return Layer(
            thickness=layer.thickness,
            tensor_material=rotated_tensor_material
        )


def rotation_matrix_z(angle: float) -> np.ndarray:
    """
    Create rotation matrix for rotation about z-axis.
    
    :param angle: Rotation angle in radians
    :return: 3x3 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def rotation_matrix_y(angle: float) -> np.ndarray:
    """
    Create rotation matrix for rotation about y-axis.
    
    :param angle: Rotation angle in radians
    :return: 3x3 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_x(angle: float) -> np.ndarray:
    """
    Create rotation matrix for rotation about x-axis.
    
    :param angle: Rotation angle in radians
    :return: 3x3 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
