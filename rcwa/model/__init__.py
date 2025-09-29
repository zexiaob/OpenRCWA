"""
Data models and validation for RCWA.

This module contains the physical data models (materials, layers, stacks)
and geometric transformations. It provides strong validation using Pydantic
and does not depend on numerical computation kernels.
"""

from .material import Material, TensorMaterial, nm, um, mm, deg
from .layer import Layer, LayerStack
from .transforms import rotate_layer

__all__ = ['Material', 'TensorMaterial', 'Layer', 'LayerStack', 'rotate_layer', 'nm', 'um', 'mm', 'deg']