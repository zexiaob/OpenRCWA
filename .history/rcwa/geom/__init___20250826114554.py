"""
Geometry module for OpenRCWA.

This module provides declarative geometry definitions for complex photonic structures.
It includes Shape classes for basic and complex geometries, and PatternedLayer for
combining shapes with materials to create RCWA-compatible layers.
"""

from .shape import (
    Shape,
    Rectangle, 
    Circle, 
    Ellipse, 
    Polygon, 
    RegularPolygon,
    ComplexShape,
    UnionShape,
    IntersectionShape, 
    DifferenceShape
)

from .patterned import PatternedLayer

__all__ = [
    'Shape',
    'Rectangle', 
    'Circle', 
    'Ellipse', 
    'Polygon', 
    'RegularPolygon',
    'ComplexShape',
    'UnionShape',
    'IntersectionShape', 
    'DifferenceShape',
    'PatternedLayer'
]
