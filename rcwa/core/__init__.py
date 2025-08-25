"""
Core numerical computation modules.

This module contains the numerical kernels for RCWA computations,
including solvers, matrix operations, and core adapters.
"""

from .solver import Solver
from .matrices import *

__all__ = ['Solver']
