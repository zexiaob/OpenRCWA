__version__ = '0.0.1'
__author__ = 'Jordan Edmunds'
__email__ = 'edmundsj@uci.edu'
import os

file_location = os.path.dirname(__file__)
nk_dir = os.path.join(file_location, 'nkData/')
test_dir = os.path.join(file_location, 'test')
example_dir = os.path.join(file_location, 'examples')

# Core modules (new architecture)
from rcwa.core.solver import Solver
from rcwa.core.matrices import MatrixCalculator

# Model modules (new architecture)
from rcwa.model.material import Material, TensorMaterial, nm, um, mm, deg, make_n_from_table, make_epsilon_tensor_from_table
from rcwa.model.layer import (
    LayerStack, Layer, HalfSpace, Air, Vacuum, Stack, freeSpaceLayer,
    Substrate, Silicon, SiO2, Glass,
)
from rcwa.model.transforms import rotate_layer

# Geometry layer (new declarative API)
try:
    from rcwa.geom.patterned import PatternedLayer, rectangular_lattice, square_lattice, hexagonal_lattice
    from rcwa.geom.shape import (
        Rectangle, Circle, Ellipse, Polygon, RegularPolygon,
        UnionShape, IntersectionShape, DifferenceShape, TaperedPolygon
    )
except Exception:
    # Keep imports optional if geom is not available in some environments
    PatternedLayer = None
    rectangular_lattice = square_lattice = hexagonal_lattice = None
    Rectangle = Circle = Ellipse = Polygon = RegularPolygon = None
    UnionShape = IntersectionShape = DifferenceShape = TaperedPolygon = None

# Legacy support (backward compatibility)
from rcwa.legacy.crystal import Crystal
from rcwa.legacy.grating import RectangularGrating, TriangularGrating, Grating

# Results and workflow modules
from rcwa.solve.source import Source, zeroSource
from rcwa.solve.results import (
    Results,
    Result,
    ResultGrid,
    build_result_grid_from_sweep,
)
from rcwa.solve import Sweep, LCP, RCP
from rcwa.solve.simulate import simulate
from rcwa.solve import compute_circular_dichroism as compute_cd
from rcwa.slicer import Slicer
from rcwa.shorthand import complexArray
from rcwa.viz import show_stack3d

# Import utils modules that may have circular dependencies later
try:
    from rcwa import utils
    from rcwa.utils import Plotter
    from rcwa.utils import rTE, rTM
except ImportError:
    # Handle missing dependencies gracefully
    utils = None
    Plotter = None
    rTE = None
    rTM = None

# Public API surface for facade exports (Stage 4.1)
__all__ = [
    # meta
    "__version__",
    "__author__",
    "__email__",
    "nk_dir",
    "test_dir",
    "example_dir",
    # core
    "Solver",
    "MatrixCalculator",
    # model
    "Material",
    "TensorMaterial",
    "nm",
    "um",
    "mm",
    "deg",
    "make_n_from_table",
    "make_epsilon_tensor_from_table",
    "Layer",
    "LayerStack",
    "HalfSpace",
    "Air",
    "Vacuum",
    "Stack",
    "freeSpaceLayer",
    "Substrate",
    "Silicon",
    "SiO2",
    "Glass",
    "rotate_layer",
    # geometry (optional)
    "PatternedLayer",
    "rectangular_lattice",
    "square_lattice",
    "hexagonal_lattice",
    "Rectangle",
    "Circle",
    "Ellipse",
    "Polygon",
    "RegularPolygon",
    "UnionShape",
    "IntersectionShape",
    "DifferenceShape",
    "TaperedPolygon",
    # legacy
    "Crystal",
    "RectangularGrating",
    "TriangularGrating",
    "Grating",
    # workflow / results
    "Source",
    "zeroSource",
    "Results",
    "Result",
    "ResultGrid",
    "build_result_grid_from_sweep",
    "Sweep",
    "LCP",
    "RCP",
    "simulate",
    "compute_cd",
    # visualization
    "show_stack3d",
    # utils
    "Slicer",
    "complexArray",
    "utils",
    "Plotter",
    "rTE",
    "rTM",
]
