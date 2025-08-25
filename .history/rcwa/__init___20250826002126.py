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
from rcwa.model.material import Material, TensorMaterial
from rcwa.model.layer import LayerStack, Layer, freeSpaceLayer
from rcwa.model.transforms import rotate_layer

# Legacy support (backward compatibility)
from rcwa.legacy.crystal import Crystal
from rcwa.legacy.grating import RectangularGrating, TriangularGrating, Grating

# Results and workflow modules
from rcwa.solve.source import Source, zeroSource
from rcwa.solve.results import Results
from rcwa.slicer import Slicer
from rcwa.shorthand import complexArray

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
