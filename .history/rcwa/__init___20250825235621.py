__version__ = '0.0.1'
__author__ = 'Jordan Edmunds'
__email__ = 'edmundsj@uci.edu'
import os

file_location = os.path.dirname(__file__)
nk_dir = os.path.join(file_location, 'nkData/')
test_dir = os.path.join(file_location, 'test')
example_dir = os.path.join(file_location, 'examples')

from rcwa.material import Material
from rcwa.crystal import Crystal
from rcwa.matrices import MatrixCalculator
from rcwa.layer import LayerStack, Layer, freeSpaceLayer
from rcwa.solve.source import Source, zeroSource
from rcwa.solve.results import Results
from rcwa.slicer import Slicer
from rcwa.grating import RectangularGrating, TriangularGrating, Grating
from rcwa.solver import Solver
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
