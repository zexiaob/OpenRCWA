import os

file_location = os.path.dirname(__file__)
nk_location = os.path.join(file_location, os.pardir, 'nkData/')

from rcwa.utils.nk_loaders import *
from rcwa.utils.fresnel import *
from rcwa.utils.dispersion import tabulated_dispersion

try:  # Optional plotting helpers
    from rcwa.utils.plotter import *
except Exception:  # pragma: no cover
    pass
