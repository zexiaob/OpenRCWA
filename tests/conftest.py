import os
import sys

# Ensure repository root is on sys.path so 'rcwa' imports work when running tests
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
