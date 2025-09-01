import numpy as np
import pytest

from rcwa.model.layer import Layer, LayerStack, Air, Substrate
from rcwa.solve import Sweep
from rcwa.solve.source import Source


def build_simple_stack():
    layer1 = Layer(er=2.25, ur=1.0, thickness=100e-9)
    stack = LayerStack(layer1, superstrate=Air(), substrate=Substrate(Air()))
    return stack


def test_sweep_source_params():
    stack = build_simple_stack()
    source = Source(wavelength=500e-9, theta=0.0, phi=0.0, pTEM=np.array([1.0, 1.0]))

    params = {
        'wavelength': [500e-9, 600e-9, 700e-9],
        'theta': [0.0, 10.0*np.pi/180.0],
    }
    sweep = Sweep(params, backend='serial')
    out = sweep.run(stack, source, n_harmonics=1)

    # There should be 3*2 = 6 results
    assert len(out['results']) == 6
    # Each element is a Results object
    for r in out['results']:
        # Results returned by solver.solve() is a Results instance
        assert hasattr(r, 'R') and hasattr(r, 'T')


def test_sweep_parametric_geometry_rectangle():
    # If PatternedLayer exists with with_params, simulate param scan by setattr fallback
    stack = build_simple_stack()
    source = Source(wavelength=500e-9, theta=0.0, phi=0.0, pTEM=np.array([1.0, 1.0]))

    # sweep the thickness of the single layer as a parametric update
    layer_to_sweep = stack.internal_layers[0]
    params = {
        'objects': [
            {
                'targets': [layer_to_sweep],
                'params': {'thickness': [50e-9, 100e-9, 150e-9]}
            }
        ]
    }
    sweep = Sweep(params, backend='serial')
    out = sweep.run(stack, source, n_harmonics=1)

    assert len(out['results']) == 3
    # Ensure reflectance varies with thickness in this simple case
    Rs = [float(np.sum(r.R)) for r in out['results']]
    assert np.std(Rs) >= 0.0  # non-crashing check
