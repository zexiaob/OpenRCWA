import numpy as np
from rcwa.solve import Sweep, LCP, RCP, compute_circular_dichroism
from rcwa import Layer, LayerStack, Source


def test_compute_cd_helper_matches_manual_difference():
    # Simple vacuum stack; CD should be ~0 but function should execute and match manual
    reflection = Layer(er=1, ur=1)
    transmission = Layer(er=1, ur=1)
    stack = LayerStack(incident_layer=reflection, transmission_layer=transmission)

    params = {
        'wavelength': [0.5],
        'pTEM': [LCP(), RCP()],
    }
    # Base source; polarization will be set by sweep params
    source = Source(wavelength=0.5)
    grid_dict = Sweep(params=params, backend='serial').run(stack, source)
    grid = grid_dict['result_grid']

    # Manual
    l = grid.sel(pTEM=LCP())
    r = grid.sel(pTEM=RCP())

    def totT(x):
        if hasattr(x, 'data'):
            T = x.get('T')  # stack over grid; trailing axis is orders
            return np.sum(T)
        else:
            return np.sum(x.T) if hasattr(x.T, '__iter__') else x.T

    manual_cd = totT(r) - totT(l)

    # Helper
    helper_cd = compute_circular_dichroism(grid)

    assert np.allclose(helper_cd, manual_cd)
