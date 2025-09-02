import numpy as np

from OpenRCWA import Layer, LayerStack
from OpenRCWA import simulate


def test_simulate_single_point_returns_result():
    stack = LayerStack(Layer(2.0, 1.0, thickness=0.2), incident_layer=Layer(1.0,1.0), transmission_layer=Layer(1.0,1.0))
    res = simulate(stack, wavelength=0.5, theta=0.0, phi=0.0, polarization='TE', n_harmonics=1)
    # Result should expose complex amplitudes and intensities
    assert hasattr(res, 'rx') and hasattr(res, 'R')
    assert np.isfinite(res.RTot)


def test_simulate_sweep_returns_grid():
    stack = LayerStack(Layer(2.0, 1.0, thickness=0.2), incident_layer=Layer(1.0,1.0), transmission_layer=Layer(1.0,1.0))
    grid = simulate(stack, wavelength=[0.5, 0.6], theta=0.0, phi=0.0, polarization=['TE','TM'], n_harmonics=1)
    # Should be a ResultGrid-like with dims and coords
    assert hasattr(grid, 'dims') and hasattr(grid, 'coords')
    assert 'wavelength' in grid.coords and 'pTEM' in grid.coords
    # And we can compute a quick array of totals
    T = grid.get('T')
    assert T.shape[0] == 2  # wavelength dim
