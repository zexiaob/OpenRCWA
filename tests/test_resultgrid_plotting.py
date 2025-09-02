import pytest

from rcwa.solve import Sweep
from rcwa.solve.source import Source
from rcwa.model.layer import Layer, LayerStack, Air, SiO2


def build_stack(thickness=100e-9):
    superstrate = Air(); substrate = Air()
    film = Layer(material=SiO2(), thickness=thickness)
    return LayerStack(layers=[film], superstrate=superstrate, substrate=substrate)


def test_plot_2d_raises_for_now():
    stack = build_stack()
    source = Source(wavelength=550e-9)
    params = {
        'wavelength': [500e-9, 550e-9, 600e-9],
        'theta': [0.0, 0.1],
    }
    out = Sweep(params, backend='serial').run(stack, source, n_harmonics=1)
    grid = out['result_grid']
    with pytest.raises(ValueError):
        grid.plot('wavelength', y='RTot')