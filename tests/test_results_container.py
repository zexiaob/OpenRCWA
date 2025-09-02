import numpy as np

from rcwa.core.solver import Solver
from rcwa.solve.source import Source
from rcwa.model.layer import Layer, LayerStack, Air, SiO2


def build_stack():
    superstrate = Air()
    substrate = Air()
    film = Layer(material=SiO2(), thickness=100e-9)
    stack = LayerStack(layers=[film], superstrate=superstrate, substrate=substrate)
    return stack


def test_legacy_results_energy_and_complex_consistency():
    stack = build_stack()
    source = Source(wavelength=550e-9)
    res = Solver(stack, source, n_harmonics=1).solve()
    # Legacy container exposes totals and checks
    assert res.verify_energy_conservation(tolerance=1e-6)
    assert res.verify_complex_consistency(tolerance=1e-4)