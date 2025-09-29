import importlib


def test_facade_basic_import_and_objects():
    orcwa = importlib.import_module("OpenRCWA")

    # Basic attributes should exist
    assert hasattr(orcwa, "Layer")
    assert hasattr(orcwa, "Source")
    assert hasattr(orcwa, "Sweep")

    # Construct trivial objects to ensure re-exports are functional
    L = orcwa.Layer(1.0, 1.0)
    S = orcwa.Source(wavelength=0.5, theta=0.0, phi=0.0, pTEM=[1, 0], layer=L)
    assert L is not None and S is not None
