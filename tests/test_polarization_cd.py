import numpy as np

from rcwa.solve import Sweep, LCP, RCP
from rcwa.solve.source import Source
from rcwa.model.layer import Layer, LayerStack, Air, SiO2


def build_planar_stack(thickness=100e-9):
    superstrate = Air()
    substrate = Air()
    film = Layer(material=SiO2(), thickness=thickness)
    stack = LayerStack(layers=[film], superstrate=superstrate, substrate=substrate)
    return stack


def test_lcp_rcp_helpers_shape():
    l = LCP(); r = RCP()
    assert l.shape == (2,) and r.shape == (2,)
    # Normalization check
    assert np.allclose(np.linalg.norm(l), 1.0)
    assert np.allclose(np.linalg.norm(r), 1.0)


def test_cd_sanity_on_planar_is_near_zero():
    # Planar isotropic film should have negligible CD at normal incidence
    stack = build_planar_stack(100e-9)
    source = Source(wavelength=550e-9)
    params = {
        'wavelength': [550e-9],
        'pTEM': [LCP(), RCP()],
    }
    out = Sweep(params, backend='serial').run(stack, source, n_harmonics=1)
    grid = out['result_grid']
    r_lcp = grid.sel(pTEM=LCP())  # label-based selection
    r_rcp = grid.sel(pTEM=RCP())

    # If dims ordering differs, both are 0-D ResultGrid or Result; normalize to totals
    def totT(x):
        if hasattr(x, 'data'):  # ResultGrid of size 1
            pt = x.isel(**{d: 0 for d in x.dims})
        else:
            pt = x
        return np.sum(pt.T) if hasattr(pt.T, '__iter__') else pt.T

    cd = totT(r_rcp) - totT(r_lcp)
    assert abs(cd) < 1e-6