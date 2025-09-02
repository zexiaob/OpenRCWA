import numpy as np

from rcwa.solve import Sweep
from rcwa.solve.source import Source
from rcwa.model.layer import Layer, LayerStack, Air, SiO2


def build_planar_stack(thickness=100e-9):
    superstrate = Air()
    substrate = Air()
    film = Layer(material=SiO2(), thickness=thickness)
    stack = LayerStack(layers=[film], superstrate=superstrate, substrate=substrate)
    return stack, film


def test_sweep_source_and_object_params_returns_result_grid():
    stack, film = build_planar_stack(100e-9)
    source = Source(wavelength=500e-9)

    params = {
        'wavelength': [500e-9, 600e-9],
        'objects': [
            {'targets': [film], 'params': {'thickness': [50e-9, 100e-9]}},
        ],
    }

    out = Sweep(params, backend='serial').run(stack, source, n_harmonics=1)

    # coords contain both source and object axes
    coords = out['coords']
    assert 'wavelength' in coords
    # Dim naming uses class name of the object
    assert any(k.endswith('.thickness') for k in coords.keys())
    thickness_key = [k for k in coords.keys() if k.endswith('.thickness')][0]

    # result_grid exists and matches sizes
    grid = out['result_grid']
    assert grid is not None
    assert set(grid.dims) >= {'wavelength', 'Layer.thickness'}
    assert len(coords['wavelength']) == 2
    assert len(coords[thickness_key]) == 2
    # Shape is product of dims (order is source first, then object)
    assert grid.shape == (len(coords['wavelength']), len(coords[thickness_key]))

    # Selection works
    sub = grid.sel(wavelength=coords['wavelength'][0])
    assert sub.shape == (len(coords[thickness_key]),)
    pt = grid.isel(wavelength=0, **{thickness_key: 0})
    # pt is a single Result point
    assert hasattr(pt, 'R') and hasattr(pt, 'T')

    # DataFrame conversion preserves complex amplitudes in columns
    df = grid.to_dataframe()
    # If pandas unavailable, df may be list of dicts
    if isinstance(df, list):
        row0 = df[0]
        assert 'r_complex' in row0 and 't_complex' in row0
    else:
        assert 'r_complex' in df.columns and 't_complex' in df.columns


def test_multi_object_targets_coord_naming_and_stack_helpers():
    stack, film1 = build_planar_stack(80e-9)
    # Second film
    film2 = Layer(material=SiO2(), thickness=120e-9)
    stack.internal_layers.append(film2)
    source = Source(wavelength=550e-9)

    params = {
        'objects': [
            {'targets': [film1, film2], 'params': {'thickness': [60e-9, 100e-9]}},
        ]
    }
    out = Sweep(params, backend='serial').run(stack, source, n_harmonics=1)
    coords = out['coords']
    # Coordinate name should include both class names joined by '+'
    key = [k for k in coords if k.endswith('.thickness')][0]
    assert '+' in key
    grid = out['result_grid']
    # Stacking helpers return arrays with trailing axis of length 3
    rC, tC = grid.get_complex_amplitudes()
    assert rC.shape[-1] == 3 and tC.shape[-1] == 3
    rP, tP = grid.get_phases()
    assert rP.shape == rC.shape and tP.shape == tC.shape


def test_resultgrid_quick_plot_1d():
    stack, film = build_planar_stack(100e-9)
    source = Source(wavelength=500e-9)

    params = {
        'wavelength': [500e-9, 550e-9, 600e-9],
    }
    out = Sweep(params, backend='serial').run(stack, source, n_harmonics=1)
    grid = out['result_grid']
    ax = grid.plot('wavelength', y='RTot')
    assert ax is not None