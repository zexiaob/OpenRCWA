import numpy as np

from rcwa.solve import Sweep
from rcwa.solve.source import Source
from rcwa.model.layer import Layer, LayerStack, Air, SiO2


def build_planar_stack(thickness=100e-9):
    superstrate = Air(); substrate = Air()
    film = Layer(material=SiO2(), thickness=thickness)
    return LayerStack(layers=[film], superstrate=superstrate, substrate=substrate)


def test_resultgrid_complex_amplitude_completeness_and_consistency():
    stack = build_planar_stack(120e-9)
    source = Source(wavelength=550e-9)
    params = {
        'wavelength': [500e-9, 600e-9],
    }
    out = Sweep(params, backend='serial').run(stack, source, n_harmonics=1)
    grid = out['result_grid']

    # Iterate all points, verify fields exist and intensities match R/T
    for idx in np.ndindex(grid.shape):
        pt = grid.isel(**{d: i for d, i in zip(grid.dims, idx)})
        # Amplitudes present
        for name in ['rx','ry','rz','tx','ty','tz','R','T']:
            assert hasattr(pt, name)
        rI = np.abs(pt.r_complex())**2
        tI = np.abs(pt.t_complex())**2
        R_sum = np.sum(pt.R) if hasattr(pt.R, '__iter__') else pt.R
        T_sum = np.sum(pt.T) if hasattr(pt.T, '__iter__') else pt.T
        assert np.isfinite(R_sum) and np.isfinite(T_sum)
        assert abs(R_sum - np.sum(rI)) < 1e-4 * max(1.0, R_sum)
        assert abs(T_sum - np.sum(tI)) < 1e-4 * max(1.0, T_sum)


def test_resultgrid_info_no_loss_dataframe_and_selection():
    stack = build_planar_stack(90e-9)
    source = Source(wavelength=550e-9)
    vals = [500e-9, 550e-9, 600e-9]
    out = Sweep({'wavelength': vals}, backend='serial').run(stack, source, n_harmonics=1)
    grid = out['result_grid']
    df = grid.to_dataframe()

    # Match first row complex amplitudes with grid point exactly
    pt = grid.isel(wavelength=0)
    if isinstance(df, list):
        row = df[0]
        rc_df = row['r_complex']
        tc_df = row['t_complex']
    else:
        rc_df = df.loc[0, 'r_complex']
        tc_df = df.loc[0, 't_complex']
    assert np.allclose(rc_df, pt.r_complex())
    assert np.allclose(tc_df, pt.t_complex())

    # Selection by label returns the same point
    sel = grid.sel(wavelength=vals[0])
    if hasattr(sel, 'data'):
        sel_pt = sel.isel(**{d: 0 for d in sel.dims})
    else:
        sel_pt = sel
    assert np.allclose(sel_pt.r_complex(), pt.r_complex())


def test_resultgrid_phase_sensitive_ops():
    stack = build_planar_stack(110e-9)
    source = Source(wavelength=550e-9)
    out = Sweep({'wavelength': [520e-9, 560e-9]}, backend='serial').run(stack, source, n_harmonics=1)
    grid = out['result_grid']
    rP, tP = grid.get_phases()
    assert rP.shape == (*grid.shape, 3)
    assert tP.shape == (*grid.shape, 3)
    # Phases should be finite numbers
    assert np.all(np.isfinite(rP)) and np.all(np.isfinite(tP))
