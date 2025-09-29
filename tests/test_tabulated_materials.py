import numpy as np
import pytest

from rcwa import Material, TensorMaterial, Source, Layer


def make_source(wl):
    # Layer needed by Source; material values irrelevant here
    air_layer = Layer()
    src = Source(wavelength=wl, layer=air_layer)
    return src


def test_material_inline_table_interpolation_and_extrapolation_flags():
    data = {
        'wavelength': np.array([1.50, 1.55, 1.60]),
        'n': np.array([1.45, 1.46, 1.47], dtype=complex),
    }

    # No interpolation: request interior non-grid point should raise
    mat_no_interp = Material(data=data, allow_interpolation=False, allow_extrapolation=False)
    mat_no_interp.source = make_source(1.552)
    with pytest.raises(ValueError):
        _ = mat_no_interp.n

    # With interpolation: should return linear between 1.55 and 1.60
    mat_interp = Material(data=data, allow_interpolation=True, allow_extrapolation=False)
    mat_interp.source = make_source(1.575)
    n_mid = mat_interp.n
    # Expected n: 1.46 + 0.5*(1.47-1.46) = 1.465
    assert np.isclose(n_mid.real, 1.465, atol=1e-6)

    # Extrapolation disabled: below table should raise
    mat_no_extra = Material(data=data, allow_interpolation=True, allow_extrapolation=False)
    mat_no_extra.source = make_source(1.40)
    with pytest.raises(ValueError):
        _ = mat_no_extra.n

    # Extrapolation enabled: below table should linearly extrapolate
    mat_extra = Material(data=data, allow_interpolation=True, allow_extrapolation=True)
    mat_extra.source = make_source(1.45)
    n_ext = mat_extra.n
    # slope from first two points (1.50->1.55): (1.46-1.45)/0.05 = 0.2 per unit wl
    # delta wl = 1.45-1.50 = -0.05 => n = 1.45 + (-0.05)*0.2 = 1.44
    assert np.isclose(n_ext.real, 1.44, atol=1e-6)


def test_tensor_material_component_table_with_flags():
    table = {
        'wavelength': np.array([1.50, 1.55, 1.60]),
        'n_xx': np.array([1.50, 1.51, 1.52]),
        'n_yy': np.array([1.48, 1.49, 1.50]),
        'n_zz': np.array([1.60, 1.60, 1.60]),
    }

    # Disallow interpolation: interior request should raise
    ani_no_interp = TensorMaterial(epsilon_tensor=table, allow_interpolation=False, allow_extrapolation=False)
    ani_no_interp.source = make_source(1.525)
    with pytest.raises(ValueError):
        _ = ani_no_interp.epsilon_tensor

    # Allow interpolation: check epsilon_xx from n_xx^2 at mid-point (1.525 between 1.50 and 1.55)
    ani_interp = TensorMaterial(epsilon_tensor=table, allow_interpolation=True, allow_extrapolation=False)
    ani_interp.source = make_source(1.525)
    eps_mid = ani_interp.epsilon_tensor
    # n_xx mid: 1.50 + 0.5*(1.51-1.50) = 1.505 => eps_xx ~ 1.505^2
    assert np.isclose(eps_mid[0, 0].real, (1.505**2), atol=1e-6)

    # Extrapolation disabled: above range should raise
    ani_no_extra = TensorMaterial(epsilon_tensor=table, allow_interpolation=True, allow_extrapolation=False)
    ani_no_extra.source = make_source(1.70)
    with pytest.raises(ValueError):
        _ = ani_no_extra.epsilon_tensor

    # Extrapolation enabled: above range should extrapolate
    ani_extra = TensorMaterial(epsilon_tensor=table, allow_interpolation=True, allow_extrapolation=True)
    ani_extra.source = make_source(1.65)
    eps_ext = ani_extra.epsilon_tensor
    # n_xx slope: (1.52-1.51)/0.05 = 0.2; delta=1.65-1.60=0.05 => n_xx=1.52+0.01=1.53 => eps=1.53^2
    assert np.isclose(eps_ext[0, 0].real, (1.53**2), atol=1e-6)
