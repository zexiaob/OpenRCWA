import numpy as np
import pytest

from rcwa import make_n_from_table, make_epsilon_tensor_from_table
from rcwa.model.material import TensorMaterial


class DummySource:
    def __init__(self, wavelength):
        self.wavelength = wavelength


def test_make_n_from_table_flags_and_values():
    table = {
        'wavelength': np.array([1.50, 1.55, 1.60]),
        'n': np.array([1.50+0j, 1.51+0j, 1.52+0j]),
    }

    # Interpolation disabled should raise for mid-point
    n_no = make_n_from_table(table, allow_interpolation=False, allow_extrapolation=False)
    with pytest.raises(ValueError):
        _ = n_no(1.525)

    # Enable interpolation, check mid value
    n_yes = make_n_from_table(table, allow_interpolation=True, allow_extrapolation=False)
    n_mid = n_yes(1.525)
    assert np.isclose(n_mid.real, 1.505, atol=1e-9)

    # Extrapolation disabled raises
    with pytest.raises(ValueError):
        _ = n_yes(1.65)

    # Enable extrapolation, check slope continuation
    n_yes2 = make_n_from_table(table, allow_interpolation=True, allow_extrapolation=True)
    n_ext = n_yes2(1.65)
    # slope = (1.52-1.51)/(1.60-1.55) = 0.01/0.05 = 0.2; delta=0.05 -> 1.52+0.01=1.53
    assert np.isclose(n_ext.real, 1.53, atol=1e-9)


def test_make_epsilon_tensor_from_table_with_tensor_material():
    table = {
        'wavelength': np.array([1.50, 1.55, 1.60]),
        'n_xx': np.array([1.50, 1.51, 1.52]),
        'n_yy': np.array([1.48, 1.49, 1.50]),
        'n_zz': np.array([1.60, 1.60, 1.60]),
        # off-diagonals omitted -> default 0
    }

    # Interpolation disabled should raise when used via TensorMaterial
    eps_fn_no = make_epsilon_tensor_from_table(table, allow_interpolation=False, allow_extrapolation=False)
    ani_no = TensorMaterial(epsilon_tensor=eps_fn_no)
    ani_no.source = DummySource(1.525)
    with pytest.raises(ValueError):
        _ = ani_no.epsilon_tensor

    # Enable interpolation and check n_xx midpoint behavior -> epsilon = n^2
    eps_fn = make_epsilon_tensor_from_table(table, allow_interpolation=True, allow_extrapolation=False)
    ani = TensorMaterial(epsilon_tensor=eps_fn)
    ani.source = DummySource(1.525)
    eps = ani.epsilon_tensor
    assert np.isclose(eps[0, 0].real, (1.505**2), atol=1e-9)
