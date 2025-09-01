import numpy as np
from numpy.typing import ArrayLike


def tabulated_dispersion(wavelengths: ArrayLike, values: ArrayLike):
    """Create a wavelength-dependent function from tabulated data.

    Parameters
    ----------
    wavelengths : array-like
        1D array of wavelengths (in meters).
    values : array-like
        Values at each wavelength. Can be scalars or arrays (e.g., 3x3 tensors).

    Returns
    -------
    callable
        Function f(wl) performing linear interpolation between data points.
        Raises ValueError if wl is outside the tabulated range.
    """
    wl_arr = np.asarray(wavelengths, dtype=float)
    val_arr = np.asarray(values)
    if wl_arr.ndim != 1:
        raise ValueError("wavelengths must be a 1D array")
    # Ensure data are sorted by wavelength
    if np.any(np.diff(wl_arr) < 0):
        idx = np.argsort(wl_arr)
        wl_arr = wl_arr[idx]
        val_arr = val_arr[idx]

    def f(wl: float):
        wl = float(wl)
        if wl < wl_arr[0] or wl > wl_arr[-1]:
            raise ValueError(
                f"Wavelength {wl} outside data range {wl_arr[0]} - {wl_arr[-1]}"
            )
        idx = np.searchsorted(wl_arr, wl)
        if wl == wl_arr[idx-1]:
            return val_arr[idx-1]
        if idx < len(wl_arr) and wl == wl_arr[idx]:
            return val_arr[idx]
        w1, w2 = wl_arr[idx-1], wl_arr[idx]
        v1, v2 = val_arr[idx-1], val_arr[idx]
        t = (wl - w1) / (w2 - w1)
        return v1 + t * (v2 - v1)

    return f