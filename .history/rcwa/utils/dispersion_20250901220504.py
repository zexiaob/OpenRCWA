"""
Dispersion utilities: tabulated dispersion for scalar or tensor n(λ).
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence
import numpy as np


def tabulated_dispersion(wavelengths: Sequence[float], values: Sequence[np.ndarray]) -> Callable[[float], np.ndarray]:
    """Create an interpolating function for refractive index (scalar or tensor) vs wavelength.

    - wavelengths: strictly increasing sequence
    - values: same length, each is a scalar or a 3x3 tensor (numpy array)

    Returns a function n(wl) that linearly interpolates element-wise. Raises ValueError when wl
    is outside the provided domain.
    """
    wl = np.asarray(wavelengths, dtype=float)
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError("wavelengths must be a 1D array with at least two points")
    if not np.all(np.diff(wl) > 0):
        raise ValueError("wavelengths must be strictly increasing")

    vals = list(values)
    if len(vals) != wl.size:
        raise ValueError("values length must match wavelengths length")

    # Normalize shapes: allow scalars or arrays; for arrays, enforce consistent shape
    first = np.asarray(vals[0])
    shape = first.shape
    for i in range(1, len(vals)):
        vi = np.asarray(vals[i])
        if vi.shape != shape:
            raise ValueError("All values must have the same shape for interpolation")

    stacked = np.stack([np.asarray(v) for v in vals], axis=0)  # shape: (N, ...)

    def interp_fn(wl_query: float):
        x = float(wl_query)
        if x < wl[0] or x > wl[-1]:
            raise ValueError("Query wavelength outside tabulated range")
        # Find right index for interpolation interval
        idx = np.searchsorted(wl, x)
        if idx == 0:
            return stacked[0]
        if idx == wl.size:
            return stacked[-1]
        x0, x1 = wl[idx-1], wl[idx]
        y0, y1 = stacked[idx-1], stacked[idx]
        t = (x - x0) / (x1 - x0)
        return (1.0 - t) * y0 + t * y1

    return interp_fn
