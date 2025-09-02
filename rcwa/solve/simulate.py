from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

from rcwa.model.layer import LayerStack, Layer
from rcwa.solve.source import Source, LCP as _LCP, RCP as _RCP
from rcwa.solve.sweep import Sweep
from rcwa.solve.results import Result, ResultGrid, Result as _Result, build_result_grid_from_sweep


def _is_seq(x: Any) -> bool:
    return isinstance(x, (list, tuple, np.ndarray)) and not isinstance(x, (str, bytes))


def _normalize_polarization(pol: Union[str, Sequence[Union[str, complex, float]], np.ndarray, None]) -> Union[np.ndarray, List[np.ndarray], None]:
    if pol is None:
        return None
    def one(p):
        if isinstance(p, str):
            s = p.strip().upper()
            if s == 'TE':
                return np.array([1.0, 0.0])
            if s == 'TM':
                return np.array([0.0, 1.0])
            if s == 'LCP':
                return _LCP()
            if s == 'RCP':
                return _RCP()
            raise ValueError(f"Unknown polarization string: {p}")
        p = np.asarray(p)
        if p.shape != (2,):
            raise ValueError("polarization must be a 2-vector [pTE, pTM]")
        return p
    if _is_seq(pol):
        return [one(p) for p in pol]  # type: ignore
    return one(pol)  # type: ignore


def simulate(
    stack: LayerStack,
    wavelength: Union[float, Sequence[float], np.ndarray],
    *,
    theta: Union[float, Sequence[float]] = 0.0,
    phi: Union[float, Sequence[float]] = 0.0,
    polarization: Union[str, Sequence[Union[str, complex, float]], np.ndarray, None] = None,
    n_harmonics: Union[int, Tuple[int, int]] = 1,
    backend: str = 'serial',
    return_grid: bool = True,
) -> Union[_Result, ResultGrid, Dict[str, Any]]:
    """High-level one-liner simulation API.

    - For scalar inputs, returns a Result (single-point).
    - For any vector input, performs a sweep and returns a ResultGrid.

    Args:
        stack: LayerStack describing the scene.
        wavelength: scalar or sequence of wavelengths.
        theta, phi: scalar or sequences (radians).
        polarization: 'TE'|'TM'|'LCP'|'RCP' or 2-vector [pTE,pTM] or list thereof.
        n_harmonics: integer or (Nx,Ny) tuple.
        backend: 'serial'|'loky'|'thread'|'process' for sweeps.
        return_grid: when sweeping, if False return the raw dict from Sweep.run.
    """
    # Build a base Source once (polarization applied per point below if sweeping)
    base_src = Source(
        wavelength=float(wavelength if not _is_seq(wavelength) else np.asarray(wavelength)[0]),
        theta=float(theta if not _is_seq(theta) else np.asarray(theta)[0]),
        phi=float(phi if not _is_seq(phi) else np.asarray(phi)[0]),
        pTEM=[1.0, 0.0],  # default; will be overridden below if provided
        layer=stack.incident_layer,
    )

    pol_norm = _normalize_polarization(polarization)

    # Determine if inputs define a sweep: sequences for scalar params; for polarization, only a list means sweep
    is_sweep = False
    if _is_seq(wavelength):
        is_sweep = True
    if _is_seq(theta):
        is_sweep = True
    if _is_seq(phi):
        is_sweep = True
    if isinstance(pol_norm, list):  # list of polarization states implies sweep
        is_sweep = True

    if not is_sweep:
        # Single point
        if pol_norm is not None:
            base_src.pTEM = pol_norm  # type: ignore
        from rcwa.core.solver import Solver
        solver = Solver(stack, base_src, n_harmonics=n_harmonics)
        res = solver.solve()
        # Wrap as unified Result
        return Result.from_solver_dict(res.inner_dict if hasattr(res, 'inner_dict') else res)  # type: ignore

    # Sweep case
    params: Dict[str, Any] = {}
    if _is_seq(wavelength):
        params['wavelength'] = list(wavelength)  # type: ignore
    if _is_seq(theta):
        params['theta'] = list(theta)  # type: ignore
    if _is_seq(phi):
        params['phi'] = list(phi)  # type: ignore
    if isinstance(pol_norm, list):
        params['pTEM'] = list(pol_norm)  # type: ignore
    # If pol_norm is a single vector and other params sweep, keep it fixed by setting base_src.pTEM
    if pol_norm is not None and not isinstance(pol_norm, list):
        base_src.pTEM = pol_norm  # type: ignore

    sweep = Sweep(params, backend=backend if backend else 'serial')
    out = sweep.run(stack, base_src, n_harmonics=n_harmonics)
    if return_grid:
        grid = out.get('result_grid')
        if grid is not None:
            return grid
        # Fallback if grid couldn't be built
        return build_result_grid_from_sweep(out.get('coords', {}), out.get('results', []))
    return out


__all__ = ["simulate"]
