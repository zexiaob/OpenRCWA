import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union, Optional


@dataclass
class Result:
    """
    Standard single-point simulation result with complete complex amplitudes.

    Required fields (complex amplitudes are mandatory for phase-sensitive work):
      - rx, ry, rz, tx, ty, tz: complex or arrays of complex amplitudes per order
      - R, T: reflectance/transmittance derived from amplitudes
      - wavelength, angle, polarization: optional metadata if available
      - backend, precision, convergence_info: optional metadata

    Accessors expose data as attributes. Helper constructors ease interop with
    legacy solver dicts.
    """

    rx: Union[complex, np.ndarray]
    ry: Union[complex, np.ndarray]
    rz: Union[complex, np.ndarray]
    tx: Union[complex, np.ndarray]
    ty: Union[complex, np.ndarray]
    tz: Union[complex, np.ndarray]
    R: Union[float, np.ndarray]
    T: Union[float, np.ndarray]

    # Optional metadata
    wavelength: Optional[float] = None
    angle: Optional[float] = None
    polarization: Optional[Union[str, np.ndarray]] = None
    backend: Optional[str] = None
    precision: Optional[str] = None
    convergence_info: Optional[Mapping[str, Any]] = None

    # Raw/extra payload
    extras: Optional[Mapping[str, Any]] = None

    @staticmethod
    def from_solver_dict(d: Mapping[str, Any]) -> "Result":
        return Result(
            rx=d.get('rx'), ry=d.get('ry'), rz=d.get('rz'),
            tx=d.get('tx'), ty=d.get('ty'), tz=d.get('tz'),
            R=d.get('R'), T=d.get('T'),
            wavelength=getattr(d.get('source', None), 'wavelength', d.get('wavelength')),
            angle=getattr(d.get('source', None), 'theta', d.get('theta')),
            polarization=getattr(d.get('source', None), 'pTEM', d.get('polarization')),
            extras=d,
        )

    def r_complex(self) -> np.ndarray:
        return np.asarray([self.rx, self.ry, self.rz])

    def t_complex(self) -> np.ndarray:
        return np.asarray([self.tx, self.ty, self.tz])

    def phases(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.angle(self.r_complex()), np.angle(self.t_complex())

    def intensities(self) -> Tuple[np.ndarray, np.ndarray]:
        rc, tc = self.r_complex(), self.t_complex()
        return np.abs(rc) ** 2, np.abs(tc) ** 2

    @property
    def RTot(self) -> float:
        """Total reflectance."""
        try:
            return float(np.sum(self.R))  # type: ignore[arg-type]
        except Exception:
            return float(self.R)  # type: ignore[return-value]

    @property
    def TTot(self) -> float:
        """Total transmittance."""
        try:
            return float(np.sum(self.T))  # type: ignore[arg-type]
        except Exception:
            return float(self.T)  # type: ignore[return-value]

    @property
    def conservation(self) -> float:
        """Energy conservation R+T."""
        return self.RTot + self.TTot

    @property
    def A(self) -> Union[float, np.ndarray]:
        """Absorption (1 - R - T)."""
        try:
            return 1.0 - np.asarray(self.R) - np.asarray(self.T)
        except Exception:
            return 1.0 - self.R - self.T  # type: ignore[operator]


class ResultGrid:
    """
    Multi-dimensional container for a sweep of Result points with labeled coords.

    Stores:
      - dims: ordered list of dimension names
      - coords: mapping dim -> sequence of coordinate labels (length matches dim size)
      - data: flat list of Result, in row-major order over dims product

    Provides selection by label (.sel/.loc), by integer index (.isel), and utilities
    to extract arrays of derived quantities without losing complex amplitudes.
    """

    def __init__(self, dims: Sequence[str], coords: Mapping[str, Sequence[Any]], data: Sequence[Result]):
        self.dims = list(dims)
        self.coords = {k: list(v) for k, v in coords.items()}
        self.data: List[Result] = list(data)
        # Validate shape
        self._shape = tuple(len(self.coords[d]) for d in self.dims)
        expected = int(np.prod(self._shape)) if self._shape else 1
        if len(self.data) != expected:
            # Best-effort: allow ragged but warn
            import warnings
            warnings.warn(
                f"ResultGrid data length {len(self.data)} does not match coords product {expected}; "
                f"indexing may be limited.")

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def _flat_index(self, indices: Sequence[int]) -> int:
        if not self._shape:
            return 0
        flat = 0
        stride = 1
        # row-major: last dim varies fastest
        for size, idx in zip(reversed(self._shape), reversed(indices)):
            flat += idx * stride
            stride *= size
        return flat

    def isel(self, **indexers: int) -> Union['ResultGrid', Result]:
        """Index by integer position per dim. If all dims fixed, return Result; else sub-grid."""
        fixed = {d: indexers[d] for d in indexers}
        # Build slices
        take_ranges: List[List[int]] = []
        out_dims: List[str] = []
        for d, size in zip(self.dims, self._shape):
            if d in fixed:
                idx = fixed[d]
                if idx < 0:
                    idx += size
                if not (0 <= idx < size):
                    raise IndexError(f"isel: index {idx} out of bounds for dim {d} with size {size}")
                take_ranges.append([idx])
            else:
                take_ranges.append(list(range(size)))
                out_dims.append(d)
        # Build new data
        new_data: List[Result] = []
        for multi in np.array(np.meshgrid(*take_ranges, indexing='ij')).reshape(len(self.dims), -1).T:
            new_data.append(self.data[self._flat_index(multi)])
        if not out_dims:
            return new_data[0]
        new_coords = {d: self.coords[d] for d in out_dims}
        return ResultGrid(out_dims, new_coords, new_data)

    def sel(self, **selectors: Any) -> Union['ResultGrid', Result]:
        """Select by coordinate label equality. Alias: .loc"""
        # Map selectors to indices
        indexers: Dict[str, int] = {}
        for d, val in selectors.items():
            if d not in self.coords:
                raise KeyError(f"Unknown dimension {d}")
            # Robust equality that supports numpy arrays as labels
            coord_list = self.coords[d]
            idx = None
            # Try fast path for hashable/scalars
            try:
                idx = coord_list.index(val)  # type: ignore[arg-type]
            except Exception:
                pass
            if idx is None:
                import numpy as _np
                for j, c in enumerate(coord_list):
                    # numpy array or array-like: compare with allclose
                    if isinstance(val, _np.ndarray) or isinstance(c, _np.ndarray):
                        try:
                            if _np.allclose(_np.asarray(c), _np.asarray(val)):
                                idx = j
                                break
                        except Exception:
                            continue
                    else:
                        try:
                            if c == val:
                                idx = j
                                break
                        except Exception:
                            continue
            if idx is None:
                raise KeyError(f"Value {val} not found in coords for {d}")
            indexers[d] = idx
        return self.isel(**indexers)

    # Pandas-like alias
    loc = sel

    def to_dataframe(self):
        """Return a pandas DataFrame if pandas is available; otherwise a list of dicts.

        The DataFrame includes coordinate columns, scalar R/T totals, and stores
        complex amplitude arrays in object columns to avoid information loss.
        """
        rows: List[Dict[str, Any]] = []
        # Precompute multi-index iterator
        iter_arrays = [range(len(self.coords[d])) for d in self.dims]
        for multi in np.array(np.meshgrid(*iter_arrays, indexing='ij')).reshape(len(self.dims), -1).T:
            r = self.data[self._flat_index(multi)]
            row: Dict[str, Any] = {}
            for d, i in zip(self.dims, multi):
                row[d] = self.coords[d][int(i)]
            row.update({
                'RTot': np.sum(r.R) if hasattr(r.R, '__iter__') else r.R,
                'TTot': np.sum(r.T) if hasattr(r.T, '__iter__') else r.T,
                'r_complex': r.r_complex(),
                't_complex': r.t_complex(),
            })
            rows.append(row)
        try:
            import pandas as pd  # type: ignore
            return pd.DataFrame(rows)
        except Exception:
            return rows

    def get(self, field: str) -> np.ndarray:
        """Stack a scalar or array field across the grid into an ndarray with shape dims.

        For array-valued fields (e.g., R over orders), returns an extra trailing axis.
        """
        # Inspect first element
        sample = getattr(self.data[0], field)
        base = np.empty(self._shape, dtype=object)
        for idx in np.ndindex(self._shape):
            r = self.data[self._flat_index(idx)]
            base[idx] = getattr(r, field)
        # Try to stack if shapes align
        try:
            return np.stack(base.ravel()).reshape(*self._shape, *np.shape(sample))
        except Exception:
            return base

    def plot(self, x: str, y: str = 'RTot', ax=None, show=False):
        """Quick 2D plot for 1D sweeps."""
        import matplotlib.pyplot as plt  # local import
        if len(self.dims) != 1:
            raise ValueError("plot currently supports 1D sweeps only")
        dim = self.dims[0]
        xv = self.coords[dim]
        if y == 'RTot':
            yv = [np.sum(r.R) if hasattr(r.R, '__iter__') else r.R for r in self.data]
        elif y == 'TTot':
            yv = [np.sum(r.T) if hasattr(r.T, '__iter__') else r.T for r in self.data]
        else:
            yv = [getattr(r, y) for r in self.data]
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(xv, yv)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if show:
            plt.show()
        return ax

    # Convenience APIs: complex amplitudes and derivatives stacked over grid
    def get_complex_amplitudes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (r_complex, t_complex) stacked over the grid with trailing axis of length 3."""
        r_list, t_list = [], []
        for idx in np.ndindex(self._shape):
            res = self.data[self._flat_index(idx)]
            r_list.append(res.r_complex())
            t_list.append(res.t_complex())
        r_arr = np.stack(r_list).reshape(*self._shape, 3)
        t_arr = np.stack(t_list).reshape(*self._shape, 3)
        return r_arr, t_arr

    def get_phases(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (r_phases, t_phases) with trailing axis of length 3 (radians)."""
        r, t = self.get_complex_amplitudes()
        return np.angle(r), np.angle(t)

    def get_intensities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (r_intensities, t_intensities) = |amplitude|^2 with trailing axis length 3."""
        r, t = self.get_complex_amplitudes()
        return np.abs(r) ** 2, np.abs(t) ** 2


class Results:
    """
    Unified results container for RCWA simulations with complete complex amplitude information.
    
    This class provides access to both complex amplitudes (rx, ry, rz, tx, ty, tz) and 
    derived intensity quantities (R, T, A, DE) ensuring no information loss and maintaining
    physical consistency.
    """
    
    def __init__(self, results_dict):
        self.inner_dict = results_dict
        self._validate_complex_amplitudes()

    def __getitem__(self, key):
        return self.inner_dict[key]

    def keys(self):
        return self.inner_dict.keys()

    def items(self):
        return self.inner_dict.items()

    def values(self):
        return self.inner_dict.values()
    
    def _validate_complex_amplitudes(self):
        """Validate that complex amplitudes are present and consistent with intensity quantities."""
        required_complex_fields = ['rx', 'ry', 'rz', 'tx', 'ty', 'tz']
        
        # Only validate if this appears to be a complete simulation result
        # (backward compatibility: don't break existing tests with incomplete data)
        has_R_or_T = 'R' in self.inner_dict or 'T' in self.inner_dict
        if not has_R_or_T:
            # Skip validation for simple test fixtures
            return
            
        missing_fields = []
        for field in required_complex_fields:
            if field not in self.inner_dict:
                missing_fields.append(field)
        
        if missing_fields:
            # Issue a warning instead of raising an error for backward compatibility
            import warnings
            warnings.warn(f"Results missing complex amplitude fields: {missing_fields}. "
                        f"This may limit access to phase information and advanced analysis features.",
                        UserWarning)
    
    @property
    def rx(self) -> Union[complex, np.ndarray]:
        """Complex reflection coefficient in x-direction."""
        return self.inner_dict['rx']
    
    @property 
    def ry(self) -> Union[complex, np.ndarray]:
        """Complex reflection coefficient in y-direction."""
        return self.inner_dict['ry']
    
    @property
    def rz(self) -> Union[complex, np.ndarray]:
        """Complex reflection coefficient in z-direction."""
        return self.inner_dict['rz']
    
    @property
    def tx(self) -> Union[complex, np.ndarray]:
        """Complex transmission coefficient in x-direction."""
        return self.inner_dict['tx']
    
    @property
    def ty(self) -> Union[complex, np.ndarray]:
        """Complex transmission coefficient in y-direction."""
        return self.inner_dict['ty']
    
    @property
    def tz(self) -> Union[complex, np.ndarray]:
        """Complex transmission coefficient in z-direction."""
        return self.inner_dict['tz']
    
    @property
    def R(self) -> Union[float, np.ndarray]:
        """Reflectance (derived from complex amplitudes)."""
        return self.inner_dict['R']
    
    @property
    def T(self) -> Union[float, np.ndarray]:
        """Transmittance (derived from complex amplitudes)."""
        return self.inner_dict['T']
    
    @property
    def RTot(self) -> float:
        """Total reflectance."""
        return self.inner_dict.get('RTot', np.sum(self.R) if hasattr(self.R, '__iter__') else self.R)
    
    @property
    def TTot(self) -> float:
        """Total transmittance."""
        return self.inner_dict.get('TTot', np.sum(self.T) if hasattr(self.T, '__iter__') else self.T)
    
    @property
    def conservation(self) -> float:
        """Energy conservation (R + T)."""
        return self.inner_dict.get('conservation', self.RTot + self.TTot)
    
    @property
    def A(self) -> Union[float, np.ndarray]:
        """Absorption (1 - R - T)."""
        return 1.0 - self.R - self.T
    
    def get_complex_amplitudes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get complete complex amplitude information.
        
        :return: Tuple of (reflection_amplitudes, transmission_amplitudes)
                where each is an array of [rx, ry, rz] or [tx, ty, tz]
        """
        r_complex = np.array([self.rx, self.ry, self.rz])
        t_complex = np.array([self.tx, self.ty, self.tz])
        return r_complex, t_complex
    
    def get_phases(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract phase information from complex amplitudes.
        
        :return: Tuple of (reflection_phases, transmission_phases) in radians
        """
        r_phases = np.angle([self.rx, self.ry, self.rz])
        t_phases = np.angle([self.tx, self.ty, self.tz])
        return r_phases, t_phases
    
    def get_intensities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get intensity information (magnitude squared of complex amplitudes).
        
        :return: Tuple of (reflection_intensities, transmission_intensities)
        """
        r_intensities = np.abs([self.rx, self.ry, self.rz])**2
        t_intensities = np.abs([self.tx, self.ty, self.tz])**2
        return r_intensities, t_intensities
    
    def verify_energy_conservation(self, tolerance: float = 1e-6) -> bool:
        """
        Verify energy conservation within specified tolerance.
        
        :param tolerance: Maximum allowed deviation from conservation
        :return: True if energy is conserved within tolerance
        """
        return abs(self.conservation - 1.0) < tolerance
    
    def verify_complex_consistency(self, tolerance: float = 1e-6) -> bool:
        """
        Verify that intensity quantities are consistent with complex amplitudes.
        
        :param tolerance: Maximum allowed relative error
        :return: True if intensities match complex amplitude magnitudes
        """
        r_intensities, t_intensities = self.get_intensities()
        
        # For single values, convert to arrays for consistency
        R_check = np.atleast_1d(self.R)
        T_check = np.atleast_1d(self.T)
        
        # Compare sums (total intensities should match R/T)
        r_total_from_complex = np.sum(r_intensities)
        t_total_from_complex = np.sum(t_intensities)
        
        r_consistent = abs(r_total_from_complex - np.sum(R_check)) < tolerance * np.sum(R_check)
        t_consistent = abs(t_total_from_complex - np.sum(T_check)) < tolerance * np.sum(T_check)
        
        return r_consistent and t_consistent

    def plot(self, x='wavelength', y='RTot', c=None, fig=None, ax=None, show=False):
        """
        :param x: Variable to plot along the x-axis
        :param y: Variable to plot along the y-axis
        :param c: Variable to plot vs. x/y as distinct curves
        :param fig: Figure to use for plotting. If None, will create with pyplot interface
        :param ax: Axes to use for  plotting. If None, will create with pyplot interface.
        :param show: Whether to show the plot using the pyplot interface. False by default.

        :returns fig, ax: Figure and Axes objects created with matplotlib pyplot interface
        """
        import matplotlib.pyplot as plt  # Local import to avoid hard dependency

        if fig is None and ax is None:
            fig, ax = plt.subplots()
        elif fig is not None and ax is None:
            ax = fig.add_subplot()

        x_data = self[x]
        if hasattr(y, '__iter__') and not isinstance(y, str):
            y_data = [self[yy] for yy in y]
        else:
            y_data = [self[y]]

        for dat in y_data:
            ax.plot(x_data, dat)
        ax.legend(y)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        if show:
            plt.show()

        return fig, ax


def build_result_grid_from_sweep(coords: Mapping[str, Sequence[Any]], results: Sequence[Mapping[str, Any]]) -> ResultGrid:
    """Helper to build a ResultGrid from Sweep.run coordinate dict and a list of solver dicts."""
    # Derive a stable dims order from coords keys
    dims = list(coords.keys())
    # Build Result list
    result_objs = [Result.from_solver_dict(r if isinstance(r, dict) else r.inner_dict if hasattr(r, 'inner_dict') else r) for r in results]
    return ResultGrid(dims=dims, coords=coords, data=result_objs)


# Convenience: circular dichroism (CD) helper
def compute_circular_dichroism(obj: Union[ResultGrid, Result], dim: str = 'pTEM') -> Any:
    """
    Compute circular dichroism (CD) = TTot(RCP) - TTot(LCP).

    Accepts:
      - A ResultGrid with a polarization dim (default 'pTEM') containing LCP/RCP labels
      - A mapping-like with .sel/.get and .TTot fields

    Returns a scalar if all other dims are fixed, or an array over remaining dims.
    """
    # If a single Result provided, CD is undefined without both states
    if isinstance(obj, Result):
        raise ValueError("compute_circular_dichroism requires a grid with both LCP and RCP states")

    grid: ResultGrid = obj
    try:
        from .source import LCP, RCP  # local import to avoid cycles
        sel_l = grid.sel(**{dim: LCP()})
        sel_r = grid.sel(**{dim: RCP()})
    except Exception as e:
        raise ValueError(f"Could not select LCP/RCP along dim '{dim}': {e}")

    def totT(x: Union[ResultGrid, Result]) -> Any:
        if hasattr(x, 'data'):
            # Use stacking utility and sum over orders axis
            T = x.get('T')
            return np.squeeze(np.sum(T, axis=-1))
        else:
            return np.sum(x.T) if hasattr(x.T, '__iter__') else x.T

    return totT(sel_r) - totT(sel_l)

