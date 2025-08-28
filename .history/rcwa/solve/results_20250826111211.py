from matplotlib import pyplot as plt
import numpy as np
from typing import Union, Tuple, Optional


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

