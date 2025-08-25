"""
I think the way this currently works is too convoluted. It needs to be refactored to be understandable.

"""
import numpy as np
import pandas as pd
import rcwa
import os
from rcwa.utils import CSVLoader, RIDatabaseLoader
import warnings
from numpy.typing import ArrayLike
from typing import Union, Callable
from pydantic import BaseModel, Field, validator, root_validator


# SI Unit Conversion Helpers (ROADMAP requirement)
def nm(value: float) -> float:
    """Convert nanometers to meters (SI units)."""
    return value * 1e-9


def um(value: float) -> float:
    """Convert micrometers to meters (SI units)."""
    return value * 1e-6


def mm(value: float) -> float:
    """Convert millimeters to meters (SI units)."""
    return value * 1e-3


def deg(value: float) -> float:
    """Convert degrees to radians."""
    return value * np.pi / 180


class Material:
    """
    Material class for defining materials permittivity / permeability / refractive index as a function of wavelength / angle.

    :param name: Material name to be looked up in database (i.e. Si)
    :param er: Complex-valued numerical permittivity value or function of wavelength
    :param ur: Complex-valued numerical permeability value or function of wavelength
    :param n: Complex-valued refractive index of material. Overrides er / ur
    :param source: Excitation source to link to material (mandatory for dispersive materials)
    :param filename: File containing n/k data for the material in question
    :param database_path: Raw file path within database
    """
    database = RIDatabaseLoader()

    def __init__(self, name=None, er=1, ur=1, n=None, database_path=None, filename=None, source=None):
        self.name = ''
        self.source = source
        self.dispersive = False
        self.loader = None

        if callable(er) or callable(ur):
            self.dispersive = True
            self._er_dispersive = er
            self._ur_dispersive = ur

        if name is not None or database_path is not None:
            self.dispersive = True
            self._load_from_database(name, filename=database_path)
        elif filename is not None:
            self.dispersive = True
            self._load_from_nk_table(filename=filename)
        elif callable(er) or callable(ur):
            self.dispersive = True
            self.dispersion_type = 'formula'
            if callable(er):
                self._er_dispersive = er
            else:
                self._er_dispersive = lambda x: er
            if callable(ur):
                self._ur_dispersive = ur
            else:
                self._ur_dispersive = lambda x: ur

        else:
            self.dispersive = False
            if n is None: # If the refractive index is not defined, go with the permittivity
                self._er = er
                self._ur = ur
                self._n = np.sqrt(er*ur)
            else: # If the refractive index is defined, ignore the permittivity and permeability
                self._n = n
                self._er = np.square(n)
                self._ur = 1

    def _set_dispersive_nk(self, data_dict):
        """
        Set our internal dispersive refractive index, permittivity, and permeability based on
        received data dictionary
        """
        self._n_dispersive = data_dict['n']
        self._er_dispersive = data_dict['er']
        self._ur_dispersive = data_dict['ur']
        if 'dispersion_type' in data_dict.keys():
            self.dispersion_type = data_dict['dispersion_type']
        if 'wavelength' in data_dict.keys():
            self.wavelengths = data_dict['wavelength']

    def _load_from_nk_table(self, filename: str):
        self.dispersion_type = 'tabulated'
        loader = CSVLoader(filename=filename)
        data_dict = loader.load()
        self._set_dispersive_nk(data_dict)

    def _load_from_database(self, material_name: str, filename: str = None):
        """
        Parses data from a CSV or database YAML file into a set of numpy arrays.

        :param filename: File containing n/k data for material in question
        """

        if filename is not None:
            file_to_load = os.path.join(rcwa.nk_dir, 'data', filename)

        if material_name in self.database.materials.keys():
            file_to_load = os.path.join(rcwa.nk_dir, 'data', self.database.materials[material_name])

        data_dict = self.database.load(file_to_load)
        self._set_dispersive_nk(data_dict)

    @property
    def n(self):
        if not self.dispersive:
            return self._n
        else:
            return self.lookupParameter(self._n_dispersive)

    @n.setter
    def n(self, n: float):
        self._er = np.square(n)
        self._ur = 1

    @property
    def er(self) -> float:
        if not self.dispersive:
            return self._er
        else:
            return self.lookupParameter(self._er_dispersive)

    @er.setter
    def er(self, er: complex):
        self._er = er

    @property
    def ur(self) -> complex:
        if not  self.dispersive:
            return self._ur
        else:
            return self.lookupParameter(self._ur_dispersive)

    @ur.setter
    def ur(self, ur: complex):
        self._ur = ur

    def lookupParameter(self, parameter: ArrayLike) -> complex:
        if self.dispersion_type == 'tabulated':
            return self.lookupNumeric(parameter)
        elif self.dispersion_type == 'formula':
            wavelength = self.source.wavelength
            return parameter(wavelength)

    def lookupNumeric(self, parameter: ArrayLike) -> complex:
        """
        Looks up a numeric value of a parameter

        :param parameter: Either _n_dispersive, _er_dispersive, or _ur_dispersive
        """
        wavelength = self.source.wavelength
        indexOfWavelength = np.searchsorted(self.wavelengths, wavelength)
        return_value = 0

        if wavelength > self.wavelengths[-1]: # Extrapolate if necessary
            slope = (parameter[-1] - parameter[-2]) / (self.wavelengths[-1] - self.wavelengths[-2])
            deltaWavelength = wavelength - self.wavelengths[-1]
            return_value = parameter[-1] + slope * deltaWavelength
            warnings.warn(f'Requested wavelength {wavelength} outside available material range {self.wavelengths[0]} - {self.wavelengths[-1]}')

        elif wavelength < self.wavelengths[0]: # Extrapolate the other direction if necessary
            slope = (parameter[1] - parameter[0]) / (self.wavelengths[1] - self.wavelengths[0])
            deltaWavelength = self.wavelengths[0] - wavelength
            return_value = parameter[0] - slope * deltaWavelength
            warnings.warn(f'Requested wavelength {wavelength} outside available material range {self.wavelengths[0]} - {self.wavelengths[-1]}')

        else: # Our wavelength is in the range over which we have data
            if wavelength == self.wavelengths[indexOfWavelength]: # We found the EXACT wavelength
                return_value = parameter[indexOfWavelength]
            else: # We need to interpolate the wavelength. The indexOfWavelength is pointing to the *next* value
                slope = (parameter[indexOfWavelength] - parameter[indexOfWavelength-1]) / (self.wavelengths[indexOfWavelength] - self.wavelengths[indexOfWavelength-1]) # wavelength spacing between two points
                deltaWavelength = wavelength - self.wavelengths[indexOfWavelength]
                return_value = parameter[indexOfWavelength] + slope * deltaWavelength

        return return_value


class TensorMaterial:
    """
    Anisotropic material class supporting 3x3 permittivity and permeability tensors.
    
    This class extends the basic Material concept to support anisotropic materials
    where the permittivity and permeability are 3x3 complex tensors rather than scalars.
    
    :param epsilon_tensor: 3x3 complex permittivity tensor, can be:
                          - Constant 3x3 array
                          - Function returning 3x3 array given wavelength
                          - Dictionary with tabulated data
    :param mu_tensor: 3x3 complex permeability tensor (default: identity matrix)
    :param source: Excitation source to link to material (mandatory for dispersive materials)
    :param name: Material name for identification
    """
    
    def __init__(self, epsilon_tensor=None, mu_tensor=None, source=None, name="anisotropic",
                 wavelength_range=None, thickness_range=(1e-12, 1e-3)):
        """
        Initialize TensorMaterial with enhanced validation.
        
        :param epsilon_tensor: 3x3 permittivity tensor (constant/function/table)
        :param mu_tensor: 3x3 permeability tensor (optional, default: identity)
        :param source: Associated source object (required for dispersive materials)
        :param name: Material identifier
        :param wavelength_range: Valid wavelength range in meters [min, max]
        :param thickness_range: Valid thickness range in meters [min, max]
        """
        # Validate inputs according to ROADMAP requirements
        self._validate_inputs(epsilon_tensor, mu_tensor, wavelength_range, thickness_range)
        
        self.name = name
        self.source = source
        self.dispersive = False
        self.wavelength_range = wavelength_range
        self.thickness_range = thickness_range
        
        # Initialize default tensors
        if epsilon_tensor is None:
            # Default isotropic case
            self._epsilon_tensor = np.eye(3, dtype=complex)
        elif callable(epsilon_tensor):
            # Function-based dispersive tensor
            self.dispersive = True
            self._epsilon_dispersive = epsilon_tensor
        elif isinstance(epsilon_tensor, dict):
            # Tabulated data
            self.dispersive = True
            self._load_tensor_from_table(epsilon_tensor)
        else:
            # Constant tensor
            self._epsilon_tensor = np.array(epsilon_tensor, dtype=complex)
            if self._epsilon_tensor.shape != (3, 3):
                raise ValueError("Epsilon tensor must be 3x3")
        
        if mu_tensor is None:
            # Default: identity matrix (non-magnetic)
            self._mu_tensor = np.eye(3, dtype=complex)
        elif callable(mu_tensor):
            self.dispersive = True
            self._mu_dispersive = mu_tensor
        elif isinstance(mu_tensor, dict):
            self.dispersive = True
            self._load_mu_tensor_from_table(mu_tensor)
        else:
            self._mu_tensor = np.array(mu_tensor, dtype=complex)
            if self._mu_tensor.shape != (3, 3):
                raise ValueError("Mu tensor must be 3x3")
        
        # Warn if dispersive material has no source
        if self.dispersive and source is None:
            warnings.warn("Dispersive materials should have an associated source")
    
    def _validate_inputs(self, epsilon_tensor, mu_tensor, wavelength_range, thickness_range):
        """Validate inputs according to ROADMAP requirements."""
        
        # Validate wavelength range (SI units - meters)
        if wavelength_range is not None:
            if len(wavelength_range) != 2:
                raise ValueError("Wavelength range must be [min, max]")
            min_wl, max_wl = wavelength_range
            if min_wl <= 0 or max_wl <= 0:
                raise ValueError("Wavelengths must be positive")
            if min_wl >= max_wl:
                raise ValueError("Min wavelength must be < max wavelength")
            # Check if values seem reasonable for SI units (meters)
            if min_wl > 1e-3:  # > 1mm seems too large for wavelength
                raise ValueError("Wavelength seems too large - ensure SI units (meters). Use nm()/um() helpers.")
            if max_wl < 1e-9:  # < 1nm seems too small
                raise ValueError("Wavelength seems too small - ensure SI units (meters). Use nm()/um() helpers.")
        
        # Validate thickness range (SI units - meters)
        if thickness_range is not None:
            min_t, max_t = thickness_range
            if min_t <= 0 or max_t <= 0:
                raise ValueError("Thicknesses must be positive")
            if min_t >= max_t:
                raise ValueError("Min thickness must be < max thickness")
            # Check if values seem reasonable for SI units (meters)
            if min_t > 1e-2:  # > 1cm seems too large for typical layers
                raise ValueError("Thickness seems too large - ensure SI units (meters). Use nm()/um() helpers.")
            if max_t < 1e-12:  # < 1pm seems too small
                raise ValueError("Thickness seems too small - ensure SI units (meters). Use nm()/um() helpers.")
        
        # Validate tensor inputs
        if epsilon_tensor is not None and isinstance(epsilon_tensor, np.ndarray):
            if epsilon_tensor.shape != (3, 3):
                raise ValueError(f"Epsilon tensor must be 3x3, got shape {epsilon_tensor.shape}")
        
        if mu_tensor is not None and isinstance(mu_tensor, np.ndarray):
            if mu_tensor.shape != (3, 3):
                raise ValueError(f"Mu tensor must be 3x3, got shape {mu_tensor.shape}")
                raise ValueError("Mu tensor must be 3x3")
    
    def _load_tensor_from_table(self, tensor_data: dict):
        """Load epsilon tensor from tabulated data"""
        if 'wavelength' not in tensor_data:
            raise ValueError("Tensor data must contain 'wavelength' key")
        
        self.wavelengths = np.array(tensor_data['wavelength'])
        
        # Handle different tensor storage formats
        if 'epsilon_xx' in tensor_data:
            # Component-wise storage
            self._epsilon_tensor_table = np.zeros((len(self.wavelengths), 3, 3), dtype=complex)
            components = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
            for i, comp in enumerate(components):
                row, col = i // 3, i % 3
                key = f'epsilon_{comp}'
                if key in tensor_data:
                    self._epsilon_tensor_table[:, row, col] = tensor_data[key]
                else:
                    # Default to zero for off-diagonal, identity for diagonal
                    if row == col:
                        self._epsilon_tensor_table[:, row, col] = 1.0
        elif 'epsilon_tensor' in tensor_data:
            # Full tensor storage
            self._epsilon_tensor_table = np.array(tensor_data['epsilon_tensor'])
        else:
            raise ValueError("Tensor data must contain either component data or 'epsilon_tensor'")
    
    def _load_mu_tensor_from_table(self, tensor_data: dict):
        """Load mu tensor from tabulated data (similar to epsilon)"""
        # Similar implementation as epsilon tensor
        pass
    
    @property
    def epsilon_tensor(self) -> np.ndarray:
        """Get the 3x3 permittivity tensor at current wavelength"""
        if not self.dispersive:
            return self._epsilon_tensor
        else:
            if hasattr(self, '_epsilon_dispersive'):
                # Function-based
                return self._epsilon_dispersive(self.source.wavelength)
            else:
                # Table-based
                return self._lookup_tensor(self._epsilon_tensor_table)
    
    @property
    def mu_tensor(self) -> np.ndarray:
        """Get the 3x3 permeability tensor at current wavelength"""
        if not self.dispersive or not hasattr(self, '_mu_dispersive'):
            return self._mu_tensor
        else:
            if hasattr(self, '_mu_dispersive'):
                return self._mu_dispersive(self.source.wavelength)
            else:
                return self._lookup_tensor(self._mu_tensor_table)
    
    def _lookup_tensor(self, tensor_table: np.ndarray) -> np.ndarray:
        """Look up tensor value at current wavelength using interpolation"""
        if self.source is None:
            raise ValueError("Source must be set for dispersive materials")
        
        wavelength = self.source.wavelength
        
        # Find the interpolation indices
        idx = np.searchsorted(self.wavelengths, wavelength)
        
        if wavelength <= self.wavelengths[0]:
            return tensor_table[0]
        elif wavelength >= self.wavelengths[-1]:
            return tensor_table[-1]
        else:
            # Linear interpolation
            w1, w2 = self.wavelengths[idx-1], self.wavelengths[idx]
            t1, t2 = tensor_table[idx-1], tensor_table[idx]
            alpha = (wavelength - w1) / (w2 - w1)
            return t1 + alpha * (t2 - t1)
    
    @classmethod
    def from_diagonal(cls, eps_xx: Union[complex, Callable], 
                           eps_yy: Union[complex, Callable] = None, 
                           eps_zz: Union[complex, Callable] = None, 
                           **kwargs):
        """
        Create TensorMaterial from diagonal elements (uniaxial/biaxial crystals)
        
        :param eps_xx: xx component of permittivity tensor
        :param eps_yy: yy component (defaults to eps_xx for uniaxial)
        :param eps_zz: zz component (defaults to eps_xx for isotropic)
        """
        if eps_yy is None:
            eps_yy = eps_xx
        if eps_zz is None:
            eps_zz = eps_xx
        
        if callable(eps_xx):
            # Dispersive diagonal tensor
            def tensor_func(wl):
                return np.diag([eps_xx(wl), eps_yy(wl), eps_zz(wl)])
            return cls(epsilon_tensor=tensor_func, **kwargs)
        else:
            # Constant diagonal tensor
            tensor = np.diag([eps_xx, eps_yy, eps_zz])
            return cls(epsilon_tensor=tensor, **kwargs)
    
    def rotated(self, rotation_matrix: np.ndarray):
        """
        Create a new TensorMaterial with rotated tensor
        
        :param rotation_matrix: 3x3 rotation matrix
        :return: New TensorMaterial instance with rotated tensors
        """
        if not self.dispersive:
            # Constant case: ε' = R ε R^T
            new_eps = rotation_matrix @ self._epsilon_tensor @ rotation_matrix.T
            new_mu = rotation_matrix @ self._mu_tensor @ rotation_matrix.T
            return TensorMaterial(epsilon_tensor=new_eps, mu_tensor=new_mu, 
                                name=f"{self.name}_rotated", source=self.source)
        else:
            # Dispersive case: wrap the function
            if hasattr(self, '_epsilon_dispersive'):
                def rotated_eps_func(wl):
                    eps = self._epsilon_dispersive(wl)
                    return rotation_matrix @ eps @ rotation_matrix.T
                epsilon_tensor = rotated_eps_func
            else:
                # Table-based rotation would be more complex
                raise NotImplementedError("Rotation of tabulated tensor materials not yet implemented")
            
            if hasattr(self, '_mu_dispersive'):
                def rotated_mu_func(wl):
                    mu = self._mu_dispersive(wl)
                    return rotation_matrix @ mu @ rotation_matrix.T
                mu_tensor = rotated_mu_func
            else:
                mu_tensor = rotation_matrix @ self._mu_tensor @ rotation_matrix.T
            
            return TensorMaterial(epsilon_tensor=epsilon_tensor, mu_tensor=mu_tensor,
                                name=f"{self.name}_rotated", source=self.source)
