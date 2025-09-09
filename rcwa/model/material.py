"""
I think the way this currently works is too convoluted. It needs to be refactored to be understandable.

"""
import numpy as np
import rcwa
import os
from rcwa.utils import CSVLoader, load_nk_database_file
import warnings
from numpy.typing import ArrayLike
from typing import Union, Callable


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
    database = type('SimpleDB', (), {'materials': {
        'Pt': 'main/Pt/Rakic-BB.yml',
        'Si': 'main/Si/Schinke.yml',
        'Ag': 'main/Ag/Johnson.yml',
        'Ti': 'main/Ti/Johnson.yml',
        'Au': 'main/Au/Johnson.yml',
        'SiO2': 'main/SiO2/Radhakrishnan-o.yml',
    }})()

    def __init__(self, name=None, er=1, ur=1, n=None, database_path=None, filename=None, source=None,
                 data: Union[None, dict] = None,
                 allow_interpolation: bool = False,
                 allow_extrapolation: bool = False):
        self.name = ''
        self.source = source
        self.dispersive = False
        self.loader = None

        # Handle direct functional definitions first (formula based)
        if callable(n):
            # Refractive index provided as function of wavelength
            self.dispersive = True
            self.dispersion_type = 'formula'
            self._n_dispersive = n
            # Derive permittivity/permeability from n (assuming non-magnetic unless ur provided)
            self._er_dispersive = lambda wl: np.square(n(wl))
            if callable(ur):
                self._ur_dispersive = ur
            else:
                self._ur_dispersive = lambda wl: ur

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
            # Derive refractive index from epsilon and mu when no explicit n is provided
            self._n_dispersive = lambda wl: np.sqrt(self._er_dispersive(wl) * self._ur_dispersive(wl))

        # Handle tabulated/database materials
        if name is not None or database_path is not None:
            self.dispersive = True
            self._load_from_database(name, filename=database_path)
        elif filename is not None:
            self.dispersive = True
            self._load_from_nk_table(filename=filename)
        elif data is not None:
            # User-supplied inline table
            self.dispersive = True
            self.dispersion_type = 'tabulated'
            self._set_from_inline_table(data)
            # Apply strict lookup policy only to inline tables
            self._lookup_policy = {
                'allow_interpolation': bool(allow_interpolation),
                'allow_extrapolation': bool(allow_extrapolation),
            }

        # Constant materials (non-dispersive)
        if not self.dispersive:
            if n is None:
                # If the refractive index is not defined, go with the permittivity
                self._er = er
                self._ur = ur
                self._n = np.sqrt(er*ur)
            else:
                # If the refractive index is defined, ignore the permittivity and permeability
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
            data_dict = load_nk_database_file(file_to_load)
        elif material_name in Material.database.materials.keys():
            file_to_load = os.path.join(rcwa.nk_dir, 'data', Material.database.materials[material_name])
            data_dict = load_nk_database_file(file_to_load)
        else:
            raise ValueError("Material not found in database")

        self._set_dispersive_nk(data_dict)

    def _set_from_inline_table(self, table: dict):
        """Initialize from in-memory table with keys: 'wavelength' and 'n' or 'er' (optional 'ur')."""
        if 'wavelength' not in table:
            raise ValueError("'data' must include 'wavelength'")
        wl = np.array(table['wavelength'], dtype=float)
        if wl.ndim != 1 or wl.size < 2:
            raise ValueError("'wavelength' must be 1D with at least 2 points")
        order = np.argsort(wl)
        wl = wl[order]

        if 'n' in table:
            n_arr = np.array(table['n'], dtype=complex)
            if n_arr.shape[0] != wl.shape[0]:
                raise ValueError("Length of 'n' must match 'wavelength'")
            n_arr = n_arr[order]
            er_arr = np.square(n_arr)
            ur_arr = np.ones_like(er_arr)
        elif 'er' in table:
            er_arr = np.array(table['er'], dtype=complex)
            if er_arr.shape[0] != wl.shape[0]:
                raise ValueError("Length of 'er' must match 'wavelength'")
            er_arr = er_arr[order]
            if 'ur' in table:
                ur_arr = np.array(table['ur'], dtype=complex)
                if ur_arr.shape[0] != wl.shape[0]:
                    raise ValueError("Length of 'ur' must match 'wavelength'")
                ur_arr = ur_arr[order]
            else:
                ur_arr = np.ones_like(er_arr)
            n_arr = np.sqrt(er_arr * ur_arr)
        else:
            raise ValueError("'data' must include either 'n' or 'er'")

        self._set_dispersive_nk({'wavelength': wl, 'n': n_arr, 'er': er_arr, 'ur': ur_arr, 'dispersion_type': 'tabulated'})

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

        # Determine policy: only active for inline user tables
        policy = getattr(self, '_lookup_policy', None)

        if wavelength > self.wavelengths[-1]: # Extrapolate if necessary
            if policy is not None and not policy.get('allow_extrapolation', False):
                raise ValueError("Extrapolation disabled. Set allow_extrapolation=True to enable.")
            slope = (parameter[-1] - parameter[-2]) / (self.wavelengths[-1] - self.wavelengths[-2])
            deltaWavelength = wavelength - self.wavelengths[-1]
            return_value = parameter[-1] + slope * deltaWavelength
            warnings.warn(f'Requested wavelength {wavelength} outside available material range {self.wavelengths[0]} - {self.wavelengths[-1]}')

        elif wavelength < self.wavelengths[0]: # Extrapolate the other direction if necessary
            if policy is not None and not policy.get('allow_extrapolation', False):
                raise ValueError("Extrapolation disabled. Set allow_extrapolation=True to enable.")
            slope = (parameter[1] - parameter[0]) / (self.wavelengths[1] - self.wavelengths[0])
            deltaWavelength = self.wavelengths[0] - wavelength
            return_value = parameter[0] - slope * deltaWavelength
            warnings.warn(f'Requested wavelength {wavelength} outside available material range {self.wavelengths[0]} - {self.wavelengths[-1]}')

        else: # Our wavelength is in the range over which we have data
            if wavelength == self.wavelengths[indexOfWavelength]: # We found the EXACT wavelength
                return_value = parameter[indexOfWavelength]
            else: # We need to interpolate the wavelength. The indexOfWavelength is pointing to the *next* value
                if policy is not None and not policy.get('allow_interpolation', False):
                    raise ValueError("Interpolation disabled. Set allow_interpolation=True to enable.")
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
    :param n_tensor: 3x3 refractive index tensor (alternative to epsilon_tensor).
                     If provided, epsilon_tensor is ignored and computed from n_tensor**2.
                     Can be constant 3x3 array or function of wavelength.
    :param mu_tensor: 3x3 complex permeability tensor (default: identity matrix)
    :param source: Excitation source to link to material (mandatory for dispersive materials)
    :param name: Material name for identification
    """
    
    def __init__(self, epsilon_tensor=None, mu_tensor=None, source=None, name="anisotropic",
                 wavelength_range=None, thickness_range=(1e-12, 1e-3), n_tensor=None,
                 allow_interpolation: bool = False,
                 allow_extrapolation: bool = False):
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
        # If refractive index tensor is provided, convert to epsilon tensor first
        if n_tensor is not None and epsilon_tensor is not None:
            raise ValueError("Specify either epsilon_tensor or n_tensor, not both")

        if n_tensor is not None:
            if callable(n_tensor):
                self.dispersive = True
                self._n_dispersive = lambda wl: np.array(n_tensor(wl), dtype=complex)
                def eps_from_n(wl):
                    n_mat = self._n_dispersive(wl)
                    return np.square(n_mat)
                epsilon_tensor = eps_from_n
            else:
                n_arr = np.array(n_tensor, dtype=complex)
                self._n_tensor = n_arr
                epsilon_tensor = np.square(n_arr)

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
            # Tabulated data (epsilon_* or n_*)
            self.dispersive = True
            self._load_tensor_from_table(epsilon_tensor)
            self._tensor_lookup_policy = {
                'allow_interpolation': bool(allow_interpolation),
                'allow_extrapolation': bool(allow_extrapolation),
            }
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
        """Load epsilon tensor from tabulated data or refractive index tensor data."""
        if 'wavelength' not in tensor_data:
            raise ValueError("Tensor data must contain 'wavelength' key")
        wl = np.array(tensor_data['wavelength'], dtype=float)
        order = np.argsort(wl)
        self.wavelengths = wl[order]

        def build_table(prefix: str) -> np.ndarray:
            table = np.zeros((len(self.wavelengths), 3, 3), dtype=complex)
            comps = ['xx','xy','xz','yx','yy','yz','zx','zy','zz']
            found = False
            for i, comp in enumerate(comps):
                r, c = i // 3, i % 3
                k = f'{prefix}_{comp}'
                if k in tensor_data:
                    arr = np.array(tensor_data[k], dtype=complex)
                    if arr.shape[0] != wl.shape[0]:
                        raise ValueError(f"'{k}' length must match 'wavelength'")
                    table[:, r, c] = arr[order]
                    found = True
                else:
                    if r == c:
                        table[:, r, c] = 1.0
            if not found:
                raise ValueError(f"No '{prefix}_*' components found")
            return table

        if 'epsilon_tensor' in tensor_data or 'n_tensor' in tensor_data:
            # Full tensor arrays provided across wavelengths
            key = 'epsilon_tensor' if 'epsilon_tensor' in tensor_data else 'n_tensor'
            arr = np.array(tensor_data[key])
            if arr.ndim != 3 or arr.shape[1:] != (3,3) or arr.shape[0] != wl.shape[0]:
                raise ValueError(f"'{key}' must have shape [N,3,3]")
            arr = arr[order]
            if key == 'n_tensor':
                # Keep n-table and derive epsilon on lookup (interpolate n first, then square)
                self._n_tensor_table = arr.astype(complex)
            else:
                self._epsilon_tensor_table = arr.astype(complex)
        elif any(k.startswith('epsilon_') for k in tensor_data.keys()):
            self._epsilon_tensor_table = build_table('epsilon')
        elif any(k.startswith('n_') for k in tensor_data.keys()):
            # Keep n-table and derive epsilon on lookup (interpolate n first, then square)
            self._n_tensor_table = build_table('n')
        else:
            raise ValueError("Tensor data must contain epsilon_* or n_* components, or epsilon_tensor/n_tensor array")
    
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
            # Unified path: evaluate at current source wavelength
            if self.source is None:
                raise ValueError("Source must be set for dispersive materials")
            return self._tensor_at_wavelength(self.source.wavelength)
    
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

    @property
    def n_tensor(self) -> np.ndarray:
        """Get the 3x3 refractive index tensor if defined"""
        if hasattr(self, '_n_tensor'):
            return self._n_tensor
        elif hasattr(self, '_n_dispersive'):
            if self.source is None:
                raise ValueError("Source must be set for dispersive materials")
            return self._n_dispersive(self.source.wavelength)
        else:
            # Approximate derivation from epsilon tensor
            return np.sqrt(self.epsilon_tensor)
    
    def _lookup_tensor(self, tensor_table: np.ndarray) -> np.ndarray:
        """Look up tensor value at current wavelength using interpolation.
        If n-tables are present, interpolate n first and square to epsilon.
        """
        if self.source is None:
            raise ValueError("Source must be set for dispersive materials")
        return self._tensor_at_wavelength(self.source.wavelength)

    def _tensor_at_wavelength(self, wavelength: float) -> np.ndarray:
        """Evaluate epsilon tensor at an explicit wavelength.

        Prefers n-table interpolation when available, then squares to epsilon.
        Respects allow_interpolation/allow_extrapolation policy.
        """
        policy = getattr(self, '_tensor_lookup_policy', None)

        def interpolate_table(table: np.ndarray) -> np.ndarray:
            idx = np.searchsorted(self.wavelengths, wavelength)
            # Left of range
            if wavelength < self.wavelengths[0]:
                if policy is not None and not policy.get('allow_extrapolation', False):
                    raise ValueError("Tensor extrapolation disabled. Set allow_extrapolation=True to enable.")
                w1, w2 = self.wavelengths[0], self.wavelengths[1]
                t1, t2 = table[0], table[1]
                alpha = (wavelength - w1) / (w2 - w1)
                return t1 + alpha * (t2 - t1)
            # Right of range
            if wavelength > self.wavelengths[-1]:
                if policy is not None and not policy.get('allow_extrapolation', False):
                    raise ValueError("Tensor extrapolation disabled. Set allow_extrapolation=True to enable.")
                w1, w2 = self.wavelengths[-2], self.wavelengths[-1]
                t1, t2 = table[-2], table[-1]
                alpha = (wavelength - w2) / (w2 - w1) + 1.0
                return t1 + alpha * (t2 - t1)
            # Exactly at first point
            if wavelength == self.wavelengths[0]:
                return table[0]
            # Exactly at some grid point (not first)
            if idx < len(self.wavelengths) and wavelength == self.wavelengths[idx]:
                return table[idx]
            # Interior interpolation
            if policy is not None and not policy.get('allow_interpolation', False):
                raise ValueError("Tensor interpolation disabled. Set allow_interpolation=True to enable.")
            w1, w2 = self.wavelengths[idx-1], self.wavelengths[idx]
            t1, t2 = table[idx-1], table[idx]
            alpha = (wavelength - w1) / (w2 - w1)
            return t1 + alpha * (t2 - t1)

        # Prefer n-table if available for physical interpolation
        if hasattr(self, '_n_tensor_table'):
            n_interp = interpolate_table(self._n_tensor_table)
            return n_interp * n_interp
        elif hasattr(self, '_epsilon_tensor_table'):
            return interpolate_table(self._epsilon_tensor_table)
        elif hasattr(self, '_epsilon_dispersive'):
            return self._epsilon_dispersive(wavelength)
        else:
            # Fallback for constant tensors
            return getattr(self, '_epsilon_tensor', np.eye(3, dtype=complex))
    
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
            # Dispersive case: support both function- and table-based by wrapping evaluation
            def rotated_eps_func(wl):
                eps = self._tensor_at_wavelength(wl)
                return rotation_matrix @ eps @ rotation_matrix.T
            epsilon_tensor = rotated_eps_func
            
            if hasattr(self, '_mu_dispersive'):
                def rotated_mu_func(wl):
                    mu = self._mu_dispersive(wl)
                    return rotation_matrix @ mu @ rotation_matrix.T
                mu_tensor = rotated_mu_func
            else:
                mu_tensor = rotation_matrix @ self._mu_tensor @ rotation_matrix.T
            
            return TensorMaterial(epsilon_tensor=epsilon_tensor, mu_tensor=mu_tensor,
                                name=f"{self.name}_rotated", source=self.source)


# --- Public helpers: build dispersion functions from tabulated data ---

def make_n_from_table(
    table: dict,
    allow_interpolation: bool = False,
    allow_extrapolation: bool = False,
):
    """Create a scalar refractive-index dispersion function n(wl) from a table.

    Table forms:
      - {'wavelength': [...], 'n': [...]} or
      - {'wavelength': [...], 'er': [...]} optionally with 'ur'
    Flags strictly control interpolation/extrapolation.
    """
    if 'wavelength' not in table:
        raise ValueError("'table' must include 'wavelength'")
    wl = np.array(table['wavelength'], dtype=float)
    order = np.argsort(wl)
    wl = wl[order]
    if 'n' in table:
        vals = np.array(table['n'], dtype=complex)[order]
        use_n = True
    elif 'er' in table:
        er = np.array(table['er'], dtype=complex)[order]
        if 'ur' in table:
            ur = np.array(table['ur'], dtype=complex)[order]
        else:
            ur = np.ones_like(er)
        vals = np.sqrt(er * ur)
        use_n = False
    else:
        raise ValueError("table must include 'n' or 'er'")

    def interp(x: float) -> complex:
        idx = np.searchsorted(wl, x)
        if x < wl[0]:
            if not allow_extrapolation:
                raise ValueError("Extrapolation disabled. Set allow_extrapolation=True to enable.")
            slope = (vals[1] - vals[0]) / (wl[1] - wl[0])
            return vals[0] + slope * (x - wl[0])
        if x > wl[-1]:
            if not allow_extrapolation:
                raise ValueError("Extrapolation disabled. Set allow_extrapolation=True to enable.")
            slope = (vals[-1] - vals[-2]) / (wl[-1] - wl[-2])
            return vals[-1] + slope * (x - wl[-1])
        if x == wl[0]:
            return vals[0]
        if idx < len(wl) and x == wl[idx]:
            return vals[idx]
        if not allow_interpolation:
            raise ValueError("Interpolation disabled. Set allow_interpolation=True to enable.")
        x1, x2 = wl[idx-1], wl[idx]
        v1, v2 = vals[idx-1], vals[idx]
        a = (x - x1) / (x2 - x1)
        return v1 + a * (v2 - v1)

    return interp


def make_epsilon_tensor_from_table(
    tensor_data: dict,
    allow_interpolation: bool = False,
    allow_extrapolation: bool = False,
):
    """Create an epsilon-tensor dispersion function eps(wl) from tensor tables.

    Accepts the same forms as TensorMaterial: epsilon_* or n_* components, or
    full arrays under 'epsilon_tensor'/'n_tensor'. When n is provided, this
    interpolates n first and squares to epsilon.
    """
    if 'wavelength' not in tensor_data:
        raise ValueError("Tensor data must contain 'wavelength'")
    wl = np.array(tensor_data['wavelength'], dtype=float)
    order = np.argsort(wl)
    wl = wl[order]

    def build_table(prefix: str) -> np.ndarray:
        table = np.zeros((len(wl), 3, 3), dtype=complex)
        comps = ['xx','xy','xz','yx','yy','yz','zx','zy','zz']
        found = False
        for i, comp in enumerate(comps):
            r, c = i // 3, i % 3
            k = f'{prefix}_{comp}'
            if k in tensor_data:
                arr = np.array(tensor_data[k], dtype=complex)
                if arr.shape[0] != len(wl):
                    raise ValueError(f"'{k}' length must match 'wavelength'")
                table[:, r, c] = arr[order]
                found = True
            else:
                if r == c and prefix == 'epsilon':
                    table[:, r, c] = 1.0
        if not found:
            raise ValueError(f"No '{prefix}_*' components found")
        return table

    n_tab = None
    eps_tab = None
    if 'epsilon_tensor' in tensor_data or 'n_tensor' in tensor_data:
        key = 'epsilon_tensor' if 'epsilon_tensor' in tensor_data else 'n_tensor'
        arr = np.array(tensor_data[key])
        if arr.ndim != 3 or arr.shape[1:] != (3,3) or arr.shape[0] != len(wl):
            raise ValueError(f"'{key}' must have shape [N,3,3]")
        arr = arr[order].astype(complex)
        if key == 'n_tensor':
            n_tab = arr
        else:
            eps_tab = arr
    elif any(k.startswith('n_') for k in tensor_data.keys()):
        n_tab = build_table('n')
    elif any(k.startswith('epsilon_') for k in tensor_data.keys()):
        eps_tab = build_table('epsilon')
    else:
        raise ValueError("Tensor data must contain epsilon_* or n_* components, or epsilon_tensor/n_tensor array")

    def interp_tensor(x: float) -> np.ndarray:
        idx = np.searchsorted(wl, x)
        def _interp_tab(tab: np.ndarray) -> np.ndarray:
            if x < wl[0]:
                if not allow_extrapolation:
                    raise ValueError("Tensor extrapolation disabled. Set allow_extrapolation=True to enable.")
                w1, w2 = wl[0], wl[1]
                t1, t2 = tab[0], tab[1]
                a = (x - w1) / (w2 - w1)
                return t1 + a * (t2 - t1)
            if x > wl[-1]:
                if not allow_extrapolation:
                    raise ValueError("Tensor extrapolation disabled. Set allow_extrapolation=True to enable.")
                w1, w2 = wl[-2], wl[-1]
                t1, t2 = tab[-2], tab[-1]
                a = (x - w2) / (w2 - w1) + 1.0
                return t1 + a * (t2 - t1)
            if x == wl[0]:
                return tab[0]
            if idx < len(wl) and x == wl[idx]:
                return tab[idx]
            if not allow_interpolation:
                raise ValueError("Tensor interpolation disabled. Set allow_interpolation=True to enable.")
            w1, w2 = wl[idx-1], wl[idx]
            t1, t2 = tab[idx-1], tab[idx]
            a = (x - w1) / (w2 - w1)
            return t1 + a * (t2 - t1)

        if n_tab is not None:
            n_interp = _interp_tab(n_tab)
            return n_interp * n_interp
        else:
            return _interp_tab(eps_tab)  # type: ignore[arg-type]

    return interp_tensor
