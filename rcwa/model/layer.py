from rcwa.shorthand import *
from .material import Material
from rcwa.legacy.crystal import Crystal
from rcwa.core.matrices import MatrixCalculator
from .material import TensorMaterial
from typing import Union, List, Tuple, TYPE_CHECKING
from numpy.typing import ArrayLike

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from matplotlib.figure import Figure, Axes
else:  # fallbacks when matplotlib is unavailable
    Figure = Axes = None

if TYPE_CHECKING:
    from .layer import HalfSpace

# TODO: Convolution matrix generation must be refactored. It's a hot mess and hard to understand.


class Layer(MatrixCalculator):
    """
    Class for defining a single layer of a layer stack used in a simulation

    :param er: Permittivity of the layer. Overridden by crystal permittivity if specified.
    :param ur: Permeability of the layer. Overridden by crystal permeability if specified.
    :param thickness: Thickness of the layer
    :param n: Refractive index of the layer. Overridden by cristal er/ur if specified.
    :param material: Material object containing the material's permittivity and permeability as a function of wavelength/angle.
    :param tensor_material: TensorMaterial object for anisotropic materials
    :param crystal: Crystal object if the layer is periodic in x and/or y. Overrides er, ur, n, and material
    """
    def __init__(self, er: complex = 1.0, ur: complex = 1.0, thickness: complex = 0.0, n: Union[complex, None] = None,
                 material: Union[None, Material] = None, tensor_material: Union[None, TensorMaterial] = None,
                 crystal: Union[None, Crystal] = None):
        
        # Handle material specification priority
        if tensor_material is not None:
            self.material = None
            self.tensor_material = tensor_material
            self.is_anisotropic = True
        elif material is None:
            self.material = Material(er=er, ur=ur, n=n)
            self.tensor_material = None
            self.is_anisotropic = False
        else:
            self.material = material
            self.tensor_material = None
            self.is_anisotropic = False

        self.thickness = thickness
        self.crystal = crystal
        self.incident = False  # Whether this is a transmission layer
        self.transmission = False  # Whether this is an incident layer

        if crystal is None:
            self.homogenous = True
        else:
            self.homogenous = False


    # Note: these are all just transparent wrappers for underlying material
    @property
    def er(self) -> Union[ArrayLike, complex]:
        if self.is_anisotropic:
            # Check if convolution matrices have been set
            if hasattr(self, '_tensor_er'):
                return self._tensor_er
            else:
                # Return the raw tensor for interface compatibility
                return self.tensor_material.epsilon_tensor
        else:
            return self.material.er

    @er.setter
    def er(self, er: complex):
        if self.is_anisotropic:
            raise ValueError("Cannot set scalar er for anisotropic layer. Use tensor_material instead.")
        else:
            self.material.er = er

    @property
    def ur(self):
        if self.is_anisotropic:
            # Check if convolution matrices have been set
            if hasattr(self, '_tensor_ur'):
                return self._tensor_ur
            else:
                # Return the raw tensor for interface compatibility  
                return self.tensor_material.mu_tensor
        else:
            return self.material.ur

    @ur.setter
    def ur(self, ur: complex):
        if self.is_anisotropic:
            raise ValueError("Cannot set scalar ur for anisotropic layer. Use tensor_material instead.")
        else:
            self.material.ur = ur

    @property
    def n(self) -> Union[ArrayLike, complex]:
        if self.is_anisotropic:
            # For anisotropic materials, n is not well-defined as a scalar
            # Return effective index or raise error
            raise ValueError("Refractive index n is not well-defined for anisotropic materials. Access epsilon_tensor directly.")
        else:
            return self.material.n

    @n.setter
    def n(self, n: complex):
        if self.is_anisotropic:
            raise ValueError("Cannot set scalar n for anisotropic layer.")
        else:
            self.material.n = n

    @property
    def source(self):
        if self.is_anisotropic:
            return self.tensor_material.source
        else:
            return self.material.source

    @source.setter
    def source(self, source):
        if self.is_anisotropic:
            self.tensor_material.source = source
        else:
            self.material.source = source

    def set_convolution_matrices(self, n_harmonics: Union[ArrayLike, int]):
        if self.crystal is not None:
            self.er = self._convolution_matrix(self.crystal.permittivityCellData, n_harmonics)
            self.ur = self._convolution_matrix(self.crystal.permeabilityCellData, n_harmonics)
        elif self.is_anisotropic:
            # For tensor materials in uniform layers, create convolution matrices
            # that represent the tensor components
            from rcwa.core.adapters import TensorToConvolutionAdapter
            
            epsilon_tensor = self.tensor_material.epsilon_tensor
            mu_tensor = self.tensor_material.mu_tensor

            # Generate convolution matrices for all tensor components
            conv_matrices = TensorToConvolutionAdapter.tensor_to_convolution_matrices(
                epsilon_tensor, mu_tensor, n_harmonics
            )

            # Store full tensor convolution matrices for downstream use
            self._tensor_conv_matrices = conv_matrices

            # For legacy interfaces expecting scalar er/ur matrices, use zz-components
            self._tensor_er = conv_matrices['er_zz']
            self._tensor_ur = conv_matrices['ur_zz']
        else:
            matrix_dim = np.prod(n_harmonics) if isinstance(n_harmonics, tuple) else n_harmonics
            self.er = self.er * complexIdentity(matrix_dim)
            self.ur = self.ur * complexIdentity(matrix_dim)

    def _convolution_matrix(self, cellData: ArrayLike, n_harmonics: Union[ArrayLike, int]) -> ArrayLike:
        dimension = self.crystal.dimensions;

        if isinstance(n_harmonics, int):
            n_harmonics = (n_harmonics,)

        if dimension == 1:
            n_harmonics = (n_harmonics + (1, 1))
        elif dimension == 2:
            n_harmonics = (n_harmonics + (1,))

        (P, Q, R) = n_harmonics

        convolutionMatrixSize = P*Q*R;
        convolutionMatrixShape = (convolutionMatrixSize, convolutionMatrixSize);
        convolutionMatrix = complexZeros(convolutionMatrixShape)

        cellData = reshapeLowDimensionalData(cellData);
        (Nx, Ny, Nz) = cellData.shape;
        zeroHarmonicsLocation = np.array([math.floor(Nx/2), math.floor(Ny/2), math.floor(Nz/2)])

        cellFourierRepresentation = fftn(cellData);
        for rrow in range(R):
            for qrow in range(Q):
                for prow in range(P):
                    row = rrow*Q*P + qrow*P + prow;
                    for rcol in range(R):
                        for qcol in range(Q):
                            for pcol in range(P):
                                col = rcol*Q*P + qcol*P + pcol;
                                # Get the desired harmonics relative to the 0th-order harmonic.
                                desiredHarmonics = np.array([prow - pcol, qrow - qcol, rrow - rcol])

                                # Get those harmonic locations from the zero harmonic location.
                                desiredHarmonicsLocation = zeroHarmonicsLocation + desiredHarmonics

                                convolutionMatrix[row][col] = \
                                    cellFourierRepresentation[desiredHarmonicsLocation[0],
                                    desiredHarmonicsLocation[1], desiredHarmonicsLocation[2]];
        if convolutionMatrix.shape == (1, 1):
            convolutionMatrix = convolutionMatrix[0][0]
        return convolutionMatrix;

    def __eq__(self, other):
        if not isinstance(other, Layer):
            return NotImplemented

        return self.er == other.er and self.ur == other.ur and self.thickness == other.thickness \
               and self.n == other.n and self.crystal == other.crystal

    def __str__(self):
        return f'Layer with\n\ter: {self.er}\n\tur: {self.ur}\n\tL: {self.thickness}\n\tn: {self.n}\n\tcrystal: {self.crystal}'

    def rotated(self, euler_angles: Tuple[float, float, float], about: str = "center", convention: str = "ZYX"):
        """
        Create a new Layer with rotated material tensors.
        
        This is a convenience method that wraps the rotate_layer function.
        
        :param euler_angles: Tuple of (alpha, beta, gamma) Euler angles in radians
        :param about: Rotation center, currently only "center" is supported
        :param convention: Euler angle convention, default "ZYX"
        :return: New Layer object with rotated material properties
        """
        from rcwa.model.transforms import rotate_layer
        return rotate_layer(self, euler_angles, about, convention)


freeSpaceLayer = Layer(1,1)


class LayerStack:
    """
    Class that defines overall geometry in terms of a stack of layers

    :param internal_layers: Layer objects, starting with the top-most layer (reflection region) and ending with the top-most region (substrate)
    :param incident_layer: Semi-infinite layer of incident region. Defaults to free space (deprecated, use superstrate)
    :param transmission_layer: Semi-infinite layer of transmission region. Defaults to free space (deprecated, use substrate)
    :param superstrate: Semi-infinite layer above the stack (preferred over incident_layer)
    :param substrate: Semi-infinite layer below the stack (preferred over transmission_layer)
    :param layers: Alternative way to specify internal layers as a list
    """
    def __init__(self, *internal_layers: Layer,
                 incident_layer: Union[Layer, 'HalfSpace'] = None, 
                 transmission_layer: Union[Layer, 'HalfSpace'] = None,
                 superstrate: Union[Layer, 'HalfSpace', Material, TensorMaterial] = None,
                 substrate: Union[Layer, 'HalfSpace', Material, TensorMaterial] = None,
                 layers: List[Layer] = None):
        self.gapLayer = Layer(er=1, ur=1)
        
        # Handle internal layers - support both positional args and 'layers' keyword
        if layers is not None:
            if internal_layers:
                raise ValueError("Cannot specify both positional internal_layers and 'layers' keyword argument")
            self.internal_layers = list(layers)
        else:
            self.internal_layers = list(internal_layers)
        
        # Handle superstrate (incident layer) - prefer new naming
        if superstrate is not None and incident_layer is not None:
            raise ValueError("Cannot specify both 'superstrate' and 'incident_layer'. Use 'superstrate' (preferred)")
        
        superstrate_layer = superstrate if superstrate is not None else incident_layer
        if superstrate_layer is None:
            self.incident_layer = Layer(er=1, ur=1)
        else:
            self.incident_layer = self._convert_to_layer(superstrate_layer)
        self.incident_layer.incident = True
        
        # Handle substrate (transmission layer) - prefer new naming
        if substrate is not None and transmission_layer is not None:
            raise ValueError("Cannot specify both 'substrate' and 'transmission_layer'. Use 'substrate' (preferred)")
            
        substrate_layer = substrate if substrate is not None else transmission_layer
        if substrate_layer is None:
            self.transmission_layer = Layer(er=1, ur=1)
        else:
            self.transmission_layer = self._convert_to_layer(substrate_layer)
        self.transmission_layer.transmission = True

        self._Kx = None
        self._Ky = None
    
    def _convert_to_layer(self, input_obj) -> Layer:
        """Convert various input types to Layer objects."""
        if isinstance(input_obj, Layer):
            # Already a Layer
            return input_obj
        elif hasattr(input_obj, 'to_layer'):
            # It's a HalfSpace
            return input_obj.to_layer()
        elif isinstance(input_obj, Material):
            # Convert Material to HalfSpace then to Layer
            halfspace = HalfSpace(material=input_obj)
            return halfspace.to_layer()
        elif isinstance(input_obj, TensorMaterial):
            # Convert TensorMaterial to HalfSpace then to Layer
            halfspace = HalfSpace(tensor_material=input_obj)
            return halfspace.to_layer()
        else:
            raise TypeError(f"Cannot convert {type(input_obj)} to Layer. "
                          f"Expected Layer, HalfSpace, Material, or TensorMaterial.")

    def __str__(self):
        top_string = f'\nReflection Layer:\n\t' + str(self.incident_layer) + \
                f'\nTransmissionLayer:\n\t' + str(self.transmission_layer) + \
                f'\nInternal Layer Count: {len(self.internal_layers)}\n'
        internal_string = ''
        for layer in self.internal_layers:
            internal_string += str(layer) + '\n'
        return top_string + internal_string

    @property
    def _k_dimension(self) -> int:
        if isinstance(self.Kx, np.ndarray):
            return self.Kx.shape[0]
        else:
            return 1

    @property
    def _s_element_dimension(self) -> int:
        s_dim = self._k_dimension * 2
        return s_dim

    @property
    def all_layers(self) -> List[Layer]:
        return [self.incident_layer, *self.internal_layers, self.transmission_layer]

    @property
    def Kx(self) -> Union[complex, ArrayLike]:
        return self._Kx

    @Kx.setter
    def Kx(self, kx: Union[complex, ArrayLike]):
        self._Kx = kx
        self.gapLayer.Kx = kx
        for layer in self.all_layers:
            layer.Kx = kx

    @property
    def Ky(self) -> Union[complex, ArrayLike]:
        return self._Ky

    @Ky.setter
    def Ky(self, ky: Union[complex, ArrayLike]):
        self._Ky = ky
        self.gapLayer.Ky = ky
        for layer in self.all_layers:
            layer.Ky = ky

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        self._source = source
        self.gapLayer.source = source
        for layer in self.all_layers:
            layer.source = self.source

    def set_gap_layer(self):
        self.gapLayer.thickness = 0
        if self._k_dimension == 1:
            self.gapLayer.er = 1 + sq(self.Kx) + sq(self.Ky)
            self.gapLayer.ur = 1
            Qg = self.gapLayer.Q_matrix()
            lambda_gap = self.gapLayer.lambda_matrix()

        else:
            Kz = self.gapLayer.Kz_gap()
            Qg = self.gapLayer.Q_matrix()
            lambda_gap = complexIdentity(self._k_dimension * 2)
            lambda_gap[:self._k_dimension, :self._k_dimension] = 1j * Kz
            lambda_gap[self._k_dimension:, self._k_dimension:] = 1j * Kz

        self.Wg = complexIdentity(self._s_element_dimension)
        self.Vg = Qg @ inv(lambda_gap)

        for layer in self.all_layers:
            layer.Wg = self.Wg
            layer.Vg = self.Vg

    # set all convolution matrices for all interior layers
    def set_convolution_matrices(self, n_harmonics: Union[int, ArrayLike]):
        for layer in self.internal_layers:
            layer.set_convolution_matrices(n_harmonics)

    @property
    def crystal(self) -> Union[None, Crystal]:
        for i in range(len(self.internal_layers)):
            if self.internal_layers[i].crystal is not None:
                return self.internal_layers[i].crystal
        return None

    def plot(self, fig: Union[None, 'Figure'] = None, ax: Union[None, 'Axes'] = None) -> Tuple['Figure', 'Axes']:
        import matplotlib.pyplot as plt  # Local import to avoid hard dependency
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        elif fig is not None and ax is None:
            ax = fig.add_subplot()

        # z = 0 will be defined at the start of the top-most layer.

        return fig, ax



emptyStack = LayerStack()


class HalfSpace:
    """
    Class for representing half-infinite spaces (superstrate or substrate).
    
    This class provides a more explicit and intuitive way to represent 
    semi-infinite media compared to using Layer(material, thickness=0).
    
    :param material: Material object for the half-infinite space
    :param tensor_material: TensorMaterial object for anisotropic half-infinite space
    """
    def __init__(self, material: Union[Material, None] = None, 
                 tensor_material: Union[TensorMaterial, None] = None):
        
        if tensor_material is not None:
            self.material = None
            self.tensor_material = tensor_material
            self.is_anisotropic = True
        elif material is not None:
            self.material = material
            self.tensor_material = None
            self.is_anisotropic = False
        else:
            # Default to air
            self.material = Material(er=1.0, ur=1.0, n=1.0)
            self.tensor_material = None
            self.is_anisotropic = False
            
        self.thickness = 0.0  # Half-spaces are always thickness=0
        self.homogenous = True  # Half-spaces are always homogeneous
        
    def to_layer(self) -> Layer:
        """Convert HalfSpace to Layer for backward compatibility."""
        if self.is_anisotropic:
            return Layer(tensor_material=self.tensor_material, thickness=0.0)
        else:
            return Layer(material=self.material, thickness=0.0)
    
    @property
    def er(self):
        """Permittivity of the half-space."""
        if self.is_anisotropic:
            return self.tensor_material.epsilon_tensor
        else:
            return self.material.er
    
    @property
    def ur(self):
        """Permeability of the half-space."""
        if self.is_anisotropic:
            return self.tensor_material.mu_tensor
        else:
            return self.material.ur
    
    @property
    def n(self):
        """Refractive index of the half-space."""
        if self.is_anisotropic:
            raise ValueError("Refractive index n is not well-defined for anisotropic half-spaces.")
        else:
            return self.material.n
    
    def __str__(self):
        if self.is_anisotropic:
            return f"HalfSpace(tensor_material={self.tensor_material})"
        else:
            return f"HalfSpace(er={self.material.er}, ur={self.material.ur})"
    
    def __repr__(self):
        return self.__str__()


# Common material factory functions

def Air(n: float = 1.0) -> Material:
    """Create Air material with given refractive index (default 1.0)."""
    return Material(er=n**2, ur=1.0, n=n)

def Vacuum(n: float = 1.0) -> Material:
    """Create Vacuum material with given refractive index (default 1.0)."""
    return Material(er=n**2, ur=1.0, n=n)


def Silicon(n: float = 3.48) -> Material:
    """Create Silicon material with given refractive index (default for 550nm)."""
    return Material(er=n**2, ur=1.0, n=n)


def SiO2(n: float = 1.46) -> Material:
    """Create SiO2 material with given refractive index (default for 550nm)."""
    return Material(er=n**2, ur=1.0, n=n)


def Glass(n: float = 1.52) -> Material:
    """Create glass material with given refractive index."""
    return Material(er=n**2, ur=1.0, n=n)


# Convenient factory functions for common half-spaces  
def Substrate(material: Union[Material, TensorMaterial]) -> HalfSpace:
    """Create a substrate half-space with the given material."""
    if isinstance(material, TensorMaterial):
        return HalfSpace(tensor_material=material)
    else:
        return HalfSpace(material=material)


# Legacy support - keep freeSpaceLayer for backward compatibility
freeSpaceLayer = Layer(1,1)

# Simplified Stack API - alias for LayerStack with new naming convention
Stack = LayerStack
