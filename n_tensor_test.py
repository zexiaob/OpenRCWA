import numpy as np
from rcwa import TensorMaterial

def epsilon_tensor_dispersion(wavelength_m):
    """
    Calculates the anisotropic epsilon tensor for hBN based on wavelength.
    This function is vectorized to handle both scalar and array inputs for wavelength.
    """
    # Ensure wavelength_nm is a numpy array for vectorized operations
    wavelength_nm = np.array(wavelength_m) * 1e9  # Convert m to nm

    B1_o, C1_o = 3.3361, 26322   # ordinary / in-plane
    B1_e, C1_e = 2.2631, 26981   # extraordinary / out-of-plane

    # Calculate n^2 using Sellmeier equation
    n_o_sq = 1 + (B1_o * wavelength_nm**2) / (wavelength_nm**2 - C1_o)
    n_e_sq = 1 + (B1_e * wavelength_nm**2) / (wavelength_nm**2 - C1_e)

    # For scalar input, return a single 3x3 array
    if wavelength_nm.ndim == 0:
        return np.array([
            [n_o_sq, 0, 0],
            [0, n_o_sq, 0],
            [0, 0, n_e_sq]
        ], dtype=complex)
    
    # For array input, construct the tensor for each wavelength
    # The result should have shape (N, 3, 3) where N is the number of wavelengths
    num_wavelengths = len(wavelength_nm)
    epsilon = np.zeros((num_wavelengths, 3, 3), dtype=complex)
    epsilon[:, 0, 0] = n_o_sq
    epsilon[:, 1, 1] = n_o_sq
    epsilon[:, 2, 2] = n_e_sq
    
    return epsilon

# 用法示例：
# 创建一个色散各向异性材料
ani_disp = TensorMaterial(epsilon_tensor=epsilon_tensor_dispersion, name="test_dispersion_tensor")

# 你可以直接用于 Layer 或仿真
# from rcwa import Layer
# layer = Layer(tensor_material=ani_disp, thickness=300e-9)
