# -*- coding: utf-8 -*-
"""Material permittivity model for MoOCl2.

The module computes anisotropic dielectric responses using a Drude–Lorentz
parameterisation and exposes a helper to retrieve the 3×3 permittivity tensor
for any wavelength of interest.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------------

# Plasma frequency (unit: eV)
omega_p_xx = 6.419
omega_p_yy = 1.361
omega_p_zz = 0.562

# Drude damping rate (unit: eV)
gamma_D_xx = 0.023
gamma_D_yy = 0.016
gamma_D_zz = 0.025

# Dielectric constant at infinite frequency (dimensionless)
epsilon_infinity_xx = 2.74
epsilon_infinity_yy = 3.69
epsilon_infinity_zz = 3.96

# Lorentz oscillator parameters (dimensionless weights and eV energies)
f1_xx = 0.42
omega_1_xx = 3.28
gamma_1_xx = 0.46

f1_yy = 1.44
omega_1_yy = 0.79
gamma_1_yy = 0.29

f2_yy = 14.0
omega_2_yy = 2.84
gamma_2_yy = 0.13

# Default sweep configuration for visualisation/export
SWEEP_ENERGY_RANGE = (0.2, 4.0)
SWEEP_NUM_SAMPLES = 1000


# -----------------------------------------------------------------------------
# Conversion helpers
# -----------------------------------------------------------------------------

def wavelength_nm_to_energy_eV(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Convert wavelength in nanometres to photon energy in electron-volts."""

    wavelengths = np.asarray(wavelength_nm, dtype=float)
    if np.any(wavelengths <= 0):
        raise ValueError("Wavelength values must be strictly positive.")
    return 1240.0 / wavelengths


def energy_eV_to_wavelength_nm(energy_eV: float | np.ndarray) -> np.ndarray:
    """Convert photon energy in electron-volts to wavelength in nanometres."""

    energies = np.asarray(energy_eV, dtype=float)
    if np.any(energies <= 0):
        raise ValueError("Photon energy values must be strictly positive.")
    return 1240.0 / energies


# -----------------------------------------------------------------------------
# Core dielectric model
# -----------------------------------------------------------------------------

def dielectric_components(energy_eV: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the diagonal dielectric components at the requested photon energies."""

    energies = np.asarray(energy_eV, dtype=np.complex128)
    if np.any(energies == 0):
        raise ValueError("Photon energy must be non-zero to avoid singularities.")

    epsilon_xx = (
        epsilon_infinity_xx
        - (omega_p_xx**2) / (energies * (energies + 1j * gamma_D_xx))
        + (f1_xx * omega_p_xx**2) / (omega_1_xx**2 - energies**2 - 1j * gamma_1_xx * energies)
    )

    epsilon_yy = (
        epsilon_infinity_yy
        - (omega_p_yy**2) / (energies * (energies + 1j * gamma_D_yy))
        + (f1_yy * omega_p_yy**2) / (omega_1_yy**2 - energies**2 - 1j * gamma_1_yy * energies)
        + (f2_yy * omega_p_yy**2) / (omega_2_yy**2 - energies**2 - 1j * gamma_2_yy * energies)
    )

    epsilon_zz = (
        epsilon_infinity_zz
        - (omega_p_zz**2) / (energies * (energies + 1j * gamma_D_zz))
    )

    return epsilon_xx, epsilon_yy, epsilon_zz


def permittivity_tensor_from_wavelength(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Return the complex 3×3 permittivity tensor for the specified wavelength.

    Parameters
    ----------
    wavelength_nm:
        Wavelength(s) in nanometres. Accepts scalars or numpy-compatible arrays.

    Returns
    -------
    numpy.ndarray
        A complex-valued tensor. For scalar input the shape is (3, 3). For an
        array-like input of shape (N, ...) the result shape is (*input_shape, 3, 3).
    """

    energies = wavelength_nm_to_energy_eV(wavelength_nm)
    epsilon_xx, epsilon_yy, epsilon_zz = dielectric_components(energies)

    epsilon_xx = np.asarray(epsilon_xx)
    epsilon_yy = np.asarray(epsilon_yy)
    epsilon_zz = np.asarray(epsilon_zz)

    tensor_shape = epsilon_xx.shape + (3, 3)
    tensor = np.zeros(tensor_shape, dtype=np.complex128)
    tensor[..., 0, 0] = epsilon_xx
    tensor[..., 1, 1] = epsilon_yy
    tensor[..., 2, 2] = epsilon_zz

    return tensor


# -----------------------------------------------------------------------------
# Visualisation and export helpers
# -----------------------------------------------------------------------------

def format_complex(z: complex) -> str:
    """Format a complex number with 4 decimal precision for consistent output."""

    return f"{z.real:.4f}{z.imag:+.4f}j"


def format_float(value: float) -> str:
    return f"{value:.2f}"


def save_formatted_data(filename: str, data: list[tuple[str, str]]) -> None:
    """Persist wavelength/eigenvalue pairs to disk with tab separation."""

    with open(filename, "w", encoding="utf-8") as handle:
        for wavelength, epsilon in data:
            handle.write(f"{wavelength}\t{epsilon}\n")


def main(show_plots: bool = False) -> None:
    """Reproduce legacy plots and console output for the defined energy sweep."""

    energy_grid = np.linspace(*SWEEP_ENERGY_RANGE, SWEEP_NUM_SAMPLES)
    epsilon_xx, epsilon_yy, epsilon_zz = dielectric_components(energy_grid)

    wavelength_nm = energy_eV_to_wavelength_nm(energy_grid)
    wavelength_um = wavelength_nm * 1e-3

    if show_plots:
        plt.figure()
        plt.plot(energy_grid, np.real(epsilon_xx), "r", linewidth=2)
        plt.plot(energy_grid, np.real(epsilon_yy), "g", linewidth=2)
        plt.plot(energy_grid, np.real(epsilon_zz), "b", linewidth=2)
        plt.title("Real Part of Dielectric Constants (ε_xx, ε_yy, ε_zz)")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Re(ε)")
        plt.legend(["ε_xx", "ε_yy", "ε_zz"], loc="best")
        plt.grid(True)
        plt.ylim([-50, 50])
        plt.show()

        plt.figure()
        plt.plot(energy_grid, np.imag(epsilon_xx), "r", linewidth=1)
        plt.plot(energy_grid, np.imag(epsilon_yy), "g", linewidth=1)
        plt.plot(energy_grid, np.imag(epsilon_zz), "b", linewidth=1)
        plt.title("Imaginary Part of Dielectric Constants (ε_xx, ε_yy, ε_zz)")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Im(ε)")
        plt.legend(["ε_xx", "ε_yy", "ε_zz"], loc="best")
        plt.grid(True)
        plt.ylim([0, 80])
        plt.show()

        plt.figure()
        plt.plot(wavelength_um, np.real(epsilon_xx), "r", linewidth=2)
        plt.plot(wavelength_um, np.real(epsilon_yy), "g", linewidth=2)
        plt.plot(wavelength_um, np.real(epsilon_zz), "b", linewidth=2)
        plt.title("Real Part of Dielectric Constants (ε_xx, ε_yy, ε_zz)")
        plt.xlabel("Wavelength (μm)")
        plt.ylabel("Re(ε)")
        plt.legend(["ε_xx", "ε_yy", "ε_zz"], loc="best")
        plt.grid(True)
        plt.ylim([-30, 40])
        plt.xlim([0.2, 2])
        plt.show()

        plt.figure()
        plt.plot(wavelength_um, np.imag(epsilon_xx), "r", linewidth=2)
        plt.plot(wavelength_um, np.imag(epsilon_yy), "g", linewidth=2)
        plt.plot(wavelength_um, np.imag(epsilon_zz), "b", linewidth=2)
        plt.title("Imaginary Part of Dielectric Constants (ε_xx, ε_yy, ε_zz)")
        plt.xlabel("Wavelength (μm)")
        plt.ylabel("Im(ε)")
        plt.legend(["ε_xx", "ε_yy", "ε_zz"], loc="best")
        plt.grid(True)
        plt.ylim([-20, 80])
        plt.xlim([0.2, 2])

    data_xx = [(format_float(wavelength_nm[i]), format_complex(epsilon_xx[i])) for i in range(len(energy_grid))]
    data_yy = [(format_float(wavelength_nm[i]), format_complex(epsilon_yy[i])) for i in range(len(energy_grid))]
    data_zz = [(format_float(wavelength_nm[i]), format_complex(epsilon_zz[i])) for i in range(len(energy_grid))]

    print("Example entries (nm, ε):")
    for label, dataset in ("xx", data_xx), ("yy", data_yy), ("zz", data_zz):
        print(f"  {label}: {dataset[:3]} ...")

    # Uncomment to persist formatted tables
    # save_formatted_data("MoOCl2_permittivity_xx.txt", data_xx)
    # save_formatted_data("MoOCl2_permittivity_yy.txt", data_yy)
    # save_formatted_data("MoOCl2_permittivity_zz.txt", data_zz)


if __name__ == "__main__":
    main(show_plots=True)

    sample_wavelength_nm = 1000.0
    tensor = permittivity_tensor_from_wavelength(sample_wavelength_nm)
    print(f"\nPermittivity tensor at {sample_wavelength_nm:.1f} nm:\n{tensor}")