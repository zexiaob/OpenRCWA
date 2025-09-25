"""Run a simple anisotropic thin-film simulation and export results.

This script builds a planar anisotropic layer with the following properties:
    - Refractive index along x: n_x = 1.5
    - Refractive index along y: n_y = 2.0
    - Refractive index along z: assumed equal to n_x for uniaxial behaviour
    - Physical thickness: 500 nm

The layer is embedded in air on both sides. We sweep the wavelength from
400 nm to 800 nm at normal incidence for left- and right-circularly polarised
input light. The script saves a CSV file containing total transmission and
reflection for each incident polarisation and produces optional figures that
plot those spectra.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

import site
import sys

# Ensure the real matplotlib package installed in site-packages takes precedence
for _site_pkg in getattr(site, "getsitepackages", lambda: [])():  # pragma: no cover - env specific
    if _site_pkg in sys.path:
        sys.path.remove(_site_pkg)
    sys.path.insert(0, _site_pkg)

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from rcwa import Air, Layer, Stack, TensorMaterial, nm
from rcwa.solve import LCP, RCP, simulate


def build_stack(thickness_m: float, n_x: float, n_y: float, n_z: float) -> Stack:
    """Create the anisotropic layer stack surrounded by air."""
    eps_xx = n_x ** 2
    eps_yy = n_y ** 2
    eps_zz = n_z ** 2

    tensor_material = TensorMaterial.from_diagonal(
        eps_xx,
        eps_yy,
        eps_zz,
        name="anisotropic_layer",
    )

    anisotropic_layer = Layer(tensor_material=tensor_material, thickness=thickness_m)
    air = Air()
    return Stack(anisotropic_layer, superstrate=air, substrate=air)


def extract_totals(results: Iterable) -> np.ndarray:
    """Return total transmission and reflection arrays from a ResultGrid."""
    totals = np.array([[res.TTot, res.RTot] for res in results], dtype=float)
    return totals


def circular_basis(direction: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalised (LCP, RCP) Jones vectors for the requested direction."""

    if direction not in {1, -1}:
        raise ValueError("direction must be +1 (forward) or -1 (backward)")

    if direction == 1:
        # Forward (+z) propagation matches the Source() definitions in rcwa.solve.source
        l_vec = np.array([1.0, 1.0j]) / np.sqrt(2.0)
        r_vec = np.array([1.0, -1.0j]) / np.sqrt(2.0)
    else:
        # For backward (-z) propagation, helicity definitions swap to remain right-handed
        l_vec = np.array([1.0, -1.0j]) / np.sqrt(2.0)
        r_vec = np.array([1.0, 1.0j]) / np.sqrt(2.0)

    return l_vec, r_vec


def project_circular(E_x: np.ndarray, E_y: np.ndarray, direction: int) -> Tuple[np.ndarray, np.ndarray]:
    """Project transverse fields onto circular polarisation basis for given propagation."""

    E_x = np.asarray(E_x)
    E_y = np.asarray(E_y)
    l_vec, r_vec = circular_basis(direction)
    l_amp = l_vec.conj()[0] * E_x + l_vec.conj()[1] * E_y
    r_amp = r_vec.conj()[0] * E_x + r_vec.conj()[1] * E_y
    return l_amp, r_amp


def extract_circular_channels(results) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project reflected/transmitted fields onto LCP/RCP channels."""

    R_LCP, R_RCP, T_LCP, T_RCP = [], [], [], []

    for res in results:
        # Use the same (+z) basis for both reflection and transmission so that
        # an isotropic stack yields a diagonal Jones matrix in this basis. This
        # makes it easy to diagnose helicity conversion introduced solely by
        # anisotropy rather than by the choice of reference frame for the
        # reflected wave (whose propagation direction is -z).
        r_l_amp, r_r_amp = project_circular(res.rx, res.ry, direction=1)
        t_l_amp, t_r_amp = project_circular(res.tx, res.ty, direction=1)

        R_LCP.append(np.sum(np.abs(r_l_amp) ** 2))
        R_RCP.append(np.sum(np.abs(r_r_amp) ** 2))
        T_LCP.append(np.sum(np.abs(t_l_amp) ** 2))
        T_RCP.append(np.sum(np.abs(t_r_amp) ** 2))

    return (
        np.asarray(R_LCP, dtype=float),
        np.asarray(R_RCP, dtype=float),
        np.asarray(T_LCP, dtype=float),
        np.asarray(T_RCP, dtype=float),
    )


def main(save_plots: bool = False) -> None:
    repo_root = Path(__file__).resolve().parent

    thickness = nm(500.0)
    n_x, n_y, n_z = 1.5, 1.5, 1.5  # assume optic axis aligned with z and matches n_x

    stack = build_stack(thickness, n_x, n_y, n_z)

    wavelengths_nm = np.linspace(400.0, 1000.0, 201)
    wavelengths_m = wavelengths_nm * 1e-9

    grid = simulate(
        stack,
        wavelength=wavelengths_m,
        polarization=["LCP", "RCP"],
        n_harmonics=1,
        return_grid=True,
    )

    wavelengths = np.asarray(grid.coords["wavelength"], dtype=float) / 1e-9

    lcp_grid = grid.sel(pTEM=LCP())
    rcp_grid = grid.sel(pTEM=RCP())

    lcp_R_LCP, lcp_R_RCP, lcp_T_LCP, lcp_T_RCP = extract_circular_channels(lcp_grid.data)
    rcp_R_LCP, rcp_R_RCP, rcp_T_LCP, rcp_T_RCP = extract_circular_channels(rcp_grid.data)

    dataframe = pd.DataFrame(
        {
            "wavelength_nm": wavelengths,
            "LCP_inc_R_LCP": lcp_R_LCP,
            "LCP_inc_R_RCP": lcp_R_RCP,
            "LCP_inc_T_LCP": lcp_T_LCP,
            "LCP_inc_T_RCP": lcp_T_RCP,
            "RCP_inc_R_LCP": rcp_R_LCP,
            "RCP_inc_R_RCP": rcp_R_RCP,
            "RCP_inc_T_LCP": rcp_T_LCP,
            "RCP_inc_T_RCP": rcp_T_RCP,
        }
    )

    csv_path = repo_root / "anisotropic_tensor_rt.csv"
    dataframe.to_csv(csv_path, index=False)

    if save_plots:
        # Plot for LCP incidence (4 lines)
        plt.figure(figsize=(8, 5))
        plt.plot(wavelengths, lcp_T_LCP, label="LCP→LCP Transmission", color="#1f77b4")
        plt.plot(wavelengths, lcp_T_RCP, label="LCP→RCP Transmission", color="#2ca02c")
        plt.plot(wavelengths, lcp_R_LCP, label="LCP→LCP Reflection", color="#ff7f0e")
        plt.plot(wavelengths, lcp_R_RCP, label="LCP→RCP Reflection", color="#d62728")
        plt.title("Anisotropic layer response for LCP incidence")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Power fraction")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        lcp_fig_path = repo_root / "lcp_incidence_rt.png"
        plt.tight_layout()
        plt.savefig(lcp_fig_path, dpi=300)
        plt.close()

        # Plot for RCP incidence (4 lines)
        plt.figure(figsize=(8, 5))
        plt.plot(wavelengths, rcp_T_LCP, label="RCP→LCP Transmission", color="#1f77b4", linestyle="--")
        plt.plot(wavelengths, rcp_T_RCP, label="RCP→RCP Transmission", color="#2ca02c", linestyle="--")
        plt.plot(wavelengths, rcp_R_LCP, label="RCP→LCP Reflection", color="#ff7f0e", linestyle="--")
        plt.plot(wavelengths, rcp_R_RCP, label="RCP→RCP Reflection", color="#d62728", linestyle="--")
        plt.title("Anisotropic layer response for RCP incidence")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Power fraction")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        rcp_fig_path = repo_root / "rcp_incidence_rt.png"
        plt.tight_layout()
        plt.savefig(rcp_fig_path, dpi=300)
        plt.close()

        print(f"Saved LCP figure to {lcp_fig_path}")
        print(f"Saved RCP figure to {rcp_fig_path}")

    print(f"Saved CSV to {csv_path}")


def parse_args():
    parser = ArgumentParser(description="Simulate circular response of an anisotropic film")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate and save diagnostic plots alongside the CSV output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(save_plots=False)
