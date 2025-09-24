"""Run a simple anisotropic thin-film simulation and export results.

This script builds a planar anisotropic layer with the following properties:
    - Refractive index along x: n_x = 1.5
    - Refractive index along y: n_y = 2.0
    - Refractive index along z: assumed equal to n_x for uniaxial behaviour
    - Physical thickness: 500 nm

The layer is embedded in air on both sides. We sweep the wavelength from
400 nm to 800 nm at normal incidence for left- and right-circularly polarised
input light. The script saves a CSV file containing total transmission and
reflection for each incident polarisation and produces two figures that plot
those spectra.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

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


def main() -> None:
    repo_root = Path(__file__).resolve().parent

    thickness = nm(500.0)
    n_x, n_y, n_z = 1.5, 2.0, 1.5  # assume optic axis aligned with z and matches n_x

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

    # 提取每个分量的反射和透射（LCP入射和RCP入射）
    def extract_LR_RT(results):
        # 用物理正交投影方式提取LCP/RCP分量
        def proj_power(E, handedness='LCP'):
            # E: [Ex, Ey, ...]
            if handedness == 'LCP':
                comp = (E[0] - 1j * E[1]) / np.sqrt(2)
            else:
                comp = (E[0] + 1j * E[1]) / np.sqrt(2)
            return np.abs(comp)**2
        R_LCP, R_RCP, T_LCP, T_RCP = [], [], [], []
        for res in results:
            r = res.r_complex()
            t = res.t_complex()
            R_LCP.append(proj_power(r, 'LCP'))
            R_RCP.append(proj_power(r, 'RCP'))
            T_LCP.append(proj_power(t, 'LCP'))
            T_RCP.append(proj_power(t, 'RCP'))
        return np.array(R_LCP), np.array(R_RCP), np.array(T_LCP), np.array(T_RCP)

    lcp_R_LCP, lcp_R_RCP, lcp_T_LCP, lcp_T_RCP = extract_LR_RT(lcp_grid.data)
    rcp_R_LCP, rcp_R_RCP, rcp_T_LCP, rcp_T_RCP = extract_LR_RT(rcp_grid.data)

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
    plt.plot(wavelengths, rcp_T_LCP, label="RCP→LCP Transmission", color="#1f77b4", linestyle='--')
    plt.plot(wavelengths, rcp_T_RCP, label="RCP→RCP Transmission", color="#2ca02c", linestyle='--')
    plt.plot(wavelengths, rcp_R_LCP, label="RCP→LCP Reflection", color="#ff7f0e", linestyle='--')
    plt.plot(wavelengths, rcp_R_RCP, label="RCP→RCP Reflection", color="#d62728", linestyle='--')
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

    print(f"Saved CSV to {csv_path}")
    print(f"Saved LCP figure to {lcp_fig_path}")
    print(f"Saved RCP figure to {rcp_fig_path}")


if __name__ == "__main__":
    main()