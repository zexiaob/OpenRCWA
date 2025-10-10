"""Simulate the anisotropic response of a MoOCl2 thin film using RCWA.

This script constructs a single anisotropic MoOCl2 layer embedded in air and
sweeps the wavelength response for circular polarisations. The complex tensor
response is provided by the Drude–Lorentz model defined in ``nk_MoOCl2 (1).py``.
Results are exported to CSV and optional diagnostic plots can be generated.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import importlib.util

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


def _load_moocl2_module():
    """Load the MoOCl2 material model defined in ``nk_MoOCl2 (1).py``."""

    module_path = Path(__file__).resolve().parent / "nk_MoOCl2 (1).py"
    if not module_path.exists():
        raise FileNotFoundError(f"Expected MoOCl2 model at {module_path}")

    spec = importlib.util.spec_from_file_location("nk_MoOCl2_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MOOCL2_MODULE = _load_moocl2_module()


def moocl2_epsilon_tensor(wavelength_m: float | np.ndarray) -> np.ndarray:
    """Return the MoOCl2 permittivity tensor for wavelengths given in meters."""

    wavelength_m = np.asarray(wavelength_m)
    wavelength_nm = wavelength_m * 1e9
    return _MOOCL2_MODULE.permittivity_tensor_from_wavelength(wavelength_nm)


def build_stack(thickness_m: float) -> Stack:
    """Create a stack with a single MoOCl2 layer between semi-infinite air."""

    tensor_material = TensorMaterial(
        epsilon_tensor=moocl2_epsilon_tensor,
        name="MoOCl2_dispersion",
    )

    layer = Layer(tensor_material=tensor_material, thickness=thickness_m)
    air = Air()
    stack = Stack(layer, superstrate=air, substrate=air)
    stack.enable_tensor_eigensolver(True)
    return stack


def circular_basis(direction: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalised (LCP, RCP) Jones vectors for the requested direction."""

    if direction not in {1, -1}:
        raise ValueError("direction must be +1 (forward) or -1 (backward)")

    if direction == 1:
        l_vec = np.array([1.0, 1.0j]) / np.sqrt(2.0)
        r_vec = np.array([1.0, -1.0j]) / np.sqrt(2.0)
    else:
        l_vec = np.array([1.0, -1.0j]) / np.sqrt(2.0)
        r_vec = np.array([1.0, 1.0j]) / np.sqrt(2.0)

    return l_vec, r_vec


def project_circular(E_x: np.ndarray, E_y: np.ndarray, direction: int) -> tuple[np.ndarray, np.ndarray]:
    """Project transverse fields onto circular polarisation basis."""

    E_x = np.asarray(E_x)
    E_y = np.asarray(E_y)
    l_vec, r_vec = circular_basis(direction)
    l_amp = l_vec.conj()[0] * E_x + l_vec.conj()[1] * E_y
    r_amp = r_vec.conj()[0] * E_x + r_vec.conj()[1] * E_y
    return l_amp, r_amp


def extract_circular_channels(results) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project reflected/transmitted fields onto LCP/RCP channels."""

    R_LCP, R_RCP, T_LCP, T_RCP = [], [], [], []

    for res in results:
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


def main(save_plots: bool = True) -> None:
    repo_root = Path(__file__).resolve().parent

    thickness = nm(100.0)
    stack = build_stack(thickness)

    wavelengths_nm = np.linspace(400.0, 600, 600)
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

    lcp_incident_power = np.asarray([res.conservation for res in lcp_grid.data], dtype=float)
    rcp_incident_power = np.asarray([res.conservation for res in rcp_grid.data], dtype=float)
    lcp_incident_power[lcp_incident_power == 0] = np.nan
    rcp_incident_power[rcp_incident_power == 0] = np.nan

    lcp_T_total = (lcp_T_LCP + lcp_T_RCP) / lcp_incident_power
    rcp_T_total = (rcp_T_LCP + rcp_T_RCP) / rcp_incident_power
    lcp_R_total = (lcp_R_LCP + lcp_R_RCP) / lcp_incident_power
    rcp_R_total = (rcp_R_LCP + rcp_R_RCP) / rcp_incident_power

    dataframe = pd.DataFrame(
        {
            "wavelength_nm": wavelengths,
            "LCP_inc_R_total": lcp_R_total,
            "LCP_inc_T_total": lcp_T_total,
            "RCP_inc_R_total": rcp_R_total,
            "RCP_inc_T_total": rcp_T_total,
        }
    )

    csv_path = repo_root / "moocl2_tensor_rt.csv"
    dataframe.to_csv(csv_path, index=False)

    if save_plots:
        fig, (ax_t, ax_r) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

        ax_t.plot(wavelengths, lcp_T_total, label="LCP", color="#1f77b4", linewidth=2)
        ax_t.plot(wavelengths, rcp_T_total, label="RCP", color="#ff7f0e", linewidth=2)
        ax_t.set_ylabel("Transmission")
        ax_t.set_title("Total Transmission")
        ax_t.grid(True, alpha=0.3)
        ax_t.legend()

        ax_r.plot(wavelengths, lcp_R_total, label="LCP", color="#1f77b4", linewidth=2)
        ax_r.plot(wavelengths, rcp_R_total, label="RCP", color="#ff7f0e", linewidth=2)
        ax_r.set_xlabel("Wavelength (nm)")
        ax_r.set_ylabel("Reflection")
        ax_r.set_title("Total Reflection")
        ax_r.grid(True, alpha=0.3)

        fig.tight_layout()
        tr_fig_path = repo_root / "moocl2_total_TR.png"
        fig.savefig(tr_fig_path, dpi=300)
        plt.close(fig)

        print(f"Saved transmission/reflection figure to {tr_fig_path}")

    print(f"Saved CSV to {csv_path}")


def parse_args():
    parser = ArgumentParser(description="Simulate circular response of a MoOCl2 anisotropic film")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate and save diagnostic plots alongside the CSV output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(save_plots=True)
