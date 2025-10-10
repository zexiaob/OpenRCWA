"""Three-layer MoOCl2 stack rotation sweep producing transmission/reflection/CD heatmaps."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import importlib.util
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

import site
import sys

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
    wavelength_m = np.asarray(wavelength_m)
    wavelength_nm = wavelength_m * 1e9
    return _MOOCL2_MODULE.permittivity_tensor_from_wavelength(wavelength_nm)


def rotation_matrix_z(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def build_three_layer_stack(theta_rad: float, thicknesses: Tuple[float, float, float]) -> Stack:
    base_material = TensorMaterial(
        epsilon_tensor=moocl2_epsilon_tensor,
        name="MoOCl2_dispersion",
    )

    rotation_plus = rotation_matrix_z(theta_rad)
    rotation_pp = rotation_matrix_z(2 * theta_rad)

    layer1 = Layer(tensor_material=base_material, thickness=thicknesses[0])
    layer2 = Layer(tensor_material=base_material.rotated(rotation_plus), thickness=thicknesses[1])
    layer3 = Layer(tensor_material=base_material.rotated(rotation_pp), thickness=thicknesses[2])

    stack = Stack(layers=[layer1, layer2, layer3], superstrate=Air(), substrate=Air())
    stack.enable_tensor_eigensolver(True)
    return stack


def circular_basis(direction: int) -> Tuple[np.ndarray, np.ndarray]:
    if direction not in {1, -1}:
        raise ValueError("direction must be +1 (forward) or -1 (backward)")

    if direction == 1:
        l_vec = np.array([1.0, 1.0j]) / np.sqrt(2.0)
        r_vec = np.array([1.0, -1.0j]) / np.sqrt(2.0)
    else:
        l_vec = np.array([1.0, -1.0j]) / np.sqrt(2.0)
        r_vec = np.array([1.0, 1.0j]) / np.sqrt(2.0)

    return l_vec, r_vec


def project_circular(E_x: np.ndarray, E_y: np.ndarray, direction: int) -> Tuple[np.ndarray, np.ndarray]:
    E_x = np.asarray(E_x)
    E_y = np.asarray(E_y)
    l_vec, r_vec = circular_basis(direction)
    l_amp = l_vec.conj()[0] * E_x + l_vec.conj()[1] * E_y
    r_amp = r_vec.conj()[0] * E_x + r_vec.conj()[1] * E_y
    return l_amp, r_amp


def extract_circular_channels(results: Iterable) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R_LCP, R_RCP, T_LCP, T_RCP = [], [], [], []
    for res in results:
        r_l_amp, r_r_amp = project_circular(res.rx, res.ry, direction=-1)
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


def run_sweep(
    theta_deg: np.ndarray,
    wavelengths_nm: np.ndarray,
    thicknesses_nm: Tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_rad = np.deg2rad(theta_deg)
    wavelengths_m = wavelengths_nm * 1e-9
    thicknesses_m = tuple(nm(t) for t in thicknesses_nm)

    lcp_T = np.zeros((theta_deg.size, wavelengths_nm.size))
    rcp_T = np.zeros_like(lcp_T)
    lcp_R = np.zeros_like(lcp_T)
    rcp_R = np.zeros_like(lcp_T)

    for idx, angle in enumerate(theta_rad):
        stack = build_three_layer_stack(angle, thicknesses_m)
        grid = simulate(
            stack,
            wavelength=wavelengths_m,
            polarization=["LCP", "RCP"],
            n_harmonics=1,
            return_grid=True,
        )

        lcp_grid = grid.sel(pTEM=LCP())
        rcp_grid = grid.sel(pTEM=RCP())

        lcp_R_LCP, lcp_R_RCP, lcp_T_LCP, lcp_T_RCP = extract_circular_channels(lcp_grid.data)
        rcp_R_LCP, rcp_R_RCP, rcp_T_LCP, rcp_T_RCP = extract_circular_channels(rcp_grid.data)

        lcp_incident = np.asarray([res.conservation for res in lcp_grid.data], dtype=float)
        rcp_incident = np.asarray([res.conservation for res in rcp_grid.data], dtype=float)
        lcp_incident[lcp_incident == 0] = np.nan
        rcp_incident[rcp_incident == 0] = np.nan

        lcp_T[idx, :] = (lcp_T_LCP + lcp_T_RCP) / lcp_incident
        rcp_T[idx, :] = (rcp_T_LCP + rcp_T_RCP) / rcp_incident
        lcp_R[idx, :] = (lcp_R_LCP + lcp_R_RCP) / lcp_incident
        rcp_R[idx, :] = (rcp_R_LCP + rcp_R_RCP) / rcp_incident

    return lcp_T, rcp_T, lcp_R, rcp_R


def save_results(
    wavelengths_nm: np.ndarray,
    theta_deg: np.ndarray,
    lcp_T: np.ndarray,
    rcp_T: np.ndarray,
    lcp_R: np.ndarray,
    rcp_R: np.ndarray,
    output_csv: Path,
) -> None:
    rows = {
        "theta_deg": np.repeat(theta_deg, wavelengths_nm.size),
        "wavelength_nm": np.tile(wavelengths_nm, theta_deg.size),
        "LCP_inc_T_total": lcp_T.reshape(-1),
        "RCP_inc_T_total": rcp_T.reshape(-1),
        "LCP_inc_R_total": lcp_R.reshape(-1),
        "RCP_inc_R_total": rcp_R.reshape(-1),
    }
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def plot_heatmaps(
    wavelengths_nm: np.ndarray,
    theta_deg: np.ndarray,
    lcp_T: np.ndarray,
    rcp_T: np.ndarray,
    lcp_R: np.ndarray,
    rcp_R: np.ndarray,
    output_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    cmap = "viridis"
    vmin = 0.0
    vmax_T = float(np.nanmax(np.concatenate([lcp_T, rcp_T])))
    vmax_R = float(np.nanmax(np.concatenate([lcp_R, rcp_R])))

    pcm0 = axes[0, 0].pcolormesh(
        wavelengths_nm,
        theta_deg,
        lcp_T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax_T,
    )
    axes[0, 0].set_title("LCP Total Transmission")
    axes[0, 0].set_ylabel("Rotation θ (deg)")
    axes[0, 0].axhline(0.0, color="white", linestyle="--", linewidth=0.8, alpha=0.6)

    pcm1 = axes[0, 1].pcolormesh(
        wavelengths_nm,
        theta_deg,
        rcp_T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax_T,
    )
    axes[0, 1].set_title("RCP Total Transmission")
    axes[0, 1].axhline(0.0, color="white", linestyle="--", linewidth=0.8, alpha=0.6)

    pcm2 = axes[1, 0].pcolormesh(
        wavelengths_nm,
        theta_deg,
        lcp_R,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax_R,
    )
    axes[1, 0].set_title("LCP Total Reflection")
    axes[1, 0].set_xlabel("Wavelength (nm)")
    axes[1, 0].set_ylabel("Rotation θ (deg)")
    axes[1, 0].axhline(0.0, color="white", linestyle="--", linewidth=0.8, alpha=0.6)

    pcm3 = axes[1, 1].pcolormesh(
        wavelengths_nm,
        theta_deg,
        rcp_R,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax_R,
    )
    axes[1, 1].set_title("RCP Total Reflection")
    axes[1, 1].set_xlabel("Wavelength (nm)")
    axes[1, 1].axhline(0.0, color="white", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.colorbar(pcm0, ax=axes[0, 0], label="Transmission")
    fig.colorbar(pcm1, ax=axes[0, 1], label="Transmission")
    fig.colorbar(pcm2, ax=axes[1, 0], label="Reflection")
    fig.colorbar(pcm3, ax=axes[1, 1], label="Reflection")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def plot_cd(
    wavelengths_nm: np.ndarray,
    theta_deg: np.ndarray,
    lcp_T: np.ndarray,
    rcp_T: np.ndarray,
    lcp_R: np.ndarray,
    rcp_R: np.ndarray,
    output_png: Path,
) -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        denom_T = lcp_T + rcp_T
        denom_R = lcp_R + rcp_R
        cd_T = np.divide(
            lcp_T - rcp_T,
            denom_T,
            out=np.full_like(denom_T, np.nan, dtype=float),
            where=denom_T != 0,
        )
        cd_R = np.divide(
            lcp_R - rcp_R,
            denom_R,
            out=np.full_like(denom_R, np.nan, dtype=float),
            where=denom_R != 0,
        )

    if np.all(np.isnan(cd_T)):
        vmax_T = 1.0
    else:
        vmax_T = float(np.nanmax(np.abs(cd_T)))

    if np.all(np.isnan(cd_R)):
        vmax_R = 1.0
    else:
        vmax_R = float(np.nanmax(np.abs(cd_R)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    cmap = "PiYG"


    pcm0 = axes[0].pcolormesh(
        wavelengths_nm,
        theta_deg,
        cd_T,
        shading="auto",
        cmap=cmap,
        vmin=-vmax_T,
        vmax=vmax_T,
    )
    axes[0].set_title("Transmission CD (LCP - RCP)")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Rotation θ (deg)")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    pcm1 = axes[1].pcolormesh(
        wavelengths_nm,
        theta_deg,
        cd_R,
        shading="auto",
        cmap=cmap,
        vmin=-vmax_R,
        vmax=vmax_R,
    )
    axes[1].set_title("Reflection CD (LCP - RCP)")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.colorbar(pcm0, ax=axes[0], label="ΔT")
    fig.colorbar(pcm1, ax=axes[1], label="ΔR")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main(
    *,
    save_plots: bool = True,
    theta_min_deg: float = -90.0,
    theta_max_deg: float = 90.0,
    theta_samples: int = 91,
    thickness_layer1_nm: float = 100.0,
    thickness_layer2_nm: float = 100.0,
    thickness_layer3_nm: float = 100.0,
) -> None:
    repo_root = Path(__file__).resolve().parent

    theta_deg = np.linspace(theta_min_deg, theta_max_deg, theta_samples)
    wavelengths_nm = np.linspace(400.0, 800.0, 201)

    lcp_T, rcp_T, lcp_R, rcp_R = run_sweep(
        theta_deg,
        wavelengths_nm,
        (thickness_layer1_nm, thickness_layer2_nm, thickness_layer3_nm),
    )

    csv_path = repo_root / "moocl2_three_layer_rt.csv"
    save_results(wavelengths_nm, theta_deg, lcp_T, rcp_T, lcp_R, rcp_R, csv_path)

    if save_plots:
        heatmap_path = repo_root / "moocl2_three_layer_TR_heatmaps.png"
        plot_heatmaps(wavelengths_nm, theta_deg, lcp_T, rcp_T, lcp_R, rcp_R, heatmap_path)
        print(f"Saved transmission/reflection heatmaps to {heatmap_path}")

        cd_path = repo_root / "moocl2_three_layer_cd_heatmap.png"
        plot_cd(wavelengths_nm, theta_deg, lcp_T, rcp_T, lcp_R, rcp_R, cd_path)
        print(f"Saved CD heatmap to {cd_path}")

    print(f"Saved sweep results to {csv_path}")


def parse_args():
    parser = ArgumentParser(description="MoOCl2 three-layer rotation sweep")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving heatmap figures.")
    parser.add_argument("--theta-min", type=float, default=-90.0, help="Minimum rotation angle in degrees.")
    parser.add_argument("--theta-max", type=float, default=90.0, help="Maximum rotation angle in degrees.")
    parser.add_argument("--theta-samples", type=int, default=91, help="Number of theta samples in sweep.")
    parser.add_argument("--thickness1", type=float, default=100.0, help="Thickness of first layer (nm).")
    parser.add_argument("--thickness2", type=float, default=100.0, help="Thickness of second layer (nm).")
    parser.add_argument("--thickness3", type=float, default=100.0, help="Thickness of third layer (nm).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        save_plots=not args.no_plots,
        theta_min_deg=args.theta_min,
        theta_max_deg=args.theta_max,
        theta_samples=args.theta_samples,
        thickness_layer1_nm=args.thickness1,
        thickness_layer2_nm=args.thickness2,
        thickness_layer3_nm=args.thickness3,
    )
