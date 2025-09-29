"""
Half-wave plate demo (no plotting).

- Builds a uniaxial anisotropic layer whose thickness gives ~pi phase delay between x and y at normal incidence.
- Uses index-matched superstrate/substrate to reduce reflections.
- Prints transmitted amplitudes (tx, ty), phase difference, amplitude ratio, and optional Jones matrix.

Note: Uses the simplified Stack API (LayerStack is deprecated; prefer Stack).

Usage (default half-wave at wl=1.0, n_x=1.45, n_y=1.55, pol=45 deg):
    python examples/half_wave_plate_demo.py

Options:
    --wavelength 1.0     Operating wavelength
    --nx 1.45            Refractive index along x
    --ny 1.55            Refractive index along y
    --nenv auto          Environment index (default average of nx, ny)
    --thickness auto     Layer thickness (default half-wave: lambda/(2*Δn))
    --pol-deg 45         Input linear polarization angle in degrees (0=x, 90=y)
    --compute-jones      Compute approximate 2x2 Jones matrix by simulating x and y inputs
"""
import argparse
import os
import sys
import numpy as np

# Allow running the script directly without installing the package
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rcwa.model.material import Material, TensorMaterial
from rcwa.model.layer import Layer, Stack
from rcwa.solve.source import Source
from rcwa.core.solver import Solver


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pTEM_for_linear_pol(angle_rad: float) -> np.ndarray:
    """
    Map desired linear polarization in (x,y) to the code's TE/TM coefficients at normal incidence.
    At normal incidence in this codebase: aTE ≈ +y, aTM ≈ -x.
    To realize p = [pX, pY] = [cos A, sin A], choose pTE=pY, pTM=-pX.
    """
    pX = np.cos(angle_rad)
    pY = np.sin(angle_rad)
    vec = np.array([pY, -pX], dtype=float)
    vec /= np.linalg.norm(vec)
    return vec


from typing import Optional


def run_once(wavelength: float, nx: float, ny: float, nenv: Optional[float], thickness: Optional[float],
             pol_deg: float, compute_jones: bool):
    wl = wavelength
    n_x, n_y = nx, ny
    n_env = 0.5 * (n_x + n_y) if nenv is None else nenv

    delta_n = n_y - n_x
    t = (wl / (2.0 * delta_n)) if thickness is None else thickness

    # Source: normal incidence, linear polarization at angle
    source = Source(wavelength=wl, theta=0.0, phi=0.0)
    source.pTEM = pTEM_for_linear_pol(np.deg2rad(pol_deg))

    # Materials
    env = Material(n=n_env)
    eps_xx, eps_yy, eps_zz = n_x**2, n_y**2, n_env**2
    tm = TensorMaterial.from_diagonal(eps_xx, eps_yy, eps_zz, source=source)

    # Stack: env | HWP | env
    hwp = Layer(tensor_material=tm, thickness=t)
    stack = Stack(hwp, superstrate=env, substrate=env)

    # Solve
    solver = Solver(stack, source, n_harmonics=1)
    res = solver.solve()

    tx = res['tx']
    ty = res['ty']
    dphi = wrap_to_pi(np.angle(ty) - np.angle(tx))
    amp_ratio = np.abs(ty) / (np.abs(tx) + 1e-30)

    print("=== Half-wave plate demo ===")
    print(f"wavelength: {wl}")
    print(f"n_x: {n_x}, n_y: {n_y}, n_env: {n_env}")
    print(f"thickness: {t}")
    print(f"input pol: {pol_deg} deg (linear)")
    print(f"tx: {tx:.6g}, ty: {ty:.6g}")
    print(f"phase(ty)-phase(tx): {dphi:.6g} rad (~pi: {np.pi:.6g})")
    print(f"|ty|/|tx|: {amp_ratio:.6g}")
    print(f"RTot: {res['RTot']:.6g}, TTot: {res['TTot']:.6g}, R+T: {res['RTot']+res['TTot']:.6g}")

    if compute_jones:
        # Approximate Jones matrix columns by exciting pure x and pure y inputs
        # Pure x: pX=1, pY=0 -> pTE=0, pTM=-1
        sx = Source(wavelength=wl, theta=0.0, phi=0.0)
        sx.pTEM = np.array([0.0, -1.0])  # normalized already
        tm_x = TensorMaterial.from_diagonal(eps_xx, eps_yy, eps_zz, source=sx)
        hwp_x = Layer(tensor_material=tm_x, thickness=t)
        stack_x = Stack(hwp_x, superstrate=env, substrate=env)
        jx = Solver(stack_x, sx, n_harmonics=1).solve()
        J_col_x = np.array([jx['tx'], jx['ty']])

        # Pure y: pX=0, pY=1 -> pTE=1, pTM=0
        sy = Source(wavelength=wl, theta=0.0, phi=0.0)
        sy.pTEM = np.array([1.0, 0.0])
        tm_y = TensorMaterial.from_diagonal(eps_xx, eps_yy, eps_zz, source=sy)
        hwp_y = Layer(tensor_material=tm_y, thickness=t)
        stack_y = Stack(hwp_y, superstrate=env, substrate=env)
        jy = Solver(stack_y, sy, n_harmonics=1).solve()
        J_col_y = np.array([jy['tx'], jy['ty']])

        J = np.column_stack([J_col_x, J_col_y])
        print("Jones matrix (columns = response to x and y inputs):")
        # Pretty print
        with np.printoptions(precision=6, suppress=True):
            print(J)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Half-wave plate demo (no plotting)")
    ap.add_argument("--wavelength", type=float, default=1.0)
    ap.add_argument("--nx", type=float, default=1.45)
    ap.add_argument("--ny", type=float, default=1.55)
    ap.add_argument("--nenv", type=float, default=None, nargs="?")
    ap.add_argument("--thickness", type=float, default=None, nargs="?")
    ap.add_argument("--pol-deg", type=float, default=45.0)
    ap.add_argument("--compute-jones", action="store_true")
    args = ap.parse_args()

    run_once(args.wavelength, args.nx, args.ny, args.nenv, args.thickness, args.pol_deg, args.compute_jones)
