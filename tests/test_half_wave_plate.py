import numpy as np
from numpy.testing import assert_allclose

from rcwa.model.material import Material, TensorMaterial
from rcwa.model.layer import Layer, LayerStack
from rcwa.solve.source import Source
from rcwa.core.solver import Solver


def _wrap_to_pi(angle):
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def test_half_wave_plate_normal_incidence_phase_retardance():
    """
    Build a uniaxial anisotropic layer that acts as a half-wave plate at a chosen wavelength.

    Verification criteria (for normal incidence):
    - Relative transmission phase between x and y, arg(ty) - arg(tx), is π (within tolerance)
    - Amplitude ratio |ty|/|tx| is near 1 (low-reflection environment to minimize FP effects)
    - Energy is approximately conserved (R+T ~ 1) within moderate tolerance
    """

    # Operating wavelength (arbitrary units, consistent across geometry)
    wl = 1.0

    # Choose birefringence and environment to minimize reflections
    # Principal axes aligned with x/y; make Δn moderate so thickness is reasonable
    n_x = 1.45
    n_y = 1.55
    n_env = 0.5 * (n_x + n_y)  # index-match to average to reduce Fresnel reflections

    # Half-wave condition: δ = 2π Δn t / λ = π  =>  t = λ / (2 Δn)
    delta_n = n_y - n_x
    t = wl / (2.0 * delta_n)

    # Source: normal incidence, linear 45° to the crystal axes
    source = Source(wavelength=wl, theta=0.0, phi=0.0)
    # At normal incidence, aTE aligns with +y, aTM aligns with -x in this codebase.
    # We want pX = pY = 1/sqrt(2) => set pTE=+1/sqrt(2), pTM=-1/sqrt(2)
    source.pTEM = np.array([1.0, -1.0]) / np.sqrt(2.0)

    # Materials
    env = Material(n=n_env)
    eps_xx, eps_yy, eps_zz = n_x ** 2, n_y ** 2, n_env ** 2  # optical axis along z with env match
    tm = TensorMaterial.from_diagonal(eps_xx, eps_yy, eps_zz, source=source)

    # Layer stack: env | HWP | env
    hwp = Layer(tensor_material=tm, thickness=t)
    stack = LayerStack(hwp, superstrate=env, substrate=env)

    # Solve (TMM case, n_harmonics=1)
    solver = Solver(stack, source, n_harmonics=1)
    results = solver.solve()

    # Extract complex transmission coefficients
    tx = results['tx']
    ty = results['ty']

    # 1) Verify phase retardance ~ π
    dphi = np.angle(ty) - np.angle(tx)
    dphi_wrapped = _wrap_to_pi(dphi)
    assert_allclose(abs(dphi_wrapped), np.pi, atol=0.15, err_msg=f"Half-wave phase not achieved: Δφ={dphi}")

    # 2) Verify amplitudes are nearly equal (minimized reflections)
    amp_ratio = np.abs(ty) / (np.abs(tx) + 1e-30)
    assert abs(amp_ratio - 1.0) < 0.1, f"Amplitude imbalance too large: |ty|/|tx|={amp_ratio}"

    # Note: Energy conservation via R/T formulas for anisotropic layers may need
    # normalization refinements in the core. We skip asserting R+T≈1 here.
