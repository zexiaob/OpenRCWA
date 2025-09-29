import numpy as np
import pandas as pd
from rcwa import (
    Material,
    TensorMaterial,
    PatternedLayer,
    Stack,
    Source,
    Solver,
    Rectangle,
    rectangular_lattice,
    Air,
    SiO2,
    HalfSpace,
    nm,
)
from n_tensor_test import epsilon_tensor_dispersion


def build_patterned_layer(S: float) -> tuple[PatternedLayer, HalfSpace, HalfSpace]:
    """Construct patterned hBN-on-SiO2 metasurface layer and half-spaces.

    Shapes use unit-cell coordinates; widths/heights are normalized by lattice periods.
    """
    # Geometry (SI units)
    h = nm(150)
    w = nm(100)
    L1 = nm(310)
    L2 = nm(360)
    px = nm(410)
    py = nm(430)

    # Lattice and normalized sizes
    lat = rectangular_lattice(px * S, py * S)
    w_norm = w / px
    L1_norm = L1 / py
    L2_norm = L2 / py

    # hBN tensor (dispersive epsilon tensor function)
    hBN_tensor = TensorMaterial(epsilon_tensor=epsilon_tensor_dispersion, name="hBN_dispersion")

    # Two rectangular bars centered at 1/4 and 3/4 along y
    rod1 = Rectangle(center=(0.5, 0.25), width=w_norm, height=L1_norm)
    rod2 = Rectangle(center=(0.5, 0.75), width=w_norm, height=L2_norm)

    # Background index matches substrate for fewer Fresnel artifacts
    patterned = PatternedLayer(
        thickness=h,
        lattice=lat,
        shapes=[(rod1, hBN_tensor), (rod2, hBN_tensor)],
        background_material=SiO2(n=1.45),
    )

    # Half-spaces
    substrate = HalfSpace(material=SiO2(n=1.45))
    superstrate = HalfSpace(material=Air())

    return patterned, substrate, superstrate


def zeroth_order_index(nh: tuple[int, int]) -> int:
    """Return flat index of (0,0) diffraction order for given harmonics (Nx, Ny)."""
    nx, ny = nh
    return (ny // 2) * nx + (nx // 2)


def simulate_spectrum(wavelengths: np.ndarray, harmonics: tuple[int, int] = (5, 5), S: float = 1.0):
    patterned, substrate, superstrate = build_patterned_layer(S)
    nh = harmonics
    zero_idx = zeroth_order_index(nh)

    # Storage
    TTot_list, RTot_list = [], []
    j11, j21, j12, j22 = [], [], [], []  # 0阶 Jones 矩阵（列1: x入射，列2: y入射）

    for wl in wavelengths:
        # Normal incidence; build two sources for x- and y-polarized incidence
        # At normal incidence: aTM aligns with +x, aTE aligns with +y in this codebase.
        src_x = Source(wavelength=wl, theta=0.2, phi=0.0, pTEM=[0, 1])  # x入射
        src_y = Source(wavelength=wl, theta=0.2, phi=0.0, pTEM=[1, 0])  # y入射

        # Stack using patterned layer; prefer P/Q eigensystem for patterned tensors
        stack_x = Stack(layers=[patterned], superstrate=superstrate, substrate=substrate)
        stack_x.enable_tensor_eigensolver(False)
        stack_y = Stack(layers=[patterned], superstrate=superstrate, substrate=substrate)
        stack_y.enable_tensor_eigensolver(False)

        # Couple dispersive material to source wavelength (PatternedLayer propagates to shape materials)
        patterned.source = src_x

        # Solve for x-incidence
        solver_x = Solver(layer_stack=stack_x, source=src_x, n_harmonics=nh)
        res_x = solver_x.solve()
        # Solve for y-incidence
        patterned.source = src_y
        solver_y = Solver(layer_stack=stack_y, source=src_y, n_harmonics=nh)
        res_y = solver_y.solve()

        # Totals (use x-incidence totals for spectrum reference; both should conserve energy similarly)
        TTot_list.append(res_x.TTot)
        RTot_list.append(res_x.RTot)

        # Zeroth-order transmitted amplitudes (flat order layout: ky major, kx minor)
        tx_x0 = np.atleast_1d(res_x['tx'])[zero_idx]
        ty_x0 = np.atleast_1d(res_x['ty'])[zero_idx]
        tx_y0 = np.atleast_1d(res_y['tx'])[zero_idx]
        ty_y0 = np.atleast_1d(res_y['ty'])[zero_idx]

        # Jones columns: [Ex_out; Ey_out] for unit x- or y-incident
        j11.append(tx_x0)
        j21.append(ty_x0)
        j12.append(tx_y0)
        j22.append(ty_y0)

    return (
        np.asarray(TTot_list),
        np.asarray(RTot_list),
        np.asarray(j11),
        np.asarray(j21),
        np.asarray(j12),
        np.asarray(j22),
    )


if __name__ == "__main__":
    # Spectrum settings (meters)
    wavelengths = np.linspace(nm(800), nm(1000), 25)
    nh = (11, 11)
    # Geometry scale factor (kept as variable S to match previous scripts)
    S = 1.7

    TTot, RTot, j11, j21, j12, j22 = simulate_spectrum(wavelengths, harmonics=nh, S=S)

    # Save CSV (magnitudes for Jones; keep complex data columns for analysis)
    df = pd.DataFrame(
        {
            'wavelength_m': wavelengths,
            'S': S,
            'TTot': TTot,
            'RTot': RTot,
            'J11_real': np.real(j11), 'J11_imag': np.imag(j11), 'J11_abs': np.abs(j11),
            'J21_real': np.real(j21), 'J21_imag': np.imag(j21), 'J21_abs': np.abs(j21),
            'J12_real': np.real(j12), 'J12_imag': np.imag(j12), 'J12_abs': np.abs(j12),
            'J22_real': np.real(j22), 'J22_imag': np.imag(j22), 'J22_abs': np.abs(j22),
        }
    )
    df.to_csv('bic_hbn_metasurface_results.csv', index=False)

    # Basic sanity checks
    assert len(TTot) == len(wavelengths)
    assert np.all(np.isfinite(TTot))
    assert np.all(TTot >= 0) & np.all(TTot <= 1 + 1e-9)  # tolerate tiny numeric noise

    print("Saved: bic_hbn_metasurface_results.csv")

# 下面的绘图可注释掉，后续可单独绘制
# fig, ax = plt.subplots()
# ax.plot(wavelengths, results.TTot)
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Transmittance (TTot)')
# ax.set_title('BIC hBN Metasurface Transmission Spectrum')
# plt.show()

# 参数化扫描示例（可选，可批量扫描 S 并绘制多组结果）
# for S in np.linspace(0.65, 1.10, 10):
#     # 重新构建几何和仿真，收集/绘制多组结果
#     hBN_tensor = TensorMaterial(epsilon_tensor=epsilon_tensor_dispersion, name="hBN_dispersion")
#     lat = rectangular_lattice(px * S, py * S)
#     rod1 = Rectangle(center=(0.0, -py/4*S), width=w*S, height=L1*S)
#     rod2 = Rectangle(center=(0.0, py/4*S), width=w*S, height=L2*S)
#     patterned = PatternedLayer(
#         thickness=h*S*1e-9,
#         lattice=lat,
#         shapes=[(rod1, hBN_tensor), (rod2, hBN_tensor)],
#         background_material=Air(),
#     )
#     substrate = HalfSpace(material=SiO2(n=n_SiO2))
#     superstrate = HalfSpace(material=Air())
#     stack = Stack(
#         substrate=substrate,
#         superstrate=superstrate,
#         layers=[patterned],
#     )
#     src = Source(
#         wavelength=wavelengths,
#         theta=0.0,
#         phi=0.0,
#         pTEM=[0, 1],
#     )
#     hBN_tensor.source = src
#     solver = Solver(layer_stack=stack, source=src, n_harmonics=(3, 3))
#     results = solver.solve(wavelength=wavelengths)
#     plt.plot(wavelengths, results.TTot, label=f'S={S:.2f}')
# plt.legend()
