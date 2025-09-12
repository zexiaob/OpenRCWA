import numpy as np
from rcwa import Material, TensorMaterial, Layer, PatternedLayer, Stack, Source, Solver, Rectangle, rectangular_lattice, Air, SiO2, HalfSpace, nm, um
from n_tensor_test import epsilon_tensor_dispersion


# 参数区
h = nm(150)  # nm, 高度
w = nm(100)  # nm, 宽度
L1 = nm(310)  # nm, 棒1长度
L2 = nm(360)  # nm, 棒2长度
px = nm(410)  # nm, 晶格x周期
py = nm(430)  # nm, 晶格y周期
n_SiO2 = 1.45
n_air = 1.0
S = 1.7

hBN_tensor = TensorMaterial(epsilon_tensor=epsilon_tensor_dispersion, name="hBN_dispersion")
lat = rectangular_lattice(px * S, py * S)

# 归一化到晶胞周期
w_norm  = w / px
L1_norm = L1 / py
L2_norm = L2 / py


# Rectangle的center参数应在归一化晶胞内，通常(0.5, y)表示x居中，y为归一化位置
rod1 = Rectangle(center=(0.5, 0.25), width=w_norm, height=L1_norm)
rod2 = Rectangle(center=(0.5, 0.75), width=w_norm, height=L2_norm)




# background_material建议用SiO2以与substrate一致
patterned = PatternedLayer(
    thickness=h,
    lattice=lat,
    shapes=[(rod1, hBN_tensor), (rod2, hBN_tensor)],
    background_material=SiO2(n=n_SiO2),
)

layer = Layer(thickness=h, material=hBN_tensor)
substrate = HalfSpace(material=SiO2(n=n_SiO2))
superstrate = HalfSpace(material=Air())
stack = Stack(
    substrate=substrate,
    superstrate=superstrate,
    layers=[layer],
)


import pandas as pd
wavelengths = np.linspace(nm(500), nm(1000), 500)  # 米
TTot_list = []
RTot_list = []


def get_first(val):
    if hasattr(val, '__len__') and not isinstance(val, str):
        return val[0]
    return val

for wl in wavelengths:
    src = Source(
        wavelength=wl,
        theta=0.01,
        phi=0.01,
        pTEM=[0, 1],
    )
    hBN_tensor = TensorMaterial(epsilon_tensor=epsilon_tensor_dispersion, name="hBN_dispersion")
    hBN_tensor.source = src
    rod1 = Rectangle(center=(0.5, 0.25), width=w_norm, height=L1_norm)
    rod2 = Rectangle(center=(0.5, 0.75), width=w_norm, height=L2_norm)
    patterned = PatternedLayer(
        thickness=h,
        lattice=lat,
        shapes=[(rod1, hBN_tensor), (rod2, hBN_tensor)],
        background_material=SiO2(n=n_SiO2),
    )
    stack = Stack(
        substrate=substrate,
        superstrate=superstrate,
        layers=[layer],
    )
    solver = Solver(layer_stack=stack, source=src, n_harmonics=(7, 7))
    result = solver.solve(wavelength=[wl])
    TTot_list.append(get_first(result.TTot) if hasattr(result, 'TTot') else np.nan)
    RTot_list.append(get_first(result.RTot) if hasattr(result, 'RTot') else np.nan)

TTot_arr = np.array(TTot_list)
RTot_arr = np.array(RTot_list)

print("TTot:", TTot_arr)
print("RTot:", RTot_arr)


# 保存数据到 CSV 文件，方便后续绘图和分析
df = pd.DataFrame({
    'wavelength': wavelengths,
    'TTot': TTot_arr,
    'RTot': RTot_arr
})
df.to_csv('bic_hbn_metasurface_results.csv', index=False)



# 合理性断言（可选）
assert len(TTot_arr) == len(wavelengths)
assert np.all(np.isfinite(TTot_arr))
assert np.all(TTot_arr >= 0) and np.all(TTot_arr <= 1)

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
