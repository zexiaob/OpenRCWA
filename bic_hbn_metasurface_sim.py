import numpy as np
from rcwa import (
    Material, TensorMaterial, Layer, PatternedLayer, Stack, Source, Solver, Sweep,
    Rectangle, square_lattice, rectangular_lattice, Air, SiO2, HalfSpace, nm
)
from n_tensor_test import epsilon_tensor_dispersion

# 参数区
h = 150  # nm, 高度
w = 100  # nm, 宽度
L1 = 310 # nm, 棒1长度
L2 = 360 # nm, 棒2长度
px = 410 # nm, 晶格x周期
py = 430 # nm, 晶格y周期
n_SiO2 = 1.45
n_air = 1.0

# 参数化缩放因子S
S = 1.0  # 可批量扫描S

hBN_tensor = TensorMaterial(epsilon_tensor=epsilon_tensor_dispersion, name="hBN_dispersion")

lat = rectangular_lattice(px * S, py * S)

# 棒1位置和尺寸
x1_min, x1_max = -w/2 * S, w/2 * S
y1_c = -py/4 * S
y1_min, y1_max = y1_c - L1/2 * S, y1_c + L1/2 * S
# 棒2位置和尺寸
x2_min, x2_max = -w/2 * S, w/2 * S
y2_c = py/4 * S
y2_min, y2_max = y2_c - L2/2 * S, y2_c + L2/2 * S

# PatternedLayer 形状
rod1 = Rectangle(center=(0.0, y1_c), width=w*S, height=L1*S)
rod2 = Rectangle(center=(0.0, y2_c), width=w*S, height=L2*S)


patterned = PatternedLayer(
    thickness=h*S*1e-9,  # 转米
    lattice=lat,
    shapes=[(rod1, hBN_tensor), (rod2, hBN_tensor)],
    background_material=Material(er=1.0, ur=1.0),
)

# 层叠结构
substrate = HalfSpace(material=SiO2(n=n_SiO2)) # 半无限
superstrate = HalfSpace(material=Air())        # 半无限
stack = Stack(
    substrate=substrate,
    superstrate=superstrate,
    layers=[patterned],
)

# 光源设置
wavelengths = np.linspace(400, 1000, 121)  # nm
src = Source(
    wavelength=wavelengths,  # nm
    theta=0.0,               # 垂直入射
    phi=0.0,
    pTEM=[0, 1],             # y偏振
)


# 显式绑定 source 给 hBN_tensor（消除警告）
hBN_tensor.source = src

# 求解器与Sweep
solver = Solver(layer_stack=stack, source=src, n_harmonics=(7, 7))
results = solver.solve(wavelength=wavelengths)

import matplotlib.pyplot as plt
plt.plot(wavelengths, results.TTot)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmittance (TTot)')
plt.title('BIC hBN Metasurface Transmission Spectrum')
plt.show()

# 参数化扫描示例（可选）
# for S in np.linspace(0.65, 1.10, 10):
#     ... # 重新构建几何和仿真，收集/绘制多组结果
