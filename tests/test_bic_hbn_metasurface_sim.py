import numpy as np
import pytest
from rcwa import (
    Material,
    TensorMaterial,
    Layer,
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

def test_bic_hbn_metasurface_sim():
    # 参数区
    h = 150  # nm, 高度
    w = 100  # nm, 宽度
    L1 = 310 # nm, 棒1长度
    L2 = 360 # nm, 棒2长度
    px = 410 # nm, 晶格x周期
    py = 430 # nm, 晶格y周期
    n_SiO2 = 1.45
    n_air = 1.0
    S = 1.0

    hBN_tensor = TensorMaterial(
        epsilon_tensor=epsilon_tensor_dispersion, name="hBN_dispersion"
    )
    lat = rectangular_lattice(px * S, py * S)

    # 使用归一化晶胞坐标 (0~1)
    w_norm = w / px
    L1_norm = L1 / py
    L2_norm = L2 / py
    rod1 = Rectangle(center=(0.5, 0.25), width=w_norm, height=L1_norm)  # 下移 py/4
    rod2 = Rectangle(center=(0.5, 0.75), width=w_norm, height=L2_norm)  # 上移 py/4

    # 验证 Rectangle.contains 在归一化坐标下工作正常
    assert rod1.contains(0.5, 0.25)
    assert not rod1.contains(0.5, 0.75)
    assert rod2.contains(0.5, 0.75)
    assert not rod2.contains(0.5, 0.25)

    patterned = PatternedLayer(
        thickness=h*S*1e-9,
        lattice=lat,
        shapes=[(rod1, hBN_tensor), (rod2, hBN_tensor)],
        background_material=Air(),
    )

    substrate = HalfSpace(material=SiO2(n=n_SiO2))
    superstrate = HalfSpace(material=Air())
    stack = Stack(
        substrate=substrate,
        superstrate=superstrate,
        layers=[patterned],
    )

    wavelengths = nm(np.linspace(400, 1000, 11))  # 用较少点加快测试
    src = Source(
        wavelength=wavelengths,
        theta=0.0,
        phi=0.0,
        pTEM=[0, 1],
    )
    hBN_tensor.source = src
    solver = Solver(layer_stack=stack, source=src, n_harmonics=(3, 3))
    results = solver.solve(wavelength=wavelengths)

    print(results.TTot)
    print(results.RTot)
    print(results.jones_matrix)
    print(results.phase_difference)
    
    # 断言结果合理
    assert hasattr(results, 'TTot')
    assert len(results.TTot) == len(wavelengths)
    assert np.all(np.isfinite(results.TTot))
    assert np.all(results.TTot >= 0) and np.all(results.TTot <= 1)
