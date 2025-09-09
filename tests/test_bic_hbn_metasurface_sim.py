import numpy as np
import pytest
from rcwa import Material, TensorMaterial, Layer, PatternedLayer, Stack, Source, Solver, Rectangle, rectangular_lattice, Air, SiO2, HalfSpace, nm
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

    rod1 = Rectangle(center=(0.0, y1_c), width=w*S, height=L1*S)
    rod2 = Rectangle(center=(0.0, y2_c), width=w*S, height=L2*S)

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

    wavelengths = np.linspace(400, 1000, 11)  # nm, 用较少点加快测试
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
