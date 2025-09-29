完整流程：从材料到求解的一站式示例
================================

本页给出一个可运行的端到端示例，覆盖以下要点：

- 材料（各向同性 Material）与各向异性材料（TensorMaterial）
- 均匀层（Layer）
- 图案层（PatternedLayer）与自动分层（auto_z_slicing）
- 叠层（Stack / LayerStack）与半空间（superstrate/substrate）
- 光源（Source，角度与偏振）
- 求解器（Solver）与参数扫描（sweep）

示例使用 rcwa 的公开 API，代码可直接复制运行。


准备工作
--------

.. code-block:: python

    import numpy as np
    from openrcwa import (
        # 核心
        Solver, Source,
        # 模型：材料与层
        Material, TensorMaterial, Layer, Stack, HalfSpace, Air, SiO2, Silicon,
        nm, um, deg,
        # 几何：图案层与形状/晶格
        PatternedLayer, square_lattice, Rectangle, Circle, DifferenceShape, TaperedPolygon,
        # 偏振辅助（可选）
        LCP, RCP,
    )


步骤 1：定义材料（Material / TensorMaterial）
-------------------------------------------

1) 各向同性（标量）材料：

.. code-block:: python

    air = Air()                 # 折射率 ~ 1
    sio2 = SiO2(n=1.46)         # 二氧化硅
    si = Silicon(n=3.48)        # 硅（此处常数示例，实际可用数据库/公式）

也可直接用 Material 指定介电常数 er、磁导率 ur 或折射率 n：

.. code-block:: python

    polymer = Material(er=2.25)   # 等效 n ~ 1.5

2) 各向异性材料（TensorMaterial）：

常量对角张量（如单轴/双轴晶体）：

.. code-block:: python

    ani_const = TensorMaterial.from_diagonal(
        eps_xx=2.4, eps_yy=2.0, eps_zz=2.2, name="biaxial_const"
    )

或用函数创建色散张量（需要在运行时提供 Source 以取用 wavelength）：

.. code-block:: python

    def eps_xx(wl):
        return 2.3  # 演示用：常数；也可写成随 wl（米）变化的函数

    ani_disp = TensorMaterial.from_diagonal(
        eps_xx=eps_xx, eps_yy=lambda wl: 2.1, eps_zz=lambda wl: 2.2,
        name="uniaxial_disp"
    )

完整 3×3（含非对角）张量：

.. code-block:: python

    import numpy as np
    from rcwa import TensorMaterial

    # 常量 3×3（含非对角项）
    eps_full = np.array([[2.30, 0.10+0.05j, 0.00],
                         [0.10+0.05j, 2.10,     0.00],
                         [0.00,       0.00,     2.20]], dtype=complex)
    ani_full = TensorMaterial(epsilon_tensor=eps_full, name="full_tensor")

    # 函数型 3×3（随波长 wl 变化，单位米）
    def eps_func(wl):
        return np.array([[2.30, 0.05j, 0.0],
                         [0.05j, 2.10, 0.0],
                         [0.0,   0.0,  2.20]], dtype=complex)

    ani_full_disp = TensorMaterial(epsilon_tensor=eps_func, name="dispersive_full")

非对角“完整版”色散（函数/表格）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) 函数型非对角色散：

.. code-block:: python

    def eps_func_full(wl):
        # 自由写随 wl 变化的非对角分量（单位：米）
        c = 2.3 + 0.02*(wl/1.55 - 1.0)
        off = 0.05j * (1 + 0.5*np.sin(2*np.pi*wl/1.55))
        return np.array([[c,     off,  0.0],
                         [off,   2.1,  0.0],
                         [0.0,   0.0,  2.2]], dtype=complex)

    ani_nd_func = TensorMaterial(epsilon_tensor=eps_func_full, name='nd_dispersive')

2) 表格型非对角（n 或 ε）：

.. code-block:: python

    ani_nd_table = {
        'wavelength': np.array([1.50, 1.55, 1.60]),
        # 提供 n_* 或 epsilon_*；未提供的分量默认 0（对角未提供时默认 1）
        'epsilon_xx': np.array([2.30, 2.32, 2.35]),
        'epsilon_xy': 1j*np.array([0.05, 0.06, 0.07]),
        'epsilon_yx': 1j*np.array([0.05, 0.06, 0.07]),
        'epsilon_yy': np.array([2.10, 2.12, 2.15]),
        'epsilon_zz': np.array([2.20, 2.20, 2.20]),
    }
    ani_nd_tab = TensorMaterial(
        epsilon_tensor=ani_nd_table,
        allow_interpolation=True,
        allow_extrapolation=False,
        name='nd_tabulated'
    )

说明：若表格提供 n_* 分量，则内部先对 n 插值，再平方得到 ε，确保物理一致性；插值/外推需显式开启。

对角 + 旋转（产生非对角项）：

.. code-block:: python

    from rcwa import TensorMaterial
    ani_diag = TensorMaterial.from_diagonal(2.4, 2.0, 2.2)
    # 绕 z 轴 ~30° 的旋转矩阵（示例）
    R = np.array([[0.866, -0.5,   0.0],
                  [0.5,    0.866, 0.0],
                  [0.0,    0.0,   1.0]], dtype=float)
    ani_rot = ani_diag.rotated(R)

层构建后的旋转（rotate_layer）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rcwa import rotate_layer, deg

    # 已有一层各向异性层 ani_layer（用上面的 ani_full/ani_nd_func/ani_nd_tab 均可）
    ani_layer = Layer(tensor_material=ani_nd_func, thickness=300e-9)

    # 对整层做欧拉角旋转（ZYX）：alpha 绕 z，beta 绕 y，gamma 绕 x
    # 注意：PatternedLayer 仅支持 z 轴平面内旋转（beta=gamma=0）
    ani_layer_rot = rotate_layer(ani_layer, euler_angles=(deg(30), 0.0, 0.0), convention='ZYX')

    # 也可直接对 TensorMaterial 做旋转再构层
    Rz = np.array([[np.cos(deg(30)), -np.sin(deg(30)), 0.0],
                   [np.sin(deg(30)),  np.cos(deg(30)), 0.0],
                   [0.0,              0.0,             1.0]])
    ani_rot2 = ani_nd_tab.rotated(Rz)
    ani_layer_rot2 = Layer(tensor_material=ani_rot2, thickness=300e-9)

3) 外部表格（非函数）数据输入与显式插值/外推开关：

- 各向同性（Material）：

.. code-block:: python

    # data 中提供 wavelength 与 n（或 er/ur），单位自定但需与 Source 一致
    iso_data = {
        'wavelength': np.array([1.50, 1.55, 1.60]),
        'n': np.array([1.45+0j, 1.46+0j, 1.47+0j]),
    }
    mat_tab = Material(data=iso_data, allow_interpolation=True, allow_extrapolation=True)

- 各向异性（TensorMaterial）：支持 epsilon_* 或 n_* 组件表，或整张量数组 [N,3,3]

.. code-block:: python

    ani_table = {
        'wavelength': np.array([1.50, 1.55, 1.60]),
        # 也可用 epsilon_xx/xy/... 或整张量 'n_tensor': np.stack([...], axis=0)
        'n_xx': np.array([1.50, 1.51, 1.52]),
        'n_yy': np.array([1.48, 1.49, 1.50]),
        'n_zz': np.array([1.60, 1.60, 1.60]),
        # 省略的非对角项默认 0
    }
    ani_tab = TensorMaterial(
        epsilon_tensor=ani_table,
        allow_interpolation=True,
        allow_extrapolation=True,
        name='ani_tabulated'
    )

说明：当使用 data（各向同性）或字典表（各向异性）时，若当前波长不在表内，只有显式传入 allow_interpolation=True 才进行内插；表外请求只有显式 allow_extrapolation=True 才进行线性外推，否则会抛错提示开启开关。

4) 用表格“生成色散函数”，再作为函数型材料使用（更灵活）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

有时你希望先把表格变成可复用的色散函数，再传给材料构造器。可以使用 make_n_from_table 与 make_epsilon_tensor_from_table：

.. code-block:: python

    from rcwa import make_n_from_table, make_epsilon_tensor_from_table

    # 标量 n(wl) 函数，由表格构造
    n_func = make_n_from_table({
        'wavelength': np.array([1.50, 1.55, 1.60]),
        'n': np.array([1.45+0j, 1.46+0j, 1.47+0j]),
    }, allow_interpolation=True)
    mat_func = Material(n=n_func)  # 作为函数型色散材料使用

    # 非对角 ε(wl) 函数，由各向异性表格构造
    eps_func2 = make_epsilon_tensor_from_table({
        'wavelength': np.array([1.50, 1.55, 1.60]),
        'n_xx': np.array([1.50, 1.51, 1.52]),
        'n_xy': 1j*np.array([0.02, 0.03, 0.04]),
        'n_yx': 1j*np.array([0.02, 0.03, 0.04]),
        'n_yy': np.array([1.48, 1.49, 1.50]),
        'n_zz': np.array([1.60, 1.60, 1.60]),
    }, allow_interpolation=True)
    ani_func2 = TensorMaterial(epsilon_tensor=eps_func2)


步骤 2：定义层（Layer）
----------------------

1) 均匀层（各向同性）：

.. code-block:: python

    thin_film = Layer(material=polymer, thickness=200e-9)  # 200 nm 薄膜

2) 各向异性均匀层：

.. code-block:: python

    ani_layer = Layer(tensor_material=ani_const, thickness=300e-9)


步骤 3：定义图案层（PatternedLayer）
----------------------------------

PatternedLayer 将形状（Shape）与材料组合，生成可用于 RCWA 的卷积矩阵。
下例在方形晶格内放置“方块减去圆孔”的图案：

.. code-block:: python

    # 方形晶格，周期 500 nm
    lat = square_lattice(500e-9)

    # 形状在晶格的单位坐标（0..1）中定义
    outer = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
    hole = Circle(center=(0.5, 0.5), radius=0.2)
    pattern = DifferenceShape(outer, [hole])  # 外矩形减内圆 = “挖孔”结构

    patterned = PatternedLayer(
        thickness=300e-9,
        lattice=lat,
        shapes=[(pattern, si)],          # 图案区域用硅
        background_material=sio2         # 背景为 SiO2
    )

提示：PatternedLayer 对 TensorMaterial 会构建九个卷积矩阵（xx、xy、xz、yx、yy、yz、zx、zy、zz），以支持完整各向异性耦合。

可选：若需要随 z 改变的几何（例如侧壁锥形），使用 TaperedPolygon 等 z-aware 形状，配合自动分层更有效：

.. code-block:: python

    # 简例：顶部/底部给定同边数顶点（演示用，顶/底相同即无锥度）
    hex_bottom = [(0.5 + 0.25*np.cos(a), 0.5 + 0.25*np.sin(a)) for a in np.linspace(0, 2*np.pi, 7)[:-1]]
    hex_top    = [(0.5 + 0.18*np.cos(a), 0.5 + 0.18*np.sin(a)) for a in np.linspace(0, 2*np.pi, 7)[:-1]]
    tapered = TaperedPolygon(hex_bottom, hex_top)

    tapered_layer = PatternedLayer(
        thickness=400e-9,
        lattice=lat,
        shapes=[(tapered, si)],
        background_material=sio2,
    )


步骤 4：构建叠层（Stack），启用自动分层（auto_z_slicing）
-----------------------------------------------

Stack 支持以 superstrate/substrate 指定上下半空间；可开启 auto_z_slicing 对 z-aware 图案层自动切片。

.. code-block:: python

    stack = Stack(
        # 内部层从上到下依次列出
        thin_film,
        patterned,
        tapered_layer,
        # 半空间（上/下）可直接传入材料，内部自动转 HalfSpace
        superstrate=air,
        substrate=sio2,
        # 自动 z 分层：
        # - True：按层建议切片；
        # - 整数 n：均匀切成 n 片；
        # - 浮点列表：显式切片位置（米，0..thickness 之间的内部点）。
        auto_z_slicing=True,
        # 可选：限制最大切片数
        max_slices=8,
    )


步骤 5：定义光源（Source）
------------------------

波长、入射角（theta、phi，弧度）和偏振（TE/TM 分量）均可扫描：

.. code-block:: python

    src = Source(
        wavelength=1.55,        # 单位与几何一致即可（标度不变性）；也可用 1550e-9
        theta=0.0,               # 入射与法线夹角
        phi=0.0,                 # 方位角
        pTEM=RCP(),              # 右旋圆偏振；也可自定义复 TE/TM 向量
    )


步骤 6：创建求解器（Solver）
---------------------------

.. code-block:: python

    solver = Solver(layer_stack=stack, source=src, n_harmonics=(7, 7))

说明：

- 对均匀薄膜（TMM）可设 n_harmonics=1；
- 对 1D/2D 周期结构用奇数（如 5、7、9）以包含 0 阶；
- 若启用收敛检查（check_convergence=True），求解会自动提升谐波截断直到收敛或到达迭代上限。


步骤 7：参数扫描与求解（sweep + solve）
--------------------------------------

有两种推荐方式：

A. 使用 Solver.solve 直接扫描（简单、线性列表）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) 扫描光源参数（1D 或多维笛卡尔积）：

.. code-block:: python

    wavelengths = np.linspace(1.2, 1.8, 31)
    thetas = np.linspace(0.0, deg(45), 7)

    # 注意：传入多个序列时，内部按参数给入顺序做笛卡尔积
    results = solver.solve(wavelength=wavelengths, theta=thetas)

    # results 为 Results 对象；多点扫描会把每个字段打平成同长度列表
    # 例如：results.R 是一个长度为 len(wavelengths)*len(thetas) 的列表
    # 若只 1D 扫描，可直接用 results.RTot 画图：
    if len(thetas) == 1:
        import matplotlib.pyplot as plt
        plt.plot(results['wavelength'], results.RTot)  # RTot 为列表
        plt.xlabel('wavelength')
        plt.ylabel('RTot')
        plt.show()

    print("能量守恒: ", np.allclose(np.array(results.RTot) + np.array(results.TTot), 1.0, atol=1e-2))

2) 同时扫描几何/材料参数（对象属性）：

.. code-block:: python

    # 与光源参数组合成笛卡尔积；顺序 = 你传入的参数顺序
    results2 = solver.solve(
        (thin_film, {"thickness": [100e-9, 150e-9, 200e-9]}),
        wavelength=[1.3, 1.55, 1.7],
        max_iters=30, atol=1e-3, rtol=1e-2, check_convergence=False,
    )

    # 多参数扫描下，results.<key> 为同长度列表（展平后的顺序与参数顺序一致）
    # 例如第 i 项对应一个具体组合，可自己 reshape 或成网格之后再可视化


B. 使用 Sweep 获取“有坐标的网格结果”（推荐用于多维扫描）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rcwa import Sweep

    # 1) 仅光源参数
    sweep = Sweep({
        'wavelength': np.linspace(1.2, 1.8, 31),
        'theta': np.linspace(0.0, deg(45), 7),
    }, backend='serial')  # 可选 'loky' 并行（若安装 joblib）

    out = sweep.run(stack, src, n_harmonics=(7, 7))
    grid = out['result_grid']   # ResultGrid，带 dims/coords，可选择/切片

    # 快速作图（仅 1D 网格时）
    try:
        grid.plot(x='wavelength', y='RTot', show=True)
    except Exception:
        pass

    # 选择某个 theta 截面
    if grid is not None:
        cut = grid.sel(theta=grid.coords['theta'][0])  # 固定第一条 theta
        # cut 若为 1D，则可取字段数组：
        RTot_line = cut.get('R').sum(axis=-1)  # 按衍射级次求和

    # 2) 加对象参数（几何/材料）
    sweep2 = Sweep({
        'wavelength': [1.3, 1.55, 1.7],
        (thin_film,): { 'thickness': [100e-9, 150e-9, 200e-9] },
    })
    out2 = sweep2.run(stack, src, n_harmonics=(7, 7))
    grid2 = out2['result_grid']
    # 坐标里会出现 "Layer.thickness" 这类复合名，便于索引

提示与常见问题（扫描）
~~~~~~~~~~~~~~~~~~~~~~
- Solver.solve 返回展平列表，更适合轻量场景；多维绘图/索引更建议用 Sweep 的 ResultGrid。
- 参数顺序决定展平次序；如需网格结构，可用 numpy.reshape 或直接用 Sweep。
- 并行：Sweep(backend='loky', n_jobs=-1) 需要安装 joblib。


结果对象：拿到什么、怎么用
--------------------------

单点：Result（solver.solve 的元素）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

每个求解点都会生成一个 `Result`，包含完整复振幅与派生量：

- 复振幅（按衍射级次/偏振展开）：rx, ry, rz, tx, ty, tz
- 反/透强度：R, T（若为数组，表示各级次；合计见 RTot/TTot）
- 汇总：RTot, TTot, conservation (= RTot + TTot), A (= 1 - R - T)
- 便捷：r_complex()/t_complex()、get_phases()、get_intensities()

示例：

.. code-block:: python

    res = solver.solve(wavelength=[1.55])[0]  # 单点
    print(res.RTot, res.TTot, res.conservation)
    r_phase, t_phase = res.get_phases()
    r_I, t_I = res.get_intensities()

多点/多维：ResultGrid（Sweep.run 返回）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ResultGrid` 带有 dims/coords 并保留每个点的 Result：

- 选择：grid.sel(theta=..., wavelength=...) 或 grid.isel(theta=0)
- 取数组：grid.get('R') -> 形如 [dims..., orders]
- 提取总量：grid.get('T') 沿最后轴求和即可得到 TTot 网格
- 快速作图：grid.plot(x='wavelength', y='RTot')（限 1D）

示例：

.. code-block:: python

    grid = out['result_grid']
    # 取透过率总量的二维网格（若有两个扫描维度）
    T_arr = grid.get('T')           # [..., orders]
    T_tot = T_arr.sum(axis=-1)      # [...]

    # 选取某个坐标点，得到单个 Result
    res_pt = grid.sel(theta=grid.coords['theta'][0], wavelength=1.55)
    print(res_pt.RTot)

圆二色性（CD）助手
~~~~~~~~~~~~~~~~~~~

提供 `compute_circular_dichroism` 计算 TTot(RCP) - TTot(LCP)：

.. code-block:: python

    from rcwa.solve.results import compute_circular_dichroism

    # grid 需在某一维包含 LCP()/RCP() 两个偏振态（默认维名 'pTEM'）
    cd = compute_circular_dichroism(grid, dim='pTEM')
    # 若其它维都被固定/选择，cd 为标量；否则为随余下维度变化的数组


几何参数扫描示例（PatternedLayer）
---------------------------------

PatternedLayer 支持参数化几何：其 with_params 会把传入的关键字参数下发到内部 Shape（若 Shape 实现了 with_params）。因此只需把 PatternedLayer 作为目标对象，即可扫描图案层尺寸、旋转、晶格等。

1) 扫描图案中矩形的宽度与旋转角
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rcwa import Sweep
    import numpy as np

    # 使用前文定义的 patterned（包含一个外矩形与内圆孔的 DifferenceShape）
    sweep_geom = Sweep({
        (patterned,): {
            'width': np.linspace(0.3, 0.7, 5),      # 作用到 Rectangle.with_params
            'rotation': np.linspace(0.0, deg(60), 4) # 作用到 Rectangle.with_params
        },
        'wavelength': [1.55],
    }, backend='serial')

    outg = sweep_geom.run(stack, src, n_harmonics=(7, 7))
    gridg = outg['result_grid']
    # 维度坐标包含：'wavelength'、'PatternedLayer.width'、'PatternedLayer.rotation'
    # 可选取某个 rotation 截面并随 width 作图
    if gridg is not None:
        cut = gridg.sel(**{'PatternedLayer.rotation': gridg.coords['PatternedLayer.rotation'][0]})
        # 取总透过率随宽度变化
        import matplotlib.pyplot as plt
        y = [np.sum(r.T) if hasattr(r.T, '__iter__') else r.T for r in cut.data]
        x = cut.coords['PatternedLayer.width']
        plt.plot(x, y)
        plt.xlabel('width (lattice units)')
        plt.ylabel('TTot')
        plt.show()

2) 扫描晶格周期（lattice）
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    periods = [400e-9, 500e-9, 600e-9]
    sweep_lattice = Sweep({
        (patterned,): {
            'lattice': [square_lattice(p) for p in periods],
        },
        'wavelength': [1.55],
    })
    outL = sweep_lattice.run(stack, src, n_harmonics=(7, 7))
    gridL = outL['result_grid']

3) 同时扫描两个图案层（多目标）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # 构造第二个图案层 patterned2：同晶格，圆孔占空比可调
    hole2 = Circle(center=(0.5, 0.5), radius=0.15)
    patterned2 = PatternedLayer(
        thickness=200e-9,
        lattice=square_lattice(500e-9),
        shapes=[(hole2, si)],
        background_material=sio2,
    )

    # 放入叠层：置于 patterned 之下（示例）
    stack2 = Stack(
        thin_film,
        patterned,
        patterned2,
        superstrate=air,
        substrate=sio2,
        auto_z_slicing=True,
        max_slices=8,
    )

    sweep_multi = Sweep({
        'objects': [
            { 'targets': [patterned],  'params': { 'width': [0.4, 0.6] } },
            { 'targets': [patterned2], 'params': { 'radius': [0.10, 0.20, 0.30] } },
        ],
        'wavelength': np.linspace(1.4, 1.7, 7),
    })

    outM = sweep_multi.run(stack2, src, n_harmonics=(7, 7))
    gridM = outM['result_grid']
    # 现在 coords 包含：wavelength, PatternedLayer.width, PatternedLayer.radius（目标名按类名聚合）

说明：
- 只有实现了 with_params 的对象/形状才会响应对应参数（Rectangle: width/height/rotation；Circle: radius；Polygon: 顶点模板等）。
- PatternedLayer.with_params 会把 kwargs 透传给其包含的 Shape 并更新 thickness/lattice/background_material/rotation_z 等层级参数。
- 若需要对多个 PatternedLayer 同时扫描，使用 Sweep 的 'objects' 形式分别指定 targets 和 params。


附：小贴士与常见问题
--------------------

- 单位：建议统一使用 SI（米、弧度）。可用 nm()/um()/deg() 进行转换。
- 半空间：优先使用 superstrate/substrate（比 incident_layer/transmission_layer 更直观）。
- 图案层切片：auto_z_slicing 对 z-aware 形状（如 TaperedPolygon）最有效；纯 z-均匀图案层切片与否等价。
- 收敛：对 2D 周期结构，提高 n_harmonics 可改善精度；开启 check_convergence=True 可自动迭代。
- 偏振：RCP()/LCP() 返回归一化的 TE/TM 复向量；也可手动给定 pTEM=[pTE, pTM]（复数）。
