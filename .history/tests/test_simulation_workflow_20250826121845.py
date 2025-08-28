"""
完整RCWA模拟工作流程测试用例

测试从材料创建到结果获取的完整模拟调用流程，
验证新架构下的各种模拟场景。
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rcwa.model.material import Material, TensorMaterial
from rcwa.model.layer import Layer
from rcwa.geom.shape import Rectangle, Circle, UnionShape, DifferenceShape
from rcwa.geom.patterned import PatternedLayer, square_lattice, rectangular_lattice
from numpy.testing import assert_allclose


class TestBasicSimulationWorkflow:
    """测试基础RCWA模拟工作流程"""

    def test_simple_homogeneous_simulation(self):
        """测试最简单的均匀层模拟"""
        # 创建材料
        air = Material(er=1.0, ur=1.0)
        silicon = Material(er=12.0, ur=1.0)
        
        # 创建均匀层
        layer = Layer(
            thickness=200e-9,  # 200nm
            material=silicon
        )
        
        # 验证layer创建成功
        assert isinstance(layer, Layer)
        assert layer.homogenous == True
        assert layer.thickness == 200e-9
        assert layer.material.er == 12.0
        
        print("✅ 基础均匀层创建成功")

    def test_multilayer_stack_creation(self):
        """测试多层结构创建"""
        # 创建材料
        air = Material(er=1.0, ur=1.0)
        silicon = Material(er=12.0, ur=1.0)
        sio2 = Material(er=2.25, ur=1.0)
        
        # 创建多层结构
        layers = [
            Layer(thickness=10.0, material=silicon),    # 厚衬底
            Layer(thickness=100e-9, material=sio2),     # SiO2层
            Layer(thickness=200e-9, material=silicon),  # Si层
            Layer(thickness=50e-9, material=air)        # 空气间隙
        ]
        
        # 验证所有层都是Layer实例
        for i, layer in enumerate(layers):
            assert isinstance(layer, Layer)
            assert layer.homogenous == True
            print(f"✅ 第{i+1}层创建成功: {layer.material.er=}, {layer.thickness=}")
        
        assert len(layers) == 4
        print("✅ 多层结构创建成功")

    def test_material_property_access(self):
        """测试材料属性访问"""
        # 创建材料
        silicon = Material(er=12.0, ur=1.0)
        
        # 测试属性访问
        assert silicon.er == 12.0
        assert silicon.ur == 1.0
        assert silicon.n == np.sqrt(12.0 * 1.0)
        
        # 测试在不同波长下的属性（非色散材料）
        wavelength = 1.55e-6
        assert silicon.er == 12.0  # 应该不变
        
        print("✅ 材料属性访问正常")


class TestPatternedLayerSimulation:
    """测试图案化层模拟工作流程"""
    
    def test_simple_patterned_layer(self):
        """测试简单图案化层"""
        # 材料
        air = Material(er=1.0, ur=1.0)
        silicon = Material(er=12.0, ur=1.0)
        
        # 简单矩形图案
        rect = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
        
        # 创建图案化层
        patterned_layer = PatternedLayer(
            thickness=220e-9,
            lattice=square_lattice(600e-9),
            background_material=air,
            shapes=[(rect, silicon)]
        )
        
        # 验证PatternedLayer是Layer
        assert isinstance(patterned_layer, Layer)
        assert patterned_layer.homogenous == False  # 非均匀
        assert patterned_layer.thickness == 220e-9
        
        print("✅ 简单图案化层创建成功")
        
        # 测试栅格化
        er_field, ur_field = patterned_layer.rasterize_tensor_field()
        assert er_field.shape == (256, 256)
        assert np.any(np.isclose(er_field, 1.0))   # 空气背景
        assert np.any(np.isclose(er_field, 12.0))  # 硅图案
        
        print("✅ 图案栅格化成功")

    def test_complex_boolean_pattern(self):
        """测试复杂布尔运算图案"""
        # 材料
        air = Material(er=1.0, ur=1.0)
        gold = Material(er=-10.0+1.0j, ur=1.0)  # 金属
        
        # 复杂图案：带孔的方块 + 小圆特征
        base = Rectangle(center=(0.5, 0.5), width=0.8, height=0.8)
        hole = Circle(center=(0.5, 0.5), radius=0.15)
        pattern_with_hole = DifferenceShape(base, [hole])
        
        # 添加小特征
        feature1 = Circle(center=(0.3, 0.3), radius=0.08)
        feature2 = Circle(center=(0.7, 0.7), radius=0.08)
        features = UnionShape([feature1, feature2])
        
        # 最终图案
        final_pattern = UnionShape([pattern_with_hole, features])
        
        # 创建图案化层
        complex_layer = PatternedLayer(
            thickness=50e-9,  # 50nm金属层
            lattice=square_lattice(1.0e-6),  # 1μm周期
            background_material=air,
            shapes=[(final_pattern, gold)]
        )
        
        # 验证
        assert isinstance(complex_layer, Layer)
        assert not complex_layer.homogenous
        
        # 测试边界
        bounds = complex_layer.get_bounds()
        assert len(bounds) == 4
        assert bounds[0] < bounds[1]  # x_min < x_max
        assert bounds[2] < bounds[3]  # y_min < y_max
        
        print("✅ 复杂布尔图案创建成功")
        print(f"   图案边界: x∈[{bounds[0]:.2f}, {bounds[1]:.2f}], y∈[{bounds[2]:.2f}, {bounds[3]:.2f}]")

    def test_mixed_layer_stack(self):
        """测试混合层栈（普通层+图案化层）"""
        # 材料
        air = Material(er=1.0, ur=1.0)
        silicon = Material(er=12.0, ur=1.0)
        sio2 = Material(er=2.25, ur=1.0)
        
        # 普通层
        substrate = Layer(thickness=10.0, material=silicon)
        capping = Layer(thickness=100e-9, material=sio2)
        
        # 图案化层
        grating = Rectangle(center=(0.5, 0.5), width=0.4, height=1.0)
        patterned_layer = PatternedLayer(
            thickness=200e-9,
            lattice=rectangular_lattice(500e-9, 500e-9),
            background_material=air,
            shapes=[(grating, silicon)]
        )
        
        # 混合栈
        mixed_stack = [substrate, patterned_layer, capping]
        
        # 验证所有层都是Layer实例
        for i, layer in enumerate(mixed_stack):
            assert isinstance(layer, Layer)
            print(f"✅ 层{i+1}: {'均匀' if layer.homogenous else '图案化'}, 厚度={layer.thickness:.2e}m")
        
        assert len(mixed_stack) == 3
        print("✅ 混合层栈创建成功")


class TestTensorMaterialSimulation:
    """测试张量材料模拟工作流程"""
    
    def test_anisotropic_material_creation(self):
        """测试各向异性材料创建"""
        # 创建各向异性张量
        epsilon_tensor = np.array([
            [12.0, 0.2, 0.0],
            [0.2, 11.5, 0.0],
            [0.0, 0.0, 12.5]
        ], dtype=complex)
        
        # 创建张量材料
        aniso_material = TensorMaterial(
            epsilon_tensor=epsilon_tensor,
            mu_tensor=np.eye(3, dtype=complex)
        )
        
        # 验证张量属性
        assert aniso_material.epsilon_tensor.shape == (3, 3)
        assert aniso_material.mu_tensor.shape == (3, 3)
        assert np.allclose(aniso_material.epsilon_tensor, epsilon_tensor)
        
        print("✅ 各向异性材料创建成功")
        print(f"   εxx={aniso_material.epsilon_tensor[0,0]}")
        print(f"   εyy={aniso_material.epsilon_tensor[1,1]}")
        print(f"   εzz={aniso_material.epsilon_tensor[2,2]}")

    def test_tensor_material_layer(self):
        """测试张量材料层"""
        # 创建张量材料
        eps_tensor = np.diag([10.0, 12.0, 14.0]).astype(complex)
        tensor_material = TensorMaterial(epsilon_tensor=eps_tensor)
        
        # 创建张量材料层
        tensor_layer = Layer(
            thickness=300e-9,
            tensor_material=tensor_material  # 使用tensor_material参数
        )
        
        # 验证层属性
        assert isinstance(tensor_layer, Layer)
        assert tensor_layer.thickness == 300e-9
        assert tensor_layer.is_anisotropic == True
        
        print("✅ 张量材料层创建成功")

    def test_rotated_tensor_material(self):
        """测试旋转后的张量材料"""
        # 原始张量（对角）
        original_tensor = np.diag([12.0, 11.0, 13.0]).astype(complex)
        material = TensorMaterial(epsilon_tensor=original_tensor)
        
        # 旋转材料（绕z轴30度）
        angle = np.pi / 6  # 30度
        rotated_material = material.rotate(alpha=0, beta=0, gamma=angle)
        
        # 验证旋转后的材料
        assert isinstance(rotated_material, TensorMaterial)
        rotated_tensor = rotated_material.epsilon_tensor
        assert rotated_tensor.shape == (3, 3)
        
        # 检查旋转后xy分量不为零（耦合）
        assert abs(rotated_tensor[0, 1]) > 1e-10
        assert abs(rotated_tensor[1, 0]) > 1e-10
        
        print("✅ 张量材料旋转成功")
        print(f"   旋转后εxy = {rotated_tensor[0,1]:.3f}")


class TestConvolutionMatrixGeneration:
    """测试卷积矩阵生成"""
    
    def test_convolution_matrix_interface(self):
        """测试卷积矩阵接口"""
        # 创建简单图案化层
        air = Material(er=1.0, ur=1.0)
        si = Material(er=12.0, ur=1.0)
        
        circle = Circle(center=(0.5, 0.5), radius=0.3)
        layer = PatternedLayer(
            thickness=200e-9,
            lattice=square_lattice(800e-9),
            background_material=air,
            shapes=[(circle, si)]
        )
        
        # 创建谐波数组
        harmonics_x = np.array([-2, -1, 0, 1, 2])
        harmonics_y = np.array([-2, -1, 0, 1, 2])
        
        # 测试卷积矩阵生成
        conv_matrix = layer.convolution_matrix(
            harmonics_x, harmonics_y, 'eps_xx'
        )
        
        # 验证矩阵属性
        expected_size = len(harmonics_x) * len(harmonics_y)  # 25
        assert conv_matrix.shape == (expected_size, expected_size)
        assert np.all(np.isfinite(conv_matrix))
        assert np.iscomplexobj(conv_matrix)
        
        # 测试其他张量分量
        for component in ['eps_yy', 'eps_zz']:
            conv_mat = layer.convolution_matrix(harmonics_x, harmonics_y, component)
            assert conv_mat.shape == (expected_size, expected_size)
            assert np.all(np.isfinite(conv_mat))
        
        print("✅ 卷积矩阵生成成功")
        print(f"   矩阵尺寸: {conv_matrix.shape}")
        print(f"   是否有限: {np.all(np.isfinite(conv_matrix))}")

    def test_harmonics_suggestion(self):
        """测试谐波数建议功能"""
        from rcwa.core.adapters import suggest_harmonics_for_pattern
        
        # 创建包含小特征的图案
        air = Material(er=1.0, ur=1.0)
        si = Material(er=12.0, ur=1.0)
        
        small_circle = Circle(center=(0.5, 0.5), radius=0.1)  # 小特征
        layer = PatternedLayer(
            thickness=100e-9,
            lattice=square_lattice(2.0e-6),  # 大周期
            background_material=air,
            shapes=[(small_circle, si)]
        )
        
        # 获取建议谐波数
        suggested_harmonics = suggest_harmonics_for_pattern(
            layer, wavelength=1.55e-6, target_accuracy=0.01
        )
        
        # 验证建议
        assert isinstance(suggested_harmonics, tuple)
        assert len(suggested_harmonics) == 2
        assert all(h >= 3 for h in suggested_harmonics)  # 合理的最小值
        assert all(h % 2 == 1 for h in suggested_harmonics)  # 奇数
        
        print("✅ 谐波数建议成功")
        print(f"   建议谐波数: {suggested_harmonics}")


class TestParametricSimulation:
    """测试参数化模拟"""
    
    def test_parametric_geometry(self):
        """测试参数化几何"""
        # 材料
        air = Material(er=1.0, ur=1.0)
        si = Material(er=12.0, ur=1.0)
        
        # 创建不同填充因子的层
        fill_factors = [0.2, 0.5, 0.8]
        layers = []
        
        for ff in fill_factors:
            # 根据填充因子调整宽度
            width = ff * 0.8
            rect = Rectangle(center=(0.5, 0.5), width=width, height=0.8)
            
            layer = PatternedLayer(
                thickness=200e-9,
                lattice=square_lattice(600e-9),
                background_material=air,
                shapes=[(rect, si)]
            )
            layers.append(layer)
        
        # 验证不同填充因子产生不同的栅格化结果
        silicon_fractions = []
        for layer in layers:
            er_field, _ = layer.rasterize_tensor_field()
            si_fraction = np.sum(np.isclose(er_field, 12.0)) / er_field.size
            silicon_fractions.append(si_fraction)
        
        # 填充因子越大，硅占比应该越大
        assert silicon_fractions[0] < silicon_fractions[1] < silicon_fractions[2]
        
        print("✅ 参数化几何测试成功")
        print(f"   硅填充比例: {[f'{frac:.3f}' for frac in silicon_fractions]}")

    def test_wavelength_sweep_setup(self):
        """测试波长扫描设置"""
        # 创建一个固定的结构
        air = Material(er=1.0, ur=1.0)
        si = Material(er=12.0, ur=1.0)
        
        grating = Rectangle(center=(0.5, 0.5), width=0.5, height=1.0)
        layer = PatternedLayer(
            thickness=220e-9,
            lattice=rectangular_lattice(600e-9, 600e-9),
            background_material=air,
            shapes=[(grating, si)]
        )
        
        # 模拟多个波长点的设置
        wavelengths = np.linspace(1400e-9, 1700e-9, 10)
        
        # 验证每个波长都能正确设置
        for wl in wavelengths:
            # 在实际模拟中，这里会创建Source和Solver
            # 现在我们只验证波长参数传递
            er_field, ur_field = layer.rasterize_tensor_field(wavelength=wl)
            
            assert er_field.shape == (256, 256)
            assert ur_field.shape == (256, 256)
            assert np.all(np.isfinite(er_field))
            assert np.all(np.isfinite(ur_field))
        
        print("✅ 波长扫描设置成功")
        print(f"   波长范围: {wavelengths[0]*1e9:.0f}-{wavelengths[-1]*1e9:.0f} nm")


class TestSimulationValidation:
    """测试模拟结果验证"""
    
    def test_energy_conservation_setup(self):
        """测试能量守恒验证的设置"""
        # 创建无损材料
        air = Material(er=1.0, ur=1.0)
        si = Material(er=12.0, ur=1.0)  # 无损硅
        
        # 创建简单结构
        layer = Layer(thickness=200e-9, material=si)
        
        # 验证材料是无损的
        assert np.isreal(si.er)
        assert np.isreal(si.ur)
        assert si.er > 0
        assert si.ur > 0
        
        print("✅ 无损材料验证成功")
        
        # 对于有损材料
        gold = Material(er=-10.0+1.0j, ur=1.0)
        assert not np.isreal(gold.er)
        print("✅ 有损材料识别成功")

    def test_physical_bounds_validation(self):
        """测试物理边界验证"""
        # 创建测试层
        air = Material(er=1.0, ur=1.0)
        si = Material(er=12.0, ur=1.0)
        
        circle = Circle(center=(0.5, 0.5), radius=0.4)
        layer = PatternedLayer(
            thickness=100e-9,
            lattice=square_lattice(800e-9),
            background_material=air,
            shapes=[(circle, si)]
        )
        
        # 验证几何边界
        bounds = layer.get_bounds()
        assert 0.0 <= bounds[0] < bounds[1] <= 1.0  # x边界
        assert 0.0 <= bounds[2] < bounds[3] <= 1.0  # y边界
        
        # 验证厚度为正
        assert layer.thickness > 0
        
        # 验证材料参数合理
        er_field, ur_field = layer.rasterize_tensor_field()
        assert np.all(er_field.real >= 1.0)  # 介电常数 >= 1
        assert np.all(ur_field.real >= 1.0)  # 磁导率 >= 1
        
        print("✅ 物理边界验证成功")


if __name__ == '__main__':
    # 运行所有测试
    test_classes = [
        TestBasicSimulationWorkflow,
        TestPatternedLayerSimulation, 
        TestTensorMaterialSimulation,
        TestConvolutionMatrixGeneration,
        TestParametricSimulation,
        TestSimulationValidation
    ]
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"🧪 运行 {test_class.__name__}")
        print('='*60)
        
        test_instance = test_class()
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                print(f"\n🔬 执行 {method_name}")
                try:
                    getattr(test_instance, method_name)()
                except Exception as e:
                    print(f"❌ 测试失败: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("🎉 所有模拟工作流程测试完成!")
    print('='*60)
