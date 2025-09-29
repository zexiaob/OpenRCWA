"""
基于模拟调用流程的标准测试用例

这是一个标准的pytest格式测试文件，测试RCWA模拟的核心工作流程，
从材料创建到完整的多层、图案化和张量材料模拟。
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


class TestRCWASimulationWorkflow:
    """完整RCWA模拟工作流程测试"""

    @pytest.fixture
    def basic_materials(self):
        """创建基础材料"""
        return {
            'air': Material(er=1.0, ur=1.0),
            'silicon': Material(er=12.0, ur=1.0),
            'sio2': Material(er=2.25, ur=1.0),
            'gold': Material(er=-10.0+1.0j, ur=1.0)
        }

    def test_material_creation_and_access(self, basic_materials):
        """测试材料创建和属性访问"""
        silicon = basic_materials['silicon']
        assert silicon.er == 12.0
        assert silicon.ur == 1.0
        assert silicon.n == np.sqrt(12.0)

    def test_homogeneous_layer_creation(self, basic_materials):
        """测试均匀层创建"""
        layer = Layer(
            thickness=200e-9,
            material=basic_materials['silicon']
        )
        assert isinstance(layer, Layer)
        assert layer.homogenous == True
        assert layer.thickness == 200e-9
        assert layer.material.er == 12.0

    def test_multilayer_stack(self, basic_materials):
        """测试多层结构"""
        layers = [
            Layer(thickness=10.0, material=basic_materials['silicon']),
            Layer(thickness=100e-9, material=basic_materials['sio2']),
            Layer(thickness=200e-9, material=basic_materials['silicon']),
            Layer(thickness=50e-9, material=basic_materials['air'])
        ]
        
        assert len(layers) == 4
        for layer in layers:
            assert isinstance(layer, Layer)
            assert layer.homogenous == True
            assert layer.thickness > 0

    def test_simple_patterned_layer(self, basic_materials):
        """测试基础图案化层"""
        rect = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
        
        patterned_layer = PatternedLayer(
            thickness=220e-9,
            lattice=square_lattice(600e-9),
            background_material=basic_materials['air'],
            shapes=[(rect, basic_materials['silicon'])]
        )
        
        assert isinstance(patterned_layer, Layer)
        assert patterned_layer.homogenous == False
        assert patterned_layer.thickness == 220e-9
        
        # 测试栅格化
        er_field, ur_field = patterned_layer.rasterize_tensor_field()
        assert er_field.shape == (256, 256)
        assert np.any(np.isclose(er_field, 1.0))   # 空气背景
        assert np.any(np.isclose(er_field, 12.0))  # 硅图案

    def test_complex_boolean_pattern(self, basic_materials):
        """测试复杂布尔运算图案"""
        # 复杂图案：带孔的方块 + 额外特征
        base = Rectangle(center=(0.5, 0.5), width=0.8, height=0.8)
        hole = Circle(center=(0.5, 0.5), radius=0.15)
        pattern_with_hole = DifferenceShape(base, [hole])
        
        feature = Circle(center=(0.3, 0.3), radius=0.08)
        final_pattern = UnionShape([pattern_with_hole, feature])
        
        complex_layer = PatternedLayer(
            thickness=50e-9,
            lattice=square_lattice(1.0e-6),
            background_material=basic_materials['air'],
            shapes=[(final_pattern, basic_materials['gold'])]
        )
        
        assert isinstance(complex_layer, Layer)
        assert not complex_layer.homogenous
        
        bounds = complex_layer.get_bounds()
        assert len(bounds) == 4
        assert bounds[0] < bounds[1]  # x_min < x_max
        assert bounds[2] < bounds[3]  # y_min < y_max

    def test_mixed_layer_stack(self, basic_materials):
        """测试混合层栈（普通层+图案化层）"""
        # 普通层
        substrate = Layer(thickness=10.0, material=basic_materials['silicon'])
        capping = Layer(thickness=100e-9, material=basic_materials['sio2'])
        
        # 图案化层
        grating = Rectangle(center=(0.5, 0.5), width=0.4, height=1.0)
        patterned_layer = PatternedLayer(
            thickness=200e-9,
            lattice=rectangular_lattice(500e-9, 500e-9),
            background_material=basic_materials['air'],
            shapes=[(grating, basic_materials['silicon'])]
        )
        
        # 混合栈
        mixed_stack = [substrate, patterned_layer, capping]
        
        assert len(mixed_stack) == 3
        for layer in mixed_stack:
            assert isinstance(layer, Layer)

    def test_tensor_material_creation(self):
        """测试张量材料创建"""
        epsilon_tensor = np.array([
            [12.0, 0.2, 0.0],
            [0.2, 11.5, 0.0],
            [0.0, 0.0, 12.5]
        ], dtype=complex)
        
        aniso_material = TensorMaterial(
            epsilon_tensor=epsilon_tensor,
            mu_tensor=np.eye(3, dtype=complex)
        )
        
        assert aniso_material.epsilon_tensor.shape == (3, 3)
        assert aniso_material.mu_tensor.shape == (3, 3)
        assert np.allclose(aniso_material.epsilon_tensor, epsilon_tensor)

    def test_tensor_material_layer(self):
        """测试张量材料层"""
        eps_tensor = np.diag([10.0, 12.0, 14.0]).astype(complex)
        tensor_material = TensorMaterial(epsilon_tensor=eps_tensor)
        
        tensor_layer = Layer(
            thickness=300e-9,
            tensor_material=tensor_material
        )
        
        assert isinstance(tensor_layer, Layer)
        assert tensor_layer.thickness == 300e-9
        assert tensor_layer.is_anisotropic == True

    def test_convolution_matrix_generation(self, basic_materials):
        """测试卷积矩阵生成"""
        circle = Circle(center=(0.5, 0.5), radius=0.3)
        layer = PatternedLayer(
            thickness=200e-9,
            lattice=square_lattice(800e-9),
            background_material=basic_materials['air'],
            shapes=[(circle, basic_materials['silicon'])]
        )
        
        harmonics_x = np.array([-2, -1, 0, 1, 2])
        harmonics_y = np.array([-2, -1, 0, 1, 2])
        
        conv_matrix = layer.convolution_matrix(
            harmonics_x, harmonics_y, 'eps_xx'
        )
        
        expected_size = len(harmonics_x) * len(harmonics_y)
        assert conv_matrix.shape == (expected_size, expected_size)
        assert np.all(np.isfinite(conv_matrix))
        assert np.iscomplexobj(conv_matrix)

    def test_parametric_geometry(self, basic_materials):
        """测试参数化几何"""
        fill_factors = [0.2, 0.5, 0.8]
        silicon_fractions = []
        
        for ff in fill_factors:
            width = ff * 0.8
            rect = Rectangle(center=(0.5, 0.5), width=width, height=0.8)
            
            layer = PatternedLayer(
                thickness=200e-9,
                lattice=square_lattice(600e-9),
                background_material=basic_materials['air'],
                shapes=[(rect, basic_materials['silicon'])]
            )
            
            er_field, _ = layer.rasterize_tensor_field()
            si_fraction = np.sum(np.isclose(er_field, 12.0)) / er_field.size
            silicon_fractions.append(si_fraction)
        
        # 填充因子越大，硅占比应该越大
        assert silicon_fractions[0] < silicon_fractions[1] < silicon_fractions[2]

    def test_wavelength_sweep_compatibility(self, basic_materials):
        """测试波长扫描兼容性"""
        grating = Rectangle(center=(0.5, 0.5), width=0.5, height=1.0)
        layer = PatternedLayer(
            thickness=220e-9,
            lattice=rectangular_lattice(600e-9, 600e-9),
            background_material=basic_materials['air'],
            shapes=[(grating, basic_materials['silicon'])]
        )
        
        wavelengths = np.linspace(1400e-9, 1700e-9, 5)
        
        for wl in wavelengths:
            er_field, ur_field = layer.rasterize_tensor_field(wavelength=wl)
            assert er_field.shape == (256, 256)
            assert np.all(np.isfinite(er_field))
            assert np.all(np.isfinite(ur_field))

    def test_physical_bounds_validation(self, basic_materials):
        """测试物理边界验证"""
        circle = Circle(center=(0.5, 0.5), radius=0.4)
        layer = PatternedLayer(
            thickness=100e-9,
            lattice=square_lattice(800e-9),
            background_material=basic_materials['air'],
            shapes=[(circle, basic_materials['silicon'])]
        )
        
        # 验证几何边界
        bounds = layer.get_bounds()
        assert 0.0 <= bounds[0] < bounds[1] <= 1.0  # x边界
        assert 0.0 <= bounds[2] < bounds[3] <= 1.0  # y边界
        
        # 验证厚度为正
        assert layer.thickness > 0
        
        # 验证材料参数合理
        er_field, ur_field = layer.rasterize_tensor_field()
        assert np.all(er_field.real >= 1.0)  # 介电常数 >= 1（空气）
        assert np.all(ur_field.real >= 1.0)  # 磁导率 >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
