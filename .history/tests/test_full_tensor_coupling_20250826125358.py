"""
全张量耦合实现测试

测试完整的 3x3 张量材料栅格化和卷积矩阵计算，
验证非对角分量的正确处理。
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rcwa.model.material import Material, TensorMaterial
from rcwa.geom.shape import Rectangle, Circle
from rcwa.geom.patterned import PatternedLayer, square_lattice
from numpy.testing import assert_allclose


class TestFullTensorCoupling:
    """测试完整张量耦合实现"""

    def test_isotropic_material_tensor_rasterization(self):
        """测试各向同性材料的张量栅格化"""
        air = Material(er=1.0, ur=1.0)
        si = Material(er=12.0, ur=1.0)
        
        rect = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
        layer = PatternedLayer(
            thickness=200e-9,
            lattice=square_lattice(600e-9),
            background_material=air,
            shapes=[(rect, si)]
        )
        
        # 测试完整张量场栅格化
        epsilon_field, mu_field = layer.rasterize_full_tensor_field()
        
        assert epsilon_field.shape == (256, 256, 3, 3)
        assert mu_field.shape == (256, 256, 3, 3)
        
        # 检查张量结构：各向同性材料应该是对角的
        center_y, center_x = 128, 128  # 中心位置
        
        # 硅区域的张量
        si_epsilon = epsilon_field[center_y, center_x]
        si_mu = mu_field[center_y, center_x]
        
        # 对角分量应该是材料值
        assert np.isclose(si_epsilon[0, 0], 12.0)
        assert np.isclose(si_epsilon[1, 1], 12.0)
        assert np.isclose(si_epsilon[2, 2], 12.0)
        
        # 非对角分量应该是零
        assert np.isclose(si_epsilon[0, 1], 0.0)
        assert np.isclose(si_epsilon[0, 2], 0.0)
        assert np.isclose(si_epsilon[1, 2], 0.0)
        
        print("✅ 各向同性材料张量栅格化正确")

    def test_anisotropic_material_tensor_rasterization(self):
        """测试各向异性材料的张量栅格化"""
        air = Material(er=1.0, ur=1.0)
        
        # 创建各向异性张量材料
        epsilon_tensor = np.array([
            [12.0, 0.5, 0.0],
            [0.5, 11.0, 0.2],
            [0.0, 0.2, 13.0]
        ], dtype=complex)
        
        aniso_material = TensorMaterial(
            epsilon_tensor=epsilon_tensor,
            mu_tensor=np.eye(3, dtype=complex)
        )
        
        rect = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
        layer = PatternedLayer(
            thickness=200e-9,
            lattice=square_lattice(600e-9),
            background_material=air,
            shapes=[(rect, aniso_material)]
        )
        
        # 测试完整张量场栅格化
        epsilon_field, mu_field = layer.rasterize_full_tensor_field()
        
        # 检查张量结构：各向异性材料应该有非对角项
        center_y, center_x = 128, 128  # 中心位置
        aniso_epsilon = epsilon_field[center_y, center_x]
        
        # 验证张量分量
        assert np.isclose(aniso_epsilon[0, 0], 12.0)  # εxx
        assert np.isclose(aniso_epsilon[1, 1], 11.0)  # εyy
        assert np.isclose(aniso_epsilon[2, 2], 13.0)  # εzz
        assert np.isclose(aniso_epsilon[0, 1], 0.5)   # εxy
        assert np.isclose(aniso_epsilon[1, 0], 0.5)   # εyx
        assert np.isclose(aniso_epsilon[1, 2], 0.2)   # εyz
        assert np.isclose(aniso_epsilon[2, 1], 0.2)   # εzy
        
        print("✅ 各向异性材料张量栅格化正确")

    def test_full_tensor_convolution_matrices(self):
        """测试完整张量卷积矩阵计算"""
        air = Material(er=1.0, ur=1.0)
        
        # 创建各向异性张量材料
        epsilon_tensor = np.array([
            [10.0, 1.0, 0.0],
            [1.0, 12.0, 0.0],
            [0.0, 0.0, 14.0]
        ], dtype=complex)
        
        aniso_material = TensorMaterial(epsilon_tensor=epsilon_tensor)
        
        circle = Circle(center=(0.5, 0.5), radius=0.3)
        layer = PatternedLayer(
            thickness=200e-9,
            lattice=square_lattice(800e-9),
            background_material=air,
            shapes=[(circle, aniso_material)]
        )
        
        # 计算卷积矩阵
        harmonics = (5, 5)  # 5x5 harmonics
        conv_matrices = layer.to_convolution_matrices(harmonics)
        
        # 验证所有 18 个张量分量都存在
        expected_components = [
            'er_xx', 'er_xy', 'er_xz', 'er_yx', 'er_yy', 'er_yz', 'er_zx', 'er_zy', 'er_zz',
            'ur_xx', 'ur_xy', 'ur_xz', 'ur_yx', 'ur_yy', 'ur_yz', 'ur_zx', 'ur_zy', 'ur_zz'
        ]
        
        for comp in expected_components:
            assert comp in conv_matrices, f"Missing component: {comp}"
            matrix = conv_matrices[comp]
            assert matrix.shape == (25, 25)  # 5x5 harmonics
            assert np.all(np.isfinite(matrix))
        
        # 验证非零的非对角分量
        er_xy = conv_matrices['er_xy']
        er_yx = conv_matrices['er_yx']
        
        # 非对角分量不应该全是零（因为有各向异性材料）
        assert not np.allclose(er_xy, 0.0), "εxy component should not be zero for anisotropic material"
        assert not np.allclose(er_yx, 0.0), "εyx component should not be zero for anisotropic material"
        
        # 验证对称性：εxy = εyx for symmetric tensor
        zero_harmonic_idx = 12  # Center of 5x5 harmonics (25//2)
        assert np.isclose(er_xy[zero_harmonic_idx, zero_harmonic_idx], 
                         er_yx[zero_harmonic_idx, zero_harmonic_idx]), "εxy ≠ εyx at zero harmonic"
        
        print("✅ 完整张量卷积矩阵计算正确")

    def test_convolution_matrix_interface_with_tensor_components(self):
        """测试卷积矩阵接口支持张量分量"""
        air = Material(er=1.0, ur=1.0)
        
        # 创建简单的各向异性材料
        epsilon_tensor = np.diag([12.0, 11.0, 10.0]).astype(complex)
        epsilon_tensor[0, 1] = epsilon_tensor[1, 0] = 0.8  # 添加 xy 耦合
        
        aniso_material = TensorMaterial(epsilon_tensor=epsilon_tensor)
        
        rect = Rectangle(center=(0.5, 0.5), width=0.4, height=0.4)
        layer = PatternedLayer(
            thickness=100e-9,
            lattice=square_lattice(500e-9),
            background_material=air,
            shapes=[(rect, aniso_material)]
        )
        
        harmonics_x = np.array([-1, 0, 1])
        harmonics_y = np.array([-1, 0, 1])
        
        # 测试不同张量分量的访问
        test_components = ['eps_xx', 'eps_xy', 'eps_yx', 'eps_yy', 'mu_xx']
        
        for comp in test_components:
            conv_matrix = layer.convolution_matrix(harmonics_x, harmonics_y, comp)
            assert conv_matrix.shape == (9, 9)  # 3x3 harmonics
            assert np.all(np.isfinite(conv_matrix))
            
        # 验证非对角分量不全为零
        conv_xy = layer.convolution_matrix(harmonics_x, harmonics_y, 'eps_xy')
        conv_yx = layer.convolution_matrix(harmonics_x, harmonics_y, 'eps_yx')
        
        # 至少某些元素不应该为零（因为有各向异性耦合）
        assert not np.allclose(conv_xy, 0.0), "εxy convolution matrix should not be zero"
        assert not np.allclose(conv_yx, 0.0), "εyx convolution matrix should not be zero"
        
        print("✅ 张量分量卷积矩阵接口正确")

    def test_rotated_tensor_material_coupling(self):
        """测试旋转张量材料的完整耦合"""
        air = Material(er=1.0, ur=1.0)
        
        # 创建对角张量
        original_tensor = np.diag([15.0, 10.0, 12.0]).astype(complex)
        material = TensorMaterial(epsilon_tensor=original_tensor)
        
        # 绕z轴旋转45度
        angle = np.pi / 4
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        rotated_material = material.rotated(rotation_matrix)
        
        circle = Circle(center=(0.5, 0.5), radius=0.25)
        layer = PatternedLayer(
            thickness=150e-9,
            lattice=square_lattice(1.0e-6),
            background_material=air,
            shapes=[(circle, rotated_material)]
        )
        
        # 栅格化并检查旋转后的张量
        epsilon_field, mu_field = layer.rasterize_full_tensor_field()
        
        center_y, center_x = 128, 128
        rotated_epsilon = epsilon_field[center_y, center_x]
        
        # 旋转后应该有非对角分量
        assert not np.isclose(rotated_epsilon[0, 1], 0.0), "Rotated tensor should have xy coupling"
        assert not np.isclose(rotated_epsilon[1, 0], 0.0), "Rotated tensor should have yx coupling"
        
        # 验证对称性
        assert np.isclose(rotated_epsilon[0, 1], rotated_epsilon[1, 0]), "Rotated tensor should be symmetric"
        
        # 验证对角分量的变化
        original_diag_sum = 15.0 + 10.0 + 12.0
        rotated_diag_sum = rotated_epsilon[0, 0].real + rotated_epsilon[1, 1].real + rotated_epsilon[2, 2].real
        assert np.isclose(original_diag_sum, rotated_diag_sum), "Trace should be preserved under rotation"
        
        print("✅ 旋转张量材料耦合正确")

    def test_mixed_tensor_and_scalar_materials(self):
        """测试张量材料和标量材料的混合"""
        air = Material(er=1.0, ur=1.0)
        si = Material(er=12.0, ur=1.0)
        
        # 张量材料
        epsilon_tensor = np.array([
            [8.0, 0.5, 0.0],
            [0.5, 9.0, 0.0],
            [0.0, 0.0, 10.0]
        ], dtype=complex)
        tensor_material = TensorMaterial(epsilon_tensor=epsilon_tensor)
        
        # 两个形状：一个用标量材料，一个用张量材料
        rect1 = Rectangle(center=(0.3, 0.3), width=0.3, height=0.3)
        rect2 = Rectangle(center=(0.7, 0.7), width=0.3, height=0.3)
        
        layer = PatternedLayer(
            thickness=200e-9,
            lattice=square_lattice(1.2e-6),
            background_material=air,
            shapes=[
                (rect1, si),              # 标量材料
                (rect2, tensor_material)  # 张量材料
            ]
        )
        
        # 栅格化
        epsilon_field, mu_field = layer.rasterize_full_tensor_field()
        
        # 检查不同区域的张量特性
        # 标量材料区域 (约在像素 77, 77)
        scalar_y, scalar_x = 77, 77
        scalar_epsilon = epsilon_field[scalar_y, scalar_x]
        
        # 标量材料应该是对角的
        assert np.isclose(scalar_epsilon[0, 0], 12.0)
        assert np.isclose(scalar_epsilon[0, 1], 0.0)  # 无 xy 耦合
        
        # 张量材料区域 (约在像素 179, 179)
        tensor_y, tensor_x = 179, 179
        tensor_epsilon = epsilon_field[tensor_y, tensor_x]
        
        # 张量材料应该有非对角分量
        assert np.isclose(tensor_epsilon[0, 0], 8.0)
        assert np.isclose(tensor_epsilon[0, 1], 0.5)  # 有 xy 耦合
        assert np.isclose(tensor_epsilon[1, 1], 9.0)
        
        print("✅ 混合张量和标量材料处理正确")


if __name__ == '__main__':
    test_class = TestFullTensorCoupling()
    
    print("🧪 测试完整张量耦合实现")
    print("=" * 50)
    
    test_methods = [
        'test_isotropic_material_tensor_rasterization',
        'test_anisotropic_material_tensor_rasterization', 
        'test_full_tensor_convolution_matrices',
        'test_convolution_matrix_interface_with_tensor_components',
        'test_rotated_tensor_material_coupling',
        'test_mixed_tensor_and_scalar_materials'
    ]
    
    for method_name in test_methods:
        print(f"\n🔬 执行 {method_name}")
        try:
            getattr(test_class, method_name)()
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("🎉 完整张量耦合测试完成!")
    print('='*50)
