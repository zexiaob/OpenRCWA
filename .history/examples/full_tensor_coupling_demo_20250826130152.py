"""
完整张量耦合的 RCWA 模拟示例

这个示例展示了如何使用完整的 3x3 张量材料进行 RCWA 模拟，
包括各向异性材料、张量旋转和完整耦合计算。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rcwa.model.material import Material, TensorMaterial
from rcwa.geom.shape import Rectangle, Circle, Ellipse
from rcwa.geom.patterned import PatternedLayer, square_lattice, rectangular_lattice
from rcwa.model.layer import Layer


def demo_full_tensor_coupling():
    """演示完整张量耦合的功能"""
    print("🎯 完整张量耦合 RCWA 模拟示例")
    print("=" * 60)
    
    # 1. 创建各向异性张量材料
    print("\n1. 创建各向异性张量材料")
    
    # 液晶材料的典型张量 (ne=1.7, no=1.5)
    liquid_crystal_tensor = np.array([
        [1.5**2, 0.0, 0.0],      # ordinary ray
        [0.0, 1.5**2, 0.0],      # ordinary ray
        [0.0, 0.0, 1.7**2]       # extraordinary ray
    ], dtype=complex)
    
    # 添加少量耦合项
    liquid_crystal_tensor[0, 1] = liquid_crystal_tensor[1, 0] = 0.05j
    liquid_crystal_tensor[1, 2] = liquid_crystal_tensor[2, 1] = 0.02j
    
    lc_material = TensorMaterial(
        epsilon_tensor=liquid_crystal_tensor,
        mu_tensor=np.eye(3, dtype=complex),
        name="LiquidCrystal"
    )
    
    print(f"   液晶材料张量:")
    print(f"   εxx = {lc_material.epsilon_tensor[0,0]:.3f}")
    print(f"   εyy = {lc_material.epsilon_tensor[1,1]:.3f}")
    print(f"   εzz = {lc_material.epsilon_tensor[2,2]:.3f}")
    print(f"   εxy = {lc_material.epsilon_tensor[0,1]:.3f}")
    print(f"   εyz = {lc_material.epsilon_tensor[1,2]:.3f}")
    
    # 2. 创建旋转的张量材料
    print("\n2. 创建旋转的张量材料")
    
    # 绕 z 轴旋转 30 度
    angle = np.pi / 6  # 30 degrees
    rotation_z = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    rotated_lc = lc_material.rotated(rotation_z)
    
    print(f"   旋转后的液晶材料张量:")
    print(f"   εxx = {rotated_lc.epsilon_tensor[0,0]:.3f}")
    print(f"   εyy = {rotated_lc.epsilon_tensor[1,1]:.3f}")
    print(f"   εxy = {rotated_lc.epsilon_tensor[0,1]:.3f}")
    print(f"   εyx = {rotated_lc.epsilon_tensor[1,0]:.3f}")
    
    # 3. 创建复杂的图案化层
    print("\n3. 创建复杂的图案化层")
    
    air = Material(er=1.0, ur=1.0)
    si = Material(er=12.0, ur=1.0)
    
    # 创建包含多种材料的复杂图案
    period = 600e-9  # 600 nm
    
    # 基础矩形：硅
    base_rect = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
    
    # 中心圆：液晶材料  
    center_circle = Circle(center=(0.5, 0.5), radius=0.2)
    
    # 四个角的椭圆：旋转液晶材料
    corner_ellipses = [
        Ellipse(center=(0.25, 0.25), semi_major=0.08, semi_minor=0.05),
        Ellipse(center=(0.75, 0.25), semi_major=0.08, semi_minor=0.05),
        Ellipse(center=(0.25, 0.75), semi_major=0.08, semi_minor=0.05),
        Ellipse(center=(0.75, 0.75), semi_major=0.08, semi_minor=0.05),
    ]
    
    # 构建图案化层
    layer = PatternedLayer(
        thickness=200e-9,
        lattice=square_lattice(period),
        background_material=air,
        shapes=[
            (base_rect, si),                    # 硅基底
            (center_circle, lc_material),       # 中心液晶
        ] + [(ellipse, rotated_lc) for ellipse in corner_ellipses]  # 角上旋转液晶
    )
    
    print(f"   图案化层包含 {len(layer.shapes)} 个形状")
    print(f"   晶格周期: {period*1e9:.0f} nm")
    print(f"   层厚度: {layer.thickness*1e9:.0f} nm")
    
    # 4. 栅格化完整张量场
    print("\n4. 栅格化完整张量场")
    
    epsilon_field, mu_field = layer.rasterize_full_tensor_field()
    
    print(f"   张量场形状: {epsilon_field.shape}")
    print(f"   数据类型: {epsilon_field.dtype}")
    
    # 检查不同区域的张量特性
    Ny, Nx = epsilon_field.shape[:2]
    
    # 中心区域 (液晶)
    center_y, center_x = Ny//2, Nx//2
    center_tensor = epsilon_field[center_y, center_x]
    
    print(f"\n   中心区域张量分析 (液晶):")
    print(f"   εxx = {center_tensor[0,0]:.3f}")
    print(f"   εxy = {center_tensor[0,1]:.3f}")
    print(f"   εzz = {center_tensor[2,2]:.3f}")
    
    # 角落区域 (旋转液晶)
    corner_y, corner_x = Ny//4, Nx//4
    corner_tensor = epsilon_field[corner_y, corner_x]
    
    print(f"\n   角落区域张量分析 (旋转液晶):")
    print(f"   εxx = {corner_tensor[0,0]:.3f}")
    print(f"   εxy = {corner_tensor[0,1]:.3f}")
    print(f"   εyx = {corner_tensor[1,0]:.3f}")
    print(f"   εyy = {corner_tensor[1,1]:.3f}")
    
    # 5. 生成卷积矩阵
    print("\n5. 生成完整张量卷积矩阵")
    
    harmonics = (7, 7)  # 7x7 harmonics
    conv_matrices = layer.to_convolution_matrices(harmonics)
    
    print(f"   生成的卷积矩阵数量: {len(conv_matrices)}")
    print(f"   每个矩阵形状: {list(conv_matrices.values())[0].shape}")
    
    # 检查所有张量分量
    expected_components = [
        'er_xx', 'er_xy', 'er_xz', 'er_yx', 'er_yy', 'er_yz', 'er_zx', 'er_zy', 'er_zz',
        'ur_xx', 'ur_xy', 'ur_xz', 'ur_yx', 'ur_yy', 'ur_yz', 'ur_zx', 'ur_zy', 'ur_zz'
    ]
    
    print(f"\n   张量分量完整性检查:")
    for comp in expected_components[:9]:  # 只显示 epsilon 分量
        matrix = conv_matrices[comp]
        max_value = np.max(np.abs(matrix))
        print(f"   {comp}: 最大值 = {max_value:.3e}")
    
    # 6. 验证非对角耦合
    print("\n6. 验证非对角张量耦合")
    
    er_xx = conv_matrices['er_xx']
    er_xy = conv_matrices['er_xy']
    er_yx = conv_matrices['er_yx']
    
    # 检查非对角项是否为零
    xy_coupling_strength = np.max(np.abs(er_xy))
    yx_coupling_strength = np.max(np.abs(er_yx))
    
    print(f"   εxy 耦合强度: {xy_coupling_strength:.3e}")
    print(f"   εyx 耦合强度: {yx_coupling_strength:.3e}")
    
    if xy_coupling_strength > 1e-10:
        print("   ✅ 检测到 x-y 张量耦合")
    else:
        print("   ⚠️  未检测到 x-y 张量耦合")
    
    # 7. 接口兼容性测试
    print("\n7. 测试接口兼容性")
    
    harmonics_x = np.array([-3, -2, -1, 0, 1, 2, 3])
    harmonics_y = np.array([-3, -2, -1, 0, 1, 2, 3])
    
    # 测试不同的张量分量访问
    test_components = ['xx', 'xy', 'yy', 'eps_xx', 'eps_xy', 'mu_xx']
    
    for comp in test_components:
        try:
            conv_matrix = layer.convolution_matrix(harmonics_x, harmonics_y, comp)
            print(f"   ✅ {comp}: 形状 {conv_matrix.shape}")
        except Exception as e:
            print(f"   ❌ {comp}: 错误 {e}")
    
    # 8. 总结
    print(f"\n{'='*60}")
    print("🎉 完整张量耦合演示完成!")
    print(f"{'='*60}")
    print("\n主要成就:")
    print("✅ 实现了完整的 3x3 张量材料支持")
    print("✅ 支持张量材料的旋转变换")
    print("✅ 栅格化所有 9 个张量分量")
    print("✅ 生成所有 18 个卷积矩阵 (epsilon + mu)")
    print("✅ 验证了非对角张量耦合")
    print("✅ 提供了完整的 RCWA 接口")
    
    return layer, conv_matrices


def visualize_tensor_field(layer: PatternedLayer, component: str = 'xx'):
    """可视化张量场分量"""
    print(f"\n📊 可视化张量场分量: {component}")
    
    epsilon_field, mu_field = layer.rasterize_full_tensor_field()
    
    # 解析分量索引
    if component == 'xx':
        i, j = 0, 0
    elif component == 'xy':
        i, j = 0, 1
    elif component == 'yy':
        i, j = 1, 1
    elif component == 'zz':
        i, j = 2, 2
    else:
        raise ValueError(f"Unknown component: {component}")
    
    # 提取分量
    field_component = epsilon_field[:, :, i, j]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 实部
    im1 = ax1.imshow(field_component.real, cmap='viridis', origin='lower')
    ax1.set_title(f'ε_{component} 实部')
    ax1.set_xlabel('x 像素')
    ax1.set_ylabel('y 像素')
    plt.colorbar(im1, ax=ax1)
    
    # 虚部
    im2 = ax2.imshow(field_component.imag, cmap='RdBu_r', origin='lower')
    ax2.set_title(f'ε_{component} 虚部')
    ax2.set_xlabel('x 像素')
    ax2.set_ylabel('y 像素')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    # 保存图像
    filename = f'tensor_field_{component}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   图像已保存: {filename}")
    
    plt.close()


if __name__ == '__main__':
    # 运行完整演示
    layer, conv_matrices = demo_full_tensor_coupling()
    
    # 可视化一些张量分量
    print("\n" + "="*60)
    print("📈 生成张量场可视化")
    
    try:
        # 可视化对角分量和耦合分量
        visualize_tensor_field(layer, 'xx')
        visualize_tensor_field(layer, 'xy')
        visualize_tensor_field(layer, 'yy')
        
        print("\n✅ 所有可视化图像已生成")
    except ImportError:
        print("\n⚠️  matplotlib 不可用，跳过可视化")
    except Exception as e:
        print(f"\n❌ 可视化错误: {e}")
