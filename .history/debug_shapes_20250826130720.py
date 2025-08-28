"""
调试形状覆盖问题
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rcwa.model.material import Material, TensorMaterial
from rcwa.geom.shape import Rectangle, Circle, Ellipse
from rcwa.geom.patterned import PatternedLayer, square_lattice


def debug_shape_coverage():
    """调试形状覆盖问题"""
    print("🐛 调试形状覆盖问题")
    print("=" * 40)
    
    # 重建材料
    air = Material(er=1.0, ur=1.0)
    si = Material(er=12.0, ur=1.0)
    
    liquid_crystal_tensor = np.array([
        [1.5**2, 0.0, 0.0],      
        [0.0, 1.5**2, 0.0],      
        [0.0, 0.0, 1.7**2]       
    ], dtype=complex)
    liquid_crystal_tensor[0, 1] = liquid_crystal_tensor[1, 0] = 0.05j
    liquid_crystal_tensor[1, 2] = liquid_crystal_tensor[2, 1] = 0.02j
    
    lc_material = TensorMaterial(epsilon_tensor=liquid_crystal_tensor)
    
    angle = np.pi / 6
    rotation_z = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    rotated_lc = lc_material.rotated(rotation_z)
    
    # 重建形状
    base_rect = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
    center_circle = Circle(center=(0.5, 0.5), radius=0.2)
    
    corner_ellipses = [
        Ellipse(center=(0.25, 0.25), semi_major=0.08, semi_minor=0.05),
        Ellipse(center=(0.75, 0.25), semi_major=0.08, semi_minor=0.05), 
        Ellipse(center=(0.25, 0.75), semi_major=0.08, semi_minor=0.05),
        Ellipse(center=(0.75, 0.75), semi_major=0.08, semi_minor=0.05),
    ]
    
    print("1. 检查形状参数:")
    print(f"   基础矩形: center={base_rect.center}, width={base_rect.width}, height={base_rect.height}")
    print(f"   中心圆: center={center_circle.center}, radius={center_circle.radius}")
    for i, ellipse in enumerate(corner_ellipses):
        print(f"   椭圆{i+1}: center={ellipse.center}, semi_major={ellipse.semi_major}, semi_minor={ellipse.semi_minor}")
    
    # 测试坐标网格
    Nx, Ny = 256, 256
    u = np.linspace(0, 1, Nx, endpoint=False)
    v = np.linspace(0, 1, Ny, endpoint=False)
    U, V = np.meshgrid(u, v)
    
    print(f"\n2. 坐标网格:")
    print(f"   网格大小: {Nx} x {Ny}")
    print(f"   U 范围: [{U.min():.3f}, {U.max():.3f}]")
    print(f"   V 范围: [{V.min():.3f}, {V.max():.3f}]")
    
    # 测试每个形状的覆盖
    shapes = [
        ("背景", None, air),
        ("基础矩形", base_rect, si),
        ("中心圆", center_circle, lc_material),
        ("椭圆1", corner_ellipses[0], rotated_lc),
        ("椭圆2", corner_ellipses[1], rotated_lc),
        ("椭圆3", corner_ellipses[2], rotated_lc),
        ("椭圆4", corner_ellipses[3], rotated_lc),
    ]
    
    print(f"\n3. 形状覆盖分析:")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    total_mask = np.zeros((Ny, Nx), dtype=bool)
    
    for i, (name, shape, material) in enumerate(shapes):
        if shape is None:
            mask = np.ones((Ny, Nx), dtype=bool)  # 背景覆盖全部
            coverage = 1.0
        else:
            mask = shape.contains(U, V)
            coverage = np.sum(mask) / (Nx * Ny)
        
        print(f"   {name}: 覆盖率 {coverage:.1%}, 像素数 {np.sum(mask)}")
        
        if i < len(axes):
            axes[i].imshow(mask.astype(int), cmap='viridis', origin='lower')
            axes[i].set_title(f'{name}\n覆盖率: {coverage:.1%}')
            axes[i].set_aspect('equal')
        
        total_mask |= mask
    
    # 隐藏多余的子图
    for i in range(len(shapes), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('shape_coverage_debug.png', dpi=150, bbox_inches='tight')
    print(f"   形状覆盖调试图已保存: shape_coverage_debug.png")
    plt.close()
    
    # 4. 模拟栅格化过程
    print(f"\n4. 模拟栅格化过程:")
    
    # 模拟材料ID分配
    material_id = np.zeros((Ny, Nx), dtype=int)  # 0=空气
    
    # 按顺序应用形状（模拟 PatternedLayer 的逻辑）
    shape_materials = [
        (base_rect, 1),  # 硅 = 1
        (center_circle, 2),  # 液晶 = 2  
        (corner_ellipses[0], 3),  # 旋转液晶 = 3
        (corner_ellipses[1], 3),
        (corner_ellipses[2], 3),
        (corner_ellipses[3], 3),
    ]
    
    for shape, mat_id in shape_materials:
        mask = shape.contains(U, V)
        material_id[mask] = mat_id
        covered_pixels = np.sum(mask)
        print(f"   应用形状 {mat_id}: 覆盖 {covered_pixels} 像素")
    
    # 统计最终材料分布
    unique_ids, counts = np.unique(material_id, return_counts=True)
    print(f"\n   最终材料分布:")
    material_names = ["空气", "硅", "液晶", "旋转液晶"]
    for mat_id, count in zip(unique_ids, counts):
        name = material_names[mat_id] if mat_id < len(material_names) else f"材料{mat_id}"
        percentage = count / (Nx * Ny) * 100
        print(f"   {name}: {count} 像素 ({percentage:.1f}%)")
    
    # 可视化材料分布
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(material_id, cmap='tab10', origin='lower')
    ax.set_title('材料分布 (按ID)')
    ax.set_xlabel('x 像素')
    ax.set_ylabel('y 像素')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, ticks=unique_ids)
    cbar.ax.set_yticklabels([material_names[i] for i in unique_ids])
    
    plt.savefig('material_distribution_debug.png', dpi=150, bbox_inches='tight')
    print(f"   材料分布调试图已保存: material_distribution_debug.png")
    plt.close()
    
    return material_id


if __name__ == '__main__':
    material_id = debug_shape_coverage()
    
    print(f"\n{'='*40}")
    print("🔍 问题诊断:")
    
    # 检查椭圆是否过大
    if np.sum(material_id == 3) > 0.5 * material_id.size:
        print("❌ 椭圆覆盖了过多区域，可能椭圆参数过大")
    else:
        print("✅ 椭圆覆盖范围看起来正常")
    
    # 检查是否有其他材料
    unique_materials = np.unique(material_id)
    if len(unique_materials) > 1:
        print("✅ 检测到多种材料")
        for mat in unique_materials:
            print(f"   材料 {mat}: 存在")
    else:
        print("❌ 只检测到一种材料，可能存在覆盖问题")
    
    print("="*40)
