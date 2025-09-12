#!/usr/bin/env python3
"""
空气-玻璃界面透射率随入射角变化的RCWA模拟 (使用Sweep功能)
Air-Glass Interface Transmission vs Incidence Angle RCWA Simulation (Using Sweep)

这个脚本使用OpenRCWA的Sweep功能模拟光从空气入射到玻璃界面的透射特性，
研究透射率随入射角的变化，验证菲涅尔反射定律。
"""

import numpy as np
import matplotlib.pyplot as plt
from rcwa import Source, Layer, Stack, Solver, Material
from rcwa.solve import Sweep
import sys
import os
import csv

def create_air_glass_simulation():
    """创建空气-玻璃界面的几何结构 (空气->玻璃)"""
    
    # 定义材料 - 使用Material类
    air_material = Material(er=1.0, ur=1.0)      # 空气 (n=1.0)
    glass_material = Material(er=2.25, ur=1.0)   # 玻璃 (n=1.5, n²=2.25)
    
    # 创建层
    air_layer = Layer(material=air_material, thickness=0)     # 半无限空气层
    glass_layer = Layer(material=glass_material, thickness=0) # 半无限玻璃层
    
    # 创建LayerStack (空气作为入射层，玻璃作为透射层)
    stack = Stack(superstrate=air_material, substrate=glass_material)
    
    wavelength = 550e-9  # 550nm (绿光)
    return stack, wavelength

def create_glass_air_simulation():
    """创建玻璃-空气界面的几何结构 (玻璃->空气, 反向)"""
    
    # 定义材料 - 使用Material类
    air_material = Material(er=1.0, ur=1.0)      # 空气 (n=1.0)
    glass_material = Material(er=2.25, ur=1.0)   # 玻璃 (n=1.5, n²=2.25)
    
    # 创建反向堆栈 (玻璃作为入射层，空气作为透射层)
    stack = Stack(superstrate=glass_material, substrate=air_material)
    
    wavelength = 550e-9  # 550nm (绿光)
    return stack, wavelength

def simulate_transmission_vs_angle():
    """使用Sweep计算透射率随入射角的变化"""
    
    print("创建空气-玻璃界面模拟...")
    stack, wavelength = create_air_glass_simulation()
    
    # 入射角范围 (弧度)
    angles_deg = np.linspace(0, 80, 41)  # 0度到80度，41个点
    angles_rad = np.deg2rad(angles_deg)
    
    print(f"开始计算 {len(angles_deg)} 个入射角的透射率...")
    print("使用Sweep功能进行参数扫描...")
    
    # 定义扫描参数
    params_s = {
        'theta': angles_rad.tolist(),  # 入射角扫描
        'pTEM': [[1, 0]]  # S偏振 (TE)
    }
    
    params_p = {
        'theta': angles_rad.tolist(),  # 入射角扫描  
        'pTEM': [[0, 1]]  # P偏振 (TM)
    }
    
    # 创建基础光源
    base_source = Source(wavelength=wavelength, theta=0, phi=0, pTEM=[1, 0])
    
    # 运行S偏振扫描
    print("运行S偏振(TE)扫描...")
    sweep_s = Sweep(params_s, backend='serial')
    out_s = sweep_s.run(stack, base_source, n_harmonics=(1, 1))
    
    # 运行P偏振扫描
    print("运行P偏振(TM)扫描...")
    sweep_p = Sweep(params_p, backend='serial')
    out_p = sweep_p.run(stack, base_source, n_harmonics=(1, 1))
    
    # 提取结果
    results_s = out_s['results']
    results_p = out_p['results']
    
    transmission_s = [r.TTot for r in results_s]
    reflection_s = [r.RTot for r in results_s]
    transmission_p = [r.TTot for r in results_p]
    reflection_p = [r.RTot for r in results_p]
    
    # 打印部分结果
    key_indices = [0, 15, 22, 30]  # 对应0°, 30°, 45°, 60°的索引
    for idx in key_indices:
        if idx < len(angles_deg):
            angle_deg = angles_deg[idx]
            print(f"  角度 {angle_deg:.1f}°: T_s={transmission_s[idx]:.3f}, T_p={transmission_p[idx]:.3f}, R_s={reflection_s[idx]:.3f}, R_p={reflection_p[idx]:.3f}")
    
    return angles_deg, transmission_s, transmission_p, reflection_s, reflection_p

def simulate_glass_air_transmission():
    """计算玻璃到空气的透射率随入射角的变化 (反向, 会有全内反射)"""
    
    print("创建玻璃-空气界面模拟 (反向)...")
    stack, wavelength = create_glass_air_simulation()
    
    # 计算临界角 (全内反射角度)
    n_glass = 1.5  # 玻璃折射率
    n_air = 1.0    # 空气折射率
    critical_angle_rad = np.arcsin(n_air / n_glass)
    critical_angle_deg = np.rad2deg(critical_angle_rad)
    print(f"理论临界角: {critical_angle_deg:.1f}°")
    
    # 入射角范围 (弧度) - 包含超过临界角的部分
    angles_deg = np.linspace(0, 80, 41)  # 0度到80度，41个点
    angles_rad = np.deg2rad(angles_deg)
    
    print(f"开始计算 {len(angles_deg)} 个入射角的透射率...")
    print("使用Sweep功能进行参数扫描...")
    
    # 定义扫描参数
    params_s = {
        'theta': angles_rad.tolist(),  # 入射角扫描
        'pTEM': [[1, 0]]  # S偏振 (TE)
    }
    
    params_p = {
        'theta': angles_rad.tolist(),  # 入射角扫描  
        'pTEM': [[0, 1]]  # P偏振 (TM)
    }
    
    # 创建基础光源
    base_source = Source(wavelength=wavelength, theta=0, phi=0, pTEM=[1, 0])
    
    # 运行S偏振扫描
    print("运行S偏振(TE)扫描...")
    sweep_s = Sweep(params_s, backend='serial')
    out_s = sweep_s.run(stack, base_source, n_harmonics=(1, 1))
    
    # 运行P偏振扫描
    print("运行P偏振(TM)扫描...")
    sweep_p = Sweep(params_p, backend='serial')
    out_p = sweep_p.run(stack, base_source, n_harmonics=(1, 1))
    
    # 提取结果
    results_s = out_s['results']
    results_p = out_p['results']
    
    transmission_s = [r.TTot for r in results_s]
    reflection_s = [r.RTot for r in results_s]
    transmission_p = [r.TTot for r in results_p]
    reflection_p = [r.RTot for r in results_p]
    
    # 打印部分结果，特别关注临界角附近
    key_indices = [0, 10, 15, 20, 25, 30, 35]  # 更多角度点
    for idx in key_indices:
        if idx < len(angles_deg):
            angle_deg = angles_deg[idx]
            is_beyond_critical = angle_deg > critical_angle_deg
            marker = " (全内反射)" if is_beyond_critical else ""
            print(f"  角度 {angle_deg:.1f}°: T_s={transmission_s[idx]:.3f}, T_p={transmission_p[idx]:.3f}, R_s={reflection_s[idx]:.3f}, R_p={reflection_p[idx]:.3f}{marker}")
    
    return angles_deg, transmission_s, transmission_p, reflection_s, reflection_p, critical_angle_deg

def calculate_fresnel_theoretical(angles_deg, n1=1.0, n2=1.5):
    """计算菲涅尔公式的理论值用于对比"""
    
    angles_rad = np.deg2rad(angles_deg)
    
    # 计算透射角 (斯涅尔定律)
    sin_theta_t = (n1/n2) * np.sin(angles_rad)
    
    # 避免全反射情况
    valid_mask = sin_theta_t <= 1.0
    theta_t = np.zeros_like(angles_rad)
    theta_t[valid_mask] = np.arcsin(sin_theta_t[valid_mask])
    
    # 菲涅尔公式
    cos_theta_i = np.cos(angles_rad)
    cos_theta_t = np.cos(theta_t)
    
    # S偏振反射系数
    r_s = np.zeros_like(angles_rad, dtype=complex)
    r_s[valid_mask] = ((n1*cos_theta_i - n2*cos_theta_t) / 
                       (n1*cos_theta_i + n2*cos_theta_t))[valid_mask]
    
    # P偏振反射系数
    r_p = np.zeros_like(angles_rad, dtype=complex)
    r_p[valid_mask] = ((n2*cos_theta_i - n1*cos_theta_t) / 
                       (n2*cos_theta_i + n1*cos_theta_t))[valid_mask]
    
    # 反射率和透射率
    R_s_theory = np.abs(r_s)**2
    R_p_theory = np.abs(r_p)**2
    T_s_theory = 1 - R_s_theory
    T_p_theory = 1 - R_p_theory
    
    # 全反射区域
    R_s_theory[~valid_mask] = 1.0
    R_p_theory[~valid_mask] = 1.0
    T_s_theory[~valid_mask] = 0.0
    T_p_theory[~valid_mask] = 0.0
    
    return T_s_theory, T_p_theory, R_s_theory, R_p_theory

def plot_results(angles_deg, transmission_s, transmission_p, 
                reflection_s, reflection_p):
    """绘制结果"""
    
    # 计算理论值
    T_s_theory, T_p_theory, R_s_theory, R_p_theory = calculate_fresnel_theoretical(angles_deg)
    
    print("\n绘制结果图表...")
    
    # 保存数据到文件
    import csv
    output_file = 'air_glass_transmission_data.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['角度(度)', 'T_s_RCWA', 'T_s_理论', 'T_p_RCWA', 'T_p_理论', 
                        'R_s_RCWA', 'R_s_理论', 'R_p_RCWA', 'R_p_理论'])
        for i, angle in enumerate(angles_deg):
            writer.writerow([
                angle, transmission_s[i], T_s_theory[i], 
                transmission_p[i], T_p_theory[i],
                reflection_s[i], R_s_theory[i], 
                reflection_p[i], R_p_theory[i]
            ])
    
    print(f"数据已保存至: {output_file}")
    
    # 尝试创建图表（如果matplotlib工作的话）
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 创建2x2的子图
        fig = plt.figure(figsize=(12, 10))
        
        # 子图1: S偏振透射率
        plt.subplot(2, 2, 1)
        plt.plot(angles_deg, transmission_s, 'bo-', label='RCWA模拟', markersize=4)
        plt.plot(angles_deg, T_s_theory, 'r--', label='菲涅尔理论', linewidth=2)
        plt.xlabel('入射角 (度)')
        plt.ylabel('透射率')
        plt.title('S偏振透射率')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)
        
        # 子图2: P偏振透射率
        plt.subplot(2, 2, 2)
        plt.plot(angles_deg, transmission_p, 'bo-', label='RCWA模拟', markersize=4)
        plt.plot(angles_deg, T_p_theory, 'r--', label='菲涅尔理论', linewidth=2)
        plt.xlabel('入射角 (度)')
        plt.ylabel('透射率')
        plt.title('P偏振透射率')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)
        
        # 子图3: S偏振反射率
        plt.subplot(2, 2, 3)
        plt.plot(angles_deg, reflection_s, 'go-', label='RCWA模拟', markersize=4)
        plt.plot(angles_deg, R_s_theory, 'r--', label='菲涅尔理论', linewidth=2)
        plt.xlabel('入射角 (度)')
        plt.ylabel('反射率')
        plt.title('S偏振反射率')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)
        
        # 子图4: P偏振反射率
        plt.subplot(2, 2, 4)
        plt.plot(angles_deg, reflection_p, 'go-', label='RCWA模拟', markersize=4)
        plt.plot(angles_deg, R_p_theory, 'r--', label='菲涅尔理论', linewidth=2)
        plt.xlabel('入射角 (度)')
        plt.ylabel('反射率')
        plt.title('P偏振反射率')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)
        
        plt.tight_layout()
        
        # 保存图片
        output_image = 'air_glass_transmission_analysis.png'
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"结果图片已保存: {output_image}")
        
        # 尝试显示（可能失败）
        try:
            plt.show()
        except:
            print("图表显示失败，但已保存到文件")
        
        return fig
        
    except Exception as e:
        print(f"绘图过程中出现错误: {e}")
        print("数据计算正确，只是绘图失败，请检查matplotlib版本")
        return None

def print_summary_table(angles_deg, transmission_s, transmission_p, 
                       reflection_s, reflection_p):
    """打印摘要表格"""
    
    T_s_theory, T_p_theory, R_s_theory, R_p_theory = calculate_fresnel_theoretical(angles_deg)
    
    print("\n" + "="*80)
    print("空气-玻璃界面透射反射特性摘要")
    print("="*80)
    print(f"{'角度':<6} {'T_s(RCWA)':<10} {'T_s(理论)':<10} {'T_p(RCWA)':<10} {'T_p(理论)':<10} {'误差_s':<8} {'误差_p':<8}")
    print("-"*80)
    
    key_angles = [0, 15, 30, 45, 60, 75]
    for angle in key_angles:
        if angle in angles_deg:
            idx = list(angles_deg).index(angle)
            t_s_rcwa = transmission_s[idx]
            t_p_rcwa = transmission_p[idx] 
            t_s_theory = T_s_theory[idx]
            t_p_theory = T_p_theory[idx]
            error_s = abs(t_s_rcwa - t_s_theory)
            error_p = abs(t_p_rcwa - t_p_theory)
            
            print(f"{angle:<6.0f} {t_s_rcwa:<10.3f} {t_s_theory:<10.3f} {t_p_rcwa:<10.3f} {t_p_theory:<10.3f} {error_s:<8.3f} {error_p:<8.3f}")
    
    print("-"*80)
    
    # 计算平均误差
    errors_s = [abs(t_rcwa - t_theory) for t_rcwa, t_theory in zip(transmission_s, T_s_theory)]
    errors_p = [abs(t_rcwa - t_theory) for t_rcwa, t_theory in zip(transmission_p, T_p_theory)]
    avg_error_s = np.mean(errors_s)
    avg_error_p = np.mean(errors_p)
    
    print(f"平均误差: S偏振 {avg_error_s:.4f}, P偏振 {avg_error_p:.4f}")
    print("="*80)

def print_glass_air_summary(angles_deg, transmission_s, transmission_p, 
                           reflection_s, reflection_p, critical_angle):
    """打印玻璃到空气的摘要表格 (包含全内反射信息)"""
    
    print("\n" + "="*80)
    print("玻璃-空气界面透射反射特性摘要 (反向, 全内反射)")
    print("="*80)
    print(f"理论临界角: {critical_angle:.1f}°")
    print("-"*80)
    print(f"{'角度':<6} {'T_s(RCWA)':<10} {'T_p(RCWA)':<10} {'R_s(RCWA)':<10} {'R_p(RCWA)':<10} {'状态':<15}")
    print("-"*80)
    
    key_angles = [0, 15, 30, 35, 40, 45, 50, 60, 70]
    for angle in key_angles:
        # 找到最接近的角度
        idx = np.argmin(np.abs(np.array(angles_deg) - angle))
        actual_angle = angles_deg[idx]
        
        t_s_rcwa = transmission_s[idx]
        t_p_rcwa = transmission_p[idx] 
        r_s_rcwa = reflection_s[idx]
        r_p_rcwa = reflection_p[idx]
        
        # 判断是否全内反射
        status = "全内反射" if actual_angle > critical_angle else "正常透射"
        
        print(f"{actual_angle:<6.1f} {t_s_rcwa:<10.3f} {t_p_rcwa:<10.3f} {r_s_rcwa:<10.3f} {r_p_rcwa:<10.3f} {status:<15}")
    
    print("-"*80)
    print(f"观察: 超过临界角 {critical_angle:.1f}° 后发生全内反射，透射率趋于0")
    print("="*80)

def save_to_csv(angles_deg, transmission_s, transmission_p, reflection_s, reflection_p, filename='transmission_data.csv'):
    """保存结果到CSV文件"""
    
    # 计算理论值
    T_s_theory, T_p_theory, R_s_theory, R_p_theory = calculate_fresnel_theoretical(angles_deg)
    
    # 写入CSV文件
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入标题行
        writer.writerow([
            '角度(度)', 'T_s_RCWA', 'T_s_理论', 'T_p_RCWA', 'T_p_理论',
            'R_s_RCWA', 'R_s_理论', 'R_p_RCWA', 'R_p_理论'
        ])
        
        # 写入数据行
        for i, angle in enumerate(angles_deg):
            writer.writerow([
                angle,
                transmission_s[i], T_s_theory[i], transmission_p[i], T_p_theory[i],
                reflection_s[i], R_s_theory[i], reflection_p[i], R_p_theory[i]
            ])
    
    print(f"数据已保存至: {filename}")

def main():
    """主函数"""
    
    print("开始界面透射率模拟...")
    print("="*50)
    
    # 询问模拟方向
    print("选择模拟类型:")
    print("1. 空气 -> 玻璃 (正向)")
    print("2. 玻璃 -> 空气 (反向, 全内反射)")
    print("3. 两种都运行")
    
    choice = 3
    if not choice:
        choice = "2"
    
    try:
        if choice in ["1", "3"]:
            print("\n" + "="*50)
            print("运行正向模拟: 空气 -> 玻璃")
            print("="*50)
            
            # 运行正向模拟
            angles_deg, transmission_s, transmission_p, reflection_s, reflection_p = simulate_transmission_vs_angle()
            
            # 保存数据
            save_to_csv(angles_deg, transmission_s, transmission_p, reflection_s, reflection_p, 
                       filename='air_glass_transmission_data.csv')
            
            # 打印摘要表格
            print_summary_table(angles_deg, transmission_s, transmission_p, reflection_s, reflection_p)
            
            # 绘制结果
            plot_results(angles_deg, transmission_s, transmission_p, reflection_s, reflection_p)
            
        if choice in ["2", "3"]:
            print("\n" + "="*50)
            print("运行反向模拟: 玻璃 -> 空气")
            print("="*50)
            
            # 运行反向模拟
            angles_deg_rev, transmission_s_rev, transmission_p_rev, reflection_s_rev, reflection_p_rev, critical_angle = simulate_glass_air_transmission()
            
            # 保存数据
            save_to_csv(angles_deg_rev, transmission_s_rev, transmission_p_rev, reflection_s_rev, reflection_p_rev, 
                       filename='glass_air_transmission_data.csv')
            
            # 特殊的反向摘要表格 (考虑全内反射)
            print_glass_air_summary(angles_deg_rev, transmission_s_rev, transmission_p_rev, 
                                   reflection_s_rev, reflection_p_rev, critical_angle)
        
        print("\n模拟完成!")
        print("主要观察:")
        if choice in ["1", "3"]:
            print("正向 (空气->玻璃):")
            print("  1. 垂直入射时透射率最高")
            print("  2. P偏振在布儒斯特角附近反射率最小")
        if choice in ["2", "3"]:
            print("反向 (玻璃->空气):")
            print("  1. 超过临界角时发生全内反射")
            print("  2. 临界角约为41.8°")
            print("  3. 全内反射时透射率为0，反射率为1")
        
    except Exception as e:
        print(f"模拟过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
