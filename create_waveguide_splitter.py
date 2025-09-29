#!/usr/bin/env python3
"""
Create a realistic waveguide coupler structure visualization.
"""

import subprocess

def create_waveguide_coupler_visualization():
    """Create a realistic silicon photonics waveguide coupler structure"""
    
    script_content = '''
import sys
import os
import numpy as np

os.chdir('/tmp')

try:
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    print("Creating waveguide coupler 3D visualization...")
    
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set background
    fig.patch.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False  
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    def create_box(ax, x0, x1, y0, y1, z0, z1, color, alpha=0.7, label="", edge_color='black', edge_width=0.5):
        vertices = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
        ])
        
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]]
        ]
        
        poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, 
                               edgecolor=edge_color, linewidth=edge_width)
        ax.add_collection3d(poly)
        
        if label:
            ax.text((x0+x1)/2, (y0+y1)/2, (z0+z1)/2, label, fontsize=9, 
                    ha='center', va='center', weight='bold')

    def create_tapered_waveguide(ax, x_start, x_end, y_center, z0, z1, 
                                width_start, width_end, color, alpha=0.8):
        """Create a tapered waveguide structure"""
        segments = 20
        dx = (x_end - x_start) / segments
        
        for i in range(segments):
            x0 = x_start + i * dx
            x1 = x_start + (i + 1) * dx
            
            # Linear taper
            w0 = width_start + (width_end - width_start) * i / segments
            w1 = width_start + (width_end - width_start) * (i + 1) / segments
            
            # Create trapezoid segment
            y0_0, y1_0 = y_center - w0/2, y_center + w0/2
            y0_1, y1_1 = y_center - w1/2, y_center + w1/2
            
            vertices = np.array([
                [x0, y0_0, z0], [x1, y0_1, z0], [x1, y1_1, z0], [x0, y1_0, z0],
                [x0, y0_0, z1], [x1, y0_1, z1], [x1, y1_1, z1], [x0, y1_0, z1]
            ])
            
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[0], vertices[3], vertices[7], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]]
            ]
            
            poly = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                   edgecolor='darkred', linewidth=0.2)
            ax.add_collection3d(poly)

    # Structure dimensions
    Lx, Ly = 12.0, 8.0  # 12x8 micron area
    
    # Layer 1: Air cladding (top)
    z0, z1 = 2.5, 4.0
    create_box(ax, -Lx/2, Lx/2, -Ly/2, Ly/2, z0, z1, 
               (0.8, 0.9, 1.0), alpha=0.15, label="Air Cladding", edge_color='lightblue', edge_width=0.3)
    
    # Layer 2: SiO2 upper cladding
    z0, z1 = 2.0, 2.5
    create_box(ax, -Lx/2, Lx/2, -Ly/2, Ly/2, z0, z1, 
               (1.0, 0.9, 0.6), alpha=0.4, label="SiO2\\nUpper Clad", edge_color='orange', edge_width=0.4)
    
    # Layer 3: Silicon device layer - with waveguides and coupler
    z0, z1 = 1.8, 2.0  # 220nm silicon layer
    
    # Background silicon (partial etch regions)
    create_box(ax, -Lx/2, Lx/2, -Ly/2, Ly/2, z0, z0+0.07, 
               (0.7, 0.4, 0.4), alpha=0.3, label="", edge_color='darkred', edge_width=0.2)
    
    # Input waveguide
    create_box(ax, -Lx/2, -2.0, -0.25, 0.25, z0, z1,
               (0.6, 0.2, 0.2), alpha=0.9, label="", edge_color='black', edge_width=0.3)
    
    # Tapered coupler section
    create_tapered_waveguide(ax, -2.0, 2.0, 0.0, z0, z1, 0.5, 2.0, (0.6, 0.2, 0.2))
    
    # Output waveguide 1 (top)
    create_box(ax, 2.0, Lx/2, 1.0, 1.5, z0, z1,
               (0.6, 0.2, 0.2), alpha=0.9, label="", edge_color='black', edge_width=0.3)
    
    # Output waveguide 2 (bottom)  
    create_box(ax, 2.0, Lx/2, -1.5, -1.0, z0, z1,
               (0.6, 0.2, 0.2), alpha=0.9, label="", edge_color='black', edge_width=0.3)
    
    # Add grating couplers at the ends
    grating_period = 0.25
    for i in range(-12, 0, 2):
        x_pos = -Lx/2 + i * grating_period
        create_box(ax, x_pos, x_pos + grating_period*0.6, -0.4, 0.4, z0, z1,
                   (0.8, 0.3, 0.3), alpha=0.8, label="", edge_color='darkred', edge_width=0.1)
    
    for i in range(0, 12, 2):
        x_pos = Lx/2 - 2.0 + i * grating_period
        # Top grating
        create_box(ax, x_pos, x_pos + grating_period*0.6, 0.8, 1.7, z0, z1,
                   (0.8, 0.3, 0.3), alpha=0.8, label="", edge_color='darkred', edge_width=0.1)
        # Bottom grating  
        create_box(ax, x_pos, x_pos + grating_period*0.6, -1.7, -0.8, z0, z1,
                   (0.8, 0.3, 0.3), alpha=0.8, label="", edge_color='darkred', edge_width=0.1)
    
    # Add text for silicon layer
    ax.text(0, Ly/2 + 0.5, (z0+z1)/2, "Silicon\\nDevice Layer", fontsize=12,
            ha='center', va='center', weight='bold', color='darkred')
    
    # Layer 4: SiO2 buried oxide (BOX)
    z0, z1 = 0.0, 1.8
    create_box(ax, -Lx/2, Lx/2, -Ly/2, Ly/2, z0, z1, 
               (1.0, 0.8, 0.4), alpha=0.5, label="SiO2 BOX\\n(2μm)", edge_color='orange', edge_width=0.4)
    
    # Layer 5: Silicon handle substrate
    z0, z1 = -1.0, 0.0
    create_box(ax, -Lx/2, Lx/2, -Ly/2, Ly/2, z0, z1, 
               (0.5, 0.3, 0.3), alpha=0.6, label="Si Handle\\nSubstrate", edge_color='darkred', edge_width=0.4)
    
    # Add directional arrows to show light path
    from mpl_toolkits.mplot3d import proj3d
    
    # Input arrow
    ax.quiver(-Lx/2-1, 0, 1.9, 1.5, 0, 0, 
              color='red', arrow_length_ratio=0.3, linewidth=3)
    ax.text(-Lx/2-1.5, 0, 2.3, "Input\\nLight", fontsize=10, ha='center', 
            weight='bold', color='red')
    
    # Output arrows
    ax.quiver(Lx/2+0.5, 1.25, 1.9, 1.0, 0, 0, 
              color='blue', arrow_length_ratio=0.3, linewidth=2)
    ax.text(Lx/2+1.2, 1.25, 2.3, "Output 1", fontsize=9, ha='center', 
            weight='bold', color='blue')
    
    ax.quiver(Lx/2+0.5, -1.25, 1.9, 1.0, 0, 0, 
              color='green', arrow_length_ratio=0.3, linewidth=2)
    ax.text(Lx/2+1.2, -1.25, 2.3, "Output 2", fontsize=9, ha='center', 
            weight='bold', color='green')
    
    # Set axis properties
    ax.set_xlim(-Lx/2*1.3, Lx/2*1.3)
    ax.set_ylim(-Ly/2*1.2, Ly/2*1.2)
    ax.set_zlim(-1.2, 4.2)
    
    ax.set_xlabel('X (μm)', fontsize=14)
    ax.set_ylabel('Y (μm)', fontsize=14)
    ax.set_zlabel('Z (μm)', fontsize=14)
    
    ax.view_init(elev=15, azim=-45)
    
    # Add comprehensive title
    plt.title('Silicon Photonics 1×2 Waveguide Splitter\\n'
              'with Grating Couplers on SOI Platform', fontsize=20, pad=25)
    
    # Add detailed legend
    legend_text = (
        "Device Components:\\n"
        "• Input grating coupler\\n"
        "• Single-mode waveguide\\n"  
        "• Adiabatic Y-splitter\\n"
        "• Output grating couplers\\n"
        "• SOI platform (220nm Si)\\n"
        "• 2μm buried oxide\\n"
        "• Si handle wafer"
    )
    ax.text2D(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=11,
              verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", 
              facecolor="lightyellow", alpha=0.9, edgecolor='gray'))
    
    # Add specifications
    specs_text = (
        "Specifications:\\n"
        "• Wavelength: 1550nm\\n"
        "• Si thickness: 220nm\\n"
        "• BOX: 2μm SiO2\\n"
        "• Waveguide: 500nm wide\\n"
        "• Splitting ratio: 50:50\\n"
        "• Grating period: 630nm"
    )
    ax.text2D(0.02, 0.65, specs_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", 
              facecolor="lightcyan", alpha=0.9, edgecolor='gray'))
    
    # Save the plot
    plt.tight_layout()
    save_path = '/Users/jinzeyuan/Documents/Reserch_Project_2/OpenRCWA/waveguide_splitter_3d.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Waveguide splitter 3D visualization saved to: {save_path}")
    
except Exception as e:
    print(f"Error creating 3D plot: {e}")
    import traceback
    traceback.print_exc()
'''
    
    # Write and run the script
    script_path = '/tmp/create_waveguide_3d.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    try:
        result = subprocess.run([
            '/Users/jinzeyuan/Documents/Reserch_Project_2/OpenRCWA/.venv/bin/python',
            script_path
        ], capture_output=True, text=True, cwd='/tmp')
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
            
    except Exception as e:
        print(f"Failed to run visualization script: {e}")
        return False

if __name__ == "__main__":
    print("=== Creating Silicon Photonics Waveguide Splitter ===")
    success = create_waveguide_coupler_visualization()
    
    if success:
        print("\\n✓ Waveguide splitter 3D visualization completed!")
        print("  Professional photonic device features:")
        print("  - Input/output grating couplers")
        print("  - Adiabatic Y-splitter")
        print("  - SOI platform structure")
        print("  - Realistic dimensions and materials")
        print("  - Light path visualization")
        print("  - Technical specifications")
    else:
        print("✗ Failed to create waveguide visualization")
