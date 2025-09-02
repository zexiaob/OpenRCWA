#!/usr/bin/env python3
"""
Create a simple structured 3D visualization with patterned layers.
"""

import os, sys
os.chdir('/tmp')
sys.path.insert(0, '/Users/jinzeyuan/Documents/Reserch_Project_2/OpenRCWA')

def create_simple_structured_visualization():
    """Create a simple structured stack with hand-coded geometry"""
    
    import subprocess
    
    # Create the visualization script
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
    
    print("Creating structured 3D visualization...")
    
    fig = plt.figure(figsize=(16, 12))
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
    
    def create_box(ax, x0, x1, y0, y1, z0, z1, color, alpha=0.7, label="", edge_color='black'):
        """Create a box/rectangular prism"""
        vertices = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # bottom
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]   # top
        ])
        
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
        ]
        
        poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, 
                               edgecolor=edge_color, linewidth=0.5)
        ax.add_collection3d(poly)
        
        # Add label at center
        if label:
            ax.text((x0+x1)/2, (y0+y1)/2, (z0+z1)/2, label, fontsize=10, 
                    ha='center', va='center')

    def create_cylinder(ax, x_center, y_center, radius, z0, z1, color, alpha=0.7, label=""):
        """Create a cylindrical structure"""
        theta = np.linspace(0, 2*np.pi, 24)
        x_circle = x_center + radius * np.cos(theta)
        y_circle = y_center + radius * np.sin(theta)
        
        # Bottom and top circles
        bottom = [(x_circle[i], y_circle[i], z0) for i in range(len(theta))]
        top = [(x_circle[i], y_circle[i], z1) for i in range(len(theta))]
        
        faces = [bottom, top]
        
        # Side faces
        for i in range(len(theta)):
            j = (i + 1) % len(theta)
            faces.append([bottom[i], bottom[j], top[j], top[i]])
        
        poly = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                               edgecolor='black', linewidth=0.3)
        ax.add_collection3d(poly)
        
        if label:
            ax.text(x_center, y_center, (z0+z1)/2, label, fontsize=8,
                    ha='center', va='center')
    
    # Define the structure dimensions
    Lx, Ly = 3.0, 3.0  # 3x3 micron cell
    
    # Layer 1: Air superstrate
    z0, z1 = 1.0, 1.5
    create_box(ax, -Lx/2, Lx/2, -Ly/2, Ly/2, z0, z1, 
               (0.7, 0.9, 1.0), alpha=0.2, label="Air\\n(Superstrate)", edge_color='blue')
    
    # Layer 2: Photonic crystal layer - Silicon with air holes
    z0, z1 = 0.5, 1.0
    # Background silicon
    create_box(ax, -Lx/2, Lx/2, -Ly/2, Ly/2, z0, z1, 
               (0.6, 0.3, 0.3), alpha=0.6, label="", edge_color='darkred')
    
    # Create air holes in a 2D array
    hole_radius = 0.2
    lattice_a = 0.6
    for i in range(-2, 3):  # 5x5 array
        for j in range(-2, 3):
            x_pos = i * lattice_a
            y_pos = j * lattice_a
            if abs(x_pos) < Lx/2 - hole_radius and abs(y_pos) < Ly/2 - hole_radius:
                create_cylinder(ax, x_pos, y_pos, hole_radius, z0, z1,
                               (0.8, 0.9, 1.0), alpha=0.4, label="")
    
    # Add text for this layer
    ax.text(0, Ly/2 + 0.3, (z0+z1)/2, "Silicon PhC\\n(Air holes)", fontsize=10,
            ha='center', va='center')
    
    # Layer 3: SiO2 layer
    z0, z1 = 0.0, 0.5
    create_box(ax, -Lx/2, Lx/2, -Ly/2, Ly/2, z0, z1, 
               (1.0, 0.8, 0.4), alpha=0.7, label="SiO2\\n(BOX)", edge_color='orange')
    
    # Layer 4: Grating layer - alternating silicon lines
    z0, z1 = -0.3, 0.0
    # Background air
    create_box(ax, -Lx/2, Lx/2, -Ly/2, Ly/2, z0, z1, 
               (0.9, 0.95, 1.0), alpha=0.2, label="", edge_color='lightblue')
    
    # Silicon grating lines
    period = 0.4
    line_width = 0.2
    for i in range(-4, 5):
        x_pos = i * period
        if abs(x_pos) < Lx/2 - line_width/2:
            create_box(ax, x_pos-line_width/2, x_pos+line_width/2, 
                      -Ly/2, Ly/2, z0, z1,
                      (0.6, 0.3, 0.3), alpha=0.8, label="", edge_color='darkred')
    
    # Add text for grating
    ax.text(0, Ly/2 + 0.3, (z0+z1)/2, "Si Grating", fontsize=10,
            ha='center', va='center')
    
    # Layer 5: Glass substrate  
    z0, z1 = -0.8, -0.3
    create_box(ax, -Lx/2, Lx/2, -Ly/2, Ly/2, z0, z1, 
               (0.7, 1.0, 0.7), alpha=0.3, label="Glass\\n(Substrate)", edge_color='green')
    
    # Set axis properties
    ax.set_xlim(-Lx/2*1.2, Lx/2*1.2)
    ax.set_ylim(-Ly/2*1.2, Ly/2*1.2)
    ax.set_zlim(-1.0, 1.6)
    
    ax.set_xlabel('X (μm)', fontsize=14)
    ax.set_ylabel('Y (μm)', fontsize=14)
    ax.set_zlabel('Z (μm)', fontsize=14)
    
    ax.view_init(elev=20, azim=-60)
    
    # Add title and legend
    plt.title('Multi-Layer Photonic Structure\\n'
              'PhC + Grating + Waveguide Stack', fontsize=18, pad=20)
    
    # Add a simple legend
    legend_text = (
        "Layers (top to bottom):\\n"
        "• Air superstrate\\n"
        "• Silicon PhC (air holes)\\n"
        "• SiO2 buried oxide\\n"
        "• Silicon grating\\n"  
        "• Glass substrate"
    )
    ax.text2D(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=12,
              verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
              facecolor="white", alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    save_path = '/Users/jinzeyuan/Documents/Reserch_Project_2/OpenRCWA/structured_stack_3d.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Structured 3D visualization saved to: {save_path}")
    
except Exception as e:
    print(f"Error creating 3D plot: {e}")
    import traceback
    traceback.print_exc()
'''
    
    # Write and run the script
    script_path = '/tmp/create_simple_structured_3d.py'
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
    print("=== Creating Multi-Layer Structured Photonic Device ===")
    success = create_simple_structured_visualization()
    
    if success:
        print("✓ Structured 3D visualization completed successfully!")
        print("  Features included:")
        print("  - Photonic crystal with air holes")
        print("  - 1D grating structure")
        print("  - Multi-layer waveguide stack")
        print("  - Realistic material colors and transparency")
    else:
        print("✗ Failed to create structured visualization")
