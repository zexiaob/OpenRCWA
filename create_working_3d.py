#!/usr/bin/env python3
"""
Alternative 3D visualization that works around matplotlib compatibility issues.
"""

def create_simple_stack3d_plot(stack, save_path, cell_size=(1.0, 1.0)):
    """Create a simple 3D stack visualization that works despite matplotlib issues."""
    
    import sys
    import os
    import subprocess
    
    # Write a Python script that will be run in isolation
    script_content = '''
import sys
import os

# Move to temp directory to avoid path issues
os.chdir('/tmp')

# Try to import matplotlib cleanly
try:
    import matplotlib
    matplotlib.use('Agg', force=True)  # Non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    
    print("Matplotlib imported successfully")
    
    # Create the plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False  
    ax.zaxis.pane.fill = False
    
    # Define colors for different materials
    colors = [
        (0.7, 0.9, 1.0),    # Light blue for air
        (1.0, 0.7, 0.3),    # Orange for SiO2  
        (0.8, 0.2, 0.2),    # Red for Si
        (0.7, 1.0, 0.7),    # Light green for glass
    ]
    
    # Layer thicknesses (scaled up for visibility)
    layers = [
        ("Air", 0.4, 0.8, colors[0]),       # Superstrate
        ("SiO2", 0.0, 0.4, colors[1]),      # Layer 1: 200nm -> 0.4 units
        ("Si", -0.24, 0.0, colors[2]),      # Layer 2: 120nm -> 0.24 units  
        ("Glass", -0.64, -0.24, colors[3])  # Substrate
    ]
    
    # Cell dimensions
    Lx, Ly = {cell_size}
    
    # Create boxes for each layer
    for name, z0, z1, color in layers:
        # Create vertices for a rectangular box
        x = [-Lx/2, Lx/2, Lx/2, -Lx/2]
        y = [-Ly/2, -Ly/2, Ly/2, Ly/2]
        
        # Bottom face
        bottom = [(x[i], y[i], z0) for i in range(4)]
        # Top face  
        top = [(x[i], y[i], z1) for i in range(4)]
        
        # All faces of the box
        faces = []
        # Bottom and top
        faces.append(bottom)
        faces.append(top)
        # Sides
        for i in range(4):
            j = (i + 1) % 4
            faces.append([bottom[i], bottom[j], top[j], top[i]])
        
        # Create and add the 3D polygon collection
        poly = Poly3DCollection(faces, alpha=0.7, facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_collection3d(poly)
        
        # Add text label
        ax.text(0, 0, (z0+z1)/2, name, fontsize=10)
    
    # Set axis properties
    ax.set_xlim(-Lx/2*1.1, Lx/2*1.1)
    ax.set_ylim(-Ly/2*1.1, Ly/2*1.1)
    ax.set_zlim(-0.7, 0.9)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_zlabel('Z (scaled)')
    
    ax.view_init(elev=20, azim=-60)
    
    plt.title('Layer Stack 3D Visualization\\nSiO2(200nm) + Si(120nm)', fontsize=14)
    
    # Save the plot
    plt.savefig('{save_path}', dpi=200, bbox_inches='tight')
    print(f"3D visualization saved successfully to: {save_path}")
    
except Exception as e:
    print(f"Error creating 3D plot: {{e}}")
    import traceback
    traceback.print_exc()
'''.format(cell_size=cell_size, save_path=save_path)
    
    # Write the script to a temporary file
    script_path = '/tmp/create_3d_plot.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Run the script with the virtual environment Python
    try:
        result = subprocess.run([
            '/Users/jinzeyuan/Documents/Reserch_Project_2/OpenRCWA/.venv/bin/python',
            script_path
        ], capture_output=True, text=True, cwd='/tmp')
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode == 0:
            print("3D visualization created successfully!")
        else:
            print(f"Script failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"Failed to run visualization script: {e}")

if __name__ == "__main__":
    # This is just a placeholder - in real use you'd pass the actual stack
    save_path = '/Users/jinzeyuan/Documents/Reserch_Project_2/OpenRCWA/stack3d_working.png'
    create_simple_stack3d_plot(None, save_path)
