#!/usr/bin/env python3
"""
Create a structured 3D visualization with patterned layers.
"""

import os, sys
os.chdir('/tmp')
sys.path.insert(0, '/Users/jinzeyuan/Documents/Reserch_Project_2/OpenRCWA')

from rcwa import Layer, LayerStack, Air, Glass, Silicon, SiO2, show_stack3d, nm
from rcwa.geom import PatternedLayer, Rectangle, Circle
from rcwa.geom.shape import Point
from rcwa.model import Material

def create_structured_example_1():
    """Create a photonic crystal structure with circular holes"""
    print("Creating photonic crystal structure...")
    
    # Materials
    air = Air(1.0)
    glass = Glass(1.52) 
    sio2 = SiO2(1.46)
    si = Silicon(3.48)
    
    # Create a patterned layer with circular holes
    # This represents a photonic crystal slab
    hole_radius = 0.15  # 150nm radius holes
    lattice_constant = 0.5  # 500nm period
    
    # Define the lattice vectors for a square lattice
    lattice = [(lattice_constant, 0.0), (0.0, lattice_constant)]
    
    # Create circular holes at lattice points
    holes = []
    for i in range(-2, 3):  # 5x5 array of holes
        for j in range(-2, 3):
            center = Point(i * lattice_constant, j * lattice_constant)
            hole = Circle(center=center, radius=hole_radius)
            holes.append((hole, air))  # Holes filled with air
    
    # Create the patterned layer
    patterned_layer = PatternedLayer(
        background_material=si,  # Silicon background
        shapes=holes,
        thickness=220*nm(1),     # 220nm thick silicon layer
        lattice=lattice
    )
    
    stack = LayerStack(
        Layer(material=sio2, thickness=100*nm(1)),  # Top oxide
        patterned_layer,                            # Patterned silicon
        Layer(material=sio2, thickness=500*nm(1)),  # BOX layer
        superstrate=air,
        substrate=glass,
    )
    
    return stack, (lattice_constant * 5, lattice_constant * 5)

def create_structured_example_2():
    """Create a grating structure"""
    print("Creating grating structure...")
    
    # Materials
    air = Air(1.0)
    glass = Glass(1.52)
    resist = Material(1.65)  # Photoresist
    si = Silicon(3.48)
    
    # Grating parameters
    period = 0.6  # 600nm period
    fill_factor = 0.5  # 50% fill factor
    line_width = period * fill_factor
    
    # Create grating lines
    lattice = [(period, 0.0), (0.0, 2.0)]  # 1D grating, 2um wide
    
    grating_lines = []
    for i in range(-5, 6):  # 11 lines
        center = Point(i * period, 0.0)
        line = Rectangle(
            center=center,
            width=line_width,
            height=1.5,  # 1.5um tall lines
            rotation=0.0
        )
        grating_lines.append((line, si))  # Silicon lines
    
    # Create the patterned grating layer
    grating_layer = PatternedLayer(
        background_material=air,  # Air background
        shapes=grating_lines,
        thickness=300*nm(1),      # 300nm thick
        lattice=lattice
    )
    
    stack = LayerStack(
        Layer(material=resist, thickness=150*nm(1)),  # Resist layer
        grating_layer,                                # Grating
        Layer(material=si, thickness=2000*nm(1)),     # Silicon substrate
        superstrate=air,
        substrate=glass,
    )
    
    return stack, (period * 11, 2.0)

def create_advanced_3d_visualization(stack, cell_size, save_path, title="Structured Layer Stack"):
    """Create an advanced 3D visualization with better rendering"""
    
    import subprocess
    
    # Convert materials and layers to simple data for the subprocess
    layer_data = []
    
    # Process superstrate
    if stack.incident_layer:
        layer_data.append({
            'name': 'Air (Superstrate)',
            'material_type': 'superstrate', 
            'thickness': 0.2,  # Scaled thickness
            'color': (0.7, 0.9, 1.0, 0.3),  # Light blue, transparent
            'shapes': None
        })
    
    # Process internal layers
    z_pos = 0.0
    for i, lyr in enumerate(stack.internal_layers):
        thickness = getattr(lyr, 'thickness', 0.0) or 0.0
        thickness_scaled = thickness * 1e6  # Convert to microns
        
        # Check if it's a patterned layer
        from rcwa.geom.patterned import PatternedLayer
        if isinstance(lyr, PatternedLayer):
            # Extract pattern information
            bg_material = getattr(lyr, 'background_material', None)
            shapes_data = []
            
            if hasattr(lyr, 'shapes') and lyr.shapes:
                for shape, material in lyr.shapes:
                    # Extract shape geometry
                    shape_info = extract_shape_info(shape)
                    if shape_info:
                        shapes_data.append({
                            'geometry': shape_info,
                            'material': get_material_name(material)
                        })
            
            layer_data.append({
                'name': f'Patterned Layer {i+1}',
                'material_type': 'patterned',
                'thickness': thickness_scaled,
                'color': get_material_color(bg_material),
                'background_material': get_material_name(bg_material),
                'shapes': shapes_data
            })
        else:
            # Regular homogeneous layer
            material = getattr(lyr, 'material', lyr)
            layer_data.append({
                'name': f'Layer {i+1}',
                'material_type': 'homogeneous',
                'thickness': thickness_scaled,
                'color': get_material_color(material),
                'shapes': None
            })
    
    # Process substrate
    if stack.transmission_layer:
        layer_data.append({
            'name': 'Glass (Substrate)', 
            'material_type': 'substrate',
            'thickness': 0.3,
            'color': (0.7, 1.0, 0.7, 0.3),  # Light green, transparent
            'shapes': None
        })
    
    # Create the visualization script
    script_content = f'''
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
    
    print("Creating advanced 3D visualization...")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set white background
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
    
    # Cell dimensions
    Lx, Ly = {cell_size}
    
    # Layer data
    layers = {layer_data}
    
    z_pos = 0.0
    
    for layer in layers:
        z0 = z_pos
        z1 = z_pos + layer['thickness']
        
        if layer['shapes'] is None:
            # Simple homogeneous layer
            create_box(ax, -Lx/2, Lx/2, -Ly/2, Ly/2, z0, z1, 
                      layer['color'], layer['name'])
        else:
            # Patterned layer - first draw background
            bg_color = layer['color']
            bg_color_faded = (bg_color[0], bg_color[1], bg_color[2], 0.3)
            create_box(ax, -Lx/2, Lx/2, -Ly/2, Ly/2, z0, z1, 
                      bg_color_faded, layer['name'] + ' (background)')
            
            # Then draw the patterns
            for shape in layer['shapes']:
                geom = shape['geometry']
                if geom['type'] == 'circle':
                    create_cylinder(ax, geom['center'][0], geom['center'][1],
                                  geom['radius'], z0, z1, (1.0, 0.5, 0.5, 0.8), 
                                  shape['material'])
                elif geom['type'] == 'rectangle':
                    x_c, y_c = geom['center']
                    w, h = geom['width'], geom['height']
                    create_box(ax, x_c-w/2, x_c+w/2, y_c-h/2, y_c+h/2, z0, z1,
                             (0.8, 0.3, 0.3, 0.9), shape['material'])
        
        z_pos = z1
    
    # Set axis properties
    ax.set_xlim(-Lx/2*1.1, Lx/2*1.1)
    ax.set_ylim(-Ly/2*1.1, Ly/2*1.1)
    ax.set_zlim(0, z_pos*1.1)
    
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_zlabel('Z (μm)', fontsize=12)
    
    ax.view_init(elev=25, azim=-65)
    
    plt.title('{title}\\nStructured Multilayer Stack', fontsize=16, pad=20)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('{save_path}', dpi=300, bbox_inches='tight')
    print(f"Advanced 3D visualization saved to: {save_path}")
    
except Exception as e:
    print(f"Error creating 3D plot: {{e}}")
    import traceback
    traceback.print_exc()

def create_box(ax, x0, x1, y0, y1, z0, z1, color, label=""):
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
    
    poly = Poly3DCollection(faces, alpha=color[3] if len(color)>3 else 0.7, 
                           facecolor=color[:3], edgecolor='black', linewidth=0.3)
    ax.add_collection3d(poly)
    
    # Add label at center
    if label:
        ax.text((x0+x1)/2, (y0+y1)/2, (z0+z1)/2, label, fontsize=8)

def create_cylinder(ax, x_center, y_center, radius, z0, z1, color, label=""):
    """Create a cylindrical hole or pillar"""
    theta = np.linspace(0, 2*np.pi, 20)
    x_circle = x_center + radius * np.cos(theta)
    y_circle = y_center + radius * np.sin(theta)
    
    # Bottom and top circles
    bottom = [(x_circle[i], y_circle[i], z0) for i in range(len(theta))]
    top = [(x_circle[i], y_circle[i], z1) for i in range(len(theta))]
    
    faces = [bottom, top]
    
    # Side faces
    for i in range(len(theta)-1):
        faces.append([bottom[i], bottom[i+1], top[i+1], top[i]])
    
    poly = Poly3DCollection(faces, alpha=color[3] if len(color)>3 else 0.7,
                           facecolor=color[:3], edgecolor='black', linewidth=0.3)
    ax.add_collection3d(poly)
'''
    
    # Write and run the script
    script_path = '/tmp/create_structured_3d.py'
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

def extract_shape_info(shape):
    """Extract geometry information from shape objects"""
    from rcwa.geom.shape import Rectangle, Circle
    
    if isinstance(shape, Circle):
        return {
            'type': 'circle',
            'center': (shape.center.x, shape.center.y),
            'radius': shape.radius
        }
    elif isinstance(shape, Rectangle):
        return {
            'type': 'rectangle', 
            'center': (shape.center.x, shape.center.y),
            'width': shape.width,
            'height': shape.height,
            'rotation': shape.rotation
        }
    return None

def get_material_name(material):
    """Get a readable name for a material"""
    if hasattr(material, 'name'):
        return material.name
    elif hasattr(material, 'n'):
        n = getattr(material, 'n', 1.0)
        if abs(n - 1.0) < 0.01:
            return 'Air'
        elif abs(n - 1.52) < 0.01:
            return 'Glass'
        elif abs(n - 1.46) < 0.01:
            return 'SiO2'
        elif abs(n - 3.48) < 0.1:
            return 'Silicon'
        else:
            return f'Material(n={n:.2f})'
    return str(type(material).__name__)

def get_material_color(material):
    """Get color for a material"""
    name = get_material_name(material)
    colors = {
        'Air': (0.7, 0.9, 1.0, 0.3),
        'Glass': (0.7, 1.0, 0.7, 0.4), 
        'SiO2': (1.0, 0.8, 0.4, 0.7),
        'Silicon': (0.6, 0.3, 0.3, 0.8),
    }
    return colors.get(name, (0.5, 0.5, 0.5, 0.7))

if __name__ == "__main__":
    # Create photonic crystal example
    print("=== Creating Photonic Crystal Structure ===")
    stack1, cell_size1 = create_structured_example_1()
    save_path1 = '/Users/jinzeyuan/Documents/Reserch_Project_2/OpenRCWA/photonic_crystal_3d.png'
    
    success1 = create_advanced_3d_visualization(
        stack1, cell_size1, save_path1, 
        "Photonic Crystal Structure"
    )
    
    if success1:
        print(f"✓ Photonic crystal visualization saved: {save_path1}")
    else:
        print("✗ Failed to create photonic crystal visualization")
    
    print("\\n=== Creating Grating Structure ===")
    # Create grating example
    stack2, cell_size2 = create_structured_example_2()
    save_path2 = '/Users/jinzeyuan/Documents/Reserch_Project_2/OpenRCWA/grating_3d.png'
    
    success2 = create_advanced_3d_visualization(
        stack2, cell_size2, save_path2,
        "1D Grating Structure"
    )
    
    if success2:
        print(f"✓ Grating visualization saved: {save_path2}")
    else:
        print("✗ Failed to create grating visualization")
