#!/usr/bin/env python3

import os, sys
sys.path.insert(0, '/Users/jinzeyuan/Documents/Reserch_Project_2/OpenRCWA')

# Import matplotlib properly
import sys
import importlib

# Remove any stubbed matplotlib modules
saved = {}
for name in list(sys.modules.keys()):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        saved[name] = sys.modules.pop(name)

try:
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    print("Matplotlib imported successfully")

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create simple rectangles (layer slabs)
    def create_box(x0, x1, y0, y1, z0, z1, color, alpha=0.7):
        # Define the vertices of a box
        vertices = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # bottom face
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]   # top face
        ])
        
        # Define the 12 triangles composing the box
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
        ]
        
        return faces

    # Create layer stack
    # Air layer (superstrate)
    faces1 = create_box(-0.5, 0.5, -0.5, 0.5, 0.2, 0.4, 'blue')
    poly1 = Poly3DCollection(faces1, facecolors='lightblue', alpha=0.3, edgecolors='blue')
    ax.add_collection3d(poly1)

    # SiO2 layer  
    faces2 = create_box(-0.5, 0.5, -0.5, 0.5, 0.0, 0.2, 'orange')
    poly2 = Poly3DCollection(faces2, facecolors='orange', alpha=0.8, edgecolors='darkorange')
    ax.add_collection3d(poly2)

    # Si layer
    faces3 = create_box(-0.5, 0.5, -0.5, 0.5, -0.12, 0.0, 'red')
    poly3 = Poly3DCollection(faces3, facecolors='red', alpha=0.8, edgecolors='darkred')
    ax.add_collection3d(poly3)

    # Glass substrate
    faces4 = create_box(-0.5, 0.5, -0.5, 0.5, -0.32, -0.12, 'green')
    poly4 = Poly3DCollection(faces4, facecolors='lightgreen', alpha=0.3, edgecolors='green')
    ax.add_collection3d(poly4)

    # Set the axes properties
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([-0.4, 0.5])

    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')

    ax.view_init(elev=20, azim=-60)

    # Set background
    fig.patch.set_facecolor('white')

    # Save
    output_path = '/Users/jinzeyuan/Documents/Reserch_Project_2/OpenRCWA/debug_3d_test.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved debug 3D plot to: {output_path}")

    # Print some debug info
    print(f"Number of collections: {len(ax.collections)}")
    print(f"Axes limits: x={ax.get_xlim()}, y={ax.get_ylim()}, z={ax.get_zlim()}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
