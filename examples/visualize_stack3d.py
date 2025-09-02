"""
Example: 3D visualization of a simple stack with a patterned layer.

Run this script to generate an image 'stack3d_example.png' in the same folder.
"""

from rcwa import (
    Layer, LayerStack, Air, Glass, Silicon, SiO2,
    PatternedLayer, rectangular_lattice, Rectangle, Circle,
    show_stack3d, nm
)


def build_example_stack():
    # Materials
    air = Air(1.0)
    sio2 = SiO2(1.46)
    si = Silicon(3.48)
    glass = Glass(1.52)

    # Homogeneous spacer
    spacer = Layer(material=sio2, thickness=200*nm(1))

    # Patterned layer: a square lattice with a silicon rectangle and a circular air hole
    period = 600e-9
    lattice = rectangular_lattice(period, period)

    rect = Rectangle(center=(0.0, 0.0), width=0.5, height=0.3)  # unit-cell coords
    hole = Circle(center=(0.15, 0.1), radius=0.12)

    # Important: PatternedLayer expects unit-cell coordinates for shapes' contains().
    # Here we focus on qualitative visualization; extrusion thickness shows along z.
    patterned = PatternedLayer(
        thickness=150*nm(1),
        lattice=lattice,
        shapes=[(rect, si), (hole, air)],
        background_material=sio2,
    )

    # Stack: air superstrate, layers, glass substrate
    stack = LayerStack(
        spacer,
        patterned,
        Layer(material=si, thickness=100*nm(1)),
        superstrate=air,
        substrate=glass,
    )
    return stack


def main():
    stack = build_example_stack()
    # Choose cell size in physical units for rendering
    Lx = Ly = 1.0  # Arbitrary units adequate for qualitative preview
    fig, ax = show_stack3d(stack, cell_size=(Lx, Ly), alpha=0.7, save='stack3d_example.png', dpi=220)
    print('Saved 3D visualization to stack3d_example.png')


if __name__ == '__main__':
    main()
