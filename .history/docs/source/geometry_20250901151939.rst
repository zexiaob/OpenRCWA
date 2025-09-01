Declarative Geometry (Patterned Layers)
======================================

This page documents the new declarative geometry API for building patterned layers with full tensor support.

PatternedLayer
--------------
``PatternedLayer`` represents a 2D pattern within a layer of given thickness. You provide:

- thickness (m)
- lattice vectors (e.g., square/rectangular/hexagonal)
- a list of (shape, material) pairs
- a background material

.. code-block:: python

    from rcwa import PatternedLayer, Rectangle, Circle, DifferenceShape, square_lattice, Material

    lat = square_lattice(1.0)
    outer = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
    hole = Circle(center=(0.5, 0.5), radius=0.2)
    pattern = DifferenceShape(outer, [hole])

    layer = PatternedLayer(
        thickness=200e-9,
        lattice=lat,
        shapes=[(pattern, Material(er=6.0))],
        background_material=Material(er=2.25),
    )

Lattices
--------
Convenience constructors:

.. code-block:: python

    from rcwa import rectangular_lattice, square_lattice, hexagonal_lattice

    a = rectangular_lattice(400e-9, 600e-9)
    b = square_lattice(500e-9)
    c = hexagonal_lattice(500e-9)

Shapes and Boolean Operations
-----------------------------
Available shapes: ``Rectangle``, ``Circle``, ``Ellipse``, ``Polygon``, ``RegularPolygon``.

Boolean composition: ``UnionShape``, ``IntersectionShape``, ``DifferenceShape``（可用于掏孔）。

.. code-block:: python

    from rcwa import Rectangle, Circle, UnionShape, DifferenceShape

    base = Rectangle(center=(0.5, 0.5), width=0.6, height=0.6)
    hole = Circle(center=(0.5, 0.5), radius=0.2)
    ring = DifferenceShape(base, [hole])

    a = Circle(center=(0.25, 0.25), radius=0.1)
    b = Circle(center=(0.75, 0.75), radius=0.1)
    features = UnionShape([a, b])

    final = UnionShape([ring, features])

Parametric Geometry
-------------------
All shapes and PatternedLayer support ``.with_params(**kwargs)`` to update parameters for sweeps.

``Polygon`` also supports template-based vertices/holes via callables or ``from_template``.

.. code-block:: python

    hole = Circle(center=(0.5, 0.5), radius=0.15)
    pattern = DifferenceShape(base, [hole])
    layer_r18 = layer.with_params(radius=0.18)  # updates hole radius

Unified Z-slicing
-----------------
对具有 z 依赖（例如锥形侧壁）的几何，可使用 Stack 的统一 Z 切片接口：

.. code-block:: python

    from rcwa import Stack, Layer

    stack = Stack(layer, superstrate=Layer(er=1.0), substrate=Layer(er=2.25), auto_z_slicing=5)

当需要自动建议切片位置时，也可传入 ``auto_z_slicing=True`` 并由几何自行建议；不需要切片时将自动跳过，接口保持统一。
