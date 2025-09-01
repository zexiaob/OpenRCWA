import numpy as np
from rcwa.geom.shape import TaperedPolygon
from rcwa.geom.patterned import PatternedLayer, square_lattice
from rcwa.model.material import Material


def test_tapered_polygon_cross_section_and_generate_slices():
    bottom = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
    top = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]
    tp = TaperedPolygon(bottom_vertices=bottom, top_vertices=top, material=None)

    # Simple isotropic materials
    bg = Material(er=1.0)
    fg = Material(er=4.0)

    # Wrap tapered poly in patterned layer
    lattice = square_lattice(1.0)
    layer = PatternedLayer(thickness=1e-6, lattice=lattice,
                           shapes=[(tp, fg)], background_material=bg)

    # Cross-section at mid-plane should be between bottom and top
    mid = layer.get_cross_section(layer.thickness / 2)
    assert isinstance(mid, PatternedLayer)
    er, ur = mid.rasterize_tensor_field()
    assert er.ndim == 2 and ur.ndim == 2

    # Generate multiple slices
    zs = [0, layer.thickness/2, layer.thickness]
    slices = layer.generate_z_slices(zs)
    assert len(slices) == 3
