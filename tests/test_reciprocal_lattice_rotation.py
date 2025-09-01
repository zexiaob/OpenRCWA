import numpy as np
import pytest

from rcwa.geom.patterned import PatternedLayer, rectangular_lattice, RasterConfig
from rcwa.geom.shape import Rectangle
from rcwa.model.material import Material


def rot2(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def test_reciprocal_lattice_rotates_consistently():
    # Base lattice: rectangular 1x2
    a = (1.0, 0.0)
    b = (0.0, 2.0)

    bg = Material(er=1.0, ur=1.0)
    mat_incl = Material(er=4.0, ur=1.0)
    rect = Rectangle(center=(0.5, 1.0), width=0.5, height=1.0)
    layer = PatternedLayer(
        thickness=100e-9,
        lattice=(a, b),
        shapes=[(rect, mat_incl)],
        background_material=bg,
        raster_config=RasterConfig(resolution=(16, 16)),
    )

    # Compute reciprocal in base orientation
    g1, g2 = layer.reciprocal_lattice()
    G = np.column_stack([g1, g2])  # 2x2

    # Rotate layer by angle theta; direct lattice rotated internally
    theta = np.deg2rad(37.0)
    layer_r = layer.rotated(theta)
    g1_r, g2_r = layer_r.reciprocal_lattice()
    G_rot = np.column_stack([g1_r, g2_r])

    # Analytic expectation: reciprocal vectors rotate by the same rotation
    R = rot2(theta)
    G_expected = R @ G

    assert np.allclose(G_rot, G_expected, rtol=1e-12, atol=1e-12)


def test_reciprocal_has_2pi_normalization():
    # Square lattice with period p: reciprocal magnitude should be 2*pi/p
    p = 1.5
    a, b = rectangular_lattice(p, p)
    bg = Material(er=1.0, ur=1.0)
    layer = PatternedLayer(
        thickness=50e-9,
        lattice=(a, b),
        shapes=[],
        background_material=bg,
        raster_config=RasterConfig(resolution=(8, 8)),
    )

    g1, g2 = layer.reciprocal_lattice()
    g1_mag = np.linalg.norm(np.array(g1))
    g2_mag = np.linalg.norm(np.array(g2))
    expected = 2 * np.pi / p

    assert np.isclose(g1_mag, expected, rtol=1e-12, atol=1e-12)
    assert np.isclose(g2_mag, expected, rtol=1e-12, atol=1e-12)
