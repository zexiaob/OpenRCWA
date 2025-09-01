import numpy as np

from rcwa.geom.patterned import PatternedLayer, rectangular_lattice, RasterConfig
from rcwa.geom.shape import Rectangle
from rcwa.model.material import Material


def small_layer(res=(16, 16)):
    a, b = rectangular_lattice(1.0, 1.0)
    bg = Material(er=1.0, ur=1.0)
    inc = Material(er=4.0, ur=1.0)
    rect = Rectangle(center=(0.5, 0.5), width=0.5, height=0.5)
    return PatternedLayer(
        thickness=100e-9,
        lattice=(a, b),
        shapes=[(rect, inc)],
        background_material=bg,
        raster_config=RasterConfig(resolution=res),
    )


def test_cache_changes_when_rotation_changes():
    layer = small_layer()
    mats1 = layer.to_convolution_matrices((3, 3), wavelength=1.0)
    key1 = layer._last_cache_key

    layer_r = layer.rotated(np.deg2rad(10.0))
    mats2 = layer_r.to_convolution_matrices((3, 3), wavelength=1.0)
    key2 = layer_r._last_cache_key

    assert key1 != key2
    # Matrices should differ for anisotropic pattern under rotation (lattice rotates)
    # Compare one representative component
    assert not np.allclose(mats1['er_xx'], mats2['er_xx'])


def test_cache_changes_when_lattice_changes():
    layer = small_layer()
    mats1 = layer.to_convolution_matrices((3, 3), wavelength=1.0)
    key1 = layer._last_cache_key

    # Change lattice period
    a2, b2 = rectangular_lattice(1.1, 0.9)
    layer2 = PatternedLayer(
        thickness=layer.thickness,
        lattice=(a2, b2),
        shapes=layer.shapes,
        background_material=layer.background_material,
        raster_config=layer.raster_config,
        **layer.params,
    )
    mats2 = layer2.to_convolution_matrices((3, 3), wavelength=1.0)
    key2 = layer2._last_cache_key

    assert key1 != key2
    assert not np.allclose(mats1['er_xx'], mats2['er_xx'])


def test_cache_changes_when_resolution_changes():
    layer = small_layer(res=(16, 16))
    mats1 = layer.to_convolution_matrices((3, 3), wavelength=1.0)
    key1 = layer._last_cache_key

    layer_hi = small_layer(res=(24, 24))
    mats2 = layer_hi.to_convolution_matrices((3, 3), wavelength=1.0)
    key2 = layer_hi._last_cache_key

    assert key1 != key2
    assert not np.allclose(mats1['er_xx'], mats2['er_xx'])
