import numpy as np
from rcwa.geom.shape import Rectangle
from rcwa.geom.patterned import PatternedLayer, square_lattice
from rcwa.model.material import TensorMaterial


def test_offdiagonal_rasterization_and_convolution_keys():
    # Anisotropic tensor with xy coupling
    eps = np.array([[2.0, 0.5, 0.0],
                    [0.5, 3.0, 0.0],
                    [0.0, 0.0, 1.5]], dtype=complex)
    tm = TensorMaterial(epsilon_tensor=eps)

    rect = Rectangle(center=(0.5, 0.5), width=0.4, height=0.4, material=tm)
    lattice = square_lattice(1.0)
    layer = PatternedLayer(thickness=1e-6, lattice=lattice,
                           shapes=[(rect, tm)], background_material=TensorMaterial(epsilon_tensor=np.eye(3)))

    # Use small resolution for speed
    layer.raster_config.resolution = (32, 32)
    E, M = layer.rasterize_full_tensor_field()
    assert E.shape == (32, 32, 3, 3)
    # xy component should be non-zero somewhere due to rectangle region
    assert np.any(np.abs(E[..., 0, 1]) > 0)

    conv = layer.to_convolution_matrices((5, 5))
    required = [
        'er_xx','er_xy','er_xz','er_yx','er_yy','er_yz','er_zx','er_zy','er_zz',
        'ur_xx','ur_xy','ur_xz','ur_yx','ur_yy','ur_yz','ur_zx','ur_zy','ur_zz'
    ]
    for k in required:
        assert k in conv
        assert conv[k].shape == (25, 25)
