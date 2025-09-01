import numpy as np
from rcwa.geom.shape import Circle, Rectangle
from rcwa.model.material import Material


def test_shape_tensor_function_broadcast_and_symmetry():
    # Isotropic material er=4 → diagonal tensor 4*I
    mat = Material(er=4, ur=1)
    circle = Circle(center=(0.5, 0.5), radius=0.2, material=mat)

    fn = circle.to_tensor_function()

    # Scalar input
    T0 = fn(0.1, 0.1)
    assert T0.shape == (3, 3)
    # Symmetry (Hermitian here is real diagonal)
    assert np.allclose(T0, T0.T.conj())

    # Grid input (broadcast)
    x = np.linspace(0, 1, 16)
    y = np.linspace(0, 1, 16)
    X, Y = np.meshgrid(x, y)
    T = fn(X, Y)
    assert T.shape == X.shape + (3, 3)

    # Inside the circle expect approx 4 on diagonal; outside ≈ 1
    mask = circle.contains(X, Y)
    diag = np.stack([T[..., 0, 0], T[..., 1, 1], T[..., 2, 2]], axis=-1)
    # Diagonals close to 4 inside
    if np.any(mask):
        assert np.allclose(diag[mask], 4.0, atol=1e-6)
    # Diagonals close to 1 outside
    if np.any(~mask):
        assert np.allclose(diag[~mask], 1.0, atol=1e-6)
