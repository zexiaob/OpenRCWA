import numpy as np
from unittest.mock import patch

from rcwa.model.material import Material, TensorMaterial
from rcwa.geom.shape import Rectangle
from rcwa.geom.patterned import PatternedLayer


def test_patterned_layer_tensor_paths():
    background = Material(er=1.0, ur=1.0)
    tensor_mat = TensorMaterial(
        epsilon_tensor=np.diag([2.0, 3.0, 4.0]),
        mu_tensor=np.eye(3)
    )
    rect = Rectangle(center=(0.5, 0.5), width=0.4, height=0.4)

    layer = PatternedLayer(
        thickness=1.0,
        lattice=((1.0, 0.0), (0.0, 1.0)),
        shapes=[(rect, tensor_mat)],
        background_material=background,
    )

    assert layer.is_anisotropic is True

    # Provide minimal k-vectors for matrix calculations
    layer.Kx = 0.0
    layer.Ky = 0.0

    with patch(
        "rcwa.core.adapters.LayerTensorAdapter.adapt_P_matrix_for_tensor",
        return_value=np.zeros((2, 2), dtype=complex),
    ) as mock_p:
        layer.P_matrix()
        assert mock_p.called

    with patch(
        "rcwa.core.adapters.LayerTensorAdapter.adapt_Q_matrix_for_tensor",
        return_value=np.zeros((2, 2), dtype=complex),
    ) as mock_q:
        layer.Q_matrix()
        assert mock_q.called
