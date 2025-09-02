import os
import tempfile

import numpy as np
import pytest
try:
    import matplotlib  # type: ignore
    HAS_MPL = hasattr(matplotlib, 'use')
    try:
        import mpl_toolkits.mplot3d  # noqa: F401
        HAS_MPL_3D = True
    except Exception:
        HAS_MPL_3D = False
except Exception:
    HAS_MPL = False
    HAS_MPL_3D = False

from rcwa import Layer, LayerStack, Air, Glass, show_stack3d


@pytest.mark.skipif(not (HAS_MPL and HAS_MPL_3D), reason="matplotlib 3D backend not available")
def test_show_stack3d_homogeneous_layers_no_error():
    # Simple two-layer stack with air superstrate and glass substrate
    air = Air(1.0)
    glass = Glass(1.5)
    l1 = Layer(material=air, thickness=100e-9)
    l2 = Layer(material=glass, thickness=200e-9)
    stack = LayerStack(l1, l2, superstrate=air, substrate=glass)

    fig, ax = show_stack3d(stack, cell_size=(1e-6, 1e-6), alpha=0.5)
    assert fig is not None and ax is not None

    # Save to temp to ensure saving works
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'stack.png')
        fig, ax = show_stack3d(stack, cell_size=(1e-6, 1e-6), save=out)
        assert os.path.exists(out)
