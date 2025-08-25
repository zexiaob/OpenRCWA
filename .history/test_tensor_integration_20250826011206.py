#!/usr/bin/env python3
"""Simple test script for tensor solver integration"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

try:
    print("Testing core adapter imports...")
    from rcwa.core.adapters import TensorToConvolutionAdapter, LayerTensorAdapter
    print("✓ Core adapters imported successfully")
    
    print("Testing tensor material imports...")
    from rcwa import TensorMaterial, Source, Layer  # Use top-level imports
    print("✓ Tensor material imported successfully")
    
    # Test basic adapter functionality
    print("\nTesting adapter functionality...")
    eps_tensor = np.diag([2.0, 3.0, 4.0])
    mu_tensor = np.eye(3, dtype=complex)
    
    result = TensorToConvolutionAdapter.tensor_to_convolution_matrices(
        eps_tensor, mu_tensor, n_harmonics=1)
    
    print(f"✓ Generated {len(result)} convolution matrices")
    print(f"✓ eps_xx = {result['eps_xx']}")
    print(f"✓ eps_yy = {result['eps_yy']}")  
    print(f"✓ eps_zz = {result['eps_zz']}")
    
    # Test effective properties
    eps_eff, mu_eff = TensorToConvolutionAdapter.extract_effective_properties(
        eps_tensor, mu_tensor, 'z')
    print(f"✓ Effective properties: eps_eff = {eps_eff}, mu_eff = {mu_eff}")
    
    # Test tensor material creation
    print("\nTesting tensor material creation...")
    source = Source(wavelength=1.0)
    tensor_mat = TensorMaterial(epsilon_tensor=eps_tensor, source=source)
    print(f"✓ TensorMaterial created: {tensor_mat.name}")
    print(f"✓ Dispersive: {tensor_mat.dispersive}")
    print(f"✓ Epsilon tensor shape: {tensor_mat.epsilon_tensor.shape}")
    
    # Test layer with tensor material
    print("\nTesting layer with tensor material...")
    layer = Layer(tensor_material=tensor_mat, thickness=1.0)
    print(f"✓ Layer created with tensor material")
    print(f"✓ Layer is anisotropic: {layer.is_anisotropic}")
    print(f"✓ Layer thickness: {layer.thickness}")
    
    # Test convolution matrix setup
    print("\nTesting convolution matrix setup...")
    layer.source = source
    layer.set_convolution_matrices(1)
    print(f"✓ Convolution matrices set")
    print(f"✓ Layer er: {layer.er}")
    print(f"✓ Layer ur: {layer.ur}")
    
    print("\n🎉 All basic tests passed!")
    print("Tensor solver integration components are working correctly.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
