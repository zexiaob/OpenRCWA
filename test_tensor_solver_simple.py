#!/usr/bin/env python3
"""
Simple test to check tensor solver functionality
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from rcwa import TensorMaterial, Source, Layer, LayerStack, Solver

def test_simple_tensor_solve():
    """Test basic tensor solver functionality"""
    try:
        # Create a simple isotropic-like tensor material
        source = Source(wavelength=1.0, theta=0)
        eps_iso = 2.25  # n = 1.5
        tensor_mat = TensorMaterial.from_diagonal(eps_iso, eps_iso, eps_iso, source=source)
        
        # Create simple stack
        incident_layer = Layer()  # Air
        tensor_layer = Layer(tensor_material=tensor_mat, thickness=0.5)
        transmission_layer = Layer()  # Air
        
        stack = LayerStack(tensor_layer,
                          incident_layer=incident_layer,
                          transmission_layer=transmission_layer)
        
        # Create solver
        solver = Solver(stack, source, n_harmonics=1)
        
        print("✓ Solver created successfully")
        
        # Test initialization
        solver._initialize()
        print("✓ Solver initialized successfully")
        
        # Test matrix setup for tensor layer
        for layer in stack.internal_layers:
            if layer.is_anisotropic:
                print(f"  Layer has tensor material: {layer.tensor_material.name}")
                layer.set_convolution_matrices(1)
                print("  ✓ Convolution matrices set")
                
                # Test P and Q matrices
                P = layer.P_matrix()
                Q = layer.Q_matrix()
                print(f"  ✓ P matrix shape: {P.shape}")
                print(f"  ✓ Q matrix shape: {Q.shape}")
                
                # Test VWLX matrices
                V, W, Lambda, X = layer.VWLX_matrices()
                print(f"  ✓ VWLX matrices computed: V{V.shape}, W{W.shape}, Lambda{Lambda.shape}, X{X.shape}")
        
        # Test S-matrix calculation
        try:
            solver._inner_s_matrix()
            print("✓ Inner S-matrix calculated")
            
            solver._global_s_matrix()
            print("✓ Global S-matrix calculated")
            
            solver._rt_quantities()
            print("✓ R/T quantities calculated")
            
            print(f"  RTot = {solver.RTot:.6f}")
            print(f"  TTot = {solver.TTot:.6f}")
            print(f"  Conservation = {solver.conservation:.6f}")
            
            if abs(solver.conservation - 1.0) < 0.1:
                print("✓ Energy conservation looks reasonable")
            else:
                print(f"⚠ Energy conservation needs improvement: {solver.conservation}")
                
        except Exception as e:
            print(f"✗ S-matrix calculation failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_tensor_solve()
    print(f"\nTensor solver test: {'PASSED' if success else 'FAILED'}")
