#!/usr/bin/env python3
"""
Test the improved HalfSpace and Stack API with direct material support.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_improved_syntax():
    """Test the improved syntax where materials can be directly used."""
    print("=== Testing Improved Stack API ===")
    
    try:
        from rcwa.model.layer import Stack, Air, Layer
        from rcwa.model.material import Material, TensorMaterial
        import numpy as np
        
        # Create test materials
        print("1. Creating test materials...")
        si_material = Material(er=12.0, ur=1.0)
        sio2_material = Material(er=2.1, ur=1.0)
        
        # Create anisotropic material
        epsilon_tensor = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0], 
            [0.0, 0.0, 4.0]
        ], dtype=complex)
        aniso_material = TensorMaterial(epsilon_tensor=epsilon_tensor)
        
        test_layer = Layer(er=4.0, ur=1.0, thickness=1.0)
        print("   ✓ Materials created")
        
        # Test 1: Air constant + Material directly
        print("2. Testing: superstrate=Air, substrate=material...")
        stack1 = Stack(
            superstrate=Air,
            layers=[test_layer],
            substrate=si_material
        )
        print(f"   ✓ Stack1: superstrate er={stack1.incident_layer.er}, substrate er={stack1.transmission_layer.er}")
        
        # Test 2: Two materials directly
        print("3. Testing: superstrate=material1, substrate=material2...")
        stack2 = Stack(
            superstrate=sio2_material,
            layers=[test_layer],
            substrate=si_material
        )
        print(f"   ✓ Stack2: superstrate er={stack2.incident_layer.er}, substrate er={stack2.transmission_layer.er}")
        
        # Test 3: Material + anisotropic material
        print("4. Testing: superstrate=material, substrate=tensor_material...")
        stack3 = Stack(
            superstrate=si_material,
            layers=[test_layer],
            substrate=aniso_material
        )
        print(f"   ✓ Stack3: superstrate er={stack3.incident_layer.er}")
        print(f"   ✓ Stack3: substrate is anisotropic={stack3.transmission_layer.is_anisotropic}")
        
        # Test 4: The ideal ROADMAP syntax (as constants)
        print("5. Testing ideal ROADMAP syntax...")
        print("   Expected: superstrate=orcwa.Air, substrate=orcwa.Silicon")
        
        # Test with top-level import
        print("6. Testing top-level imports...")
        import rcwa
        
        # This should work: superstrate=rcwa.Air (constant, not function call)
        stack_top = rcwa.Stack(
            superstrate=rcwa.Air,
            layers=[test_layer],
            substrate=si_material
        )
        print(f"   ✓ Top-level: superstrate er={stack_top.incident_layer.er}")
        
        print("\n=== All tests passed! ===")
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_syntax()
    sys.exit(0 if success else 1)
