#!/usr/bin/env python3
"""
Simple test script for HalfSpace and Stack API.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_basic_functionality():
    """Test basic HalfSpace and Stack functionality."""
    print("=== Testing HalfSpace and Stack API ===")
    
    try:
        # Test direct imports
        print("1. Testing direct imports...")
        from rcwa.model.layer import HalfSpace, Air, Substrate, Stack
        from rcwa.model.material import Material
        print("   ✓ Direct imports successful")
        
        # Test Air factory
        print("2. Testing Air() factory...")
        air = Air()
        print(f"   ✓ Air HalfSpace: {air}")
        print(f"   ✓ Air properties: er={air.er}, ur={air.ur}, n={air.n}")
        
        # Test Substrate factory
        print("3. Testing Substrate() factory...")
        si_material = Material(er=12.0, ur=1.0)
        si_substrate = Substrate(si_material)
        print(f"   ✓ Silicon Substrate: {si_substrate}")
        print(f"   ✓ Silicon properties: er={si_substrate.er}, ur={si_substrate.ur}")
        
        # Test Stack construction
        print("4. Testing Stack construction...")
        from rcwa.model.layer import Layer
        test_layer = Layer(er=4.0, ur=1.0, thickness=1.0)
        
        stack = Stack(test_layer,
                      incident_layer=air,
                      transmission_layer=si_substrate)
        print(f"   ✓ Stack created with {len(stack.internal_layers)} internal layer(s)")
        print(f"   ✓ Incident layer: er={stack.incident_layer.er}")
        print(f"   ✓ Transmission layer: er={stack.transmission_layer.er}")
        
        # Test top-level imports
        print("5. Testing top-level imports...")
        import rcwa
        air_top = rcwa.Air()
        si_sub_top = rcwa.Substrate(si_material)
        stack_top = rcwa.Stack(test_layer,
                              incident_layer=air_top,
                              transmission_layer=si_sub_top)
        print("   ✓ Top-level imports successful")
        
        print("\n=== All tests passed! ===")
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
