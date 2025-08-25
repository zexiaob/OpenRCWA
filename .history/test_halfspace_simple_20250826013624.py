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
        
        # Test Stack construction with old interface
        print("4a. Testing Stack construction (old interface)...")
        from rcwa.model.layer import Layer
        test_layer = Layer(er=4.0, ur=1.0, thickness=1.0)
        
        stack_old = Stack(test_layer,
                          incident_layer=air,
                          transmission_layer=si_substrate)
        print(f"   ✓ Stack (old) created with {len(stack_old.internal_layers)} internal layer(s)")
        print(f"   ✓ Incident layer: er={stack_old.incident_layer.er}")
        print(f"   ✓ Transmission layer: er={stack_old.transmission_layer.er}")
        
        # Test Stack construction with new interface
        print("4b. Testing Stack construction (new interface - superstrate/substrate)...")
        stack_new = Stack(superstrate=air,
                          layers=[test_layer],
                          substrate=si_substrate)
        print(f"   ✓ Stack (new) created with {len(stack_new.internal_layers)} internal layer(s)")
        print(f"   ✓ Superstrate: er={stack_new.incident_layer.er}")
        print(f"   ✓ Substrate: er={stack_new.transmission_layer.er}")
        
        # Test the ROADMAP example syntax
        print("4c. Testing ROADMAP example syntax...")
        stack_roadmap = Stack(
            superstrate=air,
            layers=[test_layer],
            substrate=si_substrate
        )
        print(f"   ✓ ROADMAP syntax works: {len(stack_roadmap.internal_layers)} layer(s)")
        print(f"   ✓ Superstrate->incident: er={stack_roadmap.incident_layer.er}")
        print(f"   ✓ Substrate->transmission: er={stack_roadmap.transmission_layer.er}")
        
        # Test top-level imports
        print("5. Testing top-level imports...")
        import rcwa
        air_top = rcwa.Air()
        si_sub_top = rcwa.Substrate(si_material)
        
        # Test new syntax with top-level imports
        stack_top_new = rcwa.Stack(
            superstrate=air_top,
            layers=[test_layer],
            substrate=si_sub_top
        )
        print("   ✓ Top-level imports successful")
        print(f"   ✓ New syntax with top-level: {len(stack_top_new.internal_layers)} layer(s)")
        
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
