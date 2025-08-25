#!/usr/bin/env python3
"""
Demo of the ideal user syntax for HalfSpace and Stack API.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def demo_ideal_syntax():
    """Demonstrate the ideal user syntax."""
    print("=== Demo: Ideal User Syntax ===")
    
    import rcwa
    
    # Create a test layer (placeholder for future PatternedLayer)
    patterned_layer = rcwa.Layer(er=4.0, ur=1.0, thickness=1.0)
    
    print("1. Basic stack with Air superstrate and Silicon substrate:")
    stack1 = rcwa.Stack(
        superstrate=rcwa.Air,          # Predefined constant
        layers=[patterned_layer],       # List of internal layers
        substrate=rcwa.Silicon()        # Material factory function
    )
    print(f"   ✓ Superstrate: er={stack1.incident_layer.er}")
    print(f"   ✓ Substrate: er={stack1.transmission_layer.er}")
    
    print("\n2. Stack with SiO2 superstrate and Glass substrate:")
    stack2 = rcwa.Stack(
        superstrate=rcwa.SiO2(),       # SiO2 material
        layers=[patterned_layer],
        substrate=rcwa.Glass()         # Glass material
    )
    print(f"   ✓ Superstrate: er={stack2.incident_layer.er}")
    print(f"   ✓ Substrate: er={stack2.transmission_layer.er}")
    
    print("\n3. Stack with custom materials:")
    custom_material = rcwa.Material(er=16.0, ur=1.0)  # Custom high-index material
    stack3 = rcwa.Stack(
        superstrate=rcwa.Air,
        layers=[patterned_layer],
        substrate=custom_material      # Direct material object
    )
    print(f"   ✓ Superstrate: er={stack3.incident_layer.er}")
    print(f"   ✓ Substrate: er={stack3.transmission_layer.er}")
    
    print("\n4. The ROADMAP example - exactly as specified:")
    si_material = rcwa.Silicon(n=3.48)  # Silicon with n=3.48
    pl = patterned_layer  # PatternedLayer placeholder
    
    # This is the exact syntax from ROADMAP
    stack = rcwa.Stack(
        superstrate=rcwa.Air,
        layers=[pl],
        substrate=si_material
    )
    print(f"   ✓ ROADMAP syntax works!")
    print(f"   ✓ Superstrate: er={stack.incident_layer.er}")
    print(f"   ✓ Substrate: er={stack.transmission_layer.er}")
    
    print("\n=== Demo completed successfully! ===")

if __name__ == "__main__":
    demo_ideal_syntax()
