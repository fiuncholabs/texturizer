#!/usr/bin/env python3
"""
Test script for double_stl blocker algorithm.
"""

from stl import mesh
from texturizer import (
    generate_simple_cube,
    generate_blocker_cylinder,
    apply_fuzzy_skin
)

def test_double_stl_algorithm():
    """Test the double_stl blocker algorithm."""
    print("\n=== Test: Double STL Algorithm ===\n")

    # Generate a simple cube
    print("Generating 20mm cube...")
    cube = generate_simple_cube(size=20)
    print(f"✓ Cube generated with {len(cube.vectors)} triangles")

    # Generate a cylinder blocker in the center
    print("\nGenerating cylinder blocker (radius=8mm, height=25mm)...")
    blocker = generate_blocker_cylinder(radius=8, height=25, position=(0, 0, 0), segments=32)
    blocker.save('test_double_stl_blocker.stl')
    print(f"✓ Blocker saved to test_double_stl_blocker.stl ({len(blocker.vectors)} triangles)")

    # Apply fuzzy skin WITH blocker using double_stl algorithm
    print("\nApplying fuzzy skin with double_stl algorithm...")
    print("Parameters: thickness=0.5mm, point_distance=2.0mm")

    output = apply_fuzzy_skin(
        cube,
        thickness=0.5,
        point_distance=2.0,
        seed=42,
        blocker_mesh=blocker,
        blocker_algorithm='double_stl'
    )

    output.save('test_double_stl_output.stl')
    print(f"\n✓ Processing complete!")
    print(f"  Output saved to: test_double_stl_output.stl")
    print(f"  Output triangles: {len(output.vectors):,}")

    return True

if __name__ == "__main__":
    try:
        success = test_double_stl_algorithm()
        if success:
            print("\n🎉 Double STL algorithm test passed!")
            exit(0)
        else:
            print("\n✗ Double STL algorithm test failed")
            exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
