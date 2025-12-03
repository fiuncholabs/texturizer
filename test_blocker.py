#!/usr/bin/env python3
"""
Test script for blocker volume functionality.
Tests the cylinder generator and blocker masking in apply_fuzzy_skin.
"""

import numpy as np
from stl import mesh
from texturizer import (
    generate_simple_cube,
    generate_blocker_cylinder,
    point_inside_mesh_volume,
    apply_fuzzy_skin
)

def test_cylinder_generation():
    """Test that cylinder generator creates a valid mesh."""
    print("\n=== Test 1: Cylinder Generation ===")

    cylinder = generate_blocker_cylinder(radius=5, height=15, position=(0, 0, 0), segments=16)

    # Verify mesh has triangles
    assert len(cylinder.vectors) > 0, "Cylinder should have triangles"
    print(f"✓ Cylinder generated with {len(cylinder.vectors)} triangles")

    # Verify mesh is closed (no NaN values)
    assert not np.any(np.isnan(cylinder.vectors)), "Cylinder should not have NaN values"
    print("✓ Cylinder mesh is valid (no NaN values)")

    # Save for visual inspection
    cylinder.save('test_cylinder.stl')
    print("✓ Cylinder saved to test_cylinder.stl")

    return cylinder

def test_point_inside_volume():
    """Test that point_inside_mesh_volume correctly identifies points."""
    print("\n=== Test 2: Point Inside Volume Detection ===")

    # Create a cylinder centered at origin
    cylinder = generate_blocker_cylinder(radius=10, height=20, position=(0, 0, 0), segments=32)

    # Test points
    test_points = np.array([
        [0, 0, 0],      # Inside (center)
        [5, 0, 0],      # Inside (within radius)
        [0, 5, 0],      # Inside (within radius)
        [15, 0, 0],     # Outside (beyond radius)
        [0, 0, 15],     # Outside (beyond height)
        [8, 8, 0],      # Outside (radius ~11.3)
        [7, 0, 0],      # Inside (within radius)
    ])

    inside = point_inside_mesh_volume(test_points, cylinder)

    expected = np.array([True, True, True, False, False, False, True])

    print("Point | Expected | Got")
    print("-" * 40)
    for i, point in enumerate(test_points):
        status = "✓" if inside[i] == expected[i] else "✗"
        print(f"{status} {point} | {expected[i]} | {inside[i]}")

    # Check that results match expectations
    matches = np.sum(inside == expected)
    print(f"\n{matches}/{len(expected)} tests passed")

    if matches == len(expected):
        print("✓ All point-in-volume tests passed")
    else:
        print("✗ Some point-in-volume tests failed")

    return matches == len(expected)

def test_fuzzy_skin_with_blocker():
    """Test that apply_fuzzy_skin respects blocker volume."""
    print("\n=== Test 3: Fuzzy Skin with Blocker ===")

    # Generate a simple cube
    print("Generating cube...")
    cube = generate_simple_cube(size=20)

    # Generate a cylinder blocker in the center
    print("Generating cylinder blocker...")
    blocker = generate_blocker_cylinder(radius=8, height=25, position=(0, 0, 0), segments=32)

    # Save blocker for inspection
    blocker.save('test_blocker_cylinder.stl')
    print("✓ Blocker saved to test_blocker_cylinder.stl")

    # Apply fuzzy skin WITHOUT blocker
    print("\nApplying fuzzy skin without blocker...")
    cube_copy1 = generate_simple_cube(size=20)
    output_no_blocker = apply_fuzzy_skin(
        cube_copy1,
        thickness=0.5,
        point_distance=2.0,
        seed=42
    )
    output_no_blocker.save('test_cube_no_blocker.stl')
    print("✓ Saved to test_cube_no_blocker.stl")

    # Apply fuzzy skin WITH blocker
    print("\nApplying fuzzy skin with blocker...")
    cube_copy2 = generate_simple_cube(size=20)
    output_with_blocker = apply_fuzzy_skin(
        cube_copy2,
        thickness=0.5,
        point_distance=2.0,
        seed=42,
        blocker_mesh=blocker
    )
    output_with_blocker.save('test_cube_with_blocker.stl')
    print("✓ Saved to test_cube_with_blocker.stl")

    # Compare vertices to verify some were blocked
    diff = np.abs(output_with_blocker.vectors - output_no_blocker.vectors)
    max_diff = np.max(diff)

    if max_diff > 0:
        print(f"✓ Blocker affected the output (max difference: {max_diff:.4f} mm)")
        return True
    else:
        print("✗ Blocker did not affect the output")
        return False

def test_simple_corner_with_blocker():
    """Test with simple-corner.stl if available."""
    print("\n=== Test 4: Simple Corner with Blocker (Optional) ===")

    try:
        # Try to load simple-corner.stl
        corner = mesh.Mesh.from_file('simple-corner.stl')
        print(f"✓ Loaded simple-corner.stl ({len(corner.vectors)} triangles)")

        # Generate a cylinder blocker
        blocker = generate_blocker_cylinder(radius=15, height=40, position=(0, 0, 20), segments=32)

        # Apply fuzzy skin with blocker
        print("Applying fuzzy skin with blocker (this may take a moment)...")
        output = apply_fuzzy_skin(
            corner,
            thickness=0.3,
            point_distance=1.5,  # Larger for faster test
            seed=42,
            blocker_mesh=blocker
        )

        output.save('test_corner_with_blocker.stl')
        print("✓ Saved to test_corner_with_blocker.stl")

        return True

    except FileNotFoundError:
        print("⊘ simple-corner.stl not found, skipping this test")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Blocker Volume Functionality")
    print("=" * 60)

    results = []

    # Test 1: Cylinder generation
    try:
        test_cylinder_generation()
        results.append(("Cylinder Generation", True))
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results.append(("Cylinder Generation", False))

    # Test 2: Point inside volume
    try:
        success = test_point_inside_volume()
        results.append(("Point Inside Volume", success))
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results.append(("Point Inside Volume", False))

    # Test 3: Fuzzy skin with blocker
    try:
        success = test_fuzzy_skin_with_blocker()
        results.append(("Fuzzy Skin with Blocker", success))
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results.append(("Fuzzy Skin with Blocker", False))

    # Test 4: Simple corner (optional)
    try:
        success = test_simple_corner_with_blocker()
        results.append(("Simple Corner with Blocker", success))
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results.append(("Simple Corner with Blocker", False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\n{passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
