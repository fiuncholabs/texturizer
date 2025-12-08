#!/usr/bin/env python3
"""Test in-plane noise to verify it only applies displacement in XY plane."""

import numpy as np
from stl import mesh
from texturizer import apply_fuzzy_skin

def create_simple_cube(size=20):
    """Create a simple cube mesh."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
    ]) * size

    faces = np.array([
        [0, 3, 1], [1, 3, 2],  # bottom
        [4, 5, 7], [5, 6, 7],  # top
        [0, 1, 4], [1, 5, 4],  # front
        [2, 3, 6], [3, 7, 6],  # back
        [0, 4, 3], [3, 4, 7],  # left
        [1, 2, 5], [2, 6, 5],  # right
    ])

    # Create mesh
    cube_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cube_mesh.vectors[i][j] = vertices[face[j]]

    return cube_mesh

def test_inplane_vs_normal_noise():
    """Test that in-plane noise only displaces in XY, while normal noise displaces in all directions."""
    print("=" * 70)
    print("TESTING: In-plane noise (XY only) vs Normal noise (all directions)")
    print("=" * 70)

    # Create test cube
    print("\n1. Creating test cube...")
    test_cube = create_simple_cube(size=20)

    # Get original Z coordinates
    original_z_coords = test_cube.vectors[:, :, 2].copy()
    print(f"   Original Z range: {original_z_coords.min():.3f} to {original_z_coords.max():.3f}")

    # Apply in-plane noise
    print("\n2. Applying in-plane noise (in_plane_noise=True)...")
    inplane_mesh = apply_fuzzy_skin(
        test_cube,
        thickness=0.5,
        point_distance=0.8,
        seed=42,
        in_plane_noise=True
    )

    inplane_z_coords = inplane_mesh.vectors[:, :, 2]
    print(f"   In-plane Z range: {inplane_z_coords.min():.3f} to {inplane_z_coords.max():.3f}")
    print(f"   Z change: {abs(inplane_z_coords.max() - original_z_coords.max()):.6f}mm")

    # Apply normal noise
    print("\n3. Applying normal noise (in_plane_noise=False)...")
    normal_mesh = apply_fuzzy_skin(
        test_cube,
        thickness=0.5,
        point_distance=0.8,
        seed=42,
        in_plane_noise=False
    )

    normal_z_coords = normal_mesh.vectors[:, :, 2]
    print(f"   Normal Z range: {normal_z_coords.min():.3f} to {normal_z_coords.max():.3f}")
    print(f"   Z change: {abs(normal_z_coords.max() - original_z_coords.max()):.6f}mm")

    # Calculate XY displacement
    print("\n4. Checking XY displacement...")

    # For in-plane noise
    inplane_xy = inplane_mesh.vectors[:, :, :2]
    original_xy = test_cube.vectors[:, :, :2]
    inplane_xy_displacement = np.linalg.norm(inplane_xy - original_xy, axis=2).max()
    print(f"   In-plane max XY displacement: {inplane_xy_displacement:.6f}mm")

    # For normal noise
    normal_xy = normal_mesh.vectors[:, :, :2]
    normal_xy_displacement = np.linalg.norm(normal_xy - original_xy, axis=2).max()
    print(f"   Normal max XY displacement: {normal_xy_displacement:.6f}mm")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & VERIFICATION")
    print("=" * 70)

    # Check if in-plane has minimal Z change (should be close to original due to subdivision only)
    z_threshold = 0.1  # Small threshold to account for subdivision artifacts
    inplane_z_ok = abs(inplane_z_coords.max() - original_z_coords.max()) < z_threshold

    # Check if normal noise has significant Z change
    normal_z_changed = abs(normal_z_coords.max() - original_z_coords.max()) > 0.2

    # Check if both have XY displacement
    both_have_xy = inplane_xy_displacement > 0.1 and normal_xy_displacement > 0.1

    if inplane_z_ok:
        print("✓ In-plane noise: Z coordinates preserved (XY only displacement)")
    else:
        print(f"✗ In-plane noise: Z coordinates changed by {abs(inplane_z_coords.max() - original_z_coords.max()):.6f}mm (expected < {z_threshold}mm)")

    if normal_z_changed:
        print("✓ Normal noise: Z coordinates changed (3D displacement)")
    else:
        print("✗ Normal noise: Z coordinates unchanged (expected change)")

    if both_have_xy:
        print("✓ Both modes: XY displacement applied")
    else:
        print("✗ XY displacement issue detected")

    print("\nConclusion:")
    if inplane_z_ok and normal_z_changed and both_have_xy:
        print("✓ PASSED: In-plane noise correctly applies only XY displacement")
        print("          (similar to OrcaSlicer fuzzy skin behavior)")
    else:
        print("✗ FAILED: In-plane noise behavior needs adjustment")

    print("=" * 70)

    # Save meshes for visual inspection
    print("\nSaving meshes for visual inspection...")
    inplane_mesh.save('test_inplane_output.stl')
    normal_mesh.save('test_normal_output.stl')
    print("Saved: test_inplane_output.stl")
    print("Saved: test_normal_output.stl")

if __name__ == "__main__":
    try:
        test_inplane_vs_normal_noise()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
