#!/usr/bin/env python3
"""Test CLI options for feature parity with web UI."""

import sys
import subprocess
import numpy as np
from stl import mesh
from texturizer import apply_fuzzy_skin, generate_test_cube

def create_test_cube():
    """Create a simple test cube."""
    return generate_test_cube(size=10)

def test_skip_small_triangles():
    """Test --skip-small-triangles option."""
    print("=" * 70)
    print("TEST: --skip-small-triangles option")
    print("=" * 70)

    test_cube = create_test_cube()

    # Without skip_small_triangles
    print("\n1. Processing without --skip-small-triangles...")
    mesh_no_skip = apply_fuzzy_skin(
        test_cube,
        thickness=0.3,
        point_distance=0.8,
        seed=42,
        skip_small_triangles=False
    )
    count_no_skip = len(mesh_no_skip.vectors)
    print(f"   Triangle count: {count_no_skip}")

    # With skip_small_triangles
    print("\n2. Processing with --skip-small-triangles...")
    mesh_skip = apply_fuzzy_skin(
        test_cube,
        thickness=0.3,
        point_distance=0.8,
        seed=42,
        skip_small_triangles=True
    )
    count_skip = len(mesh_skip.vectors)
    print(f"   Triangle count: {count_skip}")

    # Verify that skip_small_triangles reduces triangle count (or is same)
    print("\n3. Verification:")
    if count_skip <= count_no_skip:
        print(f"   ✓ PASSED: skip_small_triangles works ({count_skip} <= {count_no_skip})")
        return True
    else:
        print(f"   ✗ FAILED: skip_small_triangles increased triangles ({count_skip} > {count_no_skip})")
        return False

def test_noise_on_edges():
    """Test --noise-on-edges option."""
    print("\n" + "=" * 70)
    print("TEST: --noise-on-edges option")
    print("=" * 70)

    test_cube = create_test_cube()

    # Vertex-based noise
    print("\n1. Processing with vertex-based noise...")
    mesh_vertex = apply_fuzzy_skin(
        test_cube,
        thickness=0.3,
        point_distance=0.8,
        seed=42,
        noise_on_edges=False
    )
    print(f"   Triangle count: {len(mesh_vertex.vectors)}")

    # Edge-based noise
    print("\n2. Processing with edge-based noise...")
    mesh_edge = apply_fuzzy_skin(
        test_cube,
        thickness=0.3,
        point_distance=0.8,
        seed=42,
        noise_on_edges=True
    )
    print(f"   Triangle count: {len(mesh_edge.vectors)}")

    # Verify both produce valid meshes
    print("\n3. Verification:")
    vertex_valid = not (np.any(np.isnan(mesh_vertex.vectors)) or np.any(np.isinf(mesh_vertex.vectors)))
    edge_valid = not (np.any(np.isnan(mesh_edge.vectors)) or np.any(np.isinf(mesh_edge.vectors)))

    if vertex_valid and edge_valid:
        print("   ✓ PASSED: Both vertex and edge-based noise produce valid meshes")
        return True
    else:
        print(f"   ✗ FAILED: Invalid meshes (vertex={vertex_valid}, edge={edge_valid})")
        return False

def test_in_plane_noise():
    """Test --in-plane-noise option."""
    print("\n" + "=" * 70)
    print("TEST: --in-plane-noise option")
    print("=" * 70)

    test_cube = create_test_cube()
    original_z_min = test_cube.vectors[:, :, 2].min()
    original_z_max = test_cube.vectors[:, :, 2].max()

    # Normal 3D noise
    print("\n1. Processing with normal (3D) noise...")
    mesh_3d = apply_fuzzy_skin(
        test_cube,
        thickness=0.3,
        point_distance=0.8,
        seed=42,
        in_plane_noise=False
    )
    z_3d_min = mesh_3d.vectors[:, :, 2].min()
    z_3d_max = mesh_3d.vectors[:, :, 2].max()
    z_3d_change = max(abs(z_3d_min - original_z_min), abs(z_3d_max - original_z_max))
    print(f"   Z range: {z_3d_min:.3f} to {z_3d_max:.3f}")
    print(f"   Z change: {z_3d_change:.3f}mm")

    # In-plane (XY only) noise
    print("\n2. Processing with in-plane (XY only) noise...")
    mesh_xy = apply_fuzzy_skin(
        test_cube,
        thickness=0.3,
        point_distance=0.8,
        seed=42,
        in_plane_noise=True
    )
    z_xy_min = mesh_xy.vectors[:, :, 2].min()
    z_xy_max = mesh_xy.vectors[:, :, 2].max()
    z_xy_change = max(abs(z_xy_min - original_z_min), abs(z_xy_max - original_z_max))
    print(f"   Z range: {z_xy_min:.3f} to {z_xy_max:.3f}")
    print(f"   Z change: {z_xy_change:.3f}mm")

    # Verify in-plane preserves Z coordinates better than 3D
    print("\n3. Verification:")
    threshold = 0.15  # Allow small subdivision artifacts

    if z_xy_change < threshold:
        print(f"   ✓ PASSED: in-plane noise preserves Z coordinates (change={z_xy_change:.3f}mm < {threshold}mm)")
        return True
    else:
        print(f"   ✗ FAILED: in-plane noise changed Z too much (change={z_xy_change:.3f}mm >= {threshold}mm)")
        return False

def test_xy_plane_subdivision():
    """Test --xy-plane-subdivision option."""
    print("\n" + "=" * 70)
    print("TEST: --xy-plane-subdivision option")
    print("=" * 70)

    # Create a tall narrow box (vertical wall heavy)
    vertices = np.array([
        [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],  # bottom
        [0, 0, 20], [2, 0, 20], [2, 2, 20], [0, 2, 20],  # top (tall)
    ])

    faces = np.array([
        [0, 3, 1], [1, 3, 2],  # bottom
        [4, 5, 7], [5, 6, 7],  # top
        [0, 1, 4], [1, 5, 4],  # front (tall vertical wall)
        [2, 3, 6], [3, 7, 6],  # back (tall vertical wall)
        [0, 4, 3], [3, 4, 7],  # left (tall vertical wall)
        [1, 2, 5], [2, 6, 5],  # right (tall vertical wall)
    ])

    tall_box = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            tall_box.vectors[i][j] = vertices[face[j]]

    # 3D subdivision
    print("\n1. Processing with 3D subdivision...")
    mesh_3d = apply_fuzzy_skin(
        tall_box,
        thickness=0.3,
        point_distance=0.8,
        seed=42,
        xy_plane_subdivision=False
    )
    count_3d = len(mesh_3d.vectors)
    print(f"   Triangle count: {count_3d}")

    # XY-plane subdivision
    print("\n2. Processing with XY-plane subdivision...")
    mesh_xy = apply_fuzzy_skin(
        tall_box,
        thickness=0.3,
        point_distance=0.8,
        seed=42,
        xy_plane_subdivision=True
    )
    count_xy = len(mesh_xy.vectors)
    print(f"   Triangle count: {count_xy}")

    # Verify XY-plane subdivision produces fewer triangles on vertical walls
    print("\n3. Verification:")
    reduction = (count_3d - count_xy) / count_3d * 100 if count_3d > 0 else 0

    if count_xy < count_3d:
        print(f"   ✓ PASSED: XY-plane subdivision reduces triangles on vertical walls")
        print(f"     3D: {count_3d} triangles, XY: {count_xy} triangles ({reduction:.1f}% reduction)")
        return True
    else:
        print(f"   ✗ FAILED: XY-plane subdivision didn't reduce triangles ({count_xy} >= {count_3d})")
        return False

def test_simplify_option():
    """Test --simplify option."""
    print("\n" + "=" * 70)
    print("TEST: --simplify option")
    print("=" * 70)

    try:
        import pyfqmr
        PYFQMR_AVAILABLE = True
    except ImportError:
        PYFQMR_AVAILABLE = False
        print("\n   ⚠ SKIPPED: pyfqmr not installed")
        return True

    from texturizer import simplify_mesh

    test_cube = create_test_cube()

    # Generate fuzzy mesh
    print("\n1. Generating fuzzy mesh...")
    fuzzy_mesh = apply_fuzzy_skin(
        test_cube,
        thickness=0.3,
        point_distance=0.8,
        seed=42
    )
    original_count = len(fuzzy_mesh.vectors)
    print(f"   Original triangle count: {original_count}")

    # Simplify by 50%
    print("\n2. Simplifying mesh by 50%...")
    simplified = simplify_mesh(fuzzy_mesh, target_reduction=0.5)
    simplified_count = len(simplified.vectors)
    actual_reduction = (original_count - simplified_count) / original_count
    print(f"   Simplified triangle count: {simplified_count}")
    print(f"   Actual reduction: {actual_reduction*100:.1f}%")

    # Verify simplification reduced triangles
    print("\n3. Verification:")
    if simplified_count < original_count:
        print(f"   ✓ PASSED: Simplification reduced triangles ({simplified_count} < {original_count})")
        return True
    else:
        print(f"   ✗ FAILED: Simplification didn't reduce triangles ({simplified_count} >= {original_count})")
        return False

def test_cli_help():
    """Test that CLI --help shows all new options."""
    print("\n" + "=" * 70)
    print("TEST: CLI --help includes all new options")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, "texturizer.py", "--help"],
        capture_output=True,
        text=True
    )

    help_text = result.stdout

    required_options = [
        "--skip-small-triangles",
        "--noise-on-edges",
        "--in-plane-noise",
        "--xy-plane-subdivision",
        "--simplify"
    ]

    print("\nChecking for required options in --help output:")
    all_present = True
    for option in required_options:
        present = option in help_text
        status = "✓" if present else "✗"
        print(f"   {status} {option}")
        if not present:
            all_present = False

    if all_present:
        print("\n   ✓ PASSED: All new options present in --help")
        return True
    else:
        print("\n   ✗ FAILED: Some options missing from --help")
        return False

def test_orcaslicer_combo():
    """Test OrcaSlicer-style combination of options."""
    print("\n" + "=" * 70)
    print("TEST: OrcaSlicer-style combination (all options together)")
    print("=" * 70)

    test_cube = create_test_cube()

    print("\n1. Processing with OrcaSlicer-style options...")
    print("   (noise_on_edges + in_plane_noise + xy_plane_subdivision)")

    try:
        orca_mesh = apply_fuzzy_skin(
            test_cube,
            thickness=0.3,
            point_distance=0.8,
            seed=42,
            noise_on_edges=True,
            in_plane_noise=True,
            xy_plane_subdivision=True,
            skip_small_triangles=True
        )

        triangle_count = len(orca_mesh.vectors)
        print(f"   Triangle count: {triangle_count}")

        # Verify mesh is valid
        is_valid = not (np.any(np.isnan(orca_mesh.vectors)) or np.any(np.isinf(orca_mesh.vectors)))

        if is_valid:
            print("\n2. Verification:")
            print("   ✓ PASSED: OrcaSlicer-style combination produces valid mesh")
            return True
        else:
            print("\n2. Verification:")
            print("   ✗ FAILED: OrcaSlicer-style combination produced invalid mesh")
            return False

    except Exception as e:
        print(f"\n2. Verification:")
        print(f"   ✗ FAILED: Exception occurred: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CLI OPTIONS FEATURE PARITY TEST SUITE")
    print("=" * 70)

    tests = [
        ("CLI Help", test_cli_help),
        ("Skip Small Triangles", test_skip_small_triangles),
        ("Noise on Edges", test_noise_on_edges),
        ("In-Plane Noise", test_in_plane_noise),
        ("XY-Plane Subdivision", test_xy_plane_subdivision),
        ("Mesh Simplification", test_simplify_option),
        ("OrcaSlicer Combo", test_orcaslicer_combo),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n   ✗ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 70)
    print(f"Results: {passed_count}/{total_count} tests passed")
    print("=" * 70)

    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
