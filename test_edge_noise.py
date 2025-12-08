#!/usr/bin/env python3
"""Test edge-based noise to verify mesh remains solid."""

import numpy as np
from stl import mesh
import trimesh
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

def check_mesh_is_solid(stl_mesh):
    """Check if a mesh is a valid solid using trimesh."""
    # Convert STL mesh to trimesh
    vertices = stl_mesh.vectors.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)

    # Create trimesh object
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Merge duplicate vertices
    tm.merge_vertices()

    # Check if mesh is watertight
    is_watertight = tm.is_watertight
    is_volume = tm.is_volume

    # Check for broken faces/edges
    broken_faces = len(tm.faces) > 0 and len(tm.vertices) > 0

    return is_watertight, is_volume, broken_faces, tm

def test_edge_noise():
    """Test that edge-based noise maintains mesh solidity."""
    print("=" * 70)
    print("TESTING: Edge-based noise mesh solidity")
    print("=" * 70)

    # Create test cube
    print("\n1. Creating test cube...")
    test_cube = create_simple_cube(size=20)
    print(f"   Input mesh: {len(test_cube.vectors)} triangles")

    # Check original mesh
    print("\n2. Checking original mesh...")
    orig_watertight, orig_volume, orig_has_faces, orig_tm = check_mesh_is_solid(test_cube)
    print(f"   Original - Watertight: {orig_watertight}, Is Volume: {orig_volume}, Has Faces: {orig_has_faces}")
    print(f"   Original - Vertices: {len(orig_tm.vertices)}, Faces: {len(orig_tm.faces)}")

    # Apply edge-based noise
    print("\n3. Applying edge-based noise (noise_on_edges=True)...")
    edge_mesh = apply_fuzzy_skin(
        test_cube,
        thickness=0.3,
        point_distance=0.8,
        seed=42,
        noise_on_edges=True
    )
    print(f"   Output mesh: {len(edge_mesh.vectors)} triangles")

    # Check edge-based mesh
    print("\n4. Checking edge-based noise mesh...")
    edge_watertight, edge_volume, edge_has_faces, edge_tm = check_mesh_is_solid(edge_mesh)
    print(f"   Edge-based - Watertight: {edge_watertight}, Is Volume: {edge_volume}, Has Faces: {edge_has_faces}")
    print(f"   Edge-based - Vertices: {len(edge_tm.vertices)}, Faces: {len(edge_tm.faces)}")

    # Apply vertex-based noise for comparison
    print("\n5. Applying vertex-based noise (noise_on_edges=False) for comparison...")
    vertex_mesh = apply_fuzzy_skin(
        test_cube,
        thickness=0.3,
        point_distance=0.8,
        seed=42,
        noise_on_edges=False
    )
    print(f"   Output mesh: {len(vertex_mesh.vectors)} triangles")

    # Check vertex-based mesh
    print("\n6. Checking vertex-based noise mesh...")
    vertex_watertight, vertex_volume, vertex_has_faces, vertex_tm = check_mesh_is_solid(vertex_mesh)
    print(f"   Vertex-based - Watertight: {vertex_watertight}, Is Volume: {vertex_volume}, Has Faces: {vertex_has_faces}")
    print(f"   Vertex-based - Vertices: {len(vertex_tm.vertices)}, Faces: {len(vertex_tm.faces)}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if edge_watertight and edge_volume:
        print("✓ Edge-based noise PASSED: Mesh is watertight and forms a volume")
    else:
        print("✗ Edge-based noise FAILED: Mesh is not solid")
        print(f"  Watertight: {edge_watertight}, Is Volume: {edge_volume}")

    if vertex_watertight and vertex_volume:
        print("✓ Vertex-based noise PASSED: Mesh is watertight and forms a volume")
    else:
        print("✗ Vertex-based noise FAILED: Mesh is not solid")
        print(f"  Watertight: {vertex_watertight}, Is Volume: {vertex_volume}")

    print("=" * 70)

    # Save meshes for visual inspection
    print("\nSaving meshes for visual inspection...")
    edge_mesh.save('test_edge_noise_output.stl')
    vertex_mesh.save('test_vertex_noise_output.stl')
    print("Saved: test_edge_noise_output.stl")
    print("Saved: test_vertex_noise_output.stl")

if __name__ == "__main__":
    try:
        test_edge_noise()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
