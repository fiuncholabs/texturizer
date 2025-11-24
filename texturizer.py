#!/usr/bin/env python3
"""
Fuzzy Skin Texture Generator for STL files
Applies random surface displacement similar to slicer "fuzzy skin" feature
Requires: numpy, numpy-stl
Install: pip install numpy numpy-stl
"""

import numpy as np
from stl import mesh
import argparse
import sys

def generate_test_cube(size=20):
    """Generate a test cube STL mesh"""
    # Define the 8 vertices of a cube
    vertices = np.array([
        [-size/2, -size/2, -size/2],
        [+size/2, -size/2, -size/2],
        [+size/2, +size/2, -size/2],
        [-size/2, +size/2, -size/2],
        [-size/2, -size/2, +size/2],
        [+size/2, -size/2, +size/2],
        [+size/2, +size/2, +size/2],
        [-size/2, +size/2, +size/2]
    ])
    
    # Define the 12 triangles (2 per face)
    faces = np.array([
        [0,3,1], [1,3,2],  # Bottom
        [4,5,7], [5,6,7],  # Top
        [0,1,5], [0,5,4],  # Front
        [2,3,7], [2,7,6],  # Back
        [0,4,7], [0,7,3],  # Left
        [1,2,6], [1,6,5]   # Right
    ])
    
    # Create mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[face[j],:]
    
    return cube

def subdivide_triangle(v0, v1, v2, max_edge_length):
    """
    Recursively subdivide a triangle until all edges are below max_edge_length.
    Returns a list of triangles (each triangle is a tuple of 3 vertices).
    """
    # Calculate edge lengths
    e0 = np.linalg.norm(v1 - v0)
    e1 = np.linalg.norm(v2 - v1)
    e2 = np.linalg.norm(v0 - v2)
    max_edge = max(e0, e1, e2)

    # Base case: triangle is small enough
    if max_edge <= max_edge_length:
        return [(v0.copy(), v1.copy(), v2.copy())]

    # Subdivide by splitting all edges at midpoints (4-way split)
    m01 = (v0 + v1) / 2
    m12 = (v1 + v2) / 2
    m20 = (v2 + v0) / 2

    # Recursively subdivide the 4 new triangles
    triangles = []
    triangles.extend(subdivide_triangle(v0, m01, m20, max_edge_length))
    triangles.extend(subdivide_triangle(m01, v1, m12, max_edge_length))
    triangles.extend(subdivide_triangle(m20, m12, v2, max_edge_length))
    triangles.extend(subdivide_triangle(m01, m12, m20, max_edge_length))

    return triangles

def apply_fuzzy_skin(input_mesh, thickness=0.3, point_distance=0.8, seed=42):
    """
    Apply fuzzy skin texture to mesh by subdividing and displacing vertices.

    Args:
        input_mesh: numpy-stl mesh object
        thickness: Maximum displacement distance (mm)
        point_distance: Target distance between texture points (mm)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    print(f"Subdividing mesh (target edge length: {point_distance}mm)...")

    # Subdivide all triangles
    all_triangles = []
    for face in input_mesh.vectors:
        v0, v1, v2 = face
        subdivided = subdivide_triangle(v0, v1, v2, point_distance)
        all_triangles.extend(subdivided)

    print(f"Subdivided {len(input_mesh.vectors)} triangles into {len(all_triangles)} triangles")

    # Create new mesh from subdivided triangles
    output_mesh = mesh.Mesh(np.zeros(len(all_triangles), dtype=mesh.Mesh.dtype))
    for i, (v0, v1, v2) in enumerate(all_triangles):
        output_mesh.vectors[i][0] = v0
        output_mesh.vectors[i][1] = v1
        output_mesh.vectors[i][2] = v2

    # Build vertex map to find shared vertices
    vertices = output_mesh.vectors.reshape(-1, 3)
    # Round to avoid floating point issues when finding unique vertices
    rounded = np.round(vertices, decimals=6)
    unique_vertices, inverse_indices = np.unique(
        rounded, axis=0, return_inverse=True
    )

    print(f"Processing {len(unique_vertices)} unique vertices...")

    # Calculate vertex normals by averaging face normals
    vertex_normals = np.zeros_like(unique_vertices)
    vertex_counts = np.zeros(len(unique_vertices))

    for i in range(len(output_mesh.vectors)):
        # Calculate face normal
        v0, v1, v2 = output_mesh.vectors[i]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(face_normal)
        if norm > 0:
            face_normal = face_normal / norm

        # Add to vertex normals
        for j in range(3):
            vertex_idx = inverse_indices[i * 3 + j]
            vertex_normals[vertex_idx] += face_normal
            vertex_counts[vertex_idx] += 1

    # Normalize vertex normals
    for i in range(len(vertex_normals)):
        if vertex_counts[i] > 0:
            vertex_normals[i] /= vertex_counts[i]
            norm = np.linalg.norm(vertex_normals[i])
            if norm > 0:
                vertex_normals[i] /= norm

    # Apply random displacement to each unique vertex
    displaced_vertices = unique_vertices.copy()
    for i in range(len(unique_vertices)):
        # Random displacement amount (0 to thickness)
        displacement_amount = np.random.random() * thickness
        displaced_vertices[i] += vertex_normals[i] * displacement_amount

    # Update mesh with displaced vertices
    for i in range(len(output_mesh.vectors)):
        for j in range(3):
            vertex_idx = inverse_indices[i * 3 + j]
            output_mesh.vectors[i][j] = displaced_vertices[vertex_idx]

    return output_mesh

def main():
    parser = argparse.ArgumentParser(
        description='Apply fuzzy skin texture to STL files'
    )
    parser.add_argument(
        'input',
        nargs='?',
        help='Input STL file (omit to generate test cube)'
    )
    parser.add_argument(
        '-o', '--output',
        default='fuzzy_output.stl',
        help='Output STL file (default: fuzzy_output.stl)'
    )
    parser.add_argument(
        '-t', '--thickness',
        type=float,
        default=0.3,
        help='Fuzzy skin thickness in mm (default: 0.3)'
    )
    parser.add_argument(
        '-p', '--point-distance',
        type=float,
        default=0.8,
        help='Distance between texture points in mm (default: 0.8)'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--cube-size',
        type=float,
        default=20,
        help='Test cube size in mm (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Load or generate mesh
    if args.input:
        print(f"Loading {args.input}...")
        try:
            input_mesh = mesh.Mesh.from_file(args.input)
        except Exception as e:
            print(f"Error loading STL file: {e}")
            sys.exit(1)
    else:
        print(f"Generating {args.cube_size}mm test cube...")
        input_mesh = generate_test_cube(args.cube_size)
    
    print(f"Input mesh: {len(input_mesh.vectors)} triangles")
    
    # Apply fuzzy skin
    print(f"Applying fuzzy skin (thickness={args.thickness}mm, point_distance={args.point_distance}mm)...")
    output_mesh = apply_fuzzy_skin(
        input_mesh,
        thickness=args.thickness,
        point_distance=args.point_distance,
        seed=args.seed
    )
    
    # Save result
    print(f"Saving to {args.output}...")
    output_mesh.save(args.output)
    print("Done!")

if __name__ == '__main__':
    main()