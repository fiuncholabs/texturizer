#!/usr/bin/env python3
"""
Fuzzy Skin Texture Generator for STL files
Applies random surface displacement similar to slicer "fuzzy skin" feature
with multiple noise types matching OrcaSlicer's implementation.

Requires: numpy, numpy-stl, noise
Install: pip install numpy numpy-stl noise
"""

import numpy as np
from stl import mesh
import argparse
import sys
import math

# Try to import noise library for advanced noise types
try:
    from noise import pnoise3, snoise3
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False
    print("Warning: 'noise' library not installed. Only 'classic' noise type available.")
    print("Install with: pip install noise")


# Noise type constants matching OrcaSlicer
NOISE_CLASSIC = 'classic'
NOISE_PERLIN = 'perlin'
NOISE_BILLOW = 'billow'
NOISE_RIDGED = 'ridged'
NOISE_VORONOI = 'voronoi'

NOISE_TYPES = [NOISE_CLASSIC, NOISE_PERLIN, NOISE_BILLOW, NOISE_RIDGED, NOISE_VORONOI]


class NoiseGenerator:
    """Noise generator matching OrcaSlicer's fuzzy skin noise types."""

    def __init__(self, noise_type=NOISE_CLASSIC, scale=1.0, octaves=4, persistence=0.5, seed=42):
        self.noise_type = noise_type
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Pre-generate Voronoi points if needed
        if noise_type == NOISE_VORONOI:
            self._init_voronoi()

    def _init_voronoi(self, num_points=100):
        """Initialize Voronoi cell centers."""
        self.voronoi_points = self.rng.rand(num_points, 3) * 10

    def get_noise(self, x, y, z):
        """Get noise value at position, returns value in [-1, 1]."""
        # Apply scale
        sx = x * self.scale
        sy = y * self.scale
        sz = z * self.scale

        if self.noise_type == NOISE_CLASSIC:
            # Uniform random noise based on position hash
            return self.rng.uniform(-1, 1)

        elif self.noise_type == NOISE_PERLIN:
            if not NOISE_AVAILABLE:
                return self.rng.uniform(-1, 1)
            return pnoise3(sx, sy, sz, octaves=self.octaves,
                          persistence=self.persistence, base=self.seed)

        elif self.noise_type == NOISE_BILLOW:
            if not NOISE_AVAILABLE:
                return self.rng.uniform(-1, 1)
            # Billow is absolute value of Perlin, creates cloud-like effect
            value = pnoise3(sx, sy, sz, octaves=self.octaves,
                           persistence=self.persistence, base=self.seed)
            return abs(value) * 2 - 1  # Map [0, 1] back to [-1, 1]

        elif self.noise_type == NOISE_RIDGED:
            if not NOISE_AVAILABLE:
                return self.rng.uniform(-1, 1)
            # Ridged multifractal: 1 - abs(noise), creates sharp ridges
            value = pnoise3(sx, sy, sz, octaves=self.octaves,
                           persistence=self.persistence, base=self.seed)
            return (1 - abs(value)) * 2 - 1

        elif self.noise_type == NOISE_VORONOI:
            # Voronoi: distance to nearest cell center
            pos = np.array([sx, sy, sz])
            distances = np.linalg.norm(self.voronoi_points - pos, axis=1)
            min_dist = np.min(distances)
            # Normalize to [-1, 1] range
            return (min_dist / 2.0) * 2 - 1

        return 0.0

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
    Uses iterative approach with a queue for better performance.
    """
    # Use a queue for iterative processing instead of recursion
    queue = [(v0, v1, v2)]
    result = []

    while queue:
        v0, v1, v2 = queue.pop()

        # Calculate edge lengths
        e0 = np.linalg.norm(v1 - v0)
        e1 = np.linalg.norm(v2 - v1)
        e2 = np.linalg.norm(v0 - v2)
        max_edge = max(e0, e1, e2)

        # Base case: triangle is small enough
        if max_edge <= max_edge_length:
            result.append((v0.copy(), v1.copy(), v2.copy()))
        else:
            # Subdivide by splitting all edges at midpoints (4-way split)
            m01 = (v0 + v1) / 2
            m12 = (v1 + v2) / 2
            m20 = (v2 + v0) / 2

            # Add 4 new triangles to queue
            queue.append((v0, m01, m20))
            queue.append((m01, v1, m12))
            queue.append((m20, m12, v2))
            queue.append((m01, m12, m20))

    return result

def apply_fuzzy_skin(input_mesh, thickness=0.3, point_distance=0.8, seed=42,
                     noise_type=NOISE_CLASSIC, noise_scale=1.0, noise_octaves=4,
                     noise_persistence=0.5, skip_bottom=False):
    """
    Apply fuzzy skin texture to mesh by subdividing and displacing vertices.

    Args:
        input_mesh: numpy-stl mesh object
        thickness: Maximum displacement distance (mm)
        point_distance: Target distance between texture points (mm)
        seed: Random seed for reproducibility
        noise_type: Type of noise ('classic', 'perlin', 'billow', 'ridged', 'voronoi')
        noise_scale: Frequency scale for noise (higher = more detail)
        noise_octaves: Number of octaves for Perlin/Billow noise
        noise_persistence: Amplitude persistence for Perlin/Billow noise
        skip_bottom: If True, skip fuzzy skin on bottom layer (z ≈ min_z)
    """
    np.random.seed(seed)

    # Create noise generator
    noise_gen = NoiseGenerator(
        noise_type=noise_type,
        scale=noise_scale,
        octaves=noise_octaves,
        persistence=noise_persistence,
        seed=seed
    )

    print(f"Subdividing mesh (target edge length: {point_distance}mm)...")

    # Subdivide all triangles
    all_triangles = []
    for face in input_mesh.vectors:
        v0, v1, v2 = face
        subdivided = subdivide_triangle(v0, v1, v2, point_distance)
        all_triangles.extend(subdivided)

    print(f"Subdivided {len(input_mesh.vectors)} triangles into {len(all_triangles)} triangles")

    # Create new mesh from subdivided triangles (vectorized)
    output_mesh = mesh.Mesh(np.zeros(len(all_triangles), dtype=mesh.Mesh.dtype))
    triangle_array = np.array(all_triangles)
    output_mesh.vectors = triangle_array

    # Build vertex map to find shared vertices
    vertices = output_mesh.vectors.reshape(-1, 3)
    # Round to avoid floating point issues when finding unique vertices
    rounded = np.round(vertices, decimals=6)
    unique_vertices, inverse_indices = np.unique(
        rounded, axis=0, return_inverse=True
    )

    print(f"Processing {len(unique_vertices)} unique vertices...")

    # Find min Z for bottom layer detection
    min_z = np.min(unique_vertices[:, 2])
    bottom_threshold = min_z + 0.1  # 0.1mm tolerance for bottom layer

    # Calculate vertex normals by averaging face normals (vectorized)
    vertex_normals = np.zeros_like(unique_vertices)
    vertex_counts = np.zeros(len(unique_vertices))

    # Vectorized face normal calculation
    v0 = output_mesh.vectors[:, 0]
    v1 = output_mesh.vectors[:, 1]
    v2 = output_mesh.vectors[:, 2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = np.divide(face_normals, norms, out=np.zeros_like(face_normals), where=norms[:, 0:1] > 0)

    # Accumulate face normals to vertex normals
    indices_reshaped = inverse_indices.reshape(-1, 3)
    for j in range(3):
        np.add.at(vertex_normals, indices_reshaped[:, j], face_normals)
        np.add.at(vertex_counts, indices_reshaped[:, j], 1)

    # Normalize vertex normals (vectorized)
    mask = vertex_counts > 0
    vertex_normals[mask] /= vertex_counts[mask, np.newaxis]
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = np.divide(vertex_normals, norms, out=np.zeros_like(vertex_normals), where=norms > 0)

    # Apply displacement to each unique vertex using noise
    displaced_vertices = unique_vertices.copy()

    # Determine which vertices to process
    if skip_bottom:
        process_mask = unique_vertices[:, 2] > bottom_threshold
        skipped_count = np.sum(~process_mask)
    else:
        process_mask = np.ones(len(unique_vertices), dtype=bool)
        skipped_count = 0

    # Get noise values for all vertices to process
    noise_values = np.zeros(len(unique_vertices))
    for i in np.where(process_mask)[0]:
        vertex = unique_vertices[i]
        noise_values[i] = noise_gen.get_noise(vertex[0], vertex[1], vertex[2])

    # Map noise from [-1, 1] to [0, thickness] and apply displacement (vectorized)
    displacement_amounts = (noise_values + 1) * 0.5 * thickness
    displaced_vertices += vertex_normals * displacement_amounts[:, np.newaxis]

    # Zero out displacement for skipped vertices
    if skip_bottom:
        displaced_vertices[~process_mask] = unique_vertices[~process_mask]

    if skip_bottom and skipped_count > 0:
        print(f"Skipped {skipped_count} bottom layer vertices")

    # Update mesh with displaced vertices (vectorized)
    output_mesh.vectors = displaced_vertices[inverse_indices].reshape(-1, 3, 3)

    return output_mesh

def main():
    parser = argparse.ArgumentParser(
        description='Apply fuzzy skin texture to STL files (OrcaSlicer-compatible)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Noise Types:
  classic   - Uniform random noise (default, fast)
  perlin    - Perlin noise (smooth, organic patterns)
  billow    - Cloud-like patterns (abs of Perlin)
  ridged    - Sharp ridges (inverted Perlin)
  voronoi   - Cell-based patterns

Examples:
  %(prog)s model.stl -o fuzzy.stl
  %(prog)s model.stl -t 0.5 -p 0.6 --noise perlin
  %(prog)s model.stl --noise voronoi --noise-scale 2.0
  %(prog)s --cube-size 30 -t 0.4 --skip-bottom
"""
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

    # Advanced noise options matching OrcaSlicer
    parser.add_argument(
        '--noise',
        choices=NOISE_TYPES,
        default=NOISE_CLASSIC,
        help='Noise type (default: classic)'
    )
    parser.add_argument(
        '--noise-scale',
        type=float,
        default=1.0,
        help='Noise frequency scale - higher = more detail (default: 1.0)'
    )
    parser.add_argument(
        '--noise-octaves',
        type=int,
        default=4,
        help='Number of noise octaves for Perlin/Billow (default: 4)'
    )
    parser.add_argument(
        '--noise-persistence',
        type=float,
        default=0.5,
        help='Amplitude persistence for Perlin/Billow (default: 0.5)'
    )
    parser.add_argument(
        '--skip-bottom',
        action='store_true',
        help='Skip fuzzy skin on bottom layer (for better bed adhesion)'
    )
    parser.add_argument(
        '--ascii',
        action='store_true',
        help='Save output as ASCII STL instead of binary'
    )

    args = parser.parse_args()

    # Validate noise type availability
    if args.noise != NOISE_CLASSIC and not NOISE_AVAILABLE:
        print(f"Error: Noise type '{args.noise}' requires the 'noise' library.")
        print("Install with: pip install noise")
        print("Or use --noise classic")
        sys.exit(1)

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
    noise_info = f"noise={args.noise}"
    if args.noise in [NOISE_PERLIN, NOISE_BILLOW, NOISE_RIDGED]:
        noise_info += f", scale={args.noise_scale}, octaves={args.noise_octaves}"
    elif args.noise == NOISE_VORONOI:
        noise_info += f", scale={args.noise_scale}"

    print(f"Applying fuzzy skin (thickness={args.thickness}mm, point_distance={args.point_distance}mm, {noise_info})...")

    output_mesh = apply_fuzzy_skin(
        input_mesh,
        thickness=args.thickness,
        point_distance=args.point_distance,
        seed=args.seed,
        noise_type=args.noise,
        noise_scale=args.noise_scale,
        noise_octaves=args.noise_octaves,
        noise_persistence=args.noise_persistence,
        skip_bottom=args.skip_bottom
    )

    # Save result
    print(f"Saving to {args.output}...")
    if args.ascii:
        output_mesh.save(args.output, mode=mesh.Mode.ASCII)
    else:
        output_mesh.save(args.output)

    print("Done!")

if __name__ == '__main__':
    main()