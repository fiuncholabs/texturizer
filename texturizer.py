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
    """
    Generate default test object: Fiuncholabs Fern Coin
    A coin-like piece with embossed fern design (Fiuncholabs branding)

    Args:
        size: Diameter of the coin (default 20mm for backward compatibility)

    Returns:
        mesh.Mesh object
    """
    diameter = size  # Use size parameter as diameter
    thickness = size * 0.075  # Proportional thickness
    fern_height = size * 0.006  # Reduced emboss height (was 0.02, now 0.006 = 30% depth)
    resolution = max(80, int(size * 4))  # Higher resolution for better detail

    radius = diameter / 2
    num_radial = max(30, int(size * 1.5))  # More radial steps for smoother embossing

    def fern_function(x, y):
        """Generate realistic fern pattern with detailed fronds (Fiuncholabs mascot)"""
        # Normalize coordinates
        r = np.sqrt(x**2 + y**2) / radius

        # Return 0 if outside the fern area
        if r > 0.85:
            return 0.0

        fern_value = 0.0

        # Main stem - curved slightly for natural look
        stem_width = 0.025
        stem_curve = x * 0.05  # Slight curve
        stem_y_start = -radius * 0.4
        stem_y_end = radius * 0.5

        if y > stem_y_start and y < stem_y_end:
            stem_center = stem_curve
            stem_dist = abs(x - stem_center) / radius
            if stem_dist < stem_width:
                # Smooth stem edges
                stem_strength = 1.0 - (stem_dist / stem_width) ** 2
                fern_value = max(fern_value, stem_strength * 0.9)

        # Generate realistic fronds with pinnae (leaflets)
        num_fronds = 12  # More fronds for realism

        for i in range(num_fronds):
            # Frond position along stem
            t = i / (num_fronds - 1)
            frond_y = stem_y_start + (stem_y_end - stem_y_start) * t

            # Frond length decreases toward top
            frond_length = radius * 0.4 * (1 - t * 0.6) * (1 - (1 - t) ** 3)

            # Frond angle (upward curve)
            frond_angle = 0.6 + t * 0.2  # Steeper angle near base

            dy = y - frond_y

            # Skip if not near this frond's y position
            if abs(dy) > radius * 0.15:
                continue

            # Right side frond
            if x > 0 and x < frond_length:
                # Main frond rachis (central axis)
                frond_center_y = frond_y + x * frond_angle
                dist_to_rachis = abs(y - frond_center_y) / radius

                # Frond width tapers toward tip
                frond_width = 0.015 * (1 - x / frond_length)

                if dist_to_rachis < frond_width:
                    # Distance along frond (0 to 1)
                    frond_progress = x / frond_length
                    rachis_strength = (1.0 - frond_progress * 0.5) * (1.0 - (dist_to_rachis / frond_width) ** 2)
                    fern_value = max(fern_value, rachis_strength * 0.8)

                # Add pinnae (leaflets along the frond)
                num_pinnae = int(6 * (1 - frond_progress * 0.5))
                for p in range(num_pinnae):
                    pinna_x = x * (p + 1) / (num_pinnae + 1)
                    pinna_y = frond_center_y
                    pinna_length = radius * 0.08 * (1 - frond_progress) * (1 - p / num_pinnae * 0.5)

                    # Alternating pinnae
                    pinna_angle = 1.2 if p % 2 == 0 else 0.8

                    dx_pinna = abs(x - pinna_x)
                    dy_pinna = y - pinna_y - dx_pinna * pinna_angle

                    if dx_pinna < pinna_length and abs(dy_pinna) < radius * 0.02:
                        pinna_progress = dx_pinna / pinna_length
                        pinna_strength = (1.0 - pinna_progress ** 2) * 0.6
                        fern_value = max(fern_value, pinna_strength)

            # Left side frond (mirror)
            if x < 0 and x > -frond_length:
                frond_center_y = frond_y + abs(x) * frond_angle
                dist_to_rachis = abs(y - frond_center_y) / radius

                frond_width = 0.015 * (1 - abs(x) / frond_length)

                if dist_to_rachis < frond_width:
                    frond_progress = abs(x) / frond_length
                    rachis_strength = (1.0 - frond_progress * 0.5) * (1.0 - (dist_to_rachis / frond_width) ** 2)
                    fern_value = max(fern_value, rachis_strength * 0.8)

                # Add pinnae
                num_pinnae = int(6 * (1 - frond_progress * 0.5))
                for p in range(num_pinnae):
                    pinna_x = x * (p + 1) / (num_pinnae + 1)
                    pinna_y = frond_center_y
                    pinna_length = radius * 0.08 * (1 - frond_progress) * (1 - p / num_pinnae * 0.5)

                    pinna_angle = 1.2 if p % 2 == 0 else 0.8

                    dx_pinna = abs(x - pinna_x)
                    dy_pinna = y - pinna_y - dx_pinna * pinna_angle

                    if dx_pinna < pinna_length and abs(dy_pinna) < radius * 0.02:
                        pinna_progress = dx_pinna / pinna_length
                        pinna_strength = (1.0 - pinna_progress ** 2) * 0.6
                        fern_value = max(fern_value, pinna_strength)

        # Smooth global falloff toward edges
        edge_falloff = 1.0
        if r > 0.7:
            edge_falloff = 1.0 - ((r - 0.7) / 0.15) ** 2
            edge_falloff = max(0.0, edge_falloff)

        fern_value *= edge_falloff

        # Apply embossing with smooth gradient
        return fern_value * fern_height

    # Generate mesh
    vertices_list = []

    # Angular resolution
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    # Bottom center point
    bottom_center_idx = 0
    vertices_list.append([0, 0, 0])

    # Bottom circle
    bottom_circle_start = len(vertices_list)
    for t in theta:
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        vertices_list.append([x, y, 0])

    # Top surface with embossing
    radial_steps = np.linspace(0, radius, num_radial)
    top_vertices_grid = []

    for r_step in radial_steps:
        ring = []
        for t in theta:
            x = r_step * np.cos(t)
            y = r_step * np.sin(t)
            z = thickness + fern_function(x, y)
            ring.append(len(vertices_list))
            vertices_list.append([x, y, z])
        top_vertices_grid.append(ring)

    vertices_array = np.array(vertices_list)

    # Create faces
    faces_list = []

    # Bottom surface
    for i in range(resolution):
        next_i = (i + 1) % resolution
        faces_list.append([bottom_center_idx, bottom_circle_start + next_i, bottom_circle_start + i])

    # Side walls
    for i in range(resolution):
        next_i = (i + 1) % resolution
        bottom_v1 = bottom_circle_start + i
        bottom_v2 = bottom_circle_start + next_i
        top_v1 = top_vertices_grid[-1][i]
        top_v2 = top_vertices_grid[-1][next_i]

        faces_list.append([bottom_v1, top_v1, bottom_v2])
        faces_list.append([bottom_v2, top_v1, top_v2])

    # Top surface (embossed fern)
    for r_idx in range(len(top_vertices_grid) - 1):
        for t_idx in range(resolution):
            next_t = (t_idx + 1) % resolution

            v1 = top_vertices_grid[r_idx][t_idx]
            v2 = top_vertices_grid[r_idx][next_t]
            v3 = top_vertices_grid[r_idx + 1][t_idx]
            v4 = top_vertices_grid[r_idx + 1][next_t]

            faces_list.append([v1, v3, v2])
            faces_list.append([v2, v3, v4])

    faces_array = np.array(faces_list)

    # Create mesh
    fern_coin = mesh.Mesh(np.zeros(len(faces_array), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces_array):
        for j in range(3):
            fern_coin.vectors[i][j] = vertices_array[face[j]]

    return fern_coin

def estimate_output_size(input_mesh, point_distance=0.8):
    """
    Estimate the output STL file size and triangle count before processing.

    Args:
        input_mesh: numpy-stl mesh object
        point_distance: Target distance between texture points (mm)

    Returns:
        dict with keys:
            - estimated_triangles: Estimated number of output triangles
            - estimated_vertices: Estimated number of unique vertices
            - estimated_file_size_mb: Estimated binary STL file size in MB
            - estimated_memory_mb: Estimated peak memory usage in MB
            - input_triangles: Number of input triangles
            - avg_edge_length: Average edge length in input mesh
            - subdivision_factor: Estimated subdivision factor
    """
    # Sample edge lengths from input mesh
    sample_size = min(100, len(input_mesh.vectors))
    sample_edges = []

    for i in range(sample_size):
        v0, v1, v2 = input_mesh.vectors[i]
        sample_edges.append(np.linalg.norm(v1 - v0))
        sample_edges.append(np.linalg.norm(v2 - v1))
        sample_edges.append(np.linalg.norm(v0 - v2))

    avg_edge = np.mean(sample_edges)

    # Calculate subdivision factor
    # Each edge subdivision splits a triangle into 4 triangles
    # Subdivision level = log4(avg_edge / point_distance)
    subdivision_factor = (avg_edge / point_distance) ** 2

    # Estimate output triangle count
    input_triangles = len(input_mesh.vectors)
    estimated_triangles = int(input_triangles * subdivision_factor)

    # Estimate unique vertices (approximately triangles / 2 for a closed mesh)
    estimated_vertices = int(estimated_triangles / 2)

    # Binary STL file size calculation:
    # Header: 80 bytes
    # Triangle count: 4 bytes
    # Per triangle: 50 bytes (12 floats * 4 bytes + 2 bytes attribute)
    estimated_file_size_bytes = 84 + (estimated_triangles * 50)
    estimated_file_size_mb = estimated_file_size_bytes / (1024 * 1024)

    # Memory usage estimation:
    # - Triangle buffer: triangles * 3 vertices * 3 coords * 4 bytes (float32)
    # - Mesh object: triangles * dtype size (~96 bytes per triangle)
    # - Vertex deduplication: vertices * 3 coords * 8 bytes (float64)
    # - Normal calculations: vertices * 3 * 8 bytes
    # - Overhead and temporary arrays: ~2x multiplier
    triangle_buffer_mb = (estimated_triangles * 3 * 3 * 4) / (1024 * 1024)
    mesh_object_mb = (estimated_triangles * 96) / (1024 * 1024)
    vertex_processing_mb = (estimated_vertices * 3 * 8 * 2) / (1024 * 1024)
    estimated_memory_mb = (triangle_buffer_mb + mesh_object_mb + vertex_processing_mb) * 2

    return {
        'estimated_triangles': estimated_triangles,
        'estimated_vertices': estimated_vertices,
        'estimated_file_size_mb': round(estimated_file_size_mb, 2),
        'estimated_memory_mb': round(estimated_memory_mb, 2),
        'input_triangles': input_triangles,
        'avg_edge_length': round(avg_edge, 2),
        'subdivision_factor': round(subdivision_factor, 2)
    }


def check_processing_feasibility(input_mesh, point_distance=0.8,
                                 max_triangles=20_000_000,
                                 max_memory_mb=4096,
                                 max_file_size_mb=500):
    """
    Check if processing is feasible with given constraints.

    Args:
        input_mesh: numpy-stl mesh object
        point_distance: Target distance between texture points (mm)
        max_triangles: Maximum allowed output triangles (default: 20 million)
        max_memory_mb: Maximum allowed memory usage in MB (default: 4GB)
        max_file_size_mb: Maximum allowed output file size in MB (default: 500MB)

    Returns:
        dict with keys:
            - feasible: Boolean indicating if processing is feasible
            - reason: String explaining why not feasible (if applicable)
            - estimates: Dict from estimate_output_size()
            - suggestions: List of suggestions to make processing feasible
    """
    estimates = estimate_output_size(input_mesh, point_distance)

    feasible = True
    reason = None
    suggestions = []

    # Check triangle count
    if estimates['estimated_triangles'] > max_triangles:
        feasible = False
        reason = f"Estimated output ({estimates['estimated_triangles']:,} triangles) exceeds maximum ({max_triangles:,} triangles)"

        # Calculate suggested point_distance
        suggested_factor = max_triangles / estimates['input_triangles']
        suggested_point_distance = estimates['avg_edge_length'] / math.sqrt(suggested_factor)
        suggestions.append(f"Increase point_distance to at least {suggested_point_distance:.2f}mm")

    # Check memory usage
    if estimates['estimated_memory_mb'] > max_memory_mb:
        if not feasible:
            suggestions.append(f"OR reduce mesh complexity (current memory estimate: {estimates['estimated_memory_mb']:.0f}MB)")
        else:
            feasible = False
            reason = f"Estimated memory usage ({estimates['estimated_memory_mb']:.0f}MB) exceeds maximum ({max_memory_mb}MB)"

            # Calculate suggested point_distance for memory
            memory_ratio = max_memory_mb / estimates['estimated_memory_mb']
            suggested_point_distance = point_distance / math.sqrt(memory_ratio)
            suggestions.append(f"Increase point_distance to at least {suggested_point_distance:.2f}mm")

    # Check file size
    if estimates['estimated_file_size_mb'] > max_file_size_mb:
        if not feasible:
            suggestions.append(f"Output file size will be ~{estimates['estimated_file_size_mb']:.0f}MB")
        else:
            feasible = False
            reason = f"Estimated file size ({estimates['estimated_file_size_mb']:.0f}MB) exceeds maximum ({max_file_size_mb}MB)"

            # Calculate suggested point_distance for file size
            size_ratio = max_file_size_mb / estimates['estimated_file_size_mb']
            suggested_point_distance = point_distance / math.sqrt(size_ratio)
            suggestions.append(f"Increase point_distance to at least {suggested_point_distance:.2f}mm")

    if not feasible and not suggestions:
        suggestions.append("Try using a smaller input mesh or increasing point_distance")

    return {
        'feasible': feasible,
        'reason': reason,
        'estimates': estimates,
        'suggestions': suggestions
    }


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

    # Sanity check: estimate subdivision factor
    # Find approximate edge length of input mesh
    sample_edges = []
    for i in range(min(100, len(input_mesh.vectors))):
        v0, v1, v2 = input_mesh.vectors[i]
        sample_edges.append(np.linalg.norm(v1 - v0))
        sample_edges.append(np.linalg.norm(v2 - v1))
        sample_edges.append(np.linalg.norm(v0 - v2))
    avg_edge = np.mean(sample_edges)
    subdivision_factor = (avg_edge / point_distance) ** 2
    estimated_output_triangles = len(input_mesh.vectors) * subdivision_factor

    # Warn if subdivision will be excessive
    if estimated_output_triangles > 10_000_000:
        print(f"WARNING: Estimated output size: {estimated_output_triangles:.0f} triangles")
        print(f"This may cause memory issues. Consider increasing point_distance.")
        print(f"Average input edge length: {avg_edge:.2f}mm, target: {point_distance}mm")

    # Subdivide all triangles - estimate output size first to avoid reallocation
    # Pre-allocate with estimated size (can grow if needed)
    estimated_triangles = min(int(estimated_output_triangles * 1.2), len(input_mesh.vectors) * 1000)
    estimated_triangles = max(estimated_triangles, len(input_mesh.vectors) * 4)
    triangle_buffer = np.zeros((estimated_triangles, 3, 3), dtype=np.float32)
    triangle_count = 0

    for face_idx, face in enumerate(input_mesh.vectors):
        v0, v1, v2 = face
        subdivided = subdivide_triangle(v0, v1, v2, point_distance)

        # Expand buffer if needed
        if triangle_count + len(subdivided) > len(triangle_buffer):
            new_size = max(len(triangle_buffer) * 2, triangle_count + len(subdivided))
            new_buffer = np.zeros((new_size, 3, 3), dtype=np.float32)
            new_buffer[:triangle_count] = triangle_buffer[:triangle_count]
            triangle_buffer = new_buffer

        # Add triangles to buffer
        for tri in subdivided:
            triangle_buffer[triangle_count] = tri
            triangle_count += 1

        # Progress indicator for large meshes
        if (face_idx + 1) % 1000 == 0:
            print(f"  Processed {face_idx + 1}/{len(input_mesh.vectors)} faces...")

    # Trim buffer to actual size
    triangle_buffer = triangle_buffer[:triangle_count]

    print(f"Subdivided {len(input_mesh.vectors)} triangles into {triangle_count} triangles")

    # Create new mesh from subdivided triangles
    output_mesh = mesh.Mesh(np.zeros(triangle_count, dtype=mesh.Mesh.dtype))
    output_mesh.vectors = triangle_buffer

    # Build vertex map to find shared vertices
    print("Finding unique vertices...")
    vertices = output_mesh.vectors.reshape(-1, 3)
    # Round to avoid floating point issues when finding unique vertices
    rounded = np.round(vertices, decimals=6)

    # For very large meshes, this can be memory intensive
    try:
        unique_vertices, inverse_indices = np.unique(
            rounded, axis=0, return_inverse=True
        )
    except MemoryError:
        print("Warning: Memory error during vertex deduplication. Mesh may be too large.")
        print("Try increasing point_distance or reducing mesh size.")
        raise

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
    print("Updating mesh with displaced vertices...")
    output_mesh.vectors = displaced_vertices[inverse_indices].reshape(-1, 3, 3)

    # Validate output mesh
    if np.any(np.isnan(output_mesh.vectors)) or np.any(np.isinf(output_mesh.vectors)):
        print("ERROR: Output mesh contains invalid values (NaN or Inf)")
        print("This may indicate numerical instability with the current parameters.")
        raise ValueError("Invalid mesh output - contains NaN or Inf values")

    print("Mesh processing complete!")
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