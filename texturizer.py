#!/usr/bin/env python3
"""
Fuzzy Skin Texture Generator for STL files
Applies random surface displacement similar to slicer "fuzzy skin" feature
with multiple noise types matching OrcaSlicer's implementation.

PERFORMANCE NOTE: Why OrcaSlicer is faster
------------------------------------------
OrcaSlicer applies fuzzy skin during the slicing process by working on 2D polylines
(perimeter paths). It only adds texture points along existing slice contours without
creating a 3D volumetric mesh.

This tool, in contrast, must work on 3D STL meshes BEFORE slicing:
1. Subdivides ALL triangles in the 3D mesh to achieve the target point_distance
2. Creates a complete 3D volumetric textured surface
3. This pre-slicing approach is fundamentally more resource-intensive

The performance difference is architectural, not an optimization opportunity:
- Orca: 2D path processing during slicing (no mesh subdivision)
- This tool: 3D mesh subdivision before slicing (creates full textured geometry)

For typical settings (thickness=0.2mm, point_distance=0.2mm), a simple 20mm cube
requires ~50,000 output triangles. This is expected and correct for pre-slicing
3D mesh texturing.

Requires: numpy, numpy-stl, noise
Install: pip install numpy numpy-stl noise
"""

import numpy as np
from stl import mesh
import argparse
import sys
import math
import logging

# Set up logger
logger = logging.getLogger(__name__)

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

def generate_simple_cube(size=20):
    """
    Generate a simple cube for testing/demonstration.

    Args:
        size: Side length of the cube in mm (default 20mm)

    Returns:
        mesh.Mesh object
    """
    # Define the 8 vertices of a cube
    vertices = np.array([
        [-size/2, -size/2, -size/2],
        [+size/2, -size/2, -size/2],
        [+size/2, +size/2, -size/2],
        [-size/2, +size/2, -size/2],
        [-size/2, -size/2, +size/2],
        [+size/2, -size/2, +size/2],
        [+size/2, +size/2, +size/2],
        [-size/2, +size/2, +size/2],
    ])

    # Define the 12 triangles (2 per face, 6 faces)
    faces = np.array([
        # Bottom face (z = -size/2)
        [0, 3, 1],
        [1, 3, 2],
        # Top face (z = +size/2)
        [4, 5, 7],
        [5, 6, 7],
        # Front face (y = -size/2)
        [0, 1, 4],
        [1, 5, 4],
        # Back face (y = +size/2)
        [2, 3, 6],
        [3, 7, 6],
        # Left face (x = -size/2)
        [0, 4, 3],
        [3, 4, 7],
        # Right face (x = +size/2)
        [1, 2, 5],
        [2, 6, 5],
    ])

    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[face[j], :]

    return cube


def generate_custom_object(size=20):
    """
    Generate custom test object: Fiuncholabs Beaker Card
    A rectangular card with embossed beaker design (Fiuncholabs branding)

    Args:
        size: Reference dimension for the card (default 20mm)
              Card will be size × (size * 1.5) × (size * 0.1)

    Returns:
        mesh.Mesh object
    """
    # Card dimensions
    width = size * 1.5  # 30mm for size=20
    height = size  # 20mm for size=20
    thickness = size * 0.1  # 2mm for size=20
    emboss_height = size * 0.04  # 0.8mm embossing (much more visible than fern)

    # Resolution for mesh
    width_segments = max(40, int(width * 2))
    height_segments = max(30, int(height * 2))

    def beaker_function(x, y):
        """Generate stylized beaker design for Fiuncholabs branding"""
        # Normalize coordinates to card dimensions
        # x ranges from -width/2 to +width/2
        # y ranges from -height/2 to +height/2

        # Normalize to 0-1 range
        nx = (x + width / 2) / width  # 0 at left, 1 at right
        ny = (y + height / 2) / height  # 0 at bottom, 1 at top

        beaker_value = 0.0

        # Beaker dimensions (normalized)
        beaker_width = 0.4  # Width at top
        beaker_bottom_width = 0.3  # Width at bottom (tapers)
        beaker_height = 0.7  # Total height of beaker
        beaker_base = 0.15  # Y position of beaker bottom

        # Center the beaker
        beaker_cx = 0.5  # Center x

        # Beaker body (tapered trapezoid)
        if beaker_base <= ny <= beaker_base + beaker_height:
            # Progress from bottom (0) to top (1)
            progress = (ny - beaker_base) / beaker_height

            # Width at this height (linear taper)
            current_width = beaker_bottom_width + (beaker_width - beaker_bottom_width) * progress
            half_width = current_width / 2

            # Distance from center
            dx = abs(nx - beaker_cx)

            # Beaker walls (with thickness)
            wall_thickness = 0.02
            if half_width - wall_thickness <= dx <= half_width:
                # Calculate smooth wall strength
                wall_pos = (half_width - dx) / wall_thickness
                beaker_value = max(beaker_value, wall_pos)

        # Beaker bottom (base)
        if beaker_base <= ny <= beaker_base + 0.03:
            half_width = beaker_bottom_width / 2
            dx = abs(nx - beaker_cx)
            if dx <= half_width:
                beaker_value = max(beaker_value, 1.0)

        # Measurement marks on the side (3 marks)
        for mark_num in range(3):
            mark_y = beaker_base + beaker_height * (0.25 + mark_num * 0.25)
            if abs(ny - mark_y) < 0.01:
                progress = (mark_y - beaker_base) / beaker_height
                current_width = beaker_bottom_width + (beaker_width - beaker_bottom_width) * progress
                half_width = current_width / 2

                # Mark on left side
                mark_x_left = beaker_cx - half_width * 0.8
                if abs(nx - mark_x_left) < 0.05:
                    beaker_value = max(beaker_value, 0.7)

                # Mark on right side
                mark_x_right = beaker_cx + half_width * 0.8
                if abs(nx - mark_x_right) < 0.05:
                    beaker_value = max(beaker_value, 0.7)

        # Liquid inside (wavy top)
        liquid_top = beaker_base + beaker_height * 0.6
        # Add slight wave to liquid surface
        wave = 0.015 * np.sin(nx * 10)
        if beaker_base + 0.03 <= ny <= liquid_top + wave:
            progress = (ny - beaker_base) / beaker_height
            current_width = beaker_bottom_width + (beaker_width - beaker_bottom_width) * progress
            half_width = current_width / 2
            dx = abs(nx - beaker_cx)

            if dx < half_width - 0.02:  # Inside the beaker
                # Create gradient for liquid
                liquid_strength = 0.5
                beaker_value = max(beaker_value, liquid_strength)

        # Smooth the embossing
        return beaker_value * emboss_height

    # Generate rectangular card mesh
    vertices_list = []

    # Create grid of vertices
    x_values = np.linspace(-width / 2, width / 2, width_segments)
    y_values = np.linspace(-height / 2, height / 2, height_segments)

    # Bottom surface (flat)
    bottom_grid = []
    for j in range(height_segments):
        row = []
        for i in range(width_segments):
            x = x_values[i]
            y = y_values[j]
            row.append(len(vertices_list))
            vertices_list.append([x, y, 0])
        bottom_grid.append(row)

    # Top surface (with embossing)
    top_grid = []
    for j in range(height_segments):
        row = []
        for i in range(width_segments):
            x = x_values[i]
            y = y_values[j]
            z = thickness + beaker_function(x, y)
            row.append(len(vertices_list))
            vertices_list.append([x, y, z])
        top_grid.append(row)

    vertices_array = np.array(vertices_list)

    # Create faces
    faces_list = []

    # Bottom surface faces
    for j in range(height_segments - 1):
        for i in range(width_segments - 1):
            v1 = bottom_grid[j][i]
            v2 = bottom_grid[j][i + 1]
            v3 = bottom_grid[j + 1][i]
            v4 = bottom_grid[j + 1][i + 1]

            faces_list.append([v1, v3, v2])
            faces_list.append([v2, v3, v4])

    # Top surface faces
    for j in range(height_segments - 1):
        for i in range(width_segments - 1):
            v1 = top_grid[j][i]
            v2 = top_grid[j][i + 1]
            v3 = top_grid[j + 1][i]
            v4 = top_grid[j + 1][i + 1]

            faces_list.append([v1, v2, v3])
            faces_list.append([v2, v4, v3])

    # Side faces (4 edges)
    # Left edge
    for j in range(height_segments - 1):
        b1 = bottom_grid[j][0]
        b2 = bottom_grid[j + 1][0]
        t1 = top_grid[j][0]
        t2 = top_grid[j + 1][0]
        faces_list.append([b1, t1, b2])
        faces_list.append([b2, t1, t2])

    # Right edge
    for j in range(height_segments - 1):
        b1 = bottom_grid[j][width_segments - 1]
        b2 = bottom_grid[j + 1][width_segments - 1]
        t1 = top_grid[j][width_segments - 1]
        t2 = top_grid[j + 1][width_segments - 1]
        faces_list.append([b1, b2, t1])
        faces_list.append([b2, t2, t1])

    # Front edge
    for i in range(width_segments - 1):
        b1 = bottom_grid[0][i]
        b2 = bottom_grid[0][i + 1]
        t1 = top_grid[0][i]
        t2 = top_grid[0][i + 1]
        faces_list.append([b1, b2, t1])
        faces_list.append([b2, t2, t1])

    # Back edge
    for i in range(width_segments - 1):
        b1 = bottom_grid[height_segments - 1][i]
        b2 = bottom_grid[height_segments - 1][i + 1]
        t1 = top_grid[height_segments - 1][i]
        t2 = top_grid[height_segments - 1][i + 1]
        faces_list.append([b1, t1, b2])
        faces_list.append([b2, t1, t2])

    faces_array = np.array(faces_list)

    # Create mesh
    beaker_card = mesh.Mesh(np.zeros(len(faces_array), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces_array):
        for j in range(3):
            beaker_card.vectors[i][j] = vertices_array[face[j]]

    return beaker_card


def generate_test_cube(size=20, object_type=None):
    """
    Generate default test object based on configuration.

    Args:
        size: Size in mm (cube side length or custom object base size)
        object_type: 'cube' or 'custom'. If None, uses DEFAULT_OBJECT_TYPE from environment

    Returns:
        mesh.Mesh object
    """
    import os

    if object_type is None:
        object_type = os.environ.get('DEFAULT_OBJECT_TYPE', 'cube').lower()

    if object_type == 'custom':
        return generate_custom_object(size)
    else:
        return generate_simple_cube(size)


def generate_blocker_cylinder(radius=10, height=30, position=(0, 0, 0), rotation=(0, 0, 0), segments=32):
    """
    Generate a cylinder to use as a blocker volume.

    Args:
        radius: Cylinder radius in mm (default 10mm)
        height: Cylinder height in mm (default 30mm)
        position: (x, y, z) position of cylinder center (default (0, 0, 0))
        rotation: (rx, ry, rz) rotation in degrees around origin axes X, Y, Z (default (0, 0, 0))
        segments: Number of sides for the cylinder (default 32)

    Returns:
        mesh.Mesh object representing a closed cylinder

    Note:
        Rotation is applied BEFORE translation, so rotation happens around the origin.
        This matches the UX where rotating the cylinder causes it to orbit around the mesh center.
    """
    # Use trimesh to generate a proper manifold cylinder
    import trimesh

    # Create cylinder using trimesh (centered at origin, aligned with Z-axis)
    cyl_trimesh = trimesh.creation.cylinder(radius=radius, height=height, sections=segments)

    # Apply rotations FIRST (around origin)
    # Frontend visualization: Y-up cylinder with rotation.set(rotX, rotZ, -rotY)
    # Backend: Z-up cylinder that needs equivalent rotations
    #
    # Coordinate system relationship: Y-up = Rx(-90°) * Z-up
    # Frontend applies: Rx(rotX) * Ry(rotZ) * Rz(-rotY) to Y-up cylinder
    # This is equivalent to: Rx(rotX) * Ry(rotZ) * Rz(-rotY) * Rx(-90°) * Z-up
    #
    # To get the same result on Z-up cylinder, we need to find M such that:
    # Rx(-90°) * M * Z-up = Rx(rotX) * Ry(rotZ) * Rz(-rotY) * Rx(-90°) * Z-up
    # Therefore: M = Rx(90°) * Rx(rotX) * Ry(rotZ) * Rz(-rotY) * Rx(-90°)

    rx, ry, rz = rotation
    if rx != 0 or ry != 0 or rz != 0:
        # Build the compound transformation
        # Start: convert Z-up to Y-up
        to_yup = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])

        # Apply user rotations in Y-up space (in order: X, then Y, then Z)
        # Multiply left-to-right: result = Rz * Ry * Rx * point
        user_transform = np.eye(4)
        if rx != 0:
            user_transform = user_transform @ trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0])
        if rz != 0:
            user_transform = user_transform @ trimesh.transformations.rotation_matrix(np.radians(rz), [0, 1, 0])
        if ry != 0:
            user_transform = user_transform @ trimesh.transformations.rotation_matrix(np.radians(-ry), [0, 0, 1])

        # End: convert Y-up back to Z-up
        from_yup = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])

        # Combine: from_yup * user_transform * to_yup
        transform = from_yup @ user_transform @ to_yup
        cyl_trimesh.apply_transform(transform)

    # THEN translate to the desired position
    if position != (0, 0, 0):
        transform = trimesh.transformations.translation_matrix(position)
        cyl_trimesh.apply_transform(transform)

    # Convert to numpy-stl format
    cylinder = mesh.Mesh(np.zeros(len(cyl_trimesh.faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(cyl_trimesh.faces):
        for j in range(3):
            cylinder.vectors[i][j] = cyl_trimesh.vertices[face[j]]

    return cylinder


def generate_blocker_cube(width=10, height=10, depth=10, position=(0, 0, 0), rotation=(0, 0, 0)):
    """
    Generate a cube/box to use as a blocker volume.

    Args:
        width: Cube width (X-axis) in mm (default 10mm)
        height: Cube height (Z-axis) in mm (default 10mm)
        depth: Cube depth (Y-axis) in mm (default 10mm)
        position: (x, y, z) position of cube center (default (0, 0, 0))
        rotation: (rx, ry, rz) rotation in degrees around origin axes X, Y, Z (default (0, 0, 0))

    Returns:
        mesh.Mesh object representing a closed cube

    Note:
        Rotation is applied BEFORE translation, matching the cylinder behavior.
    """
    import trimesh

    # Create box using trimesh (centered at origin)
    cube_trimesh = trimesh.creation.box(extents=[width, depth, height])

    # Apply rotations FIRST (using the same logic as cylinder)
    rx, ry, rz = rotation
    if rx != 0 or ry != 0 or rz != 0:
        # Convert Z-up to Y-up
        to_yup = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])

        # Apply user rotations in Y-up space (left-to-right multiplication)
        user_transform = np.eye(4)
        if rx != 0:
            user_transform = user_transform @ trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0])
        if rz != 0:
            user_transform = user_transform @ trimesh.transformations.rotation_matrix(np.radians(rz), [0, 1, 0])
        if ry != 0:
            user_transform = user_transform @ trimesh.transformations.rotation_matrix(np.radians(-ry), [0, 0, 1])

        # Convert Y-up back to Z-up
        from_yup = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])

        # Combine transformations
        transform = from_yup @ user_transform @ to_yup
        cube_trimesh.apply_transform(transform)

    # Then apply translation
    if position != (0, 0, 0):
        transform = trimesh.transformations.translation_matrix(position)
        cube_trimesh.apply_transform(transform)

    # Convert to numpy-stl format
    cube = mesh.Mesh(np.zeros(len(cube_trimesh.faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(cube_trimesh.faces):
        for j in range(3):
            cube.vectors[i][j] = cube_trimesh.vertices[face[j]]

    return cube


def point_inside_mesh_volume(points, blocker_mesh, algorithm='ray_casting'):
    """
    Check if points are inside a blocker mesh volume.

    Args:
        points: Nx3 numpy array of vertex positions
        blocker_mesh: mesh.Mesh object (blocker volume)
        algorithm: Algorithm to use for point-in-volume testing:
                   'ray_casting' - Accurate for any closed mesh (default)
                   'bounding_box' - Fast, conservative (rectangular bounds)
                   'bounding_sphere' - Fast, conservative (spherical bounds)
                   'bounding_cylinder' - Fast, for cylinder-shaped blockers

    Returns:
        Boolean array of length N (True if inside volume)
    """
    if algorithm == 'ray_casting':
        # Ray casting: cast ray from point, count intersections with mesh
        # Odd number of intersections = inside, even = outside
        # This is the most accurate method for arbitrary closed meshes

        all_triangles = blocker_mesh.vectors
        inside = np.zeros(len(points), dtype=bool)

        # Ray direction: along +X axis (arbitrary choice)
        ray_dir = np.array([1.0, 0.0, 0.0])

        for idx, point in enumerate(points):
            intersection_count = 0

            # Check intersection with each triangle
            for tri in all_triangles:
                v0, v1, v2 = tri

                # Möller–Trumbore ray-triangle intersection algorithm
                edge1 = v1 - v0
                edge2 = v2 - v0
                h = np.cross(ray_dir, edge2)
                a = np.dot(edge1, h)

                # Ray is parallel to triangle
                if abs(a) < 1e-8:
                    continue

                f = 1.0 / a
                s = point - v0
                u = f * np.dot(s, h)

                if u < 0.0 or u > 1.0:
                    continue

                q = np.cross(s, edge1)
                v = f * np.dot(ray_dir, q)

                if v < 0.0 or u + v > 1.0:
                    continue

                # Intersection distance along ray
                t = f * np.dot(edge2, q)

                # Only count intersections in front of the point
                if t > 1e-8:
                    intersection_count += 1

            # Odd number of intersections = inside
            inside[idx] = (intersection_count % 2) == 1

        return inside

    elif algorithm == 'bounding_sphere':
        # Sphere: check if point is within spherical bounds
        all_vertices = blocker_mesh.vectors.reshape(-1, 3)

        # Calculate center as mean of all vertices
        center = np.mean(all_vertices, axis=0)

        # Calculate radius as max distance from center
        distances = np.linalg.norm(all_vertices - center, axis=1)
        radius = np.max(distances)

        # Check if each point is within sphere
        point_distances = np.linalg.norm(points - center, axis=1)
        inside = point_distances <= radius

        return inside

    elif algorithm == 'bounding_cylinder':
        # For cylinders: check if point is within cylindrical bounds
        # 1. Calculate cylinder center and dimensions from mesh
        all_vertices = blocker_mesh.vectors.reshape(-1, 3)

        # Get bounding box
        min_z = np.min(all_vertices[:, 2])
        max_z = np.max(all_vertices[:, 2])
        center_x = np.mean(all_vertices[:, 0])
        center_y = np.mean(all_vertices[:, 1])

        # Estimate radius as max distance from center axis in XY plane
        distances_xy = np.sqrt((all_vertices[:, 0] - center_x)**2 +
                               (all_vertices[:, 1] - center_y)**2)
        radius = np.max(distances_xy)

        # Check each point
        inside = np.zeros(len(points), dtype=bool)
        for i, point in enumerate(points):
            # Check Z bounds
            if point[2] < min_z or point[2] > max_z:
                continue

            # Check radial distance from center axis
            dist_from_axis = np.sqrt((point[0] - center_x)**2 +
                                    (point[1] - center_y)**2)
            if dist_from_axis <= radius:
                inside[i] = True

        return inside

    elif algorithm == 'bounding_box':
        # Simple bounding box check
        all_vertices = blocker_mesh.vectors.reshape(-1, 3)
        min_bounds = np.min(all_vertices, axis=0)
        max_bounds = np.max(all_vertices, axis=0)

        # Check if each point is within bounds
        inside = np.all((points >= min_bounds) & (points <= max_bounds), axis=1)
        return inside

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def split_mesh_by_blocker(input_mesh, blocker_mesh, algorithm='boolean'):
    """
    Split a mesh into two parts using boolean operations:
    - Outside: input_mesh MINUS blocker_mesh (difference)
    - Inside: input_mesh INTERSECT blocker_mesh (intersection)

    Args:
        input_mesh: mesh.Mesh object to split
        blocker_mesh: mesh.Mesh object representing the blocker volume
        algorithm: 'boolean' for CSG operations (default), or legacy algorithms

    Returns:
        tuple of (outside_mesh, inside_mesh) - two mesh.Mesh objects
        If one part has no triangles, returns an empty mesh for that part
    """
    import trimesh

    print(f"Splitting mesh by blocker volume using boolean operations...")

    try:
        # Convert numpy-stl meshes to trimesh objects
        # For numpy-stl meshes, vertices are stored as (n_triangles, 3, 3)
        # We need to properly extract unique vertices and faces
        def stl_to_trimesh(stl_mesh):
            """Convert numpy-stl mesh to trimesh with proper vertex merging"""
            # Get all vertices (reshape to list of 3D points)
            all_verts = stl_mesh.vectors.reshape(-1, 3)

            # Create trimesh (it will automatically merge duplicate vertices)
            return trimesh.Trimesh(vertices=all_verts,
                                  faces=np.arange(len(all_verts)).reshape(-1, 3),
                                  process=True)  # Enable processing to fix manifoldness

        input_trimesh = stl_to_trimesh(input_mesh)
        blocker_trimesh = stl_to_trimesh(blocker_mesh)

        # Check if meshes are watertight
        print(f"  Input mesh: {len(input_trimesh.vertices)} verts, {len(input_trimesh.faces)} faces, watertight: {input_trimesh.is_watertight}")
        print(f"  Blocker mesh: {len(blocker_trimesh.vertices)} verts, {len(blocker_trimesh.faces)} faces, watertight: {blocker_trimesh.is_watertight}")

        if not input_trimesh.is_watertight or not blocker_trimesh.is_watertight:
            raise ValueError("Meshes must be watertight for boolean operations")

        # Perform boolean operations
        # Try different engines in order of preference
        engines_to_try = ['blender', 'scad', None]  # None = auto-select

        outside_trimesh = None
        inside_trimesh = None

        for engine in engines_to_try:
            try:
                engine_name = engine if engine else 'auto'
                print(f"  Trying boolean operations with engine: {engine_name}")

                print(f"    Computing difference (outside = input - blocker)...")
                outside_trimesh = input_trimesh.difference(blocker_trimesh, engine=engine)

                print(f"    Computing intersection (inside = input ∩ blocker)...")
                inside_trimesh = input_trimesh.intersection(blocker_trimesh, engine=engine)

                print(f"  ✓ Boolean operations succeeded with engine: {engine_name}")
                break

            except Exception as engine_error:
                print(f"    Engine {engine_name} failed: {engine_error}")
                if engine == engines_to_try[-1]:
                    # Last engine failed, re-raise
                    raise
                continue

        # Convert back to numpy-stl format
        def trimesh_to_stl_mesh(tm):
            if tm is None or len(tm.faces) == 0:
                return mesh.Mesh(np.zeros(0, dtype=mesh.Mesh.dtype))

            stl_mesh = mesh.Mesh(np.zeros(len(tm.faces), dtype=mesh.Mesh.dtype))
            for i, face in enumerate(tm.faces):
                for j in range(3):
                    stl_mesh.vectors[i][j] = tm.vertices[face[j]]
            return stl_mesh

        outside_mesh = trimesh_to_stl_mesh(outside_trimesh)
        inside_mesh = trimesh_to_stl_mesh(inside_trimesh)

        print(f"  Outside (difference): {len(outside_mesh.vectors)} triangles")
        print(f"  Inside (intersection): {len(inside_mesh.vectors)} triangles")

        return outside_mesh, inside_mesh

    except Exception as e:
        print(f"  Warning: Boolean operation failed: {e}")
        print(f"  Falling back to centroid-based splitting...")

        # Fallback to simple centroid-based splitting
        outside_triangles = []
        inside_triangles = []

        for tri in input_mesh.vectors:
            centroid = np.mean(tri, axis=0).reshape(1, 3)
            is_inside = point_inside_mesh_volume(centroid, blocker_mesh, algorithm='bounding_box')[0]

            if is_inside:
                inside_triangles.append(tri)
            else:
                outside_triangles.append(tri)

        # Create meshes
        if outside_triangles:
            outside_mesh = mesh.Mesh(np.zeros(len(outside_triangles), dtype=mesh.Mesh.dtype))
            for i, tri in enumerate(outside_triangles):
                outside_mesh.vectors[i] = tri
        else:
            outside_mesh = mesh.Mesh(np.zeros(0, dtype=mesh.Mesh.dtype))

        if inside_triangles:
            inside_mesh = mesh.Mesh(np.zeros(len(inside_triangles), dtype=mesh.Mesh.dtype))
            for i, tri in enumerate(inside_triangles):
                inside_mesh.vectors[i] = tri
        else:
            inside_mesh = mesh.Mesh(np.zeros(0, dtype=mesh.Mesh.dtype))

        print(f"  Outside: {len(outside_mesh.vectors)} triangles")
        print(f"  Inside: {len(inside_mesh.vectors)} triangles")

        return outside_mesh, inside_mesh


def repair_mesh(input_mesh):
    """
    Repair a mesh to fix non-manifold edges and other issues.
    Uses trimesh to merge duplicate vertices and fix mesh topology.

    Args:
        input_mesh: numpy-stl mesh object

    Returns:
        Repaired numpy-stl mesh object with metadata preserved
    """
    import trimesh
    import tempfile
    import os

    # Store metadata if it exists
    metadata = getattr(input_mesh, 'metadata', None)

    # Convert to trimesh
    all_verts = input_mesh.vectors.reshape(-1, 3)
    tri_mesh = trimesh.Trimesh(
        vertices=all_verts,
        faces=np.arange(len(all_verts)).reshape(-1, 3),
        process=True  # This will merge duplicate vertices and fix topology
    )

    # Additional repair steps
    # Merge duplicate vertices (already done by process=True, but ensure it's done)
    tri_mesh.merge_vertices()

    # Fix normal consistency
    tri_mesh.fix_normals()

    # Try to fill holes (may fail on some meshes)
    try:
        tri_mesh.fill_holes()
    except Exception as e:
        print(f"  Note: Could not fill holes: {e}")

    # Use trimesh's STL export/import cycle to further clean up the mesh
    # This often fixes remaining non-manifold edges
    try:
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        # Export to STL
        tri_mesh.export(tmp_path)

        # Re-import with processing
        tri_mesh = trimesh.load(tmp_path, process=True)

        # Clean up temp file
        os.unlink(tmp_path)
    except Exception as e:
        print(f"  Note: Could not perform export/import cycle: {e}")

    # Convert back to numpy-stl
    repaired = mesh.Mesh(np.zeros(len(tri_mesh.faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(tri_mesh.faces):
        for j in range(3):
            repaired.vectors[i][j] = tri_mesh.vertices[face[j]]

    # Restore metadata
    if metadata is not None:
        repaired.metadata = metadata

    print(f"  Mesh repaired: {len(input_mesh.vectors)} → {len(repaired.vectors)} triangles")

    return repaired


def estimate_output_size(input_mesh, point_distance=0.8, skip_small_triangles=False):
    """
    Estimate the output STL file size and triangle count before processing.

    Args:
        input_mesh: numpy-stl mesh object
        point_distance: Target distance between texture points (mm)
        skip_small_triangles: If True, skip subdivision for triangles with at least one edge < point_distance

    Returns:
        dict with keys:
            - estimated_triangles: Estimated number of output triangles
            - estimated_vertices: Estimated number of unique vertices
            - estimated_file_size_mb: Estimated binary STL file size in MB
            - estimated_memory_mb: Estimated peak memory usage in MB
            - input_triangles: Number of input triangles
            - avg_edge_length: Average edge length in input mesh
            - subdivision_factor: Estimated subdivision factor
            - skipped_small_percent: Percentage of triangles skipped (if optimization enabled)
    """
    # Analyze ALL triangles to calculate accurate subdivision estimate
    # We need to find max edge per triangle since each triangle subdivides independently
    # Note: For large meshes (>10k triangles), we could optimize by random sampling,
    # but for accurate estimation we analyze all triangles
    sample_max_edges = []
    all_edges = []
    skipped_count = 0

    for i in range(len(input_mesh.vectors)):
        v0, v1, v2 = input_mesh.vectors[i]
        e0 = np.linalg.norm(v1 - v0)
        e1 = np.linalg.norm(v2 - v1)
        e2 = np.linalg.norm(v0 - v2)

        all_edges.extend([e0, e1, e2])

        # Check if this triangle would be skipped with optimization
        if skip_small_triangles and min(e0, e1, e2) <= point_distance:
            skipped_count += 1
        else:
            sample_max_edges.append(max(e0, e1, e2))

    avg_edge = float(np.mean(all_edges))

    # Calculate subdivision factor based on actual algorithm behavior
    # The subdivision recursively splits triangles into 4 until all edges <= point_distance
    # Each triangle subdivides based on its MAXIMUM edge length
    # Number of subdivision levels = ceil(log2(max_edge / point_distance))
    # Each level quadruples the triangle count, so factor = 4^levels

    # Use 90th percentile of max edges - this provides better estimates for varied meshes
    # The 90th percentile captures the larger triangles that drive most of the subdivision,
    # while not being overly affected by a few extremely large outliers
    # For uniform meshes (like cubes), all percentiles are equal so this remains exact
    percentile_90_max = float(np.percentile(sample_max_edges, 90))
    representative_max_edge = percentile_90_max

    if representative_max_edge <= point_distance:
        subdivision_factor = 1.0  # No subdivision needed
    else:
        # Calculate subdivision levels needed
        edge_ratio = representative_max_edge / point_distance
        subdivision_levels = np.ceil(np.log2(edge_ratio))
        subdivision_factor = 4 ** subdivision_levels

    # Estimate output triangle count
    input_triangles = len(input_mesh.vectors)

    if skip_small_triangles:
        # Only subdivide non-skipped triangles
        triangles_to_subdivide = input_triangles - skipped_count
        estimated_subdivided = int(triangles_to_subdivide * subdivision_factor)
        estimated_triangles = estimated_subdivided + skipped_count
    else:
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

    # Time estimation based on actual performance data:
    # From testing: ~100,000 output triangles/second (combined subdivision + vertex processing)
    # This rate accounts for subdivision, vertex deduplication, normal calculation, and displacement
    # File writing is negligible compared to processing time
    estimated_time_seconds = estimated_triangles / 100000

    # Add overhead for very small meshes (setup time ~0.1s)
    if estimated_time_seconds < 0.5:
        estimated_time_seconds = max(0.1, estimated_time_seconds)

    # Add overhead for very large meshes
    if estimated_triangles > 1_000_000:
        estimated_time_seconds *= 1.5

    result = {
        'estimated_triangles': estimated_triangles,
        'estimated_vertices': estimated_vertices,
        'estimated_file_size_mb': round(estimated_file_size_mb, 2),
        'estimated_memory_mb': round(estimated_memory_mb, 2),
        'estimated_time_seconds': round(estimated_time_seconds, 1),
        'input_triangles': input_triangles,
        'avg_edge_length': round(avg_edge, 2),
        'subdivision_factor': round(subdivision_factor, 2)
    }

    if skip_small_triangles:
        result['skipped_small_percent'] = round((skipped_count / input_triangles * 100) if input_triangles > 0 else 0, 1)

    return result


def check_processing_feasibility(input_mesh, point_distance=0.8,
                                 max_triangles=20_000_000,
                                 max_memory_mb=4096,
                                 max_file_size_mb=500,
                                 skip_small_triangles=False):
    """
    Check if processing is feasible with given constraints.

    Args:
        input_mesh: numpy-stl mesh object
        point_distance: Target distance between texture points (mm)
        max_triangles: Maximum allowed output triangles (default: 20 million)
        max_memory_mb: Maximum allowed memory usage in MB (default: 4GB)
        max_file_size_mb: Maximum allowed output file size in MB (default: 500MB)
        skip_small_triangles: If True, skip subdivision for triangles with at least one edge < point_distance

    Returns:
        dict with keys:
            - feasible: Boolean indicating if processing is feasible
            - reason: String explaining why not feasible (if applicable)
            - estimates: Dict from estimate_output_size()
            - suggestions: List of suggestions to make processing feasible
    """
    estimates = estimate_output_size(input_mesh, point_distance, skip_small_triangles=skip_small_triangles)

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
                     noise_persistence=0.5, skip_bottom=False, skip_small_triangles=False,
                     blocker_mesh=None, blocker_algorithm='double_stl'):
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
        skip_small_triangles: If True, skip subdivision for triangles with at least one edge < point_distance
        blocker_mesh: mesh.Mesh object representing a blocker volume (optional). Vertices inside this
                      volume will not have fuzzy skin applied.
        blocker_algorithm: Always 'double_stl' - splits mesh into inside/outside, applies fuzzy only to outside
    """
    np.random.seed(seed)

    # Handle double_stl algorithm: split mesh and process separately
    if blocker_mesh is not None and blocker_algorithm == 'double_stl':
        print("Using double STL algorithm: splitting mesh into inside/outside blocker volume...")

        # Split the mesh into outside and inside parts using boolean operations
        outside_mesh, inside_mesh = split_mesh_by_blocker(input_mesh, blocker_mesh, algorithm='boolean')

        if len(outside_mesh.vectors) == 0:
            # Everything is inside blocker - return unprocessed mesh
            print("All triangles are inside blocker volume - returning unprocessed mesh")
            return input_mesh
        elif len(inside_mesh.vectors) == 0:
            # Nothing is inside blocker - process entire mesh normally
            print("No triangles inside blocker volume - processing entire mesh")
            # Continue with normal processing (fall through)
        else:
            # Process only the outside part
            print(f"Processing outside part ({len(outside_mesh.vectors)} triangles)...")
            processed_outside = apply_fuzzy_skin(
                outside_mesh,
                thickness=thickness,
                point_distance=point_distance,
                seed=seed,
                noise_type=noise_type,
                noise_scale=noise_scale,
                noise_octaves=noise_octaves,
                noise_persistence=noise_persistence,
                skip_bottom=skip_bottom,
                skip_small_triangles=skip_small_triangles,
                blocker_mesh=None,  # No blocker for recursive call
                blocker_algorithm='ray_casting'  # Not used since blocker_mesh is None
            )

            # Combine processed outside with unprocessed inside
            print(f"Combining processed outside with unprocessed inside...")
            total_triangles = len(processed_outside.vectors) + len(inside_mesh.vectors)
            combined_mesh = mesh.Mesh(np.zeros(total_triangles, dtype=mesh.Mesh.dtype))

            # Add processed outside triangles
            combined_mesh.vectors[:len(processed_outside.vectors)] = processed_outside.vectors

            # Add unprocessed inside triangles
            combined_mesh.vectors[len(processed_outside.vectors):] = inside_mesh.vectors

            # Store metadata about the split for visualization
            # This will be used by the frontend to color the two parts differently
            combined_mesh.metadata = {
                'split_index': len(processed_outside.vectors),
                'outside_count': len(processed_outside.vectors),
                'inside_count': len(inside_mesh.vectors)
            }

            print(f"Combined mesh: {total_triangles} triangles ({len(processed_outside.vectors)} processed + {len(inside_mesh.vectors)} unprocessed)")

            # Repair mesh to fix non-manifold edges from boolean operations
            print("Repairing mesh to ensure manifold output...")
            combined_mesh = repair_mesh(combined_mesh)

            return combined_mesh

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
    skipped_small = 0

    for face_idx, face in enumerate(input_mesh.vectors):
        v0, v1, v2 = face

        # Optimization: skip subdivision if triangle is already small enough
        if skip_small_triangles:
            e0 = np.linalg.norm(v1 - v0)
            e1 = np.linalg.norm(v2 - v1)
            e2 = np.linalg.norm(v0 - v2)
            min_edge = min(e0, e1, e2)

            # If at least one edge is below point_distance, keep as-is
            if min_edge <= point_distance:
                subdivided = [(v0, v1, v2)]
                skipped_small += 1
            else:
                subdivided = subdivide_triangle(v0, v1, v2, point_distance)
        else:
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

    if skip_small_triangles and skipped_small > 0:
        print(f"Subdivided {len(input_mesh.vectors)} triangles into {triangle_count} triangles")
        print(f"  Optimization: skipped subdivision for {skipped_small} small triangles ({skipped_small/len(input_mesh.vectors)*100:.1f}%)")
    else:
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
    # Start with all vertices enabled
    process_mask = np.ones(len(unique_vertices), dtype=bool)

    # Apply skip_bottom if enabled
    if skip_bottom:
        bottom_mask = unique_vertices[:, 2] > bottom_threshold
        process_mask &= bottom_mask
        bottom_skipped = np.sum(~bottom_mask)
        if bottom_skipped > 0:
            print(f"Skipping {bottom_skipped} bottom layer vertices")

    # Apply blocker if provided
    if blocker_mesh is not None:
        # If double_stl was requested but we got here, it means we're processing the split mesh
        # In that case, we don't need to check vertices again
        if blocker_algorithm == 'double_stl':
            # This shouldn't happen - double_stl should have been handled earlier
            # But if it does, just skip blocker checking since mesh is already split
            print("Warning: double_stl algorithm reached vertex processing (mesh should already be split)")
        else:
            print(f"Checking vertices against blocker volume (algorithm: {blocker_algorithm})...")
            blocker_mask = point_inside_mesh_volume(unique_vertices, blocker_mesh, algorithm=blocker_algorithm)
            process_mask &= ~blocker_mask  # Don't process vertices inside blocker
            blocker_skipped = np.sum(blocker_mask)
            if blocker_skipped > 0:
                print(f"Blocked {blocker_skipped} vertices inside blocker volume")

    # Calculate total skipped count
    skipped_count = np.sum(~process_mask)

    # Get noise values for all vertices to process
    noise_values = np.zeros(len(unique_vertices))
    for i in np.where(process_mask)[0]:
        vertex = unique_vertices[i]
        noise_values[i] = noise_gen.get_noise(vertex[0], vertex[1], vertex[2])

    # Map noise from [-1, 1] to [0, thickness] and apply displacement (vectorized)
    displacement_amounts = (noise_values + 1) * 0.5 * thickness
    displaced_vertices += vertex_normals * displacement_amounts[:, np.newaxis]

    # Zero out displacement for skipped vertices (bottom layer and/or blocker)
    if skipped_count > 0:
        displaced_vertices[~process_mask] = unique_vertices[~process_mask]

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