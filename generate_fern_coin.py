#!/usr/bin/env python3
"""
Generate a coin-like piece with embossed fern design
For Fiuncholabs branding
"""

import numpy as np
from stl import mesh

def create_fern_coin(diameter=40, thickness=3, fern_height=0.8, resolution=100):
    """
    Create a round coin with an embossed fern pattern

    Args:
        diameter: Coin diameter in mm
        thickness: Base coin thickness in mm
        fern_height: Height of embossed fern in mm
        resolution: Number of segments around circle

    Returns:
        mesh.Mesh object
    """
    radius = diameter / 2

    # Generate points for the coin
    vertices = []
    faces = []

    # Create angular resolution for circular shape
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    # Create radial resolution (for embossing)
    num_radial = 40

    def fern_function(x, y):
        """
        Generate a stylized fern pattern
        Returns height offset based on position
        """
        # Normalize coordinates
        r = np.sqrt(x**2 + y**2) / radius
        angle = np.arctan2(y, x)

        # Main stem along y-axis
        stem_width = 0.03
        stem_dist = abs(x) / radius
        on_stem = stem_dist < stem_width and y > -radius * 0.3 and y < radius * 0.6

        # Fronds (branches)
        fern_value = 0

        if on_stem:
            fern_value = 1.0

        # Add fronds at different heights
        num_fronds = 8
        for i in range(num_fronds):
            # Frond position along stem
            frond_y = -radius * 0.2 + (radius * 0.7) * (i / num_fronds)
            frond_angle = 45 if i % 2 == 0 else -45  # Alternate sides

            # Frond length decreases toward top
            frond_length = radius * 0.35 * (1 - i / num_fronds * 0.5)

            # Transform point to frond coordinate system
            dy = y - frond_y
            dx = x

            # Rotate based on frond angle
            if frond_angle > 0:
                # Right side frond
                on_frond = (dx > 0 and dx < frond_length and
                           abs(dy - dx * 0.5) < radius * 0.02 * (1 - dx/frond_length))
            else:
                # Left side frond
                on_frond = (dx < 0 and dx > -frond_length and
                           abs(dy + dx * 0.5) < radius * 0.02 * (1 + dx/frond_length))

            if on_frond:
                # Height based on position along frond
                frond_dist = abs(dx) / frond_length
                fern_value = max(fern_value, 1.0 - frond_dist * 0.3)

        # Smooth falloff at edges
        edge_falloff = max(0, 1 - (r - 0.7) / 0.2)
        fern_value *= edge_falloff

        return fern_value * fern_height

    # Generate mesh using cylindrical approach
    # Bottom surface
    bottom_z = 0

    # Top surface with embossed fern
    top_base_z = thickness

    # Create vertices and faces
    # We'll create a grid in polar coordinates

    vertices_list = []

    # Bottom center point
    bottom_center_idx = len(vertices_list)
    vertices_list.append([0, 0, bottom_z])

    # Bottom circle
    bottom_circle_start = len(vertices_list)
    for t in theta:
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        vertices_list.append([x, y, bottom_z])

    # Top surface - create grid
    radial_steps = np.linspace(0, radius, num_radial)

    top_vertices_grid = []
    for r_step in radial_steps:
        ring = []
        for t in theta:
            x = r_step * np.cos(t)
            y = r_step * np.sin(t)
            z = top_base_z + fern_function(x, y)
            ring.append(len(vertices_list))
            vertices_list.append([x, y, z])
        top_vertices_grid.append(ring)

    # Convert to numpy array
    vertices_array = np.array(vertices_list)

    # Create faces
    faces_list = []

    # Bottom surface (flat)
    for i in range(resolution):
        next_i = (i + 1) % resolution
        faces_list.append([bottom_center_idx,
                          bottom_circle_start + next_i,
                          bottom_circle_start + i])

    # Side walls
    for i in range(resolution):
        next_i = (i + 1) % resolution
        # Bottom edge
        bottom_v1 = bottom_circle_start + i
        bottom_v2 = bottom_circle_start + next_i
        # Top edge (outer ring)
        top_v1 = top_vertices_grid[-1][i]
        top_v2 = top_vertices_grid[-1][next_i]

        # Two triangles for side quad
        faces_list.append([bottom_v1, top_v1, bottom_v2])
        faces_list.append([bottom_v2, top_v1, top_v2])

    # Top surface (embossed)
    for r_idx in range(len(top_vertices_grid) - 1):
        for t_idx in range(resolution):
            next_t = (t_idx + 1) % resolution

            # Four corners of quad
            v1 = top_vertices_grid[r_idx][t_idx]
            v2 = top_vertices_grid[r_idx][next_t]
            v3 = top_vertices_grid[r_idx + 1][t_idx]
            v4 = top_vertices_grid[r_idx + 1][next_t]

            # Two triangles
            faces_list.append([v1, v3, v2])
            faces_list.append([v2, v3, v4])

    # Create mesh
    faces_array = np.array(faces_list)

    # Create the mesh
    fern_mesh = mesh.Mesh(np.zeros(len(faces_array), dtype=mesh.Mesh.dtype))

    for i, face in enumerate(faces_array):
        for j in range(3):
            fern_mesh.vectors[i][j] = vertices_array[face[j]]

    return fern_mesh


if __name__ == '__main__':
    print("Generating Fiuncholabs Fern Coin...")

    # Generate the coin
    fern_coin = create_fern_coin(
        diameter=40,
        thickness=3,
        fern_height=0.8,
        resolution=120
    )

    # Save to file
    output_file = 'fern_coin.stl'
    fern_coin.save(output_file)

    print(f"✓ Fern coin saved to: {output_file}")
    print(f"  Triangles: {len(fern_coin.vectors):,}")
    print(f"  Dimensions: 40mm diameter × 3.8mm height")
    print(f"  Features: Embossed fern design (Fiuncholabs mascot)")
