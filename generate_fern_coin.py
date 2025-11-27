#!/usr/bin/env python3
"""
Generate a coin-like piece with embossed fern design
For Fiuncholabs branding - Improved realistic version
"""

import numpy as np
from stl import mesh

def create_fern_coin(diameter=40, thickness=3, fern_height=0.25, resolution=120):
    """
    Create a round coin with an embossed realistic fern pattern

    Args:
        diameter: Coin diameter in mm
        thickness: Base coin thickness in mm
        fern_height: Height of embossed fern in mm (shallow for aesthetics)
        resolution: Number of segments around circle

    Returns:
        mesh.Mesh object
    """
    radius = diameter / 2
    num_radial = max(30, int(diameter * 1.5))  # More radial steps for smoother embossing

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
                # Distance along frond (0 to 1)
                frond_progress = x / frond_length

                # Main frond rachis (central axis)
                frond_center_y = frond_y + x * frond_angle
                dist_to_rachis = abs(y - frond_center_y) / radius

                # Frond width tapers toward tip
                frond_width = 0.015 * (1 - frond_progress)

                if dist_to_rachis < frond_width:
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
                # Distance along frond (0 to 1)
                frond_progress = abs(x) / frond_length

                frond_center_y = frond_y + abs(x) * frond_angle
                dist_to_rachis = abs(y - frond_center_y) / radius

                frond_width = 0.015 * (1 - frond_progress)

                if dist_to_rachis < frond_width:
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

    # Create angular resolution for circular shape
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    vertices_list = []

    # Bottom center point
    bottom_center_idx = len(vertices_list)
    vertices_list.append([0, 0, 0])

    # Bottom circle
    bottom_circle_start = len(vertices_list)
    for t in theta:
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        vertices_list.append([x, y, 0])

    # Top surface - create grid with embossing
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
    print("Generating Improved Fiuncholabs Fern Coin...")

    # Generate the coin with improved parameters
    fern_coin = create_fern_coin(
        diameter=40,
        thickness=3,
        fern_height=0.25,  # Reduced from 0.8mm to 0.25mm for subtler embossing
        resolution=160     # Increased resolution for better detail
    )

    # Save to file
    output_file = 'fern_coin.stl'
    fern_coin.save(output_file)

    print(f"✓ Improved fern coin saved to: {output_file}")
    print(f"  Triangles: {len(fern_coin.vectors):,}")
    print(f"  Dimensions: 40mm diameter × ~3.25mm height")
    print(f"  Features: Realistic embossed fern with pinnae (leaflets)")
    print(f"  Embossing depth: 0.25mm (shallow for aesthetic appeal)")
