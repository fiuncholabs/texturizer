#!/usr/bin/env python3
"""
Test trimesh boolean operations directly to debug the issue.
"""

import numpy as np
import trimesh
from texturizer import generate_simple_cube, generate_blocker_cylinder

# Generate meshes
print("Generating meshes...")
cube = generate_simple_cube(size=20)
cylinder = generate_blocker_cylinder(radius=8, height=25, position=(0, 0, 0), segments=32)

print(f"Cube: {len(cube.vectors)} triangles")
print(f"Cylinder: {len(cylinder.vectors)} triangles")

# Also try trimesh's own cylinder
print("\nCreating trimesh native cylinder...")
cyl_native = trimesh.creation.cylinder(radius=8, height=25, sections=32)
print(f"Native cylinder: {len(cyl_native.faces)} faces")
print(f"  Watertight: {cyl_native.is_watertight}")
print(f"  Is volume: {cyl_native.is_volume}")

# Convert to trimesh
def stl_to_trimesh(stl_mesh):
    """Convert numpy-stl mesh to trimesh"""
    all_verts = stl_mesh.vectors.reshape(-1, 3)
    return trimesh.Trimesh(vertices=all_verts,
                          faces=np.arange(len(all_verts)).reshape(-1, 3),
                          process=True)

cube_tm = stl_to_trimesh(cube)
cyl_tm = stl_to_trimesh(cylinder)

print(f"\nCube trimesh:")
print(f"  Vertices: {len(cube_tm.vertices)}")
print(f"  Faces: {len(cube_tm.faces)}")
print(f"  Watertight: {cube_tm.is_watertight}")
print(f"  Is volume: {cube_tm.is_volume}")
print(f"  Euler number: {cube_tm.euler_number}")

print(f"\nCylinder trimesh:")
print(f"  Vertices: {len(cyl_tm.vertices)}")
print(f"  Faces: {len(cyl_tm.faces)}")
print(f"  Watertight: {cyl_tm.is_watertight}")
print(f"  Is volume: {cyl_tm.is_volume}")
print(f"  Euler number: {cyl_tm.euler_number}")

# Check if they're valid volumes
print(f"\nTrying boolean operation with our meshes...")
try:
    result = cube_tm.difference(cyl_tm)
    print(f"✓ Success! Result has {len(result.faces)} faces")
except Exception as e:
    print(f"✗ Failed: {e}")

# Try with native cylinder
print(f"\nTrying boolean operation with native cylinder...")
try:
    result = cube_tm.difference(cyl_native)
    print(f"✓ Success with native! Result has {len(result.faces)} faces")

    # Save for inspection
    from stl import mesh as stl_mesh
    result_stl = stl_mesh.Mesh(np.zeros(len(result.faces), dtype=stl_mesh.Mesh.dtype))
    for i, face in enumerate(result.faces):
        for j in range(3):
            result_stl.vectors[i][j] = result.vertices[face[j]]
    result_stl.save('test_boolean_result.stl')
    print("  Saved to test_boolean_result.stl")

except Exception as e:
    print(f"✗ Failed with native: {e}")

# Try fixing the meshes
print(f"\nTrying to fix meshes...")
cube_tm.fix_normals()
cyl_tm.fix_normals()

print(f"After fix_normals():")
print(f"  Cube is_volume: {cube_tm.is_volume}")
print(f"  Cylinder is_volume: {cyl_tm.is_volume}")

try:
    result = cube_tm.difference(cyl_tm)
    print(f"✓ Success after fix! Result has {len(result.faces)} faces")
except Exception as e:
    print(f"✗ Still failed: {e}")
