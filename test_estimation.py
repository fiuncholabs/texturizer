#!/usr/bin/env python3
"""
Test script to debug estimation with simple-corner.stl
"""
import logging
from stl import mesh
from texturizer import estimate_output_size

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Try to load simple-corner.stl (user might need to provide path)
import sys
if len(sys.argv) > 1:
    stl_file = sys.argv[1]
else:
    print("Usage: python3 test_estimation.py <path_to_simple-corner.stl>")
    print("\nTesting with default cube instead...")
    from texturizer import generate_test_cube
    test_mesh = generate_test_cube(20)
    print(f"Loaded default cube: {len(test_mesh.vectors)} triangles\n")
    stl_file = None

if stl_file:
    print(f"Loading {stl_file}...")
    test_mesh = mesh.Mesh.from_file(stl_file)
    print(f"Loaded: {len(test_mesh.vectors)} triangles\n")

# Test estimation
point_distance = 0.8
estimate = estimate_output_size(test_mesh, point_distance)

print("\n=== ESTIMATION RESULTS ===")
print(f"Input triangles: {estimate['input_triangles']:,}")
print(f"Estimated output: {estimate['estimated_triangles']:,}")
print(f"Subdivision factor: {estimate['subdivision_factor']:.0f}x")
print(f"Estimated time: {estimate['estimated_time_seconds']:.1f}s")
print(f"Estimated file size: {estimate['estimated_file_size_mb']:.1f} MB")

if stl_file and 'simple-corner' in stl_file:
    print("\n=== EXPECTED FOR SIMPLE-CORNER ===")
    print("Actual output: ~3,029,394 triangles")
    print("Actual time: ~30.5 seconds")
    print("Actual file size: ~148 MB")

    error_ratio = estimate['estimated_triangles'] / 3029394
    print(f"\nEstimation accuracy: {error_ratio:.2f}x actual")
    if error_ratio < 0.5 or error_ratio > 2.0:
        print("⚠️  WARNING: Estimation is significantly off!")
    else:
        print("✓ Estimation is reasonable")
