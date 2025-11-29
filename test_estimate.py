#!/usr/bin/env python3
"""
Test script to debug the estimation calculation
"""
import sys
from stl import mesh
import texturizer

if len(sys.argv) < 2:
    print("Usage: python3 test_estimate.py <stl_file> [point_distance]")
    sys.exit(1)

stl_file = sys.argv[1]
point_distance = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8

print(f"Loading {stl_file}...")
input_mesh = mesh.Mesh.from_file(stl_file)
print(f"Input mesh has {len(input_mesh.vectors)} triangles")

print(f"\nEstimating with point_distance={point_distance}...")
estimate = texturizer.estimate_output_size(input_mesh, point_distance)

print(f"\n=== ESTIMATION RESULTS ===")
print(f"Input triangles: {estimate['input_triangles']:,}")
print(f"Estimated output triangles: {estimate['estimated_triangles']:,}")
print(f"Subdivision factor: {estimate['subdivision_factor']:.2f}")
print(f"Average edge length: {estimate['avg_edge_length']:.4f} mm")
print(f"Estimated file size: {estimate['estimated_file_size_mb']:.2f} MB")
print(f"Estimated memory: {estimate['estimated_memory_mb']:.2f} MB")
print(f"Estimated time: {estimate['estimated_time_seconds']:.1f} seconds")
