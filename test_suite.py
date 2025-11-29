#!/usr/bin/env python3
"""
Unit tests for STL Texturizer
Tests estimation accuracy and processing correctness
"""
import unittest
import os
import tempfile
import numpy as np
from stl import mesh
from texturizer import (
    estimate_output_size,
    apply_fuzzy_skin,
    generate_test_cube,
    subdivide_triangle
)


class TestSubdivision(unittest.TestCase):
    """Test the core subdivision algorithm"""

    def test_subdivide_triangle_no_subdivision(self):
        """Test that small triangles don't get subdivided"""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([0.5, 0.0, 0.0])
        v2 = np.array([0.25, 0.43, 0.0])

        # All edges < 0.8mm, should return 1 triangle
        result = subdivide_triangle(v0, v1, v2, max_edge_length=0.8)
        self.assertEqual(len(result), 1)

    def test_subdivide_triangle_one_level(self):
        """Test single level subdivision"""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([2.0, 0.0, 0.0])
        v2 = np.array([1.0, 1.73, 0.0])

        # Edges ~2mm, ratio=2.5, needs 2 levels (16 triangles: 4^2)
        result = subdivide_triangle(v0, v1, v2, max_edge_length=0.8)
        self.assertEqual(len(result), 16)

    def test_subdivide_triangle_multiple_levels(self):
        """Test multiple levels of subdivision"""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([20.0, 0.0, 0.0])
        v2 = np.array([10.0, 17.32, 0.0])

        # Edges ~20mm, needs multiple levels
        result = subdivide_triangle(v0, v1, v2, max_edge_length=0.8)
        # Should be 1024 triangles (4^5, since 20/0.8 = 25, ceil(log2(25))=5)
        self.assertEqual(len(result), 1024)


class TestEstimation(unittest.TestCase):
    """Test estimation accuracy"""

    def test_cube_estimation(self):
        """Test estimation on default cube"""
        cube = generate_test_cube(20)
        estimate = estimate_output_size(cube, point_distance=0.8)

        # Cube should estimate 49,152 triangles exactly
        self.assertEqual(estimate['input_triangles'], 12)
        self.assertEqual(estimate['estimated_triangles'], 49152)
        self.assertEqual(estimate['subdivision_factor'], 4096)

    def test_no_subdivision_estimation(self):
        """Test estimation when no subdivision is needed"""
        # Create a tiny mesh with edges < 0.8mm
        vertices = np.array([
            [[0, 0, 0], [0.5, 0, 0], [0.25, 0.43, 0]],
            [[0, 0, 0], [0.25, 0.43, 0], [-0.25, 0.43, 0]]
        ])
        tiny_mesh = mesh.Mesh(np.zeros(2, dtype=mesh.Mesh.dtype))
        tiny_mesh.vectors = vertices

        estimate = estimate_output_size(tiny_mesh, point_distance=0.8)

        # Should estimate no subdivision
        self.assertEqual(estimate['subdivision_factor'], 1.0)
        self.assertEqual(estimate['estimated_triangles'], 2)

    def test_estimation_within_reasonable_bounds(self):
        """Test that estimations are within 10x of expected for varied meshes"""
        # This test would need actual test STL files
        # For now, we test the cube which we know is exact
        cube = generate_test_cube(20)
        estimate = estimate_output_size(cube, point_distance=0.8)

        # Known actual result for cube: 49,152
        actual = 49152
        estimated = estimate['estimated_triangles']

        # Should be exact for uniform meshes
        self.assertEqual(estimated, actual)

        # Ratio should be 1.0
        ratio = estimated / actual
        self.assertGreaterEqual(ratio, 0.1)  # Not more than 10x underestimate
        self.assertLessEqual(ratio, 10.0)    # Not more than 10x overestimate


class TestProcessing(unittest.TestCase):
    """Test full processing pipeline"""

    def test_process_cube(self):
        """Test processing the default cube"""
        cube = generate_test_cube(20)

        # Process with fuzzy skin
        result = apply_fuzzy_skin(
            cube,
            thickness=0.3,
            point_distance=0.8,
            seed=42,
            noise_type='classic'
        )

        # Should produce 49,152 triangles
        self.assertEqual(len(result.vectors), 49152)

    def test_process_preserves_mesh_structure(self):
        """Test that processing creates valid STL"""
        cube = generate_test_cube(20)
        result = apply_fuzzy_skin(cube, thickness=0.3, point_distance=0.8)

        # Check mesh is valid
        self.assertIsNotNone(result.vectors)
        self.assertEqual(result.vectors.shape[1], 3)  # 3 vertices
        self.assertEqual(result.vectors.shape[2], 3)  # 3 coordinates (x,y,z)

        # Check all coordinates are finite
        self.assertTrue(np.all(np.isfinite(result.vectors)))

    def test_process_with_different_point_distances(self):
        """Test that different point_distance values produce different subdivisions"""
        cube = generate_test_cube(20)

        # Larger point_distance = less subdivision
        result_large = apply_fuzzy_skin(cube, thickness=0.3, point_distance=2.0)
        result_small = apply_fuzzy_skin(cube, thickness=0.3, point_distance=0.4)

        # Smaller point_distance should create MORE triangles
        self.assertGreater(
            len(result_small.vectors),
            len(result_large.vectors)
        )

    def test_zero_thickness_returns_subdivided_mesh(self):
        """Test that thickness=0 still subdivides but doesn't displace"""
        cube = generate_test_cube(20)

        result_zero = apply_fuzzy_skin(cube, thickness=0.0, point_distance=0.8)
        result_normal = apply_fuzzy_skin(cube, thickness=0.3, point_distance=0.8)

        # Both should have same number of triangles (same subdivision)
        self.assertEqual(len(result_zero.vectors), len(result_normal.vectors))


class TestFileIO(unittest.TestCase):
    """Test file input/output"""

    def test_save_and_load_mesh(self):
        """Test that we can save and load processed meshes"""
        cube = generate_test_cube(20)
        result = apply_fuzzy_skin(cube, thickness=0.3, point_distance=0.8)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
            result.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Load it back
            loaded = mesh.Mesh.from_file(tmp_path)

            # Should have same number of triangles
            self.assertEqual(len(loaded.vectors), len(result.vectors))

            # Vertices should be very close (accounting for float precision)
            np.testing.assert_array_almost_equal(
                loaded.vectors,
                result.vectors,
                decimal=5
            )
        finally:
            os.unlink(tmp_path)


def run_tests():
    """Run all tests and print results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSubdivision))
    suite.addTests(loader.loadTestsFromTestCase(TestEstimation))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestFileIO))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    import sys
    sys.exit(run_tests())
