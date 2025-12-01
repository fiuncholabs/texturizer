"""
Production deployment tests for https://fiuncholabs.com
Tests estimate accuracy, processing capability, and error handling
"""

import unittest
import requests
import time
import os
from pathlib import Path


class TestProductionDeployment(unittest.TestCase):
    """Test suite for production deployment at fiuncholabs.com"""

    BASE_URL = "https://fiuncholabs.com"
    TEST_FILE = "simple-corner.stl"

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Get the path to the test STL file
        cls.test_file_path = Path(__file__).parent.parent / cls.TEST_FILE

        # Verify test file exists
        if not cls.test_file_path.exists():
            raise FileNotFoundError(f"Test file not found: {cls.test_file_path}")

        print(f"\n{'='*70}")
        print(f"Production Deployment Test Suite")
        print(f"Testing: {cls.BASE_URL}")
        print(f"Test file: {cls.TEST_FILE}")
        print(f"{'='*70}\n")

    def test_01_server_reachable(self):
        """Test that the production server is reachable"""
        print("\n[TEST] Server reachability...")
        response = requests.get(self.BASE_URL, timeout=10)
        self.assertEqual(response.status_code, 200, "Server should be reachable")
        print(f"✓ Server is reachable (status: {response.status_code})")

    def test_02_health_endpoint(self):
        """Test the health check endpoint"""
        print("\n[TEST] Health check endpoint...")
        response = requests.get(f"{self.BASE_URL}/health", timeout=10)
        self.assertEqual(response.status_code, 200, "Health endpoint should return 200")

        data = response.json()
        self.assertEqual(data['status'], 'healthy', "Status should be healthy")
        self.assertIn('service', data, "Should include service name")
        print(f"✓ Health check passed")
        print(f"  Service: {data.get('service', 'unknown')}")
        print(f"  Noise library: {data.get('noise_library_available', 'unknown')}")

    def test_03_estimate_endpoint_basic(self):
        """Test estimate endpoint with basic parameters"""
        print("\n[TEST] Estimate endpoint (basic parameters)...")

        with open(self.test_file_path, 'rb') as f:
            files = {'file': (self.TEST_FILE, f, 'application/octet-stream')}
            data = {
                'noise_type': 'classic',
                'amplitude': '0.5',
                'frequency': '1.0',
                'point_distance': '2.0',
                'skip_small_triangles': 'true'
            }

            start_time = time.time()
            response = requests.post(
                f"{self.BASE_URL}/api/estimate",
                files=files,
                data=data,
                timeout=30
            )
            estimate_time = time.time() - start_time

        self.assertEqual(response.status_code, 200, "Estimate should succeed")

        result = response.json()
        self.assertIn('can_process', result, "Should indicate if processing is possible")
        self.assertIn('estimated_time', result, "Should include time estimate")
        self.assertIn('estimated_triangles', result, "Should include triangle count estimate")

        print(f"✓ Estimate completed in {estimate_time:.2f}s")
        print(f"  Can process: {result['can_process']}")
        print(f"  Estimated time: {result['estimated_time']:.1f}s")
        print(f"  Estimated triangles: {result['estimated_triangles']:,}")

        # Store for next test
        self.__class__.basic_estimate = result
        self.__class__.basic_estimate_time = estimate_time

    def test_04_estimate_accuracy_fuzzy_skin(self):
        """Test estimate accuracy for fuzzy skin variant (point_distance=2.0)"""
        print("\n[TEST] Estimate accuracy for fuzzy skin parameters...")

        with open(self.test_file_path, 'rb') as f:
            files = {'file': (self.TEST_FILE, f, 'application/octet-stream')}
            data = {
                'noise_type': 'classic',
                'amplitude': '0.5',
                'frequency': '1.0',
                'point_distance': '2.0',  # Fuzzy skin setting
                'skip_small_triangles': 'true'
            }

            response = requests.post(
                f"{self.BASE_URL}/api/estimate",
                files=files,
                data=data,
                timeout=30
            )

        self.assertEqual(response.status_code, 200, "Estimate should succeed")

        estimate = response.json()

        # Critical: verify that the estimate says we CAN process this
        self.assertTrue(
            estimate['can_process'],
            f"Estimate should indicate fuzzy skin variant is processable. "
            f"Estimated triangles: {estimate.get('estimated_triangles', 'unknown')}"
        )

        print(f"✓ Estimate confirms fuzzy skin variant can be processed")
        print(f"  Estimated triangles: {estimate['estimated_triangles']:,}")
        print(f"  Estimated time: {estimate['estimated_time']:.1f}s")

        # Store for comparison with actual processing
        self.__class__.fuzzy_estimate = estimate

    def test_05_actual_processing_fuzzy_skin(self):
        """Test actual processing and compare with estimate"""
        print("\n[TEST] Actual processing with fuzzy skin parameters...")

        # Skip if estimate said we can't process
        if not hasattr(self.__class__, 'fuzzy_estimate'):
            self.skipTest("Estimate test not run")

        if not self.__class__.fuzzy_estimate['can_process']:
            self.skipTest("Estimate indicated processing not possible")

        with open(self.test_file_path, 'rb') as f:
            files = {'file': (self.TEST_FILE, f, 'application/octet-stream')}
            data = {
                'noise_type': 'classic',
                'amplitude': '0.5',
                'frequency': '1.0',
                'point_distance': '2.0',
                'skip_small_triangles': 'true'
            }

            start_time = time.time()
            response = requests.post(
                f"{self.BASE_URL}/api/process",
                files=files,
                data=data,
                timeout=300  # 5 minutes max
            )
            actual_time = time.time() - start_time

        self.assertEqual(response.status_code, 200, "Processing should succeed")
        self.assertEqual(
            response.headers.get('Content-Type'),
            'application/octet-stream',
            "Should return STL file"
        )

        # Verify we got a valid STL file
        stl_data = response.content
        self.assertGreater(len(stl_data), 0, "Should return non-empty STL file")

        # Compare with estimate
        estimate = self.__class__.fuzzy_estimate
        estimated_time = estimate['estimated_time']
        time_diff = abs(actual_time - estimated_time)
        time_accuracy = (1 - time_diff / max(actual_time, estimated_time)) * 100

        print(f"✓ Processing completed successfully")
        print(f"  Actual time: {actual_time:.1f}s")
        print(f"  Estimated time: {estimated_time:.1f}s")
        print(f"  Time accuracy: {time_accuracy:.1f}%")
        print(f"  Output size: {len(stl_data):,} bytes")

        # Warn if estimate is very inaccurate (more than 100% off)
        if time_accuracy < 50:
            print(f"  ⚠ WARNING: Time estimate was significantly off")

        # Store results
        self.__class__.actual_processing_time = actual_time
        self.__class__.output_size = len(stl_data)

    def test_06_estimate_with_fine_detail(self):
        """Test estimate with fine detail settings (point_distance=0.5)"""
        print("\n[TEST] Estimate with fine detail (point_distance=0.5)...")

        with open(self.test_file_path, 'rb') as f:
            files = {'file': (self.TEST_FILE, f, 'application/octet-stream')}
            data = {
                'noise_type': 'classic',
                'amplitude': '0.5',
                'frequency': '1.0',
                'point_distance': '0.5',  # Very fine detail
                'skip_small_triangles': 'true'
            }

            response = requests.post(
                f"{self.BASE_URL}/api/estimate",
                files=files,
                data=data,
                timeout=30
            )

        self.assertEqual(response.status_code, 200, "Estimate should succeed")

        estimate = response.json()

        print(f"✓ Fine detail estimate completed")
        print(f"  Can process: {estimate['can_process']}")
        print(f"  Estimated triangles: {estimate.get('estimated_triangles', 0):,}")
        print(f"  Estimated time: {estimate.get('estimated_time', 0):.1f}s")

        if not estimate['can_process']:
            print(f"  Reason: {estimate.get('reason', 'Unknown')}")

        # Store for reference
        self.__class__.fine_detail_estimate = estimate

    def test_07_error_handling_no_file(self):
        """Test error handling when no file is uploaded"""
        print("\n[TEST] Error handling (no file)...")

        data = {
            'noise_type': 'classic',
            'amplitude': '0.5',
            'frequency': '1.0',
            'point_distance': '2.0'
        }

        response = requests.post(
            f"{self.BASE_URL}/api/process",
            data=data,
            timeout=30
        )

        self.assertEqual(response.status_code, 400, "Should return 400 for missing file")
        print(f"✓ Correctly rejects request without file (status: {response.status_code})")

    def test_08_error_handling_invalid_params(self):
        """Test error handling with invalid parameters"""
        print("\n[TEST] Error handling (invalid parameters)...")

        with open(self.test_file_path, 'rb') as f:
            files = {'file': (self.TEST_FILE, f, 'application/octet-stream')}
            data = {
                'noise_type': 'invalid_noise_type',
                'amplitude': '999',
                'frequency': '-1.0',
                'point_distance': '2.0'
            }

            response = requests.post(
                f"{self.BASE_URL}/api/estimate",
                files=files,
                data=data,
                timeout=30
            )

        # Should either reject (400) or handle gracefully
        self.assertIn(response.status_code, [200, 400], "Should handle invalid params")
        print(f"✓ Handles invalid parameters (status: {response.status_code})")

    def test_09_estimate_consistency(self):
        """Test that estimate is consistent across multiple calls"""
        print("\n[TEST] Estimate consistency...")

        estimates = []

        for i in range(3):
            with open(self.test_file_path, 'rb') as f:
                files = {'file': (self.TEST_FILE, f, 'application/octet-stream')}
                data = {
                    'noise_type': 'classic',
                    'amplitude': '0.5',
                    'frequency': '1.0',
                    'point_distance': '2.0',
                    'skip_small_triangles': 'true'
                }

                response = requests.post(
                    f"{self.BASE_URL}/api/estimate",
                    files=files,
                    data=data,
                    timeout=30
                )

                self.assertEqual(response.status_code, 200)
                estimates.append(response.json())

        # All estimates should be identical for same input
        first = estimates[0]
        for est in estimates[1:]:
            self.assertEqual(
                est['estimated_triangles'],
                first['estimated_triangles'],
                "Triangle count estimate should be consistent"
            )
            self.assertEqual(
                est['can_process'],
                first['can_process'],
                "Processing capability should be consistent"
            )

        print(f"✓ Estimates are consistent across {len(estimates)} calls")
        print(f"  Triangles: {first['estimated_triangles']:,}")
        print(f"  Can process: {first['can_process']}")

    @classmethod
    def tearDownClass(cls):
        """Print summary"""
        print(f"\n{'='*70}")
        print("Test Summary")
        print(f"{'='*70}")

        if hasattr(cls, 'fuzzy_estimate'):
            print(f"\nFuzzy Skin Estimate:")
            print(f"  Can process: {cls.fuzzy_estimate['can_process']}")
            print(f"  Estimated triangles: {cls.fuzzy_estimate['estimated_triangles']:,}")
            print(f"  Estimated time: {cls.fuzzy_estimate['estimated_time']:.1f}s")

        if hasattr(cls, 'actual_processing_time'):
            print(f"\nActual Processing:")
            print(f"  Time: {cls.actual_processing_time:.1f}s")
            print(f"  Output size: {cls.output_size:,} bytes")

        if hasattr(cls, 'fine_detail_estimate'):
            print(f"\nFine Detail Estimate:")
            print(f"  Can process: {cls.fine_detail_estimate['can_process']}")
            if cls.fine_detail_estimate['can_process']:
                print(f"  Estimated triangles: {cls.fine_detail_estimate['estimated_triangles']:,}")
                print(f"  Estimated time: {cls.fine_detail_estimate['estimated_time']:.1f}s")
            else:
                print(f"  Reason: {cls.fine_detail_estimate.get('reason', 'Unknown')}")

        print(f"\n{'='*70}\n")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
