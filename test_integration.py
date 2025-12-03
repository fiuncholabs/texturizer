#!/usr/bin/env python3
"""
Integration test for blocker functionality.
Tests the complete workflow: frontend -> backend API -> processing.
"""

import requests
import os
import io
from stl import mesh

def test_default_cylinder_blocker():
    """Test processing with default cylinder blocker"""
    print("\n=== Integration Test: Default Cylinder Blocker ===")

    # Check if simple-corner.stl exists
    if not os.path.exists('simple-corner.stl'):
        print("⊘ simple-corner.stl not found, skipping test")
        return False

    # Read the STL file
    with open('simple-corner.stl', 'rb') as f:
        stl_data = f.read()

    # Prepare form data
    files = {
        'file': ('simple-corner.stl', stl_data, 'application/octet-stream')
    }

    data = {
        'use_default_cube': 'false',
        'thickness': '0.3',
        'point_distance': '2.0',  # Larger for faster test
        'seed': '42',
        'noise_type': 'classic',
        'skip_bottom': 'false',
        'skip_small_triangles': 'true',
        'use_blocker': 'true',
        'use_default_cylinder': 'true',
        'cylinder_radius': '15',
        'cylinder_height': '40'
    }

    print("Sending request to API...")
    print(f"  - Input: simple-corner.stl")
    print(f"  - Blocker: Default cylinder (radius=15mm, height=40mm)")
    print(f"  - Point distance: 2.0mm")

    try:
        response = requests.post(
            'http://localhost:5000/api/process',
            files=files,
            data=data,
            timeout=300  # 5 minute timeout
        )

        if response.status_code == 200:
            # Save output
            output_path = 'test_integration_output.stl'
            with open(output_path, 'wb') as f:
                f.write(response.content)

            # Load and verify output
            output_mesh = mesh.Mesh.from_file(output_path)
            print(f"✓ Processing successful!")
            print(f"  - Output saved to: {output_path}")
            print(f"  - Output triangles: {len(output_mesh.vectors):,}")

            return True
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("✗ Request timed out (processing took > 5 minutes)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_default_cube_with_blocker():
    """Test processing default cube with blocker"""
    print("\n=== Integration Test: Default Cube with Blocker ===")

    data = {
        'use_default_cube': 'true',
        'cube_size': '20',
        'thickness': '0.5',
        'point_distance': '2.0',
        'seed': '42',
        'noise_type': 'classic',
        'skip_bottom': 'false',
        'skip_small_triangles': 'false',
        'use_blocker': 'true',
        'use_default_cylinder': 'true',
        'cylinder_radius': '8',
        'cylinder_height': '25'
    }

    print("Sending request to API...")
    print(f"  - Input: Default cube (20mm)")
    print(f"  - Blocker: Default cylinder (radius=8mm, height=25mm)")

    try:
        response = requests.post(
            'http://localhost:5000/api/process',
            data=data,
            timeout=60
        )

        if response.status_code == 200:
            output_path = 'test_cube_integration.stl'
            with open(output_path, 'wb') as f:
                f.write(response.content)

            output_mesh = mesh.Mesh.from_file(output_path)
            print(f"✓ Processing successful!")
            print(f"  - Output saved to: {output_path}")
            print(f"  - Output triangles: {len(output_mesh.vectors):,}")

            return True
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Integration Testing: Blocker Functionality")
    print("=" * 60)
    print("\nNOTE: This test requires Flask app running on localhost:5000")
    print("Start the app with: python3 app.py")

    input("\nPress Enter to continue (or Ctrl+C to cancel)...")

    results = []

    # Test 1: Default cube with blocker
    try:
        success = test_default_cube_with_blocker()
        results.append(("Default Cube with Blocker", success))
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results.append(("Default Cube with Blocker", False))

    # Test 2: Simple corner with blocker (optional)
    try:
        success = test_default_cylinder_blocker()
        results.append(("Simple Corner with Blocker", success))
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results.append(("Simple Corner with Blocker", False))

    # Summary
    print("\n" + "=" * 60)
    print("Integration Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\n{passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print("\n⚠️  Some integration tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
