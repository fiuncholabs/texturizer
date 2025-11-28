#!/usr/bin/env python3
"""
Debug version of Flask app with better error reporting
Run this if you're getting HTML error pages instead of JSON
"""

from flask import Flask, render_template, request, send_file, jsonify
import tempfile
import os
import io
import traceback
from stl import mesh
import numpy as np

# Import from texturizer
try:
    from texturizer import (
        apply_fuzzy_skin,
        generate_test_cube,
        NOISE_TYPES,
        NOISE_CLASSIC,
        NOISE_AVAILABLE
    )
    print("✓ Successfully imported from texturizer")
except ImportError as e:
    print(f"✗ Failed to import from texturizer: {e}")
    raise

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['DEBUG'] = True

# Global error handler to always return JSON
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all exceptions and return JSON"""
    app.logger.error(f"Unhandled exception: {str(e)}")
    app.logger.error(traceback.format_exc())

    # Return JSON error instead of HTML
    return jsonify({
        'error': str(e),
        'type': type(e).__name__,
        'traceback': traceback.format_exc()
    }), 500

@app.route('/')
def index():
    try:
        return render_template('index.html', noise_types=NOISE_TYPES, config={})
    except Exception as e:
        return jsonify({'error': f'Template error: {str(e)}'}), 500

@app.route('/api/process', methods=['POST'])
def process_stl():
    tmp_input_path = None
    tmp_output_path = None

    try:
        print("\n=== Processing Request ===")

        # Get parameters
        thickness = float(request.form.get('thickness', 0.3))
        point_distance = float(request.form.get('point_distance', 0.8))
        seed = int(request.form.get('seed', 42))
        noise_type = request.form.get('noise_type', NOISE_CLASSIC)
        noise_scale = float(request.form.get('noise_scale', 1.0))
        noise_octaves = int(request.form.get('noise_octaves', 4))
        noise_persistence = float(request.form.get('noise_persistence', 0.5))
        skip_bottom = request.form.get('skip_bottom', 'false').lower() == 'true'
        use_default_cube = request.form.get('use_default_cube', 'false').lower() == 'true'
        cube_size = float(request.form.get('cube_size', 20))

        print(f"Parameters: thickness={thickness}, point_distance={point_distance}, noise={noise_type}")
        print(f"Use default cube: {use_default_cube}, cube_size={cube_size}")

        # Validate parameters
        if not (0.0 <= thickness <= 5.0):
            return jsonify({'error': 'Thickness must be between 0.0 and 5.0 mm'}), 400
        if thickness > 0 and thickness < 0.05:
            return jsonify({'error': 'Thickness must be 0 or >= 0.05 mm'}), 400
        if not (0.1 <= point_distance <= 10.0):
            return jsonify({'error': 'Point distance must be between 0.1 and 10.0 mm'}), 400

        # Validate noise type
        if noise_type not in NOISE_TYPES:
            return jsonify({'error': f'Invalid noise type: {noise_type}'}), 400

        if noise_type != NOISE_CLASSIC and not NOISE_AVAILABLE:
            return jsonify({'error': f'Noise type {noise_type} requires the noise library'}), 400

        # Load mesh
        output_filename = 'fuzzy_output.stl'

        if use_default_cube:
            print(f"Generating default cube/coin of size {cube_size}mm")
            input_mesh = generate_test_cube(cube_size)
            output_filename = f'cube_{cube_size}mm_fuzzy.stl'
        else:
            # Check for uploaded file
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded and default cube not selected'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            if not file.filename.lower().endswith('.stl'):
                return jsonify({'error': 'File must be an STL'}), 400

            print(f"Loading file: {file.filename}")

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_in:
                file.save(tmp_in.name)
                tmp_input_path = tmp_in.name

            try:
                input_mesh = mesh.Mesh.from_file(tmp_input_path)
                print(f"✓ Loaded STL with {len(input_mesh.vectors)} triangles")
            except Exception as e:
                print(f"✗ Failed to load STL: {e}")
                return jsonify({'error': f'Invalid STL file: {str(e)}'}), 400

            # Generate output filename
            base_name = os.path.splitext(file.filename)[0]
            output_filename = f"{base_name}_fuzzy.stl"

        print(f"Input mesh has {len(input_mesh.vectors)} triangles")

        # If thickness is 0, skip processing
        if thickness == 0:
            print("Thickness is 0 - returning unprocessed mesh")
            output_mesh = input_mesh
        else:
            # Apply fuzzy skin
            print("Applying fuzzy skin...")
            try:
                output_mesh = apply_fuzzy_skin(
                    input_mesh,
                    thickness=thickness,
                    point_distance=point_distance,
                    seed=seed,
                    noise_type=noise_type,
                    noise_scale=noise_scale,
                    noise_octaves=noise_octaves,
                    noise_persistence=noise_persistence,
                    skip_bottom=skip_bottom
                )
                print(f"✓ Processing complete - {len(output_mesh.vectors)} triangles")
            except MemoryError as e:
                print(f"✗ Out of memory: {e}")
                return jsonify({'error': 'Out of memory. Try increasing point_distance or using a smaller mesh.'}), 500
            except ValueError as ve:
                print(f"✗ Processing error: {ve}")
                return jsonify({'error': f'Processing error: {str(ve)}'}), 500

        # Save to temporary file
        print("Saving output file...")
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_out:
            output_mesh.save(tmp_out.name)
            tmp_output_path = tmp_out.name

        # Read the output file and send it
        with open(tmp_output_path, 'rb') as f:
            output_data = f.read()

        print(f"✓ Success - output size: {len(output_data) / 1024:.1f} KB")
        print("=== Request Complete ===\n")

        return send_file(
            io.BytesIO(output_data),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=output_filename
        )

    except Exception as e:
        print(f"✗ ERROR: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

    finally:
        # Clean up temporary files
        if tmp_input_path and os.path.exists(tmp_input_path):
            try:
                os.unlink(tmp_input_path)
            except Exception:
                pass

        if tmp_output_path and os.path.exists(tmp_output_path):
            try:
                os.unlink(tmp_output_path)
            except Exception:
                pass

@app.route('/api/info')
def get_info():
    """Return available noise types and defaults"""
    return jsonify({
        'noise_types': NOISE_TYPES,
        'noise_available': NOISE_AVAILABLE,
        'defaults': {
            'thickness': 0.3,
            'point_distance': 0.8,
            'seed': 42,
            'noise_type': NOISE_CLASSIC,
            'noise_scale': 1.0,
            'noise_octaves': 4,
            'noise_persistence': 0.5,
            'skip_bottom': True
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print("\n" + "="*50)
    print("STL Fuzzy Skin Texturizer - DEBUG MODE")
    print("="*50)
    print(f"Server starting on http://localhost:{port}")
    print("All errors will be shown with full tracebacks")
    print("="*50 + "\n")

    app.run(host='0.0.0.0', port=port, debug=True)
