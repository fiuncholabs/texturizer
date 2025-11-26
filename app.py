#!/usr/bin/env python3
"""
Flask web application for STL Fuzzy Skin Texturizer
"""

from flask import Flask, render_template, request, send_file, jsonify
import tempfile
import os
import io
from stl import mesh
import numpy as np

# Import from texturizer
from texturizer import (
    apply_fuzzy_skin,
    NOISE_TYPES,
    NOISE_CLASSIC,
    NOISE_AVAILABLE
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

@app.route('/')
def index():
    return render_template('index.html', noise_types=NOISE_TYPES)

@app.route('/api/process', methods=['POST'])
def process_stl():
    try:
        # Check for file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.stl'):
            return jsonify({'error': 'File must be an STL'}), 400

        # Get parameters
        thickness = float(request.form.get('thickness', 0.3))
        point_distance = float(request.form.get('point_distance', 0.8))
        seed = int(request.form.get('seed', 42))
        noise_type = request.form.get('noise_type', NOISE_CLASSIC)
        noise_scale = float(request.form.get('noise_scale', 1.0))
        noise_octaves = int(request.form.get('noise_octaves', 4))
        noise_persistence = float(request.form.get('noise_persistence', 0.5))
        skip_bottom = request.form.get('skip_bottom', 'false').lower() == 'true'

        # Validate noise type
        if noise_type not in NOISE_TYPES:
            return jsonify({'error': f'Invalid noise type: {noise_type}'}), 400

        if noise_type != NOISE_CLASSIC and not NOISE_AVAILABLE:
            return jsonify({'error': f'Noise type {noise_type} requires the noise library'}), 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_in:
            file.save(tmp_in.name)
            tmp_input_path = tmp_in.name

        try:
            # Load mesh
            input_mesh = mesh.Mesh.from_file(tmp_input_path)

            # Apply fuzzy skin
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
            except MemoryError:
                return jsonify({'error': 'Out of memory. Try increasing point_distance or using a smaller mesh.'}), 500
            except ValueError as ve:
                return jsonify({'error': f'Processing error: {str(ve)}'}), 500

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_out:
                output_mesh.save(tmp_out.name)
                tmp_output_path = tmp_out.name

            # Read the output file and send it
            with open(tmp_output_path, 'rb') as f:
                output_data = f.read()

            # Clean up output temp file
            os.unlink(tmp_output_path)

            # Generate output filename
            base_name = os.path.splitext(file.filename)[0]
            output_filename = f"{base_name}_fuzzy.stl"

            return send_file(
                io.BytesIO(output_data),
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name=output_filename
            )

        finally:
            # Clean up input temp file
            if os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
            'skip_bottom': False
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
