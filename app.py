#!/usr/bin/env python3
"""
Flask web application for STL Fuzzy Skin Texturizer
Production-ready version with security, logging, and monitoring
"""

from flask import Flask, render_template, request, send_file, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
import tempfile
import os
import io
import logging
import sys
import time
from functools import wraps
from stl import mesh
import numpy as np

# Import from texturizer
from texturizer import (
    apply_fuzzy_skin,
    generate_test_cube,
    estimate_output_size,
    check_processing_feasibility,
    NOISE_TYPES,
    NOISE_CLASSIC,
    NOISE_AVAILABLE
)

# Import configuration
from config import get_config

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config_class = get_config()
app.config.from_object(config_class)

# Initialize logging
def setup_logging():
    """Configure application logging"""
    log_level = getattr(logging, app.config['LOG_LEVEL'].upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure app logger
    app.logger.setLevel(log_level)
    app.logger.addHandler(console_handler)

    # File handler if specified
    if app.config.get('LOG_FILE'):
        file_handler = logging.FileHandler(app.config['LOG_FILE'])
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)

    # Suppress werkzeug logs in production
    if not app.config.get('DEBUG'):
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

setup_logging()

# Initialize CORS if enabled
if app.config.get('CORS_ENABLED'):
    CORS(app, origins=app.config.get('CORS_ORIGINS', '*'))
    app.logger.info(f"CORS enabled for origins: {app.config.get('CORS_ORIGINS')}")

# Initialize rate limiter if enabled
limiter = None
if app.config.get('RATELIMIT_ENABLED'):
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        storage_uri=app.config.get('RATELIMIT_STORAGE_URL', 'memory://'),
        default_limits=[app.config.get('RATELIMIT_DEFAULT', '10 per minute')]
    )
    app.logger.info("Rate limiting enabled")

# Initialize Talisman for security headers (only in production)
if not app.config.get('DEBUG'):
    # Configure CSP to allow inline scripts and Three.js CDN
    csp = {
        'default-src': "'self'",
        'script-src': [
            "'self'",
            "'unsafe-inline'",  # Needed for inline Three.js scripts
            'cdnjs.cloudflare.com',
            'cdn.jsdelivr.net'
        ],
        'style-src': ["'self'", "'unsafe-inline'"],
        'img-src': "'self' data:",
        'font-src': "'self'",
    }
    Talisman(app, content_security_policy=csp, force_https=False)
    app.logger.info("Security headers enabled")


# Utility decorators
def log_request(f):
    """Decorator to log requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            duration = time.time() - start_time
            app.logger.info(f"{request.method} {request.path} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            app.logger.error(f"{request.method} {request.path} failed in {duration:.2f}s: {str(e)}")
            raise
    return decorated_function


def validate_parameters(f):
    """Decorator to validate input parameters"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Validate numeric parameters
            thickness = float(request.form.get('thickness', 0.3))
            point_distance = float(request.form.get('point_distance', 0.8))
            cube_size = float(request.form.get('cube_size', 20))

            # Sanity checks
            # Allow thickness=0 for preview requests (returns unprocessed mesh)
            if not (0.0 <= thickness <= 5.0):
                return jsonify({'error': 'Thickness must be between 0.0 and 5.0 mm'}), 400
            if thickness > 0 and thickness < 0.05:
                return jsonify({'error': 'Thickness must be 0 (no processing) or between 0.05 and 5.0 mm'}), 400

            if not (0.1 <= point_distance <= 10.0):
                return jsonify({'error': 'Point distance must be between 0.1 and 10.0 mm'}), 400

            if not (5 <= cube_size <= 200):
                return jsonify({'error': 'Cube size must be between 5 and 200 mm'}), 400

            return f(*args, **kwargs)
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid parameter value: {str(e)}'}), 400

    return decorated_function


# Routes
@app.route('/')
def index():
    """Render main page"""
    return render_template(
        'index.html',
        noise_types=NOISE_TYPES,
        enable_rotation_controls=app.config.get('ENABLE_ROTATION_CONTROLS', False),
        show_support_footer=app.config.get('SHOW_SUPPORT_FOOTER', False)
    )


@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'stl-texturizer',
        'noise_library_available': NOISE_AVAILABLE
    }), 200


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
            'skip_bottom': False,
            'skip_small_triangles': False
        }
    })


@app.route('/api/estimate', methods=['POST'])
@log_request
@validate_parameters
def estimate_stl():
    """
    Estimate output size and feasibility without processing.
    Useful for preview before processing large files.
    """
    tmp_input_path = None

    try:
        # Get parameters
        point_distance = float(request.form.get('point_distance', 0.8))
        use_default_cube = request.form.get('use_default_cube', 'false').lower() == 'true'
        cube_size = float(request.form.get('cube_size', 20))

        # Load mesh
        if use_default_cube:
            input_mesh = generate_test_cube(cube_size)
        else:
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded and default cube not selected'}), 400

            file = request.files['file']
            if file.filename == '' or not file.filename.lower().endswith('.stl'):
                return jsonify({'error': 'Valid STL file required'}), 400

            # Save and load file
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_in:
                file.save(tmp_in.name)
                tmp_input_path = tmp_in.name

            try:
                input_mesh = mesh.Mesh.from_file(tmp_input_path)
            except Exception as e:
                return jsonify({'error': f'Invalid STL file: {str(e)}'}), 400

        # Get feasibility check
        max_triangles = int(os.environ.get('MAX_OUTPUT_TRIANGLES', 20_000_000))
        max_memory_mb = int(os.environ.get('MAX_MEMORY_MB', 4096))
        max_file_size_mb = int(os.environ.get('MAX_OUTPUT_FILE_SIZE_MB', 500))

        feasibility = check_processing_feasibility(
            input_mesh,
            point_distance=point_distance,
            max_triangles=max_triangles,
            max_memory_mb=max_memory_mb,
            max_file_size_mb=max_file_size_mb
        )

        return jsonify({
            'feasible': feasibility['feasible'],
            'reason': feasibility['reason'],
            'estimates': feasibility['estimates'],
            'suggestions': feasibility['suggestions'],
            'limits': {
                'max_triangles': max_triangles,
                'max_memory_mb': max_memory_mb,
                'max_file_size_mb': max_file_size_mb
            }
        })

    except Exception as e:
        app.logger.error(f"Estimation error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to estimate output size'}), 500

    finally:
        if tmp_input_path and os.path.exists(tmp_input_path):
            try:
                os.unlink(tmp_input_path)
            except Exception:
                pass


@app.route('/api/process', methods=['POST'])
@log_request
@validate_parameters
def process_stl():
    """Process STL file with fuzzy skin texture"""

    # Apply rate limiting to processing endpoint
    if limiter and app.config.get('RATELIMIT_ENABLED'):
        try:
            # Custom rate limit for processing
            limiter.limit(app.config.get('RATELIMIT_PROCESSING', '3 per minute'))(lambda: None)()
        except Exception:
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

    tmp_input_path = None
    tmp_output_path = None

    try:
        # Get parameters
        thickness = float(request.form.get('thickness', 0.3))
        point_distance = float(request.form.get('point_distance', 0.8))
        seed = int(request.form.get('seed', 42))
        noise_type = request.form.get('noise_type', NOISE_CLASSIC)
        noise_scale = float(request.form.get('noise_scale', 1.0))
        noise_octaves = int(request.form.get('noise_octaves', 4))
        noise_persistence = float(request.form.get('noise_persistence', 0.5))
        skip_bottom = request.form.get('skip_bottom', 'false').lower() == 'true'
        skip_small_triangles = request.form.get('skip_small_triangles', 'false').lower() == 'true'
        use_default_cube = request.form.get('use_default_cube', 'false').lower() == 'true'
        cube_size = float(request.form.get('cube_size', 20))

        app.logger.info(f"Processing request - use_default_cube={use_default_cube}, "
                       f"thickness={thickness}, point_distance={point_distance}, "
                       f"noise_type={noise_type}")

        # Validate noise type
        if noise_type not in NOISE_TYPES:
            return jsonify({'error': f'Invalid noise type: {noise_type}'}), 400

        if noise_type != NOISE_CLASSIC and not NOISE_AVAILABLE:
            return jsonify({'error': f'Noise type {noise_type} requires the noise library'}), 400

        # Load mesh - either from file or generate default cube
        output_filename = 'fuzzy_output.stl'

        if use_default_cube:
            # Generate default cube
            app.logger.info(f"Generating default cube of size {cube_size}mm")
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

            # Validate file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            if file_size > app.config['MAX_CONTENT_LENGTH']:
                return jsonify({'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] / (1024*1024):.0f}MB'}), 400

            app.logger.info(f"Processing uploaded file: {file.filename} ({file_size / 1024:.1f} KB)")

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_in:
                file.save(tmp_in.name)
                tmp_input_path = tmp_in.name

            # Load mesh from file
            try:
                input_mesh = mesh.Mesh.from_file(tmp_input_path)
            except Exception as e:
                app.logger.error(f"Failed to load STL file: {str(e)}")
                return jsonify({'error': f'Invalid STL file: {str(e)}'}), 400

            # Generate output filename
            base_name = os.path.splitext(file.filename)[0]
            output_filename = f"{base_name}_fuzzy.stl"

        # Log mesh info
        app.logger.info(f"Input mesh has {len(input_mesh.vectors)} triangles")

        # If thickness is 0, skip processing and return the input mesh as-is (for preview)
        if thickness == 0:
            app.logger.info("Thickness is 0 - returning unprocessed mesh for preview")
            output_mesh = input_mesh
        else:
            # Check if processing is feasible
            max_triangles = int(os.environ.get('MAX_OUTPUT_TRIANGLES', 20_000_000))
            max_memory_mb = int(os.environ.get('MAX_MEMORY_MB', 4096))
            max_file_size_mb = int(os.environ.get('MAX_OUTPUT_FILE_SIZE_MB', 500))

            feasibility = check_processing_feasibility(
                input_mesh,
                point_distance=point_distance,
                max_triangles=max_triangles,
                max_memory_mb=max_memory_mb,
                max_file_size_mb=max_file_size_mb,
                skip_small_triangles=skip_small_triangles
            )

            # Log estimates
            estimates = feasibility['estimates']
            app.logger.info(f"Processing estimates: {estimates['estimated_triangles']:,} triangles, "
                           f"{estimates['estimated_file_size_mb']:.1f}MB file, "
                           f"{estimates['estimated_memory_mb']:.0f}MB memory")

            # Reject if not feasible
            if not feasibility['feasible']:
                app.logger.warning(f"Processing rejected: {feasibility['reason']}")
                error_response = {
                    'error': feasibility['reason'],
                    'suggestions': feasibility['suggestions'],
                    'estimates': estimates
                }
                return jsonify(error_response), 400

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
                    skip_bottom=skip_bottom,
                    skip_small_triangles=skip_small_triangles
                )
                app.logger.info(f"Output mesh has {len(output_mesh.vectors)} triangles")
            except MemoryError:
                app.logger.error("Out of memory during processing")
                return jsonify({'error': 'Out of memory. Try increasing point_distance or using a smaller mesh.'}), 500
            except ValueError as ve:
                app.logger.error(f"Processing error: {str(ve)}")
                return jsonify({'error': f'Processing error: {str(ve)}'}), 500
            except Exception as e:
                app.logger.error(f"Unexpected error during processing: {str(e)}")
                return jsonify({'error': f'Processing failed: {str(e)}'}), 500

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_out:
            output_mesh.save(tmp_out.name)
            tmp_output_path = tmp_out.name

        # Read the output file and send it
        with open(tmp_output_path, 'rb') as f:
            output_data = f.read()

        app.logger.info(f"Successfully processed - output size: {len(output_data) / 1024:.1f} KB")

        return send_file(
            io.BytesIO(output_data),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=output_filename
        )

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error. Please try again.'}), 500

    finally:
        # Clean up temporary files
        if tmp_input_path and os.path.exists(tmp_input_path):
            try:
                os.unlink(tmp_input_path)
            except Exception as e:
                app.logger.warning(f"Failed to delete temp input file: {e}")

        if tmp_output_path and os.path.exists(tmp_output_path):
            try:
                os.unlink(tmp_output_path)
            except Exception as e:
                app.logger.warning(f"Failed to delete temp output file: {e}")


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    return jsonify({'error': f'File too large. Maximum size is {max_size_mb:.0f}MB'}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    app.logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# Application factory pattern support
def create_app(config_name='default'):
    """Application factory for testing and production"""
    return app


if __name__ == '__main__':
    # Development server
    # Use port 8000 by default to avoid macOS AirPlay conflict on port 5000
    port = int(os.environ.get('PORT', 8000))
    app.logger.info(f"Starting development server on port {port}")
    app.logger.info(f"Access the application at: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=app.config.get('DEBUG', False))
