# STL Fuzzy Skin Texturizer

A web application and command-line tool for applying fuzzy skin texture to STL files, similar to the "fuzzy skin" feature in OrcaSlicer. Supports multiple noise types and offers both file upload and default cube generation.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)

## Features

### Web Application
- 🎨 **Multiple noise types** (Classic, Perlin, Billow, Ridged, Voronoi)
- 📁 **Upload STL files** or use built-in default cube
- 🔧 **Adjustable parameters** (thickness, point distance, seed, etc.)
- 👁️ **Real-time 3D preview** with Three.js viewer
- 🔄 **Model rotation controls** (X, Y, Z axes)
- 📊 **Output size estimation** (prevents memory issues before processing)
- 🚀 **Production-ready** with security features
- 📦 **Easy deployment** to multiple platforms

### Command Line Tool
- **Multiple Noise Types** (matching OrcaSlicer):
  - `classic` - Uniform random noise (default, fast)
  - `perlin` - Smooth, organic Perlin noise patterns
  - `billow` - Cloud-like patterns
  - `ridged` - Sharp ridge patterns
  - `voronoi` - Cell-based patterns
- **Configurable Parameters** for fine-tuning texture
- **Bottom layer skip** option for better bed adhesion
- **Binary or ASCII STL** export

---

## Quick Start

### Web Application (Local)

1. **Clone and setup:**
   ```bash
   git clone https://github.com/your-username/texturizer.git
   cd texturizer
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open in browser:**
   ```
   http://localhost:8000
   ```

   > **Note:** The app runs on port 8000 by default to avoid conflicts with macOS AirPlay Receiver (which uses port 5000). You can use a different port with `PORT=3000 python app.py`

### Command Line Tool

```bash
# Apply fuzzy skin to an STL file
python texturizer.py model.stl -o fuzzy_model.stl

# Generate a test cube with fuzzy skin
python texturizer.py -o fuzzy_cube.stl

# Use Perlin noise for smooth organic texture
python texturizer.py model.stl --noise perlin --noise-scale 0.5
```

---

## Installation

### For Web Application

```bash
pip install -r requirements.txt
```

### For Command Line Only

```bash
pip install numpy numpy-stl noise
```

> Note: The `noise` library is optional. Without it, only the `classic` noise type will be available.

---

## Web Application Usage

1. **Start the server:**
   ```bash
   python app.py
   ```

2. **Open http://localhost:8000 in your browser**

3. **Choose input:**
   - Upload an STL file, OR
   - Check "Use default cube" and set cube size

4. **Adjust parameters:**
   - **Thickness:** Maximum displacement distance (0.05-5mm)
   - **Point Distance:** Distance between texture points (0.1-10mm)
   - **Noise Type:** Classic, Perlin, Billow, Ridged, or Voronoi
   - **Seed:** Random seed for reproducibility
   - **Skip bottom layer:** Preserve flat bottom for bed adhesion

5. **Click "Process STL"**

6. **Preview the result** in the 3D viewer

7. **Click "Download Result"** to save the textured STL

---

## Command Line Usage

### Basic Usage

```bash
# Apply fuzzy skin to an STL file
python texturizer.py model.stl -o fuzzy_model.stl

# Generate a test cube with fuzzy skin
python texturizer.py -o fuzzy_cube.stl
```

### With Noise Options

```bash
# Perlin noise for smooth organic texture
python texturizer.py model.stl --noise perlin --noise-scale 0.5

# Voronoi pattern with custom scale
python texturizer.py model.stl --noise voronoi --noise-scale 2.0

# Ridged pattern with bottom layer preserved
python texturizer.py model.stl --noise ridged --skip-bottom
```

### Full Options

```bash
python texturizer.py model.stl \
    -o output.stl \
    -t 0.5 \                    # thickness (mm)
    -p 0.6 \                    # point distance (mm)
    -s 42 \                     # random seed
    --noise perlin \            # noise type
    --noise-scale 1.0 \         # frequency scale
    --noise-octaves 4 \         # octaves (perlin/billow)
    --noise-persistence 0.5 \   # persistence (perlin/billow)
    --skip-bottom \             # preserve bottom layer
    --ascii                     # ASCII STL output
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | (test cube) | Input STL file |
| `-o, --output` | `fuzzy_output.stl` | Output STL file |
| `-t, --thickness` | `0.3` | Fuzzy skin thickness in mm |
| `-p, --point-distance` | `0.8` | Distance between texture points in mm |
| `-s, --seed` | `42` | Random seed for reproducibility |
| `--cube-size` | `20` | Test cube size in mm |
| `--noise` | `classic` | Noise type (classic/perlin/billow/ridged/voronoi) |
| `--noise-scale` | `1.0` | Noise frequency scale (higher = more detail) |
| `--noise-octaves` | `4` | Number of octaves for Perlin/Billow |
| `--noise-persistence` | `0.5` | Amplitude persistence for Perlin/Billow |
| `--skip-bottom` | `false` | Skip fuzzy skin on bottom layer |
| `--ascii` | `false` | Save as ASCII STL instead of binary |

---

## Deployment

The web application is production-ready and can be deployed to various platforms:

- **Render** (Free tier available)
- **Railway** (~$10-20/month)
- **Fly.io** (~$5-15/month)
- **DigitalOcean App Platform** ($24/month)
- **Docker/VPS** ($6-24/month)

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for detailed deployment instructions.

### Quick Deploy Options

**Deploy with Docker:**
```bash
docker build -t stl-texturizer .
docker run -p 8000:8000 -e FLASK_ENV=production stl-texturizer
# Or map to any port: docker run -p 3000:8000 -e FLASK_ENV=production stl-texturizer
```

**Deploy to Render/Railway/Fly.io:**
- Push to GitHub
- Connect repository on platform
- Platform auto-detects configuration from `Procfile` or `Dockerfile`
- Set environment variables
- Deploy!

---

## API Endpoints

The web application provides a REST API:

### `GET /`
Main web interface

### `GET /health`
Health check endpoint (returns JSON status)

### `GET /api/info`
Get available noise types and default parameters

### `POST /api/estimate`
Estimate output size and feasibility before processing (fast, no actual processing)

**Parameters (form-data):**
- `file`: STL file (optional if using default cube)
- `use_default_cube`: true/false
- `cube_size`: Size in mm
- `point_distance`: Point spacing

**Returns:** Estimated triangles, file size, memory usage, and feasibility check

### `POST /api/process`
Process STL file with fuzzy skin texture

**Parameters (form-data):**
- `file`: STL file (optional if using default cube)
- `use_default_cube`: true/false
- `cube_size`: Size in mm (if using default cube)
- `thickness`: Displacement thickness
- `point_distance`: Point spacing
- `seed`: Random seed
- `noise_type`: classic/perlin/billow/ridged/voronoi
- `noise_scale`: Noise frequency
- `noise_octaves`: Octaves for Perlin-based noise
- `noise_persistence`: Persistence for Perlin-based noise
- `skip_bottom`: true/false

---

## Configuration

Configuration is managed through environment variables. See [.env.example](.env.example) for all options.

**Key Settings:**
- `FLASK_ENV`: `development` or `production`
- `SECRET_KEY`: Secret key for sessions (required in production)
- `MAX_CONTENT_LENGTH`: Max upload size (default: 50MB)
- `RATELIMIT_ENABLED`: Enable rate limiting (default: true in production)
- `PORT`: Server port (default: 8000)

**Example .env file:**
```bash
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
PORT=8000
RATELIMIT_ENABLED=false
LOG_LEVEL=INFO
```

---

## How It Works

1. **Mesh Subdivision**: The input mesh is subdivided until all edges are smaller than the point distance, creating a denser vertex distribution.

2. **Vertex Normal Calculation**: Normals are computed for each unique vertex by averaging the normals of adjacent faces.

3. **Noise-Based Displacement**: Each vertex is displaced along its normal by an amount determined by the selected noise function.

4. **Mesh Reconstruction**: The displaced vertices are used to create the output mesh.

---

## Noise Types Explained

- **Classic**: Simple uniform random displacement. Fast and produces a rough, sandpaper-like texture.

- **Perlin**: Coherent noise that creates smooth, organic-looking patterns. Good for natural textures.

- **Billow**: Absolute value of Perlin noise, creating cloud-like, puffy patterns.

- **Ridged**: Inverted absolute Perlin noise, creating sharp ridges and valleys.

- **Voronoi**: Cell-based pattern that creates a cracked or tiled appearance.

---

## Examples

```bash
# Standard fuzzy skin (like slicer default)
python texturizer.py model.stl -t 0.3 -p 0.8

# Fine texture with small displacement
python texturizer.py model.stl -t 0.2 -p 0.4

# Coarse, aggressive texture
python texturizer.py model.stl -t 0.6 -p 1.2

# Organic pattern with Perlin noise
python texturizer.py model.stl --noise perlin --noise-scale 0.3 -t 0.4

# Cell pattern for artistic effect
python texturizer.py model.stl --noise voronoi --noise-scale 1.5 -t 0.5
```

---

## Architecture

```
texturizer/
├── app.py                 # Main Flask application (production-ready)
├── texturizer.py          # Core processing logic
├── config.py              # Configuration management
├── templates/
│   └── index.html         # Web interface with 3D viewer
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── Procfile               # Platform deployment config
├── gunicorn.conf.py       # Gunicorn production server config
├── .env.example           # Environment variables template
└── DEPLOYMENT.md          # Detailed deployment guide
```

---

## Security Features

- ✅ **Rate limiting** (configurable per endpoint)
- ✅ **CORS support** (optional, configurable)
- ✅ **Security headers** (CSP, HSTS, etc.)
- ✅ **File upload validation** (type and size)
- ✅ **Parameter validation** and sanitization
- ✅ **Comprehensive error handling**
- ✅ **Structured logging**

---

## Performance

**Recommended Resources:**
- Memory: 2GB+ RAM
- CPU: 2+ cores
- Timeout: 10 minutes for large files

**Processing Times (approximate):**
- Small models (<1MB): 5-30 seconds
- Medium models (1-10MB): 30-120 seconds
- Large models (>10MB): 2-10 minutes

**Tips for faster processing:**
- Increase `point_distance` for lower resolution
- Use `classic` noise type (fastest)
- Process locally for very large files

---

## Development

### Running in Development Mode

```bash
# Standard development
python app.py

# Or with Flask CLI
flask run
```

### Testing Production Mode Locally

```bash
# Set production environment
export FLASK_ENV=production
export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')

# Run with Gunicorn
gunicorn app:app --config gunicorn.conf.py
```

### Project Structure

- `app.py` - Production-ready Flask application
- `texturizer.py` - Core mesh processing engine
- `config.py` - Environment-based configuration
- `templates/index.html` - Web interface with Three.js viewer

---

## Dependencies

### Core
- **Flask** - Web framework
- **numpy** - Numerical operations
- **numpy-stl** - STL file handling
- **noise** - Advanced noise generation

### Production
- **gunicorn** - WSGI server
- **flask-cors** - CORS support
- **flask-limiter** - Rate limiting
- **flask-talisman** - Security headers

---

## Troubleshooting

### Memory Errors
- Increase instance memory (2GB+ recommended)
- Increase `point_distance` parameter
- Use smaller input files

### Timeout Errors
- Increase `PROCESSING_TIMEOUT` environment variable
- Increase platform-specific timeout settings
- Process locally for very large files

### Rate Limiting
- Disable in development: `RATELIMIT_ENABLED=false`
- Adjust limits: `RATELIMIT_PROCESSING=5 per minute`

See [DEPLOYMENT.md](DEPLOYMENT.md) for more troubleshooting tips.

---

## License

This project is provided as-is for educational and personal use.

---

## Acknowledgments

- Inspired by the fuzzy skin implementation in [OrcaSlicer](https://github.com/SoftFever/OrcaSlicer)
- Three.js for 3D visualization
- Flask community for excellent documentation

---

## Support

- 📖 [Deployment Guide](DEPLOYMENT.md)
- 🐛 [Report Issues](https://github.com/your-username/texturizer/issues)
- 💬 [Ask Questions](https://github.com/your-username/texturizer/discussions)
