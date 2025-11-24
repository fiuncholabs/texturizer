# Orca Texturizer

A Python tool that applies fuzzy skin texture to STL files, creating a new mesh with surface displacement similar to the "fuzzy skin" feature in OrcaSlicer.

## Features

- **Multiple Noise Types** (matching OrcaSlicer):
  - `classic` - Uniform random noise (default, fast)
  - `perlin` - Smooth, organic Perlin noise patterns
  - `billow` - Cloud-like patterns
  - `ridged` - Sharp ridge patterns
  - `voronoi` - Cell-based patterns

- **Configurable Parameters**:
  - Thickness control (displacement amount)
  - Point distance (texture resolution)
  - Noise scale, octaves, and persistence
  - Bottom layer skip option for better bed adhesion

- **Export Options**:
  - Binary STL (default)
  - ASCII STL

## Installation

```bash
pip install numpy numpy-stl noise
```

> Note: The `noise` library is optional. Without it, only the `classic` noise type will be available.

## Usage

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

## Command Line Arguments

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

## How It Works

1. **Mesh Subdivision**: The input mesh is subdivided until all edges are smaller than the point distance, creating a denser vertex distribution.

2. **Vertex Normal Calculation**: Normals are computed for each unique vertex by averaging the normals of adjacent faces.

3. **Noise-Based Displacement**: Each vertex is displaced along its normal by an amount determined by the selected noise function.

4. **Mesh Reconstruction**: The displaced vertices are used to create the output mesh.

## Noise Types Explained

- **Classic**: Simple uniform random displacement. Fast and produces a rough, sandpaper-like texture.

- **Perlin**: Coherent noise that creates smooth, organic-looking patterns. Good for natural textures.

- **Billow**: Absolute value of Perlin noise, creating cloud-like, puffy patterns.

- **Ridged**: Inverted absolute Perlin noise, creating sharp ridges and valleys.

- **Voronoi**: Cell-based pattern that creates a cracked or tiled appearance.

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

## License

This project is provided as-is for educational and personal use.

## Acknowledgments

Inspired by the fuzzy skin implementation in [OrcaSlicer](https://github.com/SoftFever/OrcaSlicer).
