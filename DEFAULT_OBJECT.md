# Default Object Configuration

The STL Texturizer allows you to configure what default object is used when users select "Use default test object".

## Configuration

Set the `DEFAULT_OBJECT_TYPE` environment variable to control which object is generated:

### Simple Cube (Default)
```bash
DEFAULT_OBJECT_TYPE=cube
DEFAULT_OBJECT_SIZE=20
```

This generates a simple 20mm × 20mm × 20mm cube with 12 triangles. It's fast to generate and perfect for quick testing.

### Custom Object
```bash
DEFAULT_OBJECT_TYPE=custom
DEFAULT_OBJECT_SIZE=20
```

This generates a custom object defined in the `generate_custom_object()` function in `texturizer.py`. The current custom object is the Fiuncholabs beaker card - a rectangular card with an embossed beaker design.

## How to Create Your Own Custom Object

1. **Set the environment variable:**
   ```bash
   DEFAULT_OBJECT_TYPE=custom
   ```

2. **Edit the `generate_custom_object()` function** in `texturizer.py` (starting around line 152):

   ```python
   def generate_custom_object(size=20):
       """
       Generate custom test object.

       Args:
           size: Reference dimension for the object (default 20mm)

       Returns:
           mesh.Mesh object
       """
       # Your custom mesh generation code here
       # Use numpy-stl to create your custom 3D object

       return your_mesh
   ```

3. **Examples of what you can create:**
   - Company logo embossed on a card
   - Custom geometric shapes
   - Branded objects for demonstrations
   - Test patterns for specific use cases

## Implementation Details

The wrapper function `generate_test_cube()` automatically routes to the correct implementation:

```python
def generate_test_cube(size=20, object_type=None):
    if object_type is None:
        object_type = os.environ.get('DEFAULT_OBJECT_TYPE', 'cube').lower()

    if object_type == 'custom':
        return generate_custom_object(size)
    else:
        return generate_simple_cube(size)
```

## Current Custom Object: Fiuncholabs Beaker Card

The current custom object is a rectangular card (30mm × 20mm × 2mm for size=20) with an embossed beaker design featuring:
- Tapered beaker body
- Measurement marks
- Wavy liquid surface
- 4% embossing depth (0.8mm for size=20)

This creates approximately 9,596 triangles and is designed to showcase the fuzzy skin effect effectively.

## Files Modified

- `config.py`: Added `DEFAULT_OBJECT_TYPE` and `DEFAULT_OBJECT_SIZE` configuration
- `texturizer.py`: Split into `generate_simple_cube()`, `generate_custom_object()`, and wrapper `generate_test_cube()`
- `templates/index.html`: Updated UI labels to be generic ("test object" instead of "card")
- `.env.example`: Added documentation for new environment variables

## Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `DEFAULT_OBJECT_TYPE` | `cube` or `custom` | `cube` | Which default object to generate |
| `DEFAULT_OBJECT_SIZE` | Number (mm) | `20` | Base size/dimension of the object |

## Testing

After changing the configuration:

1. Restart the application
2. Check the "Use default test object" checkbox
3. The preview should show your configured object
4. Adjust the "Object Size" slider to verify scaling works

## Migration from Previous Version

If you were using the beaker card and want to continue using it:

```bash
# In your .env file or environment:
DEFAULT_OBJECT_TYPE=custom
DEFAULT_OBJECT_SIZE=20
```

If you prefer the simple cube for faster loading:

```bash
DEFAULT_OBJECT_TYPE=cube
DEFAULT_OBJECT_SIZE=20
```
