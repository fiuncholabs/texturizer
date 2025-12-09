# Plan: XY-Plane Subdivision Mode (OrcaSlicer-Style)

## Problem Analysis

After reviewing the OrcaSlicer fuzzy skin source code, I've identified a fundamental difference between our current implementation and OrcaSlicer's approach:

### Current Implementation (3D-based)
1. **Edge length calculation**: Uses 3D distance `np.linalg.norm(v1 - v0)` (lines 1095-1097, 1329-1331)
   - Includes X, Y, and Z components
   - For a vertical wall edge (same X,Y but different Z), calculates full 3D distance

2. **Subdivision**: Recursively splits triangles until all 3D edges are below `point_distance`
   - Vertical surfaces get many subdivisions even though XY distance is zero
   - Results in excessive triangles on vertical walls

3. **Result**: Point density varies based on surface orientation
   - Vertical walls: High density (many subdivisions needed for 3D distance)
   - Horizontal surfaces: Lower density (XY distance ≈ 3D distance)

### OrcaSlicer Implementation (XY-plane based)
From `FuzzySkin.cpp` analysis:

1. **Distance calculation**: Uses 2D XY-plane distance only
   ```cpp
   Vec2d p0p1 = (p1 - *p0).cast<double>();  // Only X,Y components
   double p0p1_size = p0p1.norm();          // 2D norm
   ```

2. **Point spacing**: Adds points along edges based on XY-plane distance
   - Ignores Z component completely during spacing calculations
   - Point spacing: 75-125% of `point_distance` (randomized)

3. **Result**: Uniform point density in XY plane regardless of surface orientation
   - Vertical walls: Sparse points (XY distance is small/zero)
   - Horizontal surfaces: Points spaced by `point_distance`

## Key Insight

**The problem**: Our subdivision uses 3D edge lengths, causing:
- Over-subdivision on vertical/angled surfaces (to meet 3D distance threshold)
- Non-uniform point density when viewed from above
- Different fuzzy skin appearance than OrcaSlicer

**The solution**: Add option to subdivide based on XY-plane distance only, matching OrcaSlicer's behavior.

## Proposed Solution

Add a new checkbox option: **"XY-plane subdivision (OrcaSlicer-style)"**

### Algorithm Changes

When `xy_plane_subdivision=True`:

1. **Modify edge length calculation** in `subdivide_triangle()`:
   ```python
   if xy_plane_subdivision:
       # Calculate XY-plane distance only (ignore Z)
       e0 = np.linalg.norm(v1[:2] - v0[:2])  # Only X,Y
       e1 = np.linalg.norm(v2[:2] - v1[:2])
       e2 = np.linalg.norm(v0[:2] - v2[:2])
   else:
       # Current 3D distance
       e0 = np.linalg.norm(v1 - v0)
       e1 = np.linalg.norm(v2 - v1)
       e2 = np.linalg.norm(v0 - v2)
   ```

2. **Skip small triangles check** (lines 1329-1331):
   ```python
   if xy_plane_subdivision:
       e0 = np.linalg.norm(v1[:2] - v0[:2])
       e1 = np.linalg.norm(v2[:2] - v1[:2])
       e2 = np.linalg.norm(v0[:2] - v2[:2])
   else:
       e0 = np.linalg.norm(v1 - v0)
       e1 = np.linalg.norm(v2 - v1)
       e2 = np.linalg.norm(v0 - v2)
   ```

3. **Average edge estimation** (lines 1301-1305):
   ```python
   if xy_plane_subdivision:
       sample_edges.append(np.linalg.norm(v1[:2] - v0[:2]))
       sample_edges.append(np.linalg.norm(v2[:2] - v1[:2]))
       sample_edges.append(np.linalg.norm(v0[:2] - v2[:2]))
   else:
       sample_edges.append(np.linalg.norm(v1 - v0))
       sample_edges.append(np.linalg.norm(v2 - v1))
       sample_edges.append(np.linalg.norm(v0 - v2))
   ```

### Expected Behavior

| Surface Type | Current (3D) | With XY-plane Mode |
|--------------|--------------|-------------------|
| Vertical wall (10mm wide, 20mm tall) | ~20mm edge → many subdivisions | ~10mm XY edge → fewer subdivisions |
| Horizontal surface (10mm × 10mm) | ~10mm edge → subdivisions | ~10mm XY edge → same subdivisions |
| 45° angled surface (10mm × 10mm) | ~14mm edge → more subdivisions | ~10mm XY edge → fewer subdivisions |

### Combination with Other Options

This feature works independently with existing options:

- **With `in_plane_noise=True`**: Perfect OrcaSlicer match
  - Subdivision based on XY distance
  - Displacement in XY plane only

- **With `noise_on_edges=True`**: Seamless XY-plane fuzzy skin
  - XY-plane subdivision
  - Edge-based noise (no seams)

- **With both**: Maximum OrcaSlicer compatibility
  - XY subdivision + XY displacement + no seams

## Implementation Plan

### Files to Modify

1. **texturizer.py**:
   - Add `xy_plane_subdivision=False` parameter to `apply_fuzzy_skin()` (line 1198)
   - Update docstring (around line 1220)
   - Modify `subdivide_triangle()` to accept and use XY-plane mode (line 1081)
   - Update all 3 locations that calculate edge lengths in subdivision logic

2. **templates/index.html**:
   - Add checkbox in UI (around line 560): "XY-plane subdivision (OrcaSlicer-style)"
   - Pass parameter in form data (lines 1794, 1902)

3. **app.py**:
   - Parse `xy_plane_subdivision` parameter (around line 324)
   - Pass to `apply_fuzzy_skin()` call (around line 643)

### Testing Strategy

Create `test_xy_subdivision.py` to verify:

1. **Triangle count comparison**:
   - Create vertical wall (minimal XY distance)
   - Subdivide with 3D mode: expect many triangles
   - Subdivide with XY mode: expect few/no triangles

2. **Horizontal surface**:
   - Both modes should produce similar triangle counts

3. **Angled surface**:
   - XY mode should produce fewer triangles than 3D mode

## Technical Considerations

### Performance Impact

**Positive**: XY-plane subdivision will generally produce FEWER triangles
- Vertical walls won't be over-subdivided
- Lower memory usage
- Faster processing

**Memory savings example**:
- 20mm tall vertical wall with 0.8mm point_distance
- 3D mode: ~25 subdivisions (20/0.8)
- XY mode: 0-1 subdivisions (edges are vertical, XY distance ≈ 0)

### Edge Cases

1. **Pure vertical edges** (XY distance = 0):
   - Won't subdivide (correct behavior)
   - Noise will still apply if vertices differ

2. **Horizontal edges** (Z distance = 0):
   - Same behavior in both modes

3. **Mixed surfaces**:
   - Each triangle subdivided based on its XY-plane projection

## User Experience

### UI Label
"XY-plane subdivision (OrcaSlicer-style)" - Clear indication of purpose

### Tooltip/Help Text
"Calculate subdivision distance in XY plane only (like OrcaSlicer). Reduces over-subdivision on vertical walls and matches OrcaSlicer's fuzzy skin appearance."

### Recommended Combinations

For maximum OrcaSlicer similarity:
1. ✓ XY-plane subdivision
2. ✓ In-plane noise only
3. ✓ Noise on edges

## Questions to Clarify

None - the implementation is straightforward and well-defined based on OrcaSlicer source code analysis.

## Summary

This change adds a single boolean flag that modifies how edge lengths are calculated during subdivision. By using XY-plane distance instead of 3D distance, we match OrcaSlicer's behavior of:
- Uniform point density in the horizontal plane
- Reduced over-subdivision on vertical surfaces
- Better performance (fewer triangles)
- More authentic OrcaSlicer-style fuzzy skin appearance

The modification is minimal, non-breaking (defaults to current behavior), and provides clear user benefit.

---

## Sources Consulted

- [OrcaSlicer FuzzySkin.cpp](https://github.com/OrcaSlicer/OrcaSlicer/blob/main/src/libslic3r/Feature/FuzzySkin/FuzzySkin.cpp)
- [OrcaSlicer PerimeterGenerator.cpp](https://github.com/OrcaSlicer/OrcaSlicer/blob/main/src/libslic3r/PerimeterGenerator.cpp)
- [OrcaSlicer Fuzzy Skin Wiki](https://github.com/OrcaSlicer/OrcaSlicer/wiki/others_settings_fuzzy_skin)
- [Pull Request #7678 - Perlin noise fuzzy skin](https://github.com/SoftFever/OrcaSlicer/pull/7678)
