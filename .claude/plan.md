# Implementation Plan: Selective Fuzzy Skin with Blocker Volumes

## Feature Overview
Add capability to use a secondary STL file as a "blocker" that prevents fuzzy skin application where it intersects the primary STL surface. Initial implementation provides a default cylinder blocker shape.

## Current Architecture Analysis

### Processing Pipeline (texturizer.py)
- `apply_fuzzy_skin()` function (lines 647-839) handles:
  1. Mesh subdivision based on point_distance
  2. Vertex deduplication
  3. Normal calculation
  4. Displacement application with noise
- **Key insertion point**: Lines 803-809 where `process_mask` is created
  - Currently only handles `skip_bottom` logic
  - Perfect place to add blocker volume logic

### File Upload System (app.py)
- Single file upload via `request.files['file']` (line 341)
- Saves to temp file, loads with `mesh.Mesh.from_file()`
- Also supports default cube generation

### 3D Viewer (templates/index.html)
- Three.js scene with single mesh (currentMesh)
- `loadSTL()` function (line 689) loads and displays one mesh
- Material color: `0x4ade80` (green)
- Scene allows adding multiple meshes

### Default Shape Generation
- `generate_simple_cube()` exists (line 122)
- Pattern: Create vertices → faces → mesh.Mesh object
- Can create cylinder following same pattern

## Implementation Strategy

### Phase 1: Backend - Blocker Volume Support

#### 1.1 Create Cylinder Generator (texturizer.py)
```python
def generate_blocker_cylinder(radius=10, height=30, position=(0,0,0), segments=32):
    """
    Generate a cylinder to use as a blocker volume.
    Returns mesh.Mesh object.
    """
```
- Place at bottom of default shape generators (after line 385)
- Use similar pattern to `generate_simple_cube()`

#### 1.2 Point-in-Volume Check Function (texturizer.py)
```python
def point_inside_mesh_volume(points, blocker_mesh):
    """
    Check if points are inside a closed mesh volume.
    Uses ray casting algorithm.

    Args:
        points: Nx3 numpy array of vertex positions
        blocker_mesh: mesh.Mesh object (blocker volume)

    Returns:
        Boolean array of length N (True if inside volume)
    """
```
- Implement ray casting algorithm or use bounding box approximation
- Consider performance: may need spatial indexing for large meshes
- **Key Decision**: Ray casting vs. Bounding Box
  - **Ray casting**: Accurate but slower
  - **Bounding box**: Fast but approximate (good for first version)

#### 1.3 Modify apply_fuzzy_skin() (texturizer.py)
Add new parameter `blocker_mesh=None`:
```python
def apply_fuzzy_skin(input_mesh, thickness=0.3, point_distance=0.8, seed=42,
                     ..., skip_small_triangles=False, blocker_mesh=None):
```

Update process_mask logic (around line 803):
```python
# Determine which vertices to process
if blocker_mesh is not None:
    # Check which vertices are inside blocker volume
    blocker_mask = point_inside_mesh_volume(unique_vertices, blocker_mesh)
    process_mask = ~blocker_mask  # Don't process vertices inside blocker
    skipped_count = np.sum(blocker_mask)
elif skip_bottom:
    process_mask = unique_vertices[:, 2] > bottom_threshold
    skipped_count = np.sum(~process_mask)
else:
    process_mask = np.ones(len(unique_vertices), dtype=bool)
    skipped_count = 0
```

#### 1.4 Update app.py /api/process endpoint
- Add support for second file upload: `blocker_file`
- Add blocker configuration parameters:
  - `use_blocker`: boolean
  - `use_default_cylinder`: boolean
  - `cylinder_radius`, `cylinder_height`, `cylinder_position`
- Load blocker mesh (from file or generate cylinder)
- Pass `blocker_mesh` to `apply_fuzzy_skin()`

### Phase 2: Frontend - Dual Mesh Viewer & UI

#### 2.1 Add Second File Upload (index.html)
```html
<div class="form-group">
    <label>
        <input type="checkbox" id="useBlocker">
        Use Blocker Volume
    </label>
</div>

<div id="blockerSection" style="display:none">
    <div class="form-group">
        <label>
            <input type="checkbox" id="useDefaultCylinder" checked>
            Use default cylinder
        </label>
    </div>

    <div id="blockerUploadSection" style="display:none">
        <input type="file" id="blockerFile" accept=".stl">
    </div>

    <div id="cylinderOptions">
        <label>Cylinder Radius (mm)</label>
        <input type="number" id="cylinderRadius" value="10">
        <label>Cylinder Height (mm)</label>
        <input type="number" id="cylinderHeight" value="30">
    </div>
</div>
```

#### 2.2 Dual Mesh Viewer (index.html JavaScript)
Create new variables:
```javascript
let currentMesh = null;     // Primary mesh
let blockerMesh = null;     // Blocker mesh
let uploadedBlockerFile = null;
```

Update `loadSTL()` to `loadPrimarySTL()` and create `loadBlockerSTL()`:
```javascript
function loadBlockerSTL(arrayBuffer) {
    if (blockerMesh) {
        scene.remove(blockerMesh);
    }

    const loader = new THREE.STLLoader();
    const geometry = loader.parse(arrayBuffer);

    // Semi-transparent red material for blocker
    const material = new THREE.MeshPhongMaterial({
        color: 0xff4757,  // Red
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide
    });

    blockerMesh = new THREE.Mesh(geometry, material);
    // Position relative to primary mesh
    scene.add(blockerMesh);
}
```

#### 2.3 Default Cylinder Preview
```javascript
function generateDefaultCylinderPreview() {
    const radius = parseFloat(document.getElementById('cylinderRadius').value);
    const height = parseFloat(document.getElementById('cylinderHeight').value);

    // Create cylinder geometry in Three.js
    const geometry = new THREE.CylinderGeometry(radius, radius, height, 32);

    // Semi-transparent red material
    const material = new THREE.MeshPhongMaterial({
        color: 0xff4757,
        transparent: true,
        opacity: 0.5
    });

    if (blockerMesh) {
        scene.remove(blockerMesh);
    }
    blockerMesh = new THREE.Mesh(geometry, material);
    scene.add(blockerMesh);
}
```

#### 2.4 Update processSTL() Function
Add blocker parameters to FormData:
```javascript
if (useBlocker.checked) {
    formData.append('use_blocker', 'true');
    if (useDefaultCylinder.checked) {
        formData.append('use_default_cylinder', 'true');
        formData.append('cylinder_radius', cylinderRadius.value);
        formData.append('cylinder_height', cylinderHeight.value);
    } else if (uploadedBlockerFile) {
        formData.append('blocker_file', uploadedBlockerFile);
    }
}
```

### Phase 3: API Endpoints

#### 3.1 Add Preview Endpoint (app.py)
```python
@app.route('/api/preview-with-blocker', methods=['POST'])
def preview_with_blocker():
    """
    Return both primary mesh and blocker mesh for preview.
    Returns JSON with two STL files (base64 encoded or URLs).
    """
```
- Optional: Could be useful for server-side preview generation
- For MVP, client-side preview with Three.js is sufficient

#### 3.2 Update /api/estimate Endpoint
- Include blocker mesh in feasibility calculations
- Estimate won't change significantly (same triangle count)

### Phase 4: Testing & Polish

#### 4.1 Test Cases
1. Default cylinder with simple-corner.stl
2. Custom blocker STL with simple-corner.stl
3. Verify masked regions have no displacement
4. Visual verification in viewer (two colors)

#### 4.2 UX Polish
- Toggle blocker visibility
- Adjust blocker position/rotation controls
- Show blocked vertex count in UI
- Add blocker to analytics tracking

## Technical Considerations

### Point-in-Volume Algorithm Options

**Option 1: Bounding Box (Recommended for MVP)**
- Pros: Fast (O(1) per point), simple to implement
- Cons: Only works for box-shaped blockers, approximate
- Implementation: Check if point is within min/max bounds

**Option 2: Ray Casting**
- Pros: Accurate for any closed volume
- Cons: Slower (O(n) per point), more complex
- Implementation: Cast ray, count intersections (odd = inside)

**Option 3: Signed Distance Field**
- Pros: Very fast lookups after preprocessing, smooth falloff possible
- Cons: Memory intensive, preprocessing required
- Implementation: Voxelize blocker mesh, query SDF

**Recommendation**: Start with bounding box for cylinder (simple and fast), add ray casting option later for arbitrary blocker meshes.

### Performance Considerations
- Blocker intersection check happens once per unique vertex
- For simple-corner.stl: ~850K vertices with point_distance=0.6
- Bounding box check: ~1ms
- Ray casting: ~100-500ms depending on blocker complexity
- **Impact**: Minimal performance hit for MVP

### Memory Considerations
- Need to load two meshes simultaneously
- Blocker mesh should be simple (low triangle count)
- Add validation: max blocker triangles = 10,000

## Implementation Order

1. **Backend Foundation** (1-2 hours)
   - Create `generate_blocker_cylinder()`
   - Implement `point_inside_mesh_volume()` with bounding box
   - Add `blocker_mesh` parameter to `apply_fuzzy_skin()`
   - Test with Python script

2. **Backend API** (1 hour)
   - Update `/api/process` to handle blocker parameters
   - Add file upload for blocker STL
   - Add cylinder parameter handling

3. **Frontend Viewer** (2 hours)
   - Add dual mesh rendering
   - Implement `loadBlockerSTL()` and cylinder preview
   - Different colors/transparency for primary vs blocker

4. **Frontend UI** (1-2 hours)
   - Add blocker section with checkbox
   - Add file upload for blocker
   - Add cylinder parameters (radius, height)
   - Wire up event handlers

5. **Integration & Testing** (1-2 hours)
   - Test end-to-end workflow
   - Verify masking works correctly
   - Visual verification
   - Test with various blocker positions

**Total Estimate**: 6-9 hours

## Files to Modify

### Backend
- `texturizer.py`: Add cylinder generator, point-in-volume check, modify apply_fuzzy_skin()
- `app.py`: Update /api/process endpoint for blocker support

### Frontend
- `templates/index.html`: Add UI controls, dual mesh viewer, blocker handling

### New Files
- None (all changes to existing files)

## Open Questions / User Decisions Needed

1. **Blocker Positioning**: How should users position the blocker?
   - Option A: Manual input (x, y, z coordinates)
   - Option B: Visual dragging in 3D viewer (more complex)
   - Option C: Automatic centering (simple)
   - **Recommendation**: Start with Option C (centered), add Option A

2. **Cylinder Orientation**: Vertical (Z-axis) or horizontal?
   - **Recommendation**: Vertical (Z-axis aligned) - most intuitive

3. **Multiple Blockers**: Support multiple blocker volumes?
   - **Recommendation**: Not for MVP, easy to add later

4. **Blocker Export**: Save blocker configuration with project?
   - **Recommendation**: Not for MVP

5. **Falloff/Transition**: Sharp cutoff or gradual transition at boundary?
   - **Recommendation**: Sharp cutoff for MVP, add falloff later

## Success Criteria

1. User can upload primary STL + blocker STL
2. User can use default cylinder blocker
3. Preview shows both meshes in different colors
4. Processed output has no fuzzy skin where blocker intersects
5. Processing time impact < 10%
6. Works with existing test file (simple-corner.stl)

## Risk Mitigation

1. **Risk**: Ray casting too slow for large meshes
   - Mitigation: Start with bounding box, profile before optimizing

2. **Risk**: UI complexity confuses users
   - Mitigation: Hide blocker section by default, add tooltips

3. **Risk**: Blocker positioning unclear without visual feedback
   - Mitigation: Always show preview before processing

4. **Risk**: Memory issues with two large STLs
   - Mitigation: Add file size validation, limit blocker complexity
