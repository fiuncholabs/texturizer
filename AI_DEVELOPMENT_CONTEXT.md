# AI Development Context for STL Texturizer

> **⚠️ IMPORTANT: AI ASSISTANT INSTRUCTIONS**
>
> This file serves as a living development context for AI assistants (Claude, GPT, etc.) working on this project.
>
> **WHEN WORKING ON THIS PROJECT:**
> 1. READ this file FIRST before making any changes
> 2. UPDATE this file after completing significant work
> 3. ADD new sections as architectural decisions are made
> 4. MAINTAIN chronological order in the changelog
> 5. DOCUMENT any gotchas, bugs fixed, or important design decisions
>
> This ensures continuity across different AI sessions and helps prevent regression of fixed issues.

---

## Project Overview

**STL Fuzzy Skin Texturizer** - A web application that applies fuzzy skin texture to STL files by modifying the mesh geometry, replicating OrcaSlicer's fuzzy skin feature but at the STL level instead of G-code level.

### Core Purpose
- Apply randomized surface texture to 3D models (STL files)
- Provide a web interface for easy configuration
- Estimate processing requirements before execution
- Support multiple noise types and configurable parameters

### Key Differences from OrcaSlicer
- **STL-based**: Modifies mesh geometry directly (permanent)
- **G-code-based (OrcaSlicer)**: Adds texture during slicing (adjustable)
- **Trade-off**: Larger files and longer processing, but works with any slicer

---

## Architecture

### Technology Stack
- **Backend**: Python 3, Flask web framework
- **3D Processing**: numpy-stl for mesh manipulation
- **Frontend**: Vanilla JavaScript, Three.js (r128) for 3D visualization
- **Styling**: Custom CSS with dark forest green theme
- **Configuration**: Environment variables via .env file

### Core Files

#### Backend
- `app.py` - Flask application, API endpoints, request handling
- `texturizer.py` - Core mesh processing logic (subdivision, noise generation)
- `config.py` - Configuration management (development/production)
- `gunicorn.conf.py` - Production server configuration

#### Frontend
- `templates/index.html` - Main UI (single-page application)
- Uses Three.js for 3D preview with OrbitControls

#### Configuration
- `.env.example` - Template for environment variables
- `config.py` - Configuration classes for different environments

#### Testing & Debugging
- `test_suite.py` - Unit tests for subdivision algorithms
- `app_debug.py` - Simplified version for debugging
- `app_simple.py` - Minimal version for testing

#### Documentation
- `README.md` - User-facing documentation
- `DEPLOYMENT.md` - Production deployment guide
- `SIZE_ESTIMATION.md` - Estimation algorithm documentation
- `DEFAULT_OBJECT.md` - Custom default object configuration
- `ANALYTICS.md` - Analytics integration guide
- `TROUBLESHOOTING.md` - Common issues and solutions

---

## Key Algorithms

### Triangle Subdivision
- **Location**: `texturizer.py:subdivide_triangle()`
- **Purpose**: Subdivide large triangles to create sufficient vertices for texture
- **Method**: Recursive 4-way subdivision until edges < point_distance
- **Optimization**: Skip triangles already smaller than point_distance (when enabled)

### Noise Generation
- **Location**: `texturizer.py:get_noise_value()`
- **Noise Types**:
  - `classic`: Perlin-like noise (default)
  - `simplex`: OpenSimplex noise
  - `voronoi`: Cellular/Voronoi noise
- **Parameters**: scale, octaves, persistence for layered noise

### Size Estimation
- **Location**: `texturizer.py:estimate_output_size()`
- **Method**:
  - Analyze edge length distribution (90th percentile)
  - Calculate subdivision levels needed
  - Estimate triangle count, memory, file size
- **Used by**: `/api/estimate` endpoint for pre-flight checks

### Feasibility Checking
- **Location**: `texturizer.py:check_processing_feasibility()`
- **Checks**: Max triangles (20M), max memory (4GB), max file size (500MB)
- **Returns**: Feasibility boolean, estimates, suggestions

---

## API Endpoints

### `GET /`
- Renders main application page
- Passes configuration (noise types, feature flags)

### `POST /api/estimate`
- **Purpose**: Estimate output size before processing
- **Parameters**: point_distance, skip_small_triangles, file/cube settings
- **Returns**: Feasibility, estimates (triangles, memory, file size), suggestions
- **Note**: MUST include skip_small_triangles parameter (bug fixed in commit 93f5ccf)

### `POST /api/process`
- **Purpose**: Apply fuzzy skin texture to STL
- **Parameters**: All fuzzy skin settings (thickness, point_distance, noise_type, etc.)
- **Returns**: Modified STL file as binary download
- **Special**: thickness=0 returns unprocessed mesh for preview

### `GET /health`
- Health check endpoint for monitoring
- Returns service status and configuration

---

## Configuration System

### Environment Variables (from `.env.example`)

#### Processing Limits
```bash
MAX_OUTPUT_TRIANGLES=20000000  # 20 million triangles
MAX_MEMORY_MB=4096             # 4GB
MAX_OUTPUT_FILE_SIZE_MB=500    # 500MB
```

#### UI Features (Toggle Features)
```bash
ENABLE_ROTATION_CONTROLS=false  # Show XYZ rotation controls in UI
SHOW_SUPPORT_FOOTER=false       # Show Ko-fi/support links
```

#### Default Object
```bash
DEFAULT_OBJECT_TYPE=cube        # 'cube' or 'custom'
DEFAULT_OBJECT_SIZE=20          # Size in mm
```

#### Server
```bash
PORT=8000                       # Default port (avoids macOS AirPlay on 5000)
FLASK_ENV=development           # development, production, testing
SECRET_KEY=your-secret-key      # Change in production
```

#### Analytics
```bash
GA_MEASUREMENT_ID=G-XXXXXXXXXX  # Google Analytics 4 (optional)
```

### Feature Flags
Controlled via environment variables, passed to template in `app.py`:
- `enable_rotation_controls` - Show model rotation controls
- `show_support_footer` - Display support/donation links

---

## UI Design

### Color Scheme (Dark Forest Green Theme)
- **Primary Background**: `#1a2e1a`
- **Panel Background**: `#1e3e21`
- **Input Background**: `#0f4a30`
- **Primary Accent**: `#4ade80` (bright green)
- **Hover Accent**: `#22c55e`
- **Borders**: `#3a6b4a`
- **Text**: `#eee` (light gray)

### Layout Structure
1. **Left Panel**: Settings and controls
   - File upload / default object selection
   - Fuzzy Skin Settings (noise type, thickness, point distance)
   - Advanced Options (seed, noise parameters)
   - Bottom Options (skip bottom, triangle optimization)
   - Action buttons (Preview, Process)
2. **Right Panel**: 3D viewer and info
   - Three.js 3D visualization
   - Expandable "More Info" section

### Three.js Viewer Configuration
- **Scene Background**: `0x1a2e1a` (matches UI)
- **Grid Colors**: `0x3a6b4a`, `0x2a5a3a`
- **Mesh Material**: `MeshPhongMaterial` with color `0x4ade80`
- **Lighting**: Ambient + 2 directional lights for good visibility
- **Controls**: OrbitControls for rotation/zoom

---

## Development History & Key Decisions

### Phase 1: Initial Development (Commits ab3257b - 40fd5f2)
- Created basic mesh texturing based on OrcaSlicer analysis
- Initial calibration for file size and processing

### Phase 2: Web Interface (Commits ea0e73e - 9c613a7)
- Built Flask web application
- Added Three.js 3D preview
- Joined web and CLI versions

### Phase 3: Optimization (Commit 721403c)
- Optimized for large meshes
- Performance improvements for production use

### Phase 4: Production Features (Commits 8bc9785 - 1a8eddc)
- Added output size estimation
- Ko-fi integration for support
- Google Analytics support
- Port configuration (default 8000 to avoid macOS conflicts)

### Phase 5: Branding & Custom Objects (Commits dc26739 - d1d8ed6)
- Added custom default object support (Fiuncholabs beaker card)
- Fern coin branding experiments
- Preview improvements (thickness=0 for unprocessed preview)

### Phase 6: Estimation Improvements (Commits e31e958 - 7e9ca7d)
- Configurable default object system
- Fixed estimation to analyze all triangles (not just first 100)
- Switched from mean to 90th percentile for better accuracy

### Phase 7: Major Optimizations (Commit a623853)
- **Triangle Size Optimization**: Skip subdivision for small triangles
- Significant performance improvement for detailed models
- Configurable via `skip_small_triangles` parameter

### Phase 8: UI Refinements (Commits 550cd7d - e9ca8b8)
- Removed object size input (simplified UX)
- Consolidated settings headers
- **Dark Forest Green Theme**: Complete UI redesign from blue to green
- Moved noise type to top of fuzzy skin settings
- Updated Three.js viewer colors to match theme

### Phase 9: Documentation & Info (Commit c3785df)
- Added expandable "More Info" section in UI
- Documented STL vs G-code differences
- Explained triangle optimization feature
- Listed processing limits for user awareness

### Phase 10: Bug Fixes (Commit 93f5ccf)
- **CRITICAL FIX**: Estimate endpoint now respects triangle optimization setting
- Bug: `/api/estimate` was not reading or passing `skip_small_triangles`
- Impact: Estimates were always pessimistic, didn't reflect optimization benefit
- Fixed in `app.py:213,237`

---

## Known Issues & Gotchas

### 1. Estimate Endpoint Parameters ⚠️
**Issue**: The `/api/estimate` endpoint MUST receive and pass the `skip_small_triangles` parameter.

**Context**: This was a bug (fixed in 93f5ccf) where estimates didn't change when optimization was toggled.

**Code Location**: `app.py:199-248`
```python
# MUST retrieve from form:
skip_small_triangles = request.form.get('skip_small_triangles', 'false').lower() == 'true'

# MUST pass to feasibility check:
feasibility = check_processing_feasibility(
    input_mesh,
    point_distance=point_distance,
    skip_small_triangles=skip_small_triangles  # Don't forget this!
)
```

### 2. Three.js Material Colors
**Issue**: Material colors use hex format `0x4ade80`, not CSS format `#4ade80`

**Location**: `templates/index.html` - search for `MeshPhongMaterial`

**Why**: Three.js uses numeric hex (0x prefix), not string hex (# prefix)

### 3. macOS Port Conflict
**Issue**: macOS AirPlay uses port 5000 by default

**Solution**: Default to port 8000 instead
- Set in `app.py` and `.env.example`
- Documented in `PORT_GUIDE.md`

### 4. Background Bash Processes
**Issue**: Multiple background Flask processes can accumulate during development

**Solution**: Use `lsof -ti:8000 | xargs kill -9` before starting new server

### 5. Estimation Algorithm Sensitivity
**Issue**: Mean edge length underestimates, max overestimates

**Solution**: Use 90th percentile of edge lengths (commit 48ea42d)

**Why**: Models often have mix of large and tiny triangles; 90th percentile balances accuracy

### 6. Triangle Subdivision Depth
**Issue**: Deep recursion for very large triangles vs small point_distance

**Solution**: Iterative subdivision in `subdivide_triangle()` with level limits

**Location**: `texturizer.py:subdivide_triangle()`

---

## Testing

### Unit Tests
**File**: `test_suite.py`

**Coverage**:
- Triangle subdivision (no subdivision, single level, multiple levels)
- Edge length calculations
- Subdivision level estimation

**Run**: `python3 test_suite.py`

### Manual Testing Checklist
- [ ] Upload STL file and preview
- [ ] Use default cube and preview
- [ ] Toggle triangle optimization, verify estimates change
- [ ] Process with different noise types
- [ ] Check file size estimates vs actual output
- [ ] Test with very large meshes (check memory limits)
- [ ] Verify dark green theme in all UI states
- [ ] Test expandable info section

### Test Files
- `simple-corner.stl` - Good for testing optimization (mix of triangle sizes)
- Default cube - Basic functionality testing

---

## Deployment

See `DEPLOYMENT.md` for full production deployment guide.

### Quick Start (Development)
```bash
# Install dependencies
pip3 install -r requirements.txt

# Create .env from example
cp .env.example .env

# Edit .env as needed
nano .env

# Run development server
python3 app.py
# Access at http://localhost:8000
```

### Production (Gunicorn)
```bash
gunicorn --config gunicorn.conf.py app:app
```

---

## Future Considerations

### Potential Improvements
1. **Streaming Processing**: For very large files, process in chunks
2. **Progress Indicator**: Real-time progress updates during processing
3. **Batch Processing**: Process multiple files at once
4. **Preset Profiles**: Save/load common setting combinations
5. **Advanced Noise**: More noise types (ridged, fractal, etc.)
6. **Mesh Repair**: Auto-fix non-manifold meshes before processing
7. **WebGL 2.0**: Upgrade Three.js viewer for better performance

### Architecture Notes
- Keep estimation logic synchronized with actual processing
- Always pass all parameters to estimate AND process endpoints
- Maintain feature flag consistency between backend and frontend
- Update AI_DEVELOPMENT_CONTEXT.md with significant changes

### Performance Optimization Ideas
- Cache subdivision calculations for identical triangles
- Parallel processing for independent triangle operations
- WebAssembly for client-side preview generation

---

## Recent Session Work Log

### Session: 2025-11-30 (Morning)
**Work Completed**:
1. Fixed estimate endpoint bug (skip_small_triangles parameter)
2. Created comprehensive AI development context document
3. Created v0.1 "First Leaf" release with git tag
4. Added version display in UI
5. Set up development branch for future work

**Commits**:
- `93f5ccf` - Fix estimate endpoint to respect triangle size optimization setting
- `c3785df` - Add expandable info section with documentation about the tool
- `e9ca8b8` - Update UI with dark forest green theme and improved layout
- `99bc29b` - Add AI_DEVELOPMENT_CONTEXT.md for project continuity
- `002038e` - Add version v0.1 (First Leaf) to application

**Git Tags**:
- `v0.1` - First Leaf release (stable baseline with core functionality)

**Branches**:
- `main` - Stable releases (tagged v0.1)
- `development` - New feature development

**Notes**:
- Estimates now correctly reflect optimization benefits
- UI theme is fully green (no blue remnants)
- Info section provides user documentation inline
- Version system in place for future releases
- Development branch ready for new features

### Session: 2025-11-30 (Afternoon - OAuth Configuration)
**Work Completed**:
1. Added Google OAuth configuration infrastructure
2. Updated requirements.txt with authentication dependencies
3. Created comprehensive GOOGLE_AUTH.md documentation
4. Updated AI_DEVELOPMENT_CONTEXT.md with OAuth feature

**Files Modified**:
- `.env.example` - Added OAuth environment variables (lines 59-65)
- `config.py` - Added OAuth configuration (lines 55-59)
- `requirements.txt` - Added flask-login, requests, oauthlib packages
- `GOOGLE_AUTH.md` - Complete setup and usage documentation

**Configuration Added**:
```bash
ENABLE_GOOGLE_AUTH=false  # Feature flag for optional authentication
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret
GOOGLE_DISCOVERY_URL=https://accounts.google.com/.well-known/openid-configuration
```

**Next Steps**:
- Implement auth module with OAuth flow
- Add login/logout routes to app.py
- Update UI with authentication buttons (when enabled)
- Test OAuth flow with test credentials

**Notes**:
- Authentication is fully optional (feature-flagged)
- Session-based implementation (no database in v0.1)
- App works identically when ENABLE_GOOGLE_AUTH=false
- Documentation covers setup, security, and troubleshooting

---

## Instructions for Future AI Sessions

### Before Starting Work
1. Read this entire file to understand context
2. Check git log for recent changes: `git log --oneline -10`
3. Review any open issues or TODOs
4. Check server status: `lsof -ti:8000`

### During Work
1. Test changes with both default cube and uploaded STL
2. Verify estimates match actual processing results
3. Check UI in browser after any frontend changes
4. Run unit tests if modifying core algorithms

### After Completing Work
1. Update this file's "Recent Session Work Log" section
2. Add any new gotchas or important decisions to "Known Issues"
3. Update architecture sections if design changes
4. Commit this file along with code changes
5. Include clear commit messages referencing this context file

### Communication with User
- Reference specific commit hashes when discussing changes
- Use file paths with line numbers (e.g., `app.py:213`)
- Explain WHY architectural decisions were made, not just WHAT changed
- Suggest testing steps for verification

---

## Glossary

- **Fuzzy Skin**: Randomized surface texture that hides layer lines
- **Point Distance**: Minimum distance between texture points (mm)
- **Triangle Subdivision**: Splitting triangles into smaller ones (4-way split)
- **Mesh**: 3D model represented as triangular faces
- **STL**: Standard Triangle Language - 3D model file format
- **G-code**: Machine instructions for 3D printers
- **Feasibility Check**: Pre-processing validation of resource requirements
- **Skip Small Triangles**: Optimization to avoid subdividing tiny triangles

---

*This document should be updated by AI assistants as the project evolves. Treat it as a living document that captures institutional knowledge across sessions.*
