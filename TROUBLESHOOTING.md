# Troubleshooting Guide

This guide helps diagnose and fix common issues with the STL Fuzzy Skin Texturizer.

## Quick Diagnostics

### Run the Debug App
For any issues, start with the debug version to see detailed error messages:

```bash
python app_debug.py
```

This will show:
- Full stack traces for errors
- Detailed console logging
- JSON error responses (not HTML)
- Processing step-by-step output

---

## Common Issues

### 1. "Unexpected token '<', " <!DOCTYPE "... is not valid JSON"

**Symptoms:**
- Error appears in browser console when processing
- Processing seems to fail silently

**Cause:**
Server returned HTML error page instead of JSON response.

**Solutions:**

**A. Check server console for actual error**
The real error is shown in the terminal where you ran the app. Look for Python stack traces.

**B. Missing dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# Or just core dependencies
pip install flask numpy numpy-stl noise
```

**C. Use simple app for testing**
```bash
# Minimal version without extra dependencies
python app_simple.py
```

**D. Run debug app**
```bash
# Shows full error details
python app_debug.py
```

---

### 2. Out of Memory / Processing Fails

**Symptoms:**
- Server crashes during processing
- "MemoryError" in console
- Process killed by system

**Diagnosis:**
Check the warning in console:
```
WARNING: Estimated output size: 25000000 triangles
This may cause memory issues. Consider increasing point_distance.
Average input edge length: 2.50mm, target: 0.5mm
```

**Solutions:**

**A. Increase point_distance** (primary solution)
The relationship is quadratic: `output_triangles ≈ input_triangles × (avg_edge / point_distance)²`

```
Current:  point_distance=0.5mm  → 25M triangles
Solution: point_distance=1.0mm  → 6.25M triangles (75% reduction!)
Solution: point_distance=1.5mm  → 2.78M triangles (89% reduction!)
```

**B. Check file size**
Rough estimate: 1 million triangles ≈ 50MB file

If your input is 10MB with average edge 5mm:
- point_distance=0.8mm → ~250M triangles → 12GB file (too large!)
- point_distance=2.0mm → ~40M triangles → 2GB file (feasible)

**C. Use estimation endpoint first**
```bash
# Check before processing (fast, no actual processing)
curl -X POST http://localhost:8000/api/estimate \
  -F "file=@model.stl" \
  -F "point_distance=0.8"

# Response shows if feasible:
{
  "feasible": false,
  "reason": "Estimated output (25000000 triangles) exceeds maximum (20000000 triangles)",
  "suggestions": ["Increase point_distance to at least 1.2mm"],
  "estimates": {
    "estimated_triangles": 25000000,
    "estimated_file_size_mb": 1250.5,
    "estimated_memory_mb": 3200
  }
}
```

**D. Adjust environment limits** (if you have more resources)
```bash
export MAX_OUTPUT_TRIANGLES=50000000  # 50M instead of 20M
export MAX_MEMORY_MB=8192             # 8GB instead of 4GB
export MAX_OUTPUT_FILE_SIZE_MB=1000   # 1GB instead of 500MB

python app.py
```

---

### 3. Processing Takes Too Long

**Symptoms:**
- Processing never completes
- Timeout errors
- Browser shows "pending" indefinitely

**Solutions:**

**A. Increase timeout settings**
```bash
# In .env or environment
export PROCESSING_TIMEOUT=1200  # 20 minutes (default is 10)
export UPLOAD_TIMEOUT=600       # 10 minutes (default is 5)
```

**B. Platform-specific timeout adjustments**

**Render:**
```yaml
# render.yaml
services:
  - type: web
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn app:app --timeout 1200"
```

**Railway:**
Settings → Environment → Variables → Add `GUNICORN_TIMEOUT=1200`

**Fly.io:**
```toml
# fly.toml
[http_service]
  http_checks = []
  internal_port = 8000

[http_service.timeouts]
  hard_limit = "20m"
  soft_limit = "19m"
```

**C. Use faster parameters**
```
# Slower (higher quality)
thickness=0.3, point_distance=0.4, noise=perlin

# Faster (lower quality)
thickness=0.3, point_distance=1.2, noise=classic
```

Classic noise is ~2x faster than Perlin/Billow/Ridged.

**D. Process locally for very large files**
```bash
# Direct command-line processing (no server overhead)
python texturizer.py large_model.stl -o output.stl -t 0.3 -p 1.0
```

---

### 4. Invalid STL Output / Corrupted File

**Symptoms:**
- Downloaded STL won't open in slicer
- "Invalid STL" errors
- File size is 0 or very small

**Solutions:**

**A. Check for NaN/Inf errors**
Look in console for:
```
ERROR: Output mesh contains invalid values (NaN or Inf)
This may indicate numerical instability with the current parameters.
```

**B. Validate input STL**
```bash
# Use an STL validator first
python -c "from stl import mesh; m = mesh.Mesh.from_file('input.stl'); print('Valid')"
```

**C. Try different parameters**
Sometimes extreme parameter combinations cause issues:
```bash
# Avoid extremely small point_distance with large thickness
thickness=0.5, point_distance=0.1  # May cause issues

# Better balance
thickness=0.3, point_distance=0.8  # Safer
```

**D. Check file permissions**
Ensure the app can write to the temp directory:
```bash
# Linux/Mac
ls -la /tmp/

# Windows
dir %TEMP%
```

---

### 5. Port Already in Use

**Symptoms:**
```
OSError: [Errno 48] Address already in use
```

**Solutions:**

**A. Use different port**
```bash
PORT=8080 python app.py
```

**B. Kill process on port**
```bash
# Find process
lsof -ti:8000

# Kill it
kill -9 $(lsof -ti:8000)

# Or on Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**C. macOS AirPlay conflict**
Port 5000 is used by AirPlay Receiver on macOS. That's why we use 8000 by default.

```bash
# Disable AirPlay Receiver in System Preferences
# Or use a different port
PORT=3000 python app.py
```

---

### 6. Deployment Issues

#### Render

**Build fails:**
```bash
# Check Python version in render.yaml
python-version: "3.11"  # Ensure this matches your local version
```

**Out of memory during build:**
```yaml
# Use smaller instance for build, larger for runtime
service:
  type: web
  plan: free  # or starter
```

**Slow cold starts:**
```yaml
# Add health check to keep instance warm
healthCheckPath: /health
```

#### Railway

**Deploy fails:**
```bash
# Check logs in Railway dashboard
# Common issue: missing environment variables

# Set these in Railway dashboard:
FLASK_ENV=production
SECRET_KEY=<random-string>
```

**Memory limits:**
Railway free tier has 512MB limit. Use:
```bash
MAX_MEMORY_MB=400  # Leave headroom for system
MAX_OUTPUT_TRIANGLES=5000000  # Reduce limits
```

#### Fly.io

**Region selection:**
```bash
# Deploy to region with more resources
fly regions add iad  # US East (often has more capacity)
```

**Scale up memory:**
```bash
fly scale memory 2048  # 2GB
```

---

### 7. Rate Limiting Errors

**Symptoms:**
```json
{"error": "Rate limit exceeded. Please try again later."}
```

**Solutions:**

**A. Disable for development**
```bash
export RATELIMIT_ENABLED=false
python app.py
```

**B. Adjust limits**
```bash
export RATELIMIT_DEFAULT="50 per minute"      # General requests
export RATELIMIT_PROCESSING="10 per minute"   # Processing endpoint
```

**C. Use Redis for distributed limiting** (production)
```bash
# Install Redis
pip install redis

# Set storage URL
export RATELIMIT_STORAGE_URL="redis://localhost:6379"
```

---

### 8. CORS Errors (Cross-Origin Requests)

**Symptoms:**
```
Access to fetch at 'http://localhost:8000/api/process' from origin 'http://localhost:3000'
has been blocked by CORS policy
```

**Solutions:**

**A. Enable CORS**
```bash
export CORS_ENABLED=true
export CORS_ORIGINS="http://localhost:3000,https://yourdomain.com"
python app.py
```

**B. Allow all origins (development only)**
```bash
export CORS_ENABLED=true
export CORS_ORIGINS="*"
python app.py
```

---

### 9. Viewer Not Loading / Black Screen

**Symptoms:**
- 3D viewer shows black screen
- Model doesn't appear
- Console errors about Three.js

**Solutions:**

**A. Check CDN access**
Open browser console (F12) and check for errors loading:
- three.min.js
- OrbitControls.js
- STLLoader.js

**B. Check CSP headers**
If you see Content Security Policy errors:
```python
# In app.py, ensure CSP allows CDN:
csp = {
    'script-src': [
        "'self'",
        "'unsafe-inline'",
        'cdnjs.cloudflare.com',
        'cdn.jsdelivr.net'
    ],
}
```

**C. Try local Three.js**
Download Three.js libraries and serve them locally instead of CDN.

---

### 10. File Upload Fails

**Symptoms:**
- "File too large" error
- Upload hangs
- No file selected error when file is selected

**Solutions:**

**A. Check file size limit**
```bash
export MAX_CONTENT_LENGTH=104857600  # 100MB (default is 50MB)
python app.py
```

**B. Platform-specific limits**

**Render/Railway:** Default 100MB request limit
**Fly.io:** 100MB limit
**Nginx (if using):** Add `client_max_body_size 100M;`

**C. Chunk large files** (future feature)
Currently not supported. For >50MB files, use command-line tool:
```bash
python texturizer.py large_file.stl -o output.stl
```

---

## Debug Checklist

When reporting issues or debugging, collect this information:

### Environment
```bash
# Python version
python --version

# Installed packages
pip list | grep -E "(flask|numpy|stl|noise)"

# OS and architecture
uname -a  # Linux/Mac
systeminfo  # Windows
```

### Configuration
```bash
# Current environment variables
env | grep -E "(FLASK|PORT|MAX_|RATELIMIT)"

# App version
head -20 CHANGELOG.md
```

### Error Information
```bash
# Full error from console
# Include stack trace

# Server logs (last 50 lines)
tail -50 <log-file>

# Browser console errors (F12 → Console tab)
```

### Request Details
```bash
# Input file info
ls -lh input.stl

# Parameters used
{
  "thickness": 0.3,
  "point_distance": 0.8,
  "noise_type": "classic",
  ...
}

# Expected vs actual behavior
```

---

## Getting Help

1. **Check this troubleshooting guide first**
2. **Run `python app_debug.py`** to see detailed errors
3. **Search existing issues**: [GitHub Issues](https://github.com/your-username/texturizer/issues)
4. **Create new issue** with debug checklist information
5. **Ask in discussions**: [GitHub Discussions](https://github.com/your-username/texturizer/discussions)

---

## Development Tips

### Fast Iteration
```bash
# Use simple app for quick testing
python app_simple.py

# Use test cube instead of uploading files
# Check "Use default cube" in UI

# Use classic noise for faster processing
# Classic is ~2x faster than other noise types
```

### Profiling
```bash
# Time command-line processing
time python texturizer.py model.stl -o output.stl

# Profile with cProfile
python -m cProfile -s cumtime texturizer.py model.stl -o output.stl
```

### Memory Profiling
```bash
# Install memory profiler
pip install memory_profiler

# Run with profiling
python -m memory_profiler app_debug.py
```

---

## Performance Tuning

### For Large Deployments

**Gunicorn workers:**
```bash
# Calculate workers: (2 * CPU cores) + 1
gunicorn app:app --workers 5 --threads 2 --timeout 1200
```

**Memory per worker:**
Each worker can use ~500MB-2GB depending on file sizes.
```bash
# For 4GB instance with 4 workers:
# 4000MB / 4 workers = 1000MB per worker
export MAX_MEMORY_MB=900  # Leave headroom
```

**Database for rate limiting** (if high traffic):
```bash
# Use Redis instead of memory
export RATELIMIT_STORAGE_URL="redis://localhost:6379"
```

---

## Still Need Help?

If none of these solutions work:

1. **Gather all info** from Debug Checklist section
2. **Create detailed issue** on GitHub with:
   - Exact error message
   - Steps to reproduce
   - Environment details
   - What you've already tried
3. **Be patient** - this is open source and maintained by volunteers

**Priority support available** for sponsors on [Ko-fi](https://ko-fi.com/igloopup)
