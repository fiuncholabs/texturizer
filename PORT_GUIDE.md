# Port Configuration Guide

## Default Port: 8000

The STL Texturizer uses **port 8000** by default to avoid conflicts with macOS AirPlay Receiver (which uses port 5000).

---

## Running on Different Ports

### Quick Start (Default Port 8000)
```bash
python app.py
# Access at: http://localhost:8000
```

### Custom Port
```bash
PORT=3000 python app.py
# Access at: http://localhost:3000
```

### Using Port 5000 (if AirPlay is disabled)
```bash
PORT=5000 python app.py
# Access at: http://localhost:5000
```

---

## Fixing Port Conflicts

### macOS AirPlay Conflict (Port 5000)

If you get "port already in use" error:

**Option 1: Use Default Port 8000**
```bash
python app.py  # Automatically uses port 8000
```

**Option 2: Disable AirPlay Receiver**
1. Open **System Settings**
2. Go to **General** → **AirDrop & Handoff**
3. Turn off **AirPlay Receiver**

**Option 3: Kill Process on Port**
```bash
# Find and kill process using port 5000
lsof -ti:5000 | xargs kill -9

# Or for any port
lsof -ti:3000 | xargs kill -9
```

---

## Finding What's Using a Port

```bash
# Check what's using port 8000
lsof -i :8000

# Check what's using port 5000
lsof -i :5000

# Find all Flask/Python processes
ps aux | grep -E "flask|gunicorn|app.py"
```

---

## Killing Flask Processes

```bash
# Kill by port (recommended)
lsof -ti:8000 | xargs kill -9

# Kill by process name
pkill -f app.py
pkill -f flask
pkill -f gunicorn

# Kill specific process by PID
kill -9 <PID>
```

---

## Environment Variable

Set the port via environment variable:

**Temporary (current session):**
```bash
export PORT=3000
python app.py
```

**In .env file:**
```bash
PORT=8000
```

**Docker:**
```bash
docker run -p 8000:8000 -e PORT=8000 stl-texturizer
# Map container port 8000 to host port 3000:
docker run -p 3000:8000 stl-texturizer
```

---

## Platform-Specific Ports

Different deployment platforms may require specific ports:

| Platform | Port Configuration |
|----------|-------------------|
| **Local Development** | 8000 (default) |
| **Heroku/Render** | Uses `$PORT` env var automatically |
| **Railway** | Uses `$PORT` env var automatically |
| **Fly.io** | Set in `fly.toml` (default 8080) |
| **Docker** | Set via `-p` flag or docker-compose |

---

## Troubleshooting

### "Address already in use"
```bash
# Find what's using the port
lsof -i :8000

# Kill it
lsof -ti:8000 | xargs kill -9

# Or use a different port
PORT=9000 python app.py
```

### Can't access from other devices
Make sure the app binds to `0.0.0.0` (it does by default):
```python
app.run(host='0.0.0.0', port=8000)  # Already configured
```

Then access from other devices:
```
http://YOUR_IP:8000
```

### Firewall blocking port
```bash
# macOS - Allow port in firewall
# System Settings → Network → Firewall → Options

# Linux - Allow port with ufw
sudo ufw allow 8000/tcp
```

---

## Quick Reference

```bash
# Start on default port (8000)
python app.py

# Start on custom port
PORT=3000 python app.py

# Check what's using a port
lsof -i :8000

# Kill process on port
lsof -ti:8000 | xargs kill -9

# Find Flask processes
ps aux | grep flask

# Kill all Flask processes
pkill -f flask
```
