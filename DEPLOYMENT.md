# Deployment Guide - STL Fuzzy Skin Texturizer

This guide covers deployment options for the STL Texturizer web application.

## Table of Contents
- [Local Development](#local-development)
- [Environment Configuration](#environment-configuration)
- [Deployment Options](#deployment-options)
  - [Render](#render)
  - [Railway](#railway)
  - [Fly.io](#flyio)
  - [DigitalOcean App Platform](#digitalocean-app-platform)
  - [Docker (Self-Hosted)](#docker-self-hosted)
- [Production Checklist](#production-checklist)
- [Monitoring & Troubleshooting](#monitoring--troubleshooting)

---

## Local Development

### Setup

1. **Clone the repository and navigate to the project:**
   ```bash
   cd /path/to/texturizer
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create environment file:**
   ```bash
   cp .env.example .env
   # Edit .env with your local settings
   ```

5. **Run the development server:**
   ```bash
   python app.py
   ```

   Or with Flask directly:
   ```bash
   flask run
   ```

6. **Access the application:**
   Open http://localhost:5000 in your browser

### Testing Production Mode Locally

To test production configuration locally:

```bash
# Set production environment
export FLASK_ENV=production
export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')

# Run with Gunicorn
gunicorn app:app --config gunicorn.conf.py
```

---

## Environment Configuration

### Required Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FLASK_ENV` | Environment (development/production) | `development` | No |
| `SECRET_KEY` | Secret key for sessions | Auto-generated | Yes (prod) |
| `PORT` | Server port | `5000` | No |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_CONTENT_LENGTH` | Max upload size (bytes) | `52428800` (50MB) |
| `PROCESSING_TIMEOUT` | Processing timeout (seconds) | `600` |
| `RATELIMIT_ENABLED` | Enable rate limiting | `true` (prod), `false` (dev) |
| `RATELIMIT_DEFAULT` | Default rate limit | `10 per minute` |
| `RATELIMIT_PROCESSING` | Processing rate limit | `3 per minute` |
| `CORS_ENABLED` | Enable CORS | `false` |
| `CORS_ORIGINS` | Allowed CORS origins | `*` |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## Deployment Options

### Render

**Cost:** Free tier available, Paid from $7/month

**Recommended Plan:** $21/month (2GB RAM)

#### Steps:

1. **Push code to GitHub/GitLab**

2. **Create new Web Service on Render:**
   - Connect your repository
   - Name: `stl-texturizer`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --config gunicorn.conf.py`

3. **Set Environment Variables:**
   ```
   FLASK_ENV=production
   SECRET_KEY=<generate-random-string>
   RATELIMIT_ENABLED=true
   ```

4. **Deploy!**

**Render Configuration File (optional):**

Create `render.yaml`:
```yaml
services:
  - type: web
    name: stl-texturizer
    env: python
    plan: starter  # or professional
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --config gunicorn.conf.py
    envVars:
      - key: FLASK_ENV
        value: production
      - key: SECRET_KEY
        generateValue: true
      - key: RATELIMIT_ENABLED
        value: true
      - key: MAX_CONTENT_LENGTH
        value: 52428800
```

---

### Railway

**Cost:** $5/month base + usage (~$10-20/month total)

#### Steps:

1. **Push code to GitHub**

2. **Create new project on Railway:**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

3. **Railway will auto-detect and deploy using Procfile**

4. **Set Environment Variables in Railway dashboard:**
   ```
   FLASK_ENV=production
   SECRET_KEY=<generate-random-string>
   RATELIMIT_ENABLED=true
   ```

5. **Generate domain or use custom domain**

**Railway Configuration File (optional):**

Create `railway.json`:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "gunicorn app:app --config gunicorn.conf.py",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

---

### Fly.io

**Cost:** Free tier available, ~$5-15/month for production

#### Steps:

1. **Install Fly CLI:**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login to Fly:**
   ```bash
   fly auth login
   ```

3. **Launch app:**
   ```bash
   fly launch
   ```

   This will create a `fly.toml` configuration file.

4. **Update `fly.toml`:**
   ```toml
   app = "stl-texturizer"
   primary_region = "sjc"

   [build]
     dockerfile = "Dockerfile"

   [env]
     FLASK_ENV = "production"
     PORT = "8080"
     RATELIMIT_ENABLED = "true"

   [http_service]
     internal_port = 8080
     force_https = true
     auto_stop_machines = true
     auto_start_machines = true
     min_machines_running = 0
     processes = ["app"]

   [[services.ports]]
     port = 80
     handlers = ["http"]
     force_https = true

   [[services.ports]]
     port = 443
     handlers = ["tls", "http"]

   [services.concurrency]
     type = "connections"
     hard_limit = 25
     soft_limit = 20

   [[services.tcp_checks]]
     interval = "15s"
     timeout = "2s"
     grace_period = "5s"
     restart_limit = 0

   [[services.http_checks]]
     interval = 10000
     grace_period = "5s"
     method = "get"
     path = "/health"
     protocol = "http"
     timeout = 2000
     tls_skip_verify = false
   ```

5. **Set secrets:**
   ```bash
   fly secrets set SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
   ```

6. **Deploy:**
   ```bash
   fly deploy
   ```

---

### DigitalOcean App Platform

**Cost:** $24/month (2GB RAM recommended)

#### Steps:

1. **Push code to GitHub**

2. **Create new App on DigitalOcean:**
   - Select your GitHub repository
   - Choose "Python" as the type
   - Configure build settings:
     - Build Command: `pip install -r requirements.txt`
     - Run Command: `gunicorn app:app --config gunicorn.conf.py`

3. **Set Environment Variables:**
   ```
   FLASK_ENV=production
   SECRET_KEY=<generate-random-string>
   RATELIMIT_ENABLED=true
   ```

4. **Choose plan** (Professional - 2GB RAM recommended)

5. **Deploy!**

**App Spec File (optional):**

Create `.do/app.yaml`:
```yaml
name: stl-texturizer
region: sfo
services:
  - name: web
    github:
      repo: your-username/texturizer
      branch: main
      deploy_on_push: true
    build_command: pip install -r requirements.txt
    run_command: gunicorn app:app --config gunicorn.conf.py
    environment_slug: python
    instance_count: 1
    instance_size_slug: professional-xs
    http_port: 5000
    health_check:
      http_path: /health
    envs:
      - key: FLASK_ENV
        value: production
      - key: RATELIMIT_ENABLED
        value: "true"
      - key: SECRET_KEY
        type: SECRET
```

---

### Docker (Self-Hosted)

**Cost:** $6-24/month (VPS hosting)

#### Local Docker Testing:

```bash
# Build image
docker build -t stl-texturizer .

# Run container
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -e SECRET_KEY=your-secret-key \
  stl-texturizer
```

#### Docker Compose:

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=${SECRET_KEY}
      - RATELIMIT_ENABLED=true
      - MAX_CONTENT_LENGTH=52428800
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Run with:
```bash
docker-compose up -d
```

#### Deploy to VPS (DigitalOcean, Linode, Vultr):

1. **SSH into your VPS:**
   ```bash
   ssh user@your-server-ip
   ```

2. **Install Docker:**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   ```

3. **Clone repository:**
   ```bash
   git clone https://github.com/your-username/texturizer.git
   cd texturizer
   ```

4. **Set environment variables:**
   ```bash
   export SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
   ```

5. **Run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

6. **Setup Nginx reverse proxy (optional but recommended):**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;

           # Increase timeouts for large file processing
           proxy_read_timeout 600s;
           proxy_send_timeout 600s;

           # Increase max body size
           client_max_body_size 50M;
       }
   }
   ```

7. **Setup SSL with Let's Encrypt:**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

---

## Production Checklist

Before deploying to production:

- [ ] Set `FLASK_ENV=production`
- [ ] Generate strong `SECRET_KEY`
- [ ] Enable rate limiting (`RATELIMIT_ENABLED=true`)
- [ ] Configure appropriate memory/CPU resources (2GB+ RAM recommended)
- [ ] Set up monitoring/logging
- [ ] Configure custom domain (optional)
- [ ] Enable HTTPS
- [ ] Test health check endpoint (`/health`)
- [ ] Test error handling with invalid files
- [ ] Test with large STL files
- [ ] Set up backup/monitoring if self-hosting
- [ ] Review security headers in production
- [ ] Set appropriate file size limits

---

## Monitoring & Troubleshooting

### Health Check

Access the health check endpoint:
```bash
curl https://your-domain.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "stl-texturizer",
  "noise_library_available": true
}
```

### Logs

**View logs in production:**

- **Render:** Dashboard → Logs tab
- **Railway:** Dashboard → Deployments → View logs
- **Fly.io:** `fly logs`
- **DigitalOcean:** Console → Runtime Logs
- **Docker:** `docker logs <container-name>`

### Common Issues

**1. Memory errors during processing:**
- Increase instance memory (2GB minimum recommended)
- Adjust `point_distance` parameter higher
- Check `MAX_CONTENT_LENGTH` setting

**2. Timeout errors:**
- Increase `PROCESSING_TIMEOUT`
- Increase Gunicorn timeout in `gunicorn.conf.py`
- Check platform-specific timeout settings

**3. Rate limiting too aggressive:**
- Adjust `RATELIMIT_PROCESSING` in environment
- Disable for testing: `RATELIMIT_ENABLED=false`

**4. CORS errors:**
- Enable CORS: `CORS_ENABLED=true`
- Set allowed origins: `CORS_ORIGINS=https://your-domain.com`

### Performance Tuning

**For high traffic:**
- Increase Gunicorn workers (in `gunicorn.conf.py` or via `GUNICORN_WORKERS` env var)
- Consider using Redis for rate limiting: `RATELIMIT_STORAGE_URL=redis://localhost:6379`
- Scale horizontally (multiple instances)

**For large files:**
- Increase memory allocation
- Increase timeouts
- Consider queue-based processing for very large files

---

## Cost Comparison Summary

| Platform | Monthly Cost | Ease of Setup | Best For |
|----------|-------------|---------------|----------|
| Render (Free) | $0 | ⭐⭐⭐⭐⭐ | Testing, demos |
| Fly.io | $5-15 | ⭐⭐⭐⭐ | Cost-effective production |
| Railway | $10-20 | ⭐⭐⭐⭐⭐ | Quick production deploy |
| DigitalOcean App | $24 | ⭐⭐⭐⭐ | Reliable production |
| VPS (Self-hosted) | $6-24 | ⭐⭐ | Full control needed |

---

## Support

For issues or questions:
- Check logs first
- Review this guide
- Open an issue on GitHub
- Check platform-specific documentation
