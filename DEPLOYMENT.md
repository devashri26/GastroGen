# GastroGen — Production Deployment Guide

This guide provides step-by-step instructions for deploying the **GastroGen Generative AI Research Platform** in production environments, from local workstations to cloud GPU instances (AWS EC2, GCP, Azure, RunPod, Lambda Labs) and Docker containers.

---

## 1. System Architecture

In production, GastroGen operates as a unified, high-performance platform:
* **Frontend**: React 18 + TypeScript + Tailwind CSS bundled into optimized static assets (`frontend/dist`).
* **Backend**: FastAPI with PyTorch CUDA GPU acceleration serving both the neural network inference endpoints (`/api/reconstruct`, `/api/screen`, `/api/generate`) and the frontend single-page application.
* **Ports**: Single unified port (default: `8000`).

```
┌────────────────────────────────────────────────────────┐
│               Client Web Browser                      │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP / WebSocket (Port 8000)
┌──────────────────────────▼─────────────────────────────┐
│                 FastAPI Production Server               │
│                                                        │
│  ┌───────────────────────┐  ┌───────────────────────┐  │
│  │   Frontend SPA (/ )   │  │   Inference API (/api)│  │
│  │   Static Assets       │  │   • /api/reconstruct  │  │
│  │   (frontend/dist)     │  │   • /api/health       │  │
│  └───────────────────────┘  └───────────┬───────────┘  │
│                                         │              │
│                             PyTorch CUDA 12.x          │
│                             NVIDIA GPU Acceleration    │
└────────────────────────────────────────────────────────┘
```

---

## 2. Zero-Cost Cloud Deployment: GitHub + Hugging Face Spaces (Recommended)

This is the recommended free production stack for academic research and portfolio demonstration:

| Layer | Recommended Choice | Cost |
| :--- | :--- | :--- |
| **Code Repository** | **GitHub** | Free |
| **Frontend** | **React + TypeScript** (`frontend/dist`) | Free |
| **Backend** | **FastAPI** (`app.py`) | Free |
| **Cloud Hosting** | **Hugging Face Spaces (Docker SDK)** | Free (2 vCPU, 16 GB RAM) |
| **Model Weights** | PyTorch `.pth` files (<70 MB each) | Free (Committed directly) |
| **SSL / HTTPS** | Provided by Hugging Face | Free |
| **Database** | None needed | Free |
| **Training** | Trained locally on RTX 4050 GPU | Free |

### Step-by-Step Deployment Workflow:

#### Step 1: Initialize Git and Push to GitHub
Open PowerShell or your terminal in the project directory:
```bash
# 1. Initialize git (if not already done)
git init

# 2. Stage files (.gitignore automatically filters heavy zip files)
git add .

# 3. Create your initial commit
git commit -m "feat: GastroGen v2.0 production deployment"

# 4. Create a repository on GitHub (e.g. 'gastrogen-ai') and push
git branch -M main
git remote add origin https://github.com/<your-github-username>/gastrogen-ai.git
git push -u origin main
```

#### Step 2: Create a Hugging Face Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. **Space Name**: e.g., `gastrogen-ai`
3. **License**: `mit` or `apache-2.0`
4. **Space SDK**: Select **Docker** -> **Blank**
5. **Space Hardware**: Choose **CPU Basic • 2 vCPU • 16 GB RAM • Free**
6. Click **"Create Space"**.

#### Step 3: Link & Deploy
* **Option A (GitHub Sync - Automatic Updates)**:
  Under your Space Settings -> **"Connect GitHub Repository"** -> select your `gastrogen-ai` repository. Every time you push to GitHub, Hugging Face automatically rebuilds and redeploys!
* **Option B (Direct Hugging Face Push)**:
  ```bash
  git remote add space https://huggingface.co/spaces/<your-hf-username>/gastrogen-ai
  git push space main
  ```

Hugging Face will automatically:
1. Detect the multi-stage `Dockerfile`.
2. Build the React frontend distribution.
3. Install CPU PyTorch and dependencies.
4. Launch FastAPI on port `7860`.
5. Issue a permanent, public HTTPS link: `https://huggingface.co/spaces/<your-username>/gastrogen-ai`.

---

## 3. Quick Start: Local Workstation Production Mode

### Prerequisites
* Python 3.10 or 3.11 with PyTorch installed (`torch`, `torchvision`)
* Node.js 18+ and npm (for building the frontend assets)
* NVIDIA GPU with CUDA support (optional, falls back to CPU if unavailable)

### Option A: Windows Batch (One-Click)
Double-click `start_production.bat` or run in terminal:
```cmd
start_production.bat
```

### Option B: Python Launcher
```bash
# 1. Install production requirements
pip install -r requirements-prod.txt

# 2. Build frontend assets (first time or after UI modifications)
cd frontend
npm install
npm run build
cd ..

# 3. Launch production server
python run_production.py
```

The server will automatically start at **`http://localhost:8000`** and open your default browser.

---

## 3. Docker & Docker Compose Deployment

### Prerequisites
1. [Docker](https://docs.docker.com/get-docker/) installed.
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU acceleration inside Docker).

### Launching with Docker Compose
```bash
# Build and launch with GPU passthrough
docker compose up --build -d

# View real-time logs
docker compose logs -f

# Check container health status
docker compose ps
```

### Stopping the Container
```bash
docker compose down
```

---

## 4. Cloud GPU Deployment (Ubuntu 22.04 LTS)

### 1. Install NVIDIA Drivers & CUDA Toolkit
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential dkms nvidia-cuda-toolkit
nvidia-smi  # Verify GPU is recognized
```

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone <your-repo-url> /opt/gastrogen
cd /opt/gastrogen

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-prod.txt
```

### 3. Build Frontend
```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

cd frontend
npm ci
npm run build
cd ..
```

### 4. Create Systemd Service (Auto-restart on boot)
Create `/etc/systemd/system/gastrogen.service`:
```ini
[Unit]
Description=GastroGen Medical AI Production Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/gastrogen
ExecStart=/opt/gastrogen/venv/bin/python run_production.py
Restart=always
RestartSec=5
Environment=HOST=0.0.0.0
Environment=PORT=8000

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gastrogen
sudo systemctl start gastrogen
sudo systemctl status gastrogen
```

---

## 5. Nginx Reverse Proxy & SSL (HTTPS)

For domain-based production deployment with SSL encryption:

### Nginx Configuration (`/etc/nginx/sites-available/gastrogen`)
```nginx
server {
    listen 80;
    server_name gastrogen.yourdomain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
```

Enable and secure with Certbot:
```bash
sudo ln -s /etc/nginx/sites-available/gastrogen /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d gastrogen.yourdomain.com
```

---

## 6. Health & Telemetry Endpoints

* **Application Health**: `GET /api/health`
  ```json
  {
    "status": "healthy",
    "service": "GastroGen Medical AI Platform",
    "cuda_available": true,
    "device": "cuda"
  }
  ```
* **Model Telemetry**: `GET /api/info`
* **Swagger API Documentation**: `http://localhost:8000/docs`
* **ReDoc Documentation**: `http://localhost:8000/redoc`
