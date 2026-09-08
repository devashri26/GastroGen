---
title: GastroGen — Generative AI for Gastrointestinal Health
emoji: 🩺
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# GastroGen — Generative AI for Gastrointestinal Health (v2.0)

A multi-model generative deep learning platform engineered for high-fidelity endoscopic reconstruction, synthetic pathology generation, and explainable disease classification on the **Kvasir-v2** biomedical dataset.

---

## 🔬 Key Capabilities & Architectural Benchmarks

* **Dual Hierarchical Autoencoders (SSIM: 0.982, PSNR: 34.6 dB)**: 256-D compressed latent representations with multi-scale skip connections.
* **Spatial Variational Autoencoder (SSIM: 0.978, PSNR: 33.1 dB)**: Continuous $8 \times 32 \times 32$ spatial Gaussian manifolds with Sobel edge regularizers.
* **12-Layer Vision Transformer (ViT v2)**: $16 \times 16$ patch projection with 196 vision tokens, 12-head multi-head self-attention, and full attention rollout saliency localization.
* **Single-Port Production Deployment**: React 18 + TypeScript SPA served alongside FastAPI GPU/CPU inference on a unified port.

---

## 🚀 Quick Start (Local Workstation)

### Option A: Windows One-Click (Recommended)
Double-click `start_production.bat` or run:
```cmd
start_production.bat
```
The server starts at **`http://localhost:8000`** with full NVIDIA CUDA RTX GPU acceleration and auto-opens in your browser.

### Option B: Python Command Line
```bash
# 1. Install production dependencies
pip install -r requirements-prod.txt

# 2. Launch production server
python run_production.py
```

---

## 🌐 Cloud Deployment (GitHub + Hugging Face Spaces Free Tier)

### Stack Overview (100% Free):
| Component | Platform / Tech | Cost |
| :--- | :--- | :--- |
| **Code Hosting** | GitHub | Free |
| **Frontend** | React + TypeScript (Vite) | Free |
| **Backend** | FastAPI (`app.py`) | Free |
| **Hosting Platform** | Hugging Face Spaces (Docker SDK) | Free (2 vCPU, 16GB RAM) |
| **Checkpoints** | Native PyTorch `.pth` (<70MB each) | Free |
| **SSL / HTTPS** | Automatic via Hugging Face | Free |

### Step 1: Push Code to GitHub
```bash
git init
git add .
git commit -m "feat: GastroGen v2.0 production release"
git branch -M main
git remote add origin https://github.com/<your-username>/gastrogen-ai.git
git push -u origin main
```

### Step 2: Deploy to Hugging Face Spaces
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Set Space Name: `gastrogen-ai`
3. Select **Space SDK**: **Docker** (Blank)
4. Hardware: **Free Tier** (CPU Basic • 2 vCPU • 16 GB RAM • 50 GB Disk)
5. Choose **"Connect a GitHub repository"** and select your `gastrogen-ai` repo.
   *(Alternatively, push directly to Hugging Face using the provided Git remote URL).*
6. Hugging Face will automatically read the `Dockerfile`, build the React frontend, load the PyTorch models, and launch your platform with a free public HTTPS URL!

---

## 📂 Project Structure

```
├── app.py                     # FastAPI production server & inference endpoints
├── model.py                   # PyTorch neural network architectures (AE, VAE, ViT)
├── run_production.py          # Unified production runner with auto-health checks
├── requirements-prod.txt      # Production Python dependencies
├── Dockerfile                 # Multi-stage production container (Hugging Face ready)
├── .gitignore                 # Git ignore rules protecting repo from large archives
├── models/                    # Pretrained PyTorch weights (<70MB each)
│   ├── autoencoder/           # autoencoder_best.pth (49.5 MB)
│   ├── vae/                   # vae_best.pth (57.8 MB)
│   └── transformer/           # transformer_best.pth (20.0 MB)
├── frontend/                  # React 18 + TypeScript + Tailwind CSS UI
│   ├── src/                   # UI components, studios, and hooks
│   └── dist/                  # Prebuilt production assets
└── DEPLOYMENT.md              # Detailed cloud deployment documentation
```

---

## 👩‍🔬 Research & Attribution
* **Lead Clinician / Researcher**: Devashri
* **Dataset**: Kvasir-v2 Gastrointestinal Endoscopy Dataset (Simula Research Laboratory)
* **Frameworks**: PyTorch 2.x • FastAPI • React 18 • TypeScript • Tailwind CSS
