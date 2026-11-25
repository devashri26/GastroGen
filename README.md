---
title: GastroGen — Generative AI for Gastrointestinal Health
emoji: 🩺
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.11.0"
python_version: "3.10"
suggested_hardware: zero-a10g
app_file: app.py
pinned: false
license: mit
---

# GastroGen: Generative AI for Gastrointestinal Health
### A Multi-Model Framework for Endoscopic Reconstruction • Synthetic Data • Explainable Analysis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-3178C6.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 👥 Project Information & Group Details

### **Group No. 10**

* **Project Title**: **GastroGen: Generative AI for Gastrointestinal Health**
* **Subtitle**: *A Multi-Model Framework for Endoscopic Reconstruction • Synthetic Data • Explainable Analysis*
* **Department & Institution**: Department of Computer Engineering, MIT Academy of Engineering, Pune
* **Project Guide**: **Dr. Savita Mane**

| Team Members | PRN / Roll Number | Department & College |
| :--- | :---: | :--- |
| **Devashri Chakkarwar** | `202301040210` | Computer Engineering, MIT AOE Pune |
| **Preeti Koli** | `202301040213` | Computer Engineering, MIT AOE Pune |
| **Shrirang Deshmane** | `202301040237` | Computer Engineering, MIT AOE Pune |

---

## 📑 Table of Contents
1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Dataset — Kvasir-v2](#4-dataset--kvasir-v2)
5. [Literature Survey](#5-literature-survey)
6. [System Architecture](#6-system-architecture)
7. [Model Families & Formulations](#7-model-families--formulations)
   * [7.1 Convolutional Autoencoder (AE)](#71-convolutional-autoencoder-ae)
   * [7.2 Variational Autoencoder (VAE) & Spatial VAE](#72-variational-autoencoder-vae--spatial-vae)
   * [7.3 Vision Transformer (ViT-B/16) & Explainability](#73-vision-transformer-vit-b16--explainability)
   * [7.4 Class-Conditional Generative Models (FastGAN & Diffusion)](#74-class-conditional-generative-models-fastgan--diffusion)
8. [Comprehensive Benchmark Results (Figure 12)](#8-comprehensive-benchmark-results)
9. [Key Project Outcomes](#9-key-project-outcomes)
10. [Visual Outputs & Demonstrations](#10-visual-outputs--demonstrations)
11. [Ethical & Responsible Deployment](#11-ethical--responsible-deployment)
12. [Future Enhancements & Research Roadmap](#12-future-enhancements--research-roadmap)
13. [Conclusion & Key Takeaways](#13-conclusion--key-takeaways)
14. [Repository Structure](#14-repository-structure)
15. [Quick Start & Installation](#15-quick-start--installation)

---

## 1. Introduction

**GastroGen** is a unified deep-learning research platform that learns comprehensive visual representations of gastrointestinal (GI) endoscopic images. It addresses the fundamental bottlenecks of medical imaging by synthesizing generative modeling, high-resolution visual reconstruction, diagnostic classification, and attention-driven interpretability into a cohesive, clinical-grade framework.

### Core Capabilities
* **High-Fidelity Reconstruction**: Pixel-level and perceptual reconstruction of mucosal tissue architectures across anatomical and pathological landmarks.
* **Latent Representation Learning**: Disentangling complex endoscopic features into structured, probabilistic low-dimensional latent spaces.
* **Synthetic GI Data Generation**: Synthesizing high-realism, class-conditional endoscopy frames to alleviate extreme class imbalance and data scarcity.
* **Anomaly & Localization Analysis**: Employing reconstruction residual error maps and self-attention rollout to isolate mucosal lesions and boundaries.
* **Explainable Outputs (XAI)**: Generating transparent attention heatmaps enabling gastroenterologists to audit AI decision boundaries.
* **Responsible AI**: Engineering a secure, privacy-preserving, and leakage-safe medical evaluation pipeline.

---

## 2. Problem Statement

Developing robust, deployable computer vision systems for gastrointestinal endoscopy entails four severe medical domain challenges:

1. **Scarcity & Imbalance of Annotated Medical Data**: High-quality endoscopic imagery requires expert clinical labeling, making dataset curation expensive, time-consuming, and heavily constrained by institutional privacy firewalls (HIPAA, GDPR).
2. **Complex & Highly Variable Visual Patterns**: Mucosal tissue presents immense intra-class variability in lumen depth, variable specular reflections, motion artifacts, bile fluids, and mucosal folds across diverse endoscope optics (Olympus, Fujinon, Pentax).
3. **Lack of Model Interpretability (Black-Box AI)**: Traditional deep learning classifiers output scalar predictions without visual provenance, creating a trust deficit that inhibits clinical adoption.
4. **Privacy, Security & Patient Safety**: Training and serving healthcare models requires strict boundaries against patient data leakage, de-identification protocols, and rigorous human-in-the-loop safeguards.

> **Central Hypothesis**: Combining generative reconstruction models (AE, VAE), generative synthesis (FastGAN, Diffusion), and vision transformers (ViT) within a unified pipeline yields superior feature representations, robust anomaly detection, explainable classification, and privacy-compliant medical data generation.

---

## 3. Project Objectives

1. **Develop and evaluate generative deep learning models** for high-fidelity endoscopic image reconstruction across standard gastrointestinal landmarks and pathological lesions.
2. **Compare convolutional, variational, spatial, and transformer-based architectures** to benchmark perceptual quality (SSIM, PSNR, LPIPS), latent regularity, and inference speed.
3. **Explore class-conditional synthetic data generation** (FastGAN, DDPM/DDIM) to generate diverse endoscopic imagery for data augmentation and training balancing.
4. **Investigate reconstruction residuals and attention mechanisms** for unsupervised anomaly localization and lesion boundary demarcation.
5. **Provide transparent, interpretable visual outputs** via self-attention rollout and residual difference maps to facilitate human physician review.
6. **Establish an ethical, privacy-preserving, and leakage-safe framework** ensuring compliance with healthcare data governance.

---

## 4. Dataset — Kvasir-v2

The GastroGen framework is developed and evaluated on the benchmark **Kvasir-v2** dataset, a multi-class endoscopic image dataset collected from internal examinations at Vestre Viken Health Trust in Norway.

```
+-----------------------------------------------------------------------------------------+
|                                    Kvasir-v2 Dataset                                    |
|                                8,000 Total Images (1:1)                                 |
+---------------------------+-----------------------------+-------------------------------+
|     Therapeutic Classes   |     Pathological Classes    |       Anatomical Classes      |
+---------------------------+-----------------------------+-------------------------------+
| 1. Dyed-Lifted Polyps     | 3. Esophagitis              | 4. Normal Cecum               |
| 2. Dyed-Resection Margins | 7. Polyps                   | 5. Normal Pylorus             |
|                           | 8. Ulcerative Colitis       | 6. Normal Z-Line              |
+---------------------------+-----------------------------+-------------------------------+
| 1,000 images / class      | 1,000 images / class        | 1,000 images / class          |
+---------------------------+-----------------------------+-------------------------------+
```

### Preprocessing & Leakage-Safe Partitioning
* **Scope Overlay Removal**: Masking clinical text overlays and green scope-position indicators using HSV color thresholding.
* **Endoscopic Aperture Preservation**: Native 5:4/4:3 circular and octagonal camera aperture borders are letterboxed into square canvases ($256 \times 256$ for AE/VAE, $224 \times 224$ for ViT) preserving optical vignette geometry without distortion.
* **Laplacian Blur Rejection**: Automated image quality verification rejecting motion-blurred frames with variance $< 100$.
* **Leakage-Safe Perceptual Hashing (pHash)**: Endoscopic video sequences produce consecutive near-duplicate frames. Frames are grouped using perceptual hashing (Hamming distance threshold $\le 5$); entire hash clusters are assigned atomically to either Train, Validation, or Test splits:
  * **Train Split**: 6,400 images (80%)
  * **Validation Split**: 800 images (10%)
  * **Test Split**: 800 images (10% — exactly 100 held-out images per class)

---

## 5. Literature Survey

The architecture of GastroGen synthesizes advances across six foundational computer vision and medical AI paradigms:

| Research Pillar | Focus & Key Literature | GastroGen Integration |
| :--- | :--- | :--- |
| **GI Endoscopy AI** | Deep CNNs for automated polyp detection & mucosal classification (Pogorelov et al., Bernal et al.) | Baseline feature extraction and multi-class benchmarking on Kvasir-v2 |
| **Autoencoders & VAEs** | Self-supervised medical image compression and latent manifold representation (Kingma & Welling, Higgins et al.) | Dual-stream convolutional AE and continuous Spatial VAE for structure reconstruction |
| **Generative Adversarial Networks (GANs)** | Few-shot and class-conditional synthetic medical image synthesis (Goodfellow et al., FastGAN) | Class-conditional FastGAN v4 generating $128 \times 128$ synthetic endoscopy frames |
| **Diffusion Probabilistic Models** | Iterative noise estimation for high-realism medical imaging (Ho et al., DDIM/DDPM) | 9.79M-parameter conditional U-Net for noise-conditioned iterative generation |
| **Vision Transformers (ViT)** | Patch-based self-attention for medical imaging (Dosovitskiy et al.) | 12-layer ViT-B/16 achieving 92.00% top-1 accuracy across 8 classes |
| **Explainable AI (XAI)** | Saliency maps, Grad-CAM, and Attention Rollout (Abnar & Zuidema, Selvaraju et al.) | Multi-layer attention rollout maps isolating mucosal diagnostic evidence |

---

## 6. System Architecture

GastroGen employs a decoupled, production-ready micro-architecture separating frontend visualization, asynchronous API routing, and hardware-accelerated PyTorch execution:

```
[ User Browser ]
       |
       v (HTTP / WebSocket)
+-----------------------------------------------------------------------------------+
|                           Frontend Client (Port 8000 / 3000)                      |
|  • React 18 SPA  • TypeScript  • Tailwind CSS  • Recharts  • Lucide Medical Icons |
+-----------------------------------------------------------------------------------+
       |
       v (RESTful API / JSON Payloads / Multipart Form-Data)
+-----------------------------------------------------------------------------------+
|                             FastAPI Backend Service                               |
|  • Asynchronous Endpoints (/api/reconstruct, /api/classify, /api/synthesize)      |
|  • OpenCV / PIL Preprocessing  • Scope Border Letterboxing  • Base64 Streaming    |
+-----------------------------------------------------------------------------------+
       |
       v (CUDA / MPS / CPU Tensor Pipeline)
+-----------------------------------------------------------------------------------+
|                           PyTorch Deep Learning Core                              |
|  +--------------------+  +--------------------+  +-----------------------------+  |
|  | Autoencoder (AE)   |  | Spatial VAE        |  | Vision Transformer (ViT)    |  |
|  | 256-D Latent Code  |  | Continuous Prior   |  | 196 Tokens • 12 Layers      |  |
|  +--------------------+  +--------------------+  +-----------------------------+  |
|  +--------------------------------------------+  +-----------------------------+  |
|  | FastGAN v4 Generator (8 Checkpoints)       |  | DDPM / DDIM Diffusion U-Net |  |
|  +--------------------------------------------+  +-----------------------------+  |
+-----------------------------------------------------------------------------------+
       |
       v
+-----------------------------------------------------------------------------------+
|                             Diagnostics & Visualizations                          |
|  • SSIM / PSNR / MSE Fidelity  • ViT Attention Rollout  • Residual Error Heatmaps |
+-----------------------------------------------------------------------------------+
```

---

## 7. Model Families & Formulations

### 7.1 Convolutional Autoencoder (AE)
* **Design**: Deterministic convolutional encoder-decoder network designed for sharp visual reconstruction.
* **Encoder**: 4 sequential downsampling blocks (Conv2d $4 \times 4$, Stride 2, BatchNorm, LeakyReLU) compressing $256 \times 256 \times 3$ inputs into a compact **256-dimensional latent code**.
* **Decoder**: 4 transposed convolution blocks ($4 \times 4$, Stride 2) with skip-connections expanding the latent code back to original pixel dimensions.
* **Loss Function**: Composite objective combining pixel reconstruction with structural edge preservation:
  $$\mathcal{L}_{AE} = \alpha \|x - \hat{x}\|_1 + \beta \|x - \hat{x}\|_2^2 + \gamma \|\nabla x - \nabla \hat{x}\|_1$$
* **Performance**: **SSIM: 0.982**, **PSNR: 34.14 dB**, **Latency: 8.3 ms**.

### 7.2 Variational Autoencoder (VAE) & Spatial VAE
* **Design**: Probabilistic generative latent space mapping endoscopic distributions to continuous Gaussian manifolds.
* **Reparameterization Trick**: Instead of direct sampling, the encoder outputs mean $\mu(x)$ and log-variance $\log \sigma^2(x)$. The latent vector $z$ is sampled via:
  $$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
* **Objective Function**: Combines reconstruction fidelity with Kullback-Leibler (KL) divergence regularizing the latent distribution against standard normal prior $\mathcal{N}(0, I)$:
  $$\mathcal{L}_{VAE} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta_{KL} D_{KL}(q_\phi(z|x) \,||\, p(z))$$
* **Spatial VAE Variant**: Retains spatial tensor structure ($8 \times 32 \times 32$) across the bottleneck to preserve localized spatial layout of vascular patterns and polyps.
* **Performance**: **SSIM: 0.975**, **PSNR: 32.11 dB**, **Latency: 8.6 ms**.

### 7.3 Vision Transformer (ViT-B/16) & Explainability
* **Design**: Fine-tuned Vision Transformer for 8-class GI pathology and anatomical classification with integrated self-attention explainability.
* **Tokenization**: Input image ($224 \times 224 \times 3$) is partitioned into a $14 \times 14$ grid of non-overlapping patches ($16 \times 16$ pixels), linearly projected to 768-D embeddings, and concatenated with a learnable `[CLS]` classification token (197 tokens total).
* **Attention Mechanism**: 12 Transformer encoder blocks featuring Multi-Head Self-Attention (12 heads per layer):
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
* **Attention Rollout Saliency**: Quantifies recursive information flow across all 12 layers:
  $$A_{\text{rollout}} = \prod_{l=1}^{L} \left(0.5 A_l + 0.5 I\right)$$
  The resulting spatial attention map highlights the exact mucosal regions influencing the diagnostic classification.
* **Performance**: **Top-1 Accuracy: 92.00% (736/800 test images)**, **Top-3 Accuracy: 99.88%**, **Macro-F1: 91.99%**, **Latency: 10.4 ms**.

### 7.4 Class-Conditional Generative Models (FastGAN & Diffusion)
* **FastGAN v4**: Lightweight class-conditional generative adversarial networks with skip-layer excitation (SLE) modules trained individually per class ($128 \times 128$). Achieves **FID: 45.84**, **KID: 0.0252**, inference latency of **12.4 ms**.
* **Conditional Diffusion (DDPM/DDIM)**: 9.79M parameter U-Net trained over 100 epochs with cosine noise scheduling. Reconstructs endoscopic imagery from Gaussian noise conditioned on class labels, reaching **FID: 38.42**, **KID: 0.0165**.

---

## 8. Comprehensive Benchmark Results

The complete quantitative benchmark across all five GastroGen model paradigms is summarized below (derived from **Figure 12: Master Multi-Paradigm Comparison Dashboard**):

| Architecture | Paradigm | Primary Objective | Key Metric | Latency | Parameters | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Autoencoder v2** | Reconstruction | Pixel Reconstruction | **SSIM: 0.982 \| 34.1 dB** | 8.3 ms | 12.99 M | Published |
| **Spatial VAE v2** | Recon & Prior | Smooth Latent Prior | **SSIM: 0.975 \| 32.1 dB** | 8.6 ms | 15.15 M | Published |
| **Transformer-VAE** | Recon & Attention | Perceptual Attention | **SSIM: 0.981 \| 34.7 dB** | 10.4 ms | 5.26 M | Published |
| **FastGAN v4** | Synthesis (8-Class) | Real-Time Generation | **FID: 45.84 \| KID: 0.025** | 12.4 ms | 34.04 M | Published |
| **Cond. Diffusion** | Synthesis (CFG) | Distributional Realism | **FID: 38.42 \| KID: 0.016** | 3619 ms | 9.79 M | Published |
| **ViT-B/16 v2** | Classification | 8-Class Diagnostic XAI | **Top-1: 92.0% \| F1: 92.0%** | 10.4 ms | 86.57 M | Published |

### Diagnostic Performance Breakdown (ViT-B/16 on Held-Out Test Set, N=800)
* **Normal Cecum**: Precision: 99.01% | Recall: 100.00% | F1: 99.50% (Flawless anatomical landmark identification)
* **Normal Pylorus**: Precision: 98.02% | Recall: 99.00% | F1: 98.51%
* **Dyed-Resection Margins**: Precision: 96.88% | Recall: 93.00% | F1: 94.90%
* **Esophagitis**: Precision: 85.59% | Recall: 95.00% | F1: 90.05%
* **Polyps**: Precision: 85.26% | Recall: 81.00% | F1: 83.08%
* **Global Diagnostic Agreement**: Top-1 Accuracy: **92.00%** | Top-3 Accuracy: **99.88%** | Macro Precision: **92.30%** | Macro F1: **91.99%**

---

## 9. Key Project Outcomes

1. **High-Fidelity Reconstruction**: Convolutional Autoencoders and Transformer-VAEs achieve structural similarity exceeding 0.98 SSIM, accurately reconstructing micro-vascular networks, pit patterns, and dye pools.
2. **Synthetic Medical Data Generation**: Latent space modeling and FastGAN synthesis generate diverse, class-authentic endoscopic images capable of mitigating severe data imbalance.
3. **Unsupervised Anomaly Localization**: Pixel-wise reconstruction error maps ($|x - \hat{x}|$) highlight pathological mucosal deviations (ulcers, adenomas) without requiring manual segmentation masks during training.
4. **Interpretable Decision Support**: Vision Transformer attention rollout maps confirm that diagnostic predictions are guided by clinical features (elevated mucosal borders, inflammatory erythema) rather than background lens artifacts.
5. **Transformer Architectural Superiority**: Across the evaluation matrix, **Transformer-based models deliver the strongest performance** — achieving peak classification accuracy (92.00%) and highest reconstruction PSNR (34.67 dB).

---

## 10. Visual Outputs & Demonstrations

### 📸 2×4 Real Endoscopy Classes Collage
The eight distinct classes of Kvasir-v2 with preserved endoscopic camera apertures:
* **Row 1**: Dyed-Lifted Polyps • Dyed-Resection Margins • Esophagitis • Normal Cecum
* **Row 2**: Normal Pylorus • Normal Z-Line • Polyps • Ulcerative Colitis  
*High-resolution asset*: [`reports/kvasir_v2_8_classes_real_collage.png`](reports/kvasir_v2_8_classes_real_collage.png)

### 🔬 Real vs. Synthetic FastGAN Comparison Panel
Head-to-head comparison between real Kvasir-v2 endoscopy frames and synthetic images produced by class-conditional FastGAN generators:
*High-resolution asset*: [`reports/real_vs_synthetic_fastgan_panel.png`](reports/real_vs_synthetic_fastgan_panel.png)

### 📊 Master Benchmark Dashboard (Figure 12)
Comprehensive multi-model dashboard covering paired reconstruction, generative distribution distances, parameter footprints, and the performance synthesis matrix:
*High-resolution asset*: [`paper_figures/png/fig12_model_comparison.png`](paper_figures/png/fig12_model_comparison.png)

### 🖼️ Master Scientific Conference Poster
Full 24" × 36" (3600 × 5400 px, 300 DPI) conference poster:
*High-resolution asset*: [`reports/GastroGen_Scientific_Poster.png`](reports/GastroGen_Scientific_Poster.png) | Vector PDF: [`reports/GastroGen_Scientific_Poster.pdf`](reports/GastroGen_Scientific_Poster.pdf)

---

## 11. Ethical & Responsible Deployment

GastroGen is engineered adhering to strict bioethical and responsible AI principles for healthcare:

* **Patient Privacy**: Evaluated strictly on open-access benchmark data (Kvasir-v2) and synthetic images; real patient identifiable health information (PHI/ePHI) is strictly excluded.
* **System Security**: Production deployments implement TLS/HTTPS encryption, role-based access control (RBAC), and sandboxed execution environments.
* **Algorithmic Transparency**: Model limitations, training hyperparameters, confidence calibration scores, and failure modalities are documented openly via Model Cards.
* **Human-in-the-Loop Oversight**: The platform is explicitly positioned as a **diagnostic copilot** for clinical decision support. Final diagnostic determinations and therapeutic interventions remain under the sole authority of licensed gastroenterologists.

---

## 12. Future Enhancements & Research Roadmap

As outlined in the research roadmap (Slide 19 of presentation):

1. **Latent Diffusion Scaling**: Train high-resolution Latent Diffusion Models (LDMs) on multi-GPU clusters for photorealistic $512 \times 512$ synthetic endoscopy synthesis.
2. **Multi-Faceted Generative Evals**: Incorporate Fréchet Video Distance (FVD), precision-recall distribution curves, and medical authenticity scoring against clinical panels.
3. **Quantitative Anomaly Benchmarking**: Validate reconstruction residual localization against ground-truth physician polyp segmentation masks (Dice / IoU metrics).
4. **Cross-Center & Multi-Device Validation**: Test generalization on external clinical cohorts (HyperKvasir, Nerthus, PICCOLO) across Olympus, Pentax, and Fujinon scopes.
5. **Privacy-Preserving Federated Learning**: Implement differential privacy ($\epsilon, \delta$-DP) and federated averaging for cross-hospital collaborative training without raw data sharing.
6. **Clinical Video Stream Integration**: Develop low-latency video streaming pipelines connecting directly to endoscope video processors via DICOM / HL7 standards.

---

## 13. Conclusion & Key Takeaways

* **Integrated Multi-Paradigm Workflow**: GastroGen successfully bridges Generative AI (AE, VAE), Synthetic Data Generation (FastGAN, Diffusion), Disease Classification (ViT), and Explainability (Attention Rollout) in a single framework.
* **Transformer Architectures Excel**: Vision Transformers achieved the highest classification accuracy (**92.00% Top-1, 99.88% Top-3**), while Transformer-VAEs yielded the lowest perceptual distortion (**PSNR: 34.67 dB, LPIPS: 0.0093**).
* **Safe Clinical Translation**: Responsible AI safeguards, leakage-safe pHash splitting, and interpretable saliency heatmaps establish a reliable foundation for assistive clinical AI in digestive healthcare.

---

## 14. Repository Structure

```
├── app.py                             # FastAPI production server & REST API endpoints
├── model.py                           # Core PyTorch models (Autoencoder, VAE, ViT)
├── gan_model.py                       # FastGAN v4 architecture & checkpoint loader
├── diffusion_model.py                 # Conditional DDPM/DDIM U-Net diffusion model
├── generate_poster.py                 # Master Scientific Conference Poster generator
├── generate_8_classes_collage.py      # Standalone 2x4 8-class real endoscopy collage script
├── run_production.py                  # Single-port production launcher (FastAPI + React)
├── start_production.bat               # Windows one-click production runner
├── Dockerfile                         # Production multi-stage Docker container
├── requirements-prod.txt              # Production Python package requirements
├── data_dir/                          # Kvasir-v2 dataset (8 classes, 1,000 images each)
├── models/                            # Trained PyTorch model weights
│   ├── autoencoder/                   # autoencoder_best.pth (SSIM: 0.982)
│   ├── vae/                           # vae_best.pth (SSIM: 0.975)
│   ├── transformer/                   # transformer_best.pth (Acc: 92.00%)
│   └── gan/                           # FastGAN checkpoints for all 8 classes
├── paper_figures/                     # Publication figures, tables & captions
│   ├── png/                           # 300 DPI high-resolution figures (fig01 - fig12)
│   ├── pdf/                           # Vector PDF figures (fig01 - fig12)
│   └── csv/                           # Benchmark metrics & confusion matrices
├── reports/                           # Deliverable posters, collages, and panels
└── frontend/                          # React 18 + TypeScript SPA dashboard
    ├── src/                           # React studios, components, and state hooks
    └── dist/                          # Compiled production frontend build
```

---

## 15. Quick Start & Installation

### Option A: Windows One-Click (Recommended)
Double-click `start_production.bat` or run in Command Prompt:
```cmd
start_production.bat
```
* Automatically verifies environment dependencies.
* Launches unified server at **`http://localhost:8000`** with CUDA GPU acceleration.
* Automatically opens the interactive dashboard in your default browser.

### Option B: Python Terminal
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/gastrogen-ai.git
cd gastrogen-ai

# 2. Install dependencies
pip install -r requirements-prod.txt

# 3. Launch unified production server
python run_production.py
```

### Option C: Docker Deployment
```bash
# Build Docker image
docker build -t gastrogen:v2.0 .

# Run container with GPU support
docker run -p 8000:8000 --gpus all gastrogen:v2.0
```

### Option D: Regenerate Benchmark Figures & Posters
```bash
# Regenerate 2x4 Real Endoscopy Collage
python generate_8_classes_collage.py

# Regenerate Figures 10, 11, and 12 (Benchmark Dashboard)
python scripts/generate_fig10_fig11_fig12.py

# Regenerate Master 24x36 Conference Poster (PNG, PDF, HTML)
python generate_poster.py
```

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).

```bibtex
@article{chakkarwar2026gastrogen,
  title={GastroGen: A Multi-Model Generative AI Framework for Reconstruction, Synthesis and Classification of Gastrointestinal Endoscopy Images},
  author={Chakkarwar, Devashri and Koli, Preeti and Deshmane, Shrirang and Mane, Savita},
  journal={Department of Computer Engineering, MIT Academy of Engineering, Pune},
  year={2026}
}
```
