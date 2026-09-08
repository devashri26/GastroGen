# =======================================================
# GastroGen — Multi-Stage Production Dockerfile
# Optimized for Hugging Face Spaces (Free Tier) & Cloud
# =======================================================

# Stage 1: Build React Frontend Assets
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Python Runtime
FROM python:3.10-slim AS runtime
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install runtime utilities for OpenCV & Matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user (Hugging Face Spaces requirement UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /app

# Install lightweight CPU PyTorch wheel (fast download & memory efficient for free cloud tiers)
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining production dependencies
COPY --chown=user:user requirements-prod.txt .
RUN pip install --no-cache-dir --user -r requirements-prod.txt

# Copy Python application code and pretrained model checkpoints
COPY --chown=user:user app.py model.py run_production.py ./
COPY --chown=user:user models/ ./models/
COPY --chown=user:user static/ ./static/
COPY --chown=user:user templates/ ./templates/

# Copy built frontend bundle from Stage 1
COPY --chown=user:user --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:7860/api/health || exit 1

# Launch FastAPI app with uvicorn on port 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
