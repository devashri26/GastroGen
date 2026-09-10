# =======================================================
# GastroGen — Production Dockerfile for Hugging Face Spaces & Cloud
# Serving Frontend 1 Dashboard & Real PyTorch Inference
# =======================================================

FROM python:3.10-slim
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install runtime utilities for Pillow & Matplotlib
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

# Install CPU PyTorch wheel (fast download & memory efficient for cloud instances)
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code, frontend1 dashboard, and pretrained checkpoints
COPY --chown=user:user app.py model.py ./
COPY --chown=user:user models/ ./models/
COPY --chown=user:user metrics/ ./metrics/
COPY --chown=user:user frontend1/ ./frontend1/
COPY --chown=user:user static/ ./static/

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:7860/api/health || exit 1

# Launch the unified Hugging Face production server
CMD ["python", "app.py"]
