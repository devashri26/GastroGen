"""
GastroGen — Generative AI for Gastrointestinal Health (v2.0)
Official Hugging Face Spaces Deployment Entrypoint

Serves the Frontend 1 Medical AI Dashboard at root (http://0.0.0.0:7860)
Supports Hugging Face Spaces (CPU Basic & ZeroGPU) and Docker containers.
"""

import os
import sys

# Ensure repository directories are in Python module search path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND1_DIR = os.path.join(ROOT_DIR, "frontend1", "frontend")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if FRONTEND1_DIR not in sys.path:
    sys.path.insert(0, FRONTEND1_DIR)

import server

# Provide Gradio demo object for Hugging Face Spaces SDK compatibility
try:
    import gradio as gr
    with gr.Blocks(title="GastroGen — Medical AI Platform") as demo:
        gr.Markdown("""
        # GastroGen Medical AI Platform
        *The full, interactive Frontend 1 Web Dashboard is running at root.*
        """)
except Exception:
    demo = None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("=" * 65)
    print(f"  [OK] GastroGen Hugging Face Spaces Server Starting on Port {port}")
    print("  --> Serving Frontend 1 Medical AI Dashboard at: http://0.0.0.0:" + str(port))
    print("=" * 65)
    sys.stdout.flush()
    server.start_server(port=port)
