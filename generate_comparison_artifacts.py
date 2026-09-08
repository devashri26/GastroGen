import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import torchvision.transforms as transforms

from model import Autoencoder, VAE, TransformerVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

os.makedirs("./verification_output/comparisons", exist_ok=True)

def generate_visual_comparison(model_name, model, is_vae=False):
    ckpt_path = f"./models/{model_name}/{model_name}_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found for {model_name} at {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    sample_images = [
        ("polyps", "./data_dir/polyps/polyps/00072d5f-7cd8-434c-8a5a-1a0bb2c9711d.jpg"),
        ("ulcerative_colitis", "./data_dir/ulcerative-colitis/ulcerative-colitis/000eeefc-9035-4d42-92a1-2f97a3200ee0.jpg"),
        ("normal_cecum", "./data_dir/normal-cecum/normal-cecum/00b1b20b-ce2b-449c-93fc-6df854e3ab33.jpg"),
        ("esophagitis", "./data_dir/esophagitis/esophagitis/001fb927-4814-4ba5-851d-189db99291d8.jpg"),
    ]

    results = []

    for name, path in sample_images:
        if not os.path.exists(path):
            continue

        pil_img = Image.open(path).convert("RGB")
        tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            if is_vae:
                recon, _, _ = model(tensor)
            else:
                recon = model(tensor)

        recon = recon.squeeze(0).clamp(0, 1)
        recon_np = recon.permute(1, 2, 0).cpu().numpy()
        orig_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        mse = float(np.mean((orig_np - recon_np) ** 2))
        psnr = float(psnr_fn(orig_np, recon_np, data_range=1.0))
        ssim = float(ssim_fn(orig_np, recon_np, channel_axis=2, data_range=1.0))

        # Error map
        error_map = np.mean(np.abs(orig_np - recon_np), axis=2)

        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_np)
        axes[0].set_title("Original Input")
        axes[0].axis("off")

        axes[1].imshow(recon_np)
        axes[1].set_title(f"Reconstruction\nPSNR: {psnr:.2f}dB | SSIM: {ssim*100:.1f}%")
        axes[1].axis("off")

        im_err = axes[2].imshow(error_map, cmap="inferno", vmin=0, vmax=0.3)
        axes[2].set_title(f"Absolute Error Map\nMSE: {mse:.5f}")
        axes[2].axis("off")
        plt.colorbar(im_err, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_file = f"./verification_output/comparisons/{model_name}_{name}_comparison.png"
        plt.savefig(out_file, dpi=120)
        plt.close(fig)

        artifact_dir = r"C:\Users\devag\.gemini\antigravity-ide\brain\27470909-7b7f-48e2-a491-41f43db69a02\comparisons"
        os.makedirs(artifact_dir, exist_ok=True)
        import shutil
        art_path = os.path.join(artifact_dir, f"{model_name}_{name}_comparison.png")
        shutil.copy2(out_file, art_path)

        results.append({"class": name, "psnr": psnr, "ssim": ssim, "mse": mse, "saved": out_file, "artifact": art_path})

    print(f"Generated visual comparisons for {model_name} in ./verification_output/comparisons/ and artifacts")
    return results

if __name__ == "__main__":
    print("[1/4] Generating comparisons for Convolutional Autoencoder...")
    res_ae = generate_visual_comparison("autoencoder", Autoencoder(128), is_vae=False)

    print("[2/4] Generating comparisons for Normal-Only Autoencoder...")
    res_norm = generate_visual_comparison("autoencoder_normal_only", Autoencoder(128), is_vae=False)

    print("[3/4] Generating comparisons for Variational Autoencoder...")
    res_vae = generate_visual_comparison("vae", VAE(128), is_vae=True)

    print("[4/4] Generating comparisons for Transformer-VAE...")
    res_trans = generate_visual_comparison("transformer", TransformerVAE(128), is_vae=True)

    print("\nAll comparison artifacts successfully generated in ./verification_output/comparisons/")
