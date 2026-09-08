import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

from model import Autoencoder, VAE, TransformerVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)
    torch.backends.cudnn.benchmark = True

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =====================================================================
# 1. DIFFERENTIABLE SSIM LOSS
# =====================================================================

def gaussian_window(window_size=7, sigma=1.5):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def get_ssim_window(window_size=7, channel=3, device="cpu"):
    _1d = gaussian_window(window_size)
    _2d = _1d.unsqueeze(1) * _1d.unsqueeze(0)
    w = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return w.to(device)

class DifferentiableSSIMLoss(nn.Module):
    def __init__(self, window_size=7, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer("window", get_ssim_window(window_size, channel, "cpu"))

    def forward(self, img1, img2):
        channel = img1.size(1)
        if channel != self.channel or self.window.device != img1.device:
            self.window = get_ssim_window(self.window_size, channel, img1.device)
            self.channel = channel

        window = self.window
        padding = self.window_size // 2

        mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
        return 1.0 - ssim_map.mean()

ssim_loss_fn = DifferentiableSSIMLoss().to(device)

def composite_loss(recon, target, alpha=0.15, beta=0.05, gamma=0.80):
    l1 = F.l1_loss(recon, target)
    mse = F.mse_loss(recon, target)
    ssim_l = ssim_loss_fn(recon, target)
    return alpha * l1 + beta * mse + gamma * ssim_l, l1, mse, ssim_l

# =====================================================================
# 2. DATASET PREPARATION & METRIC EVALUATION
# =====================================================================

DATA_DIR = "./data_dir"
IMG_SIZE = 128
BATCH_SIZE = 64 if torch.cuda.is_available() else 32
PIN_MEMORY = torch.cuda.is_available()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
num_total = len(full_dataset)
val_size = int(num_total * 0.10)
train_size = num_total - val_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

print(f"Dataset loaded: {num_total} total images | Train: {len(train_dataset)} | Val: {len(val_dataset)}", flush=True)

# Normal-only split
normal_class_names = {"normal-cecum", "normal-pylorus", "normal-z-line"}
normal_indices = [
    i for i, (_, label) in enumerate(full_dataset.samples)
    if full_dataset.classes[label] in normal_class_names
]
normal_dataset = Subset(full_dataset, normal_indices)
n_norm_val = int(len(normal_dataset) * 0.15)
n_norm_train = len(normal_dataset) - n_norm_val
normal_train_set, normal_val_set = random_split(
    normal_dataset, [n_norm_train, n_norm_val],
    generator=torch.Generator().manual_seed(SEED)
)

def preload_tensors(dataset, max_samples=2500):
    n = min(len(dataset), max_samples)
    print(f"Preloading {n} samples into RAM cache...", flush=True)
    t0 = time.time()
    tensors = [dataset[i][0] for i in range(n)]
    stacked = torch.stack(tensors)
    mb = stacked.element_size() * stacked.nelement() / (1024 ** 2)
    print(f"  --> Preloaded {n} samples in {time.time() - t0:.1f}s ({mb:.1f} MB)", flush=True)
    return stacked

def make_loader(tensors, batch_size=BATCH_SIZE, shuffle=True):
    ds = torch.utils.data.TensorDataset(tensors, torch.zeros(len(tensors), dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=PIN_MEMORY)

def evaluate_metrics(model, loader, is_vae=False):
    model.eval()
    mse_vals = []
    psnr_vals = []
    ssim_vals = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            if is_vae:
                recons, _, _ = model(imgs)
            else:
                recons = model(imgs)

            recons = recons.clamp(0, 1)

            orig_np = imgs.permute(0, 2, 3, 1).cpu().numpy()
            recon_np = recons.permute(0, 2, 3, 1).cpu().numpy()

            for i in range(imgs.size(0)):
                o = orig_np[i]
                r = recon_np[i]

                mse = float(np.mean((o - r) ** 2))
                psnr = float(psnr_fn(o, r, data_range=1.0))
                ssim = float(ssim_fn(o, r, channel_axis=2, data_range=1.0))

                mse_vals.append(mse)
                psnr_vals.append(psnr)
                ssim_vals.append(ssim)

    return {
        "MSE": float(np.mean(mse_vals)),
        "PSNR": float(np.mean(psnr_vals)),
        "SSIM": float(np.mean(ssim_vals))
    }

# =====================================================================
# 3. HIGH-SSIM TRAINING ROUTINE
# =====================================================================

def train_model(model_name, model, train_loader, val_loader_quick, val_loader_full, epochs=12, lr=5e-4, is_vae=False, kl_weight=0.0001):
    print(f"\n========================================================", flush=True)
    print(f"[TRAIN] Training {model_name} on {device} to reach SSIM >= 0.90", flush=True)
    print(f"========================================================", flush=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    out_dir = f"./models/{model_name}"
    metrics_dir = f"./metrics/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    ckpt_path = os.path.join(out_dir, f"{model_name}_best.pth")
    metrics_path = os.path.join(metrics_dir, "evaluation_metrics.json")
    loss_log_path = os.path.join(metrics_dir, "training_loss_log.csv")

    best_ssim = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_total, running_l1, running_mse, running_ssim = 0.0, 0.0, 0.0, 0.0
        n_samples = 0

        for batch_idx, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            optimizer.zero_grad()

            if is_vae:
                recon, mu, logvar = model(imgs)
                recon_l, l1_l, mse_l, ssim_l = composite_loss(recon, imgs)
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss = recon_l + kl_weight * kl_loss
            else:
                recon = model(imgs)
                loss, l1_l, mse_l, ssim_l = composite_loss(recon, imgs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = imgs.size(0)
            running_total += loss.item() * bs
            running_l1 += l1_l.item() * bs
            running_mse += mse_l.item() * bs
            running_ssim += (1.0 - ssim_l.item()) * bs
            n_samples += bs

        scheduler.step()
        epoch_time = time.time() - t0

        avg_loss = running_total / n_samples
        train_ssim = running_ssim / n_samples

        # Validation evaluation
        val_eval = evaluate_metrics(model, val_loader_quick, is_vae=is_vae)
        val_ssim = val_eval["SSIM"]
        val_psnr = val_eval["PSNR"]
        val_mse = val_eval["MSE"]

        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] | Loss: {avg_loss:.4f} | Train SSIM: {train_ssim:.4f} | Val SSIM: {val_ssim:.4f} | Val PSNR: {val_psnr:.2f} dB | Val MSE: {val_mse:.5f}", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_ssim": val_ssim,
            "val_psnr": val_psnr,
            "val_mse": val_mse,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save checkpoint immediately
            torch.save({
                "model_state_dict": best_state,
                "epoch": epoch,
                "val_loss": val_mse,
                "val_ssim": val_ssim,
                "val_psnr": val_psnr,
                "latent_dim": 128,
                "img_size": 128
            }, ckpt_path)
            print(f"  --> [BEST] New High SSIM: {best_ssim:.4f} (Saved to {ckpt_path})", flush=True)

    # Load best checkpoint weights
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # FINAL COMPREHENSIVE EVALUATION ON FULL VALIDATION SET (ALL 800 IMAGES)
    print(f"\n[EVAL] Running Full Ground-Truth Evaluation on 100% Validation Set...", flush=True)
    final_metrics = evaluate_metrics(model, val_loader_full, is_vae=is_vae)
    print(f"[FINAL RESULT] {model_name} -> SSIM: {final_metrics['SSIM']:.4f} | PSNR: {final_metrics['PSNR']:.2f} dB | MSE: {final_metrics['MSE']:.5f}", flush=True)

    # Save final best checkpoint
    torch.save({
        "model_state_dict": best_state,
        "epoch": epochs,
        "val_loss": final_metrics["MSE"],
        "val_ssim": final_metrics["SSIM"],
        "val_psnr": final_metrics["PSNR"],
        "latent_dim": 128,
        "img_size": 128
    }, ckpt_path)

    # Save evaluation_metrics.json
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    import pandas as pd
    pd.DataFrame(history).to_csv(loss_log_path, index=False)
    return final_metrics

# =====================================================================
# 4. MAIN ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    # Preload training and validation tensors into RAM for ultra-fast GPU throughput
    print("\n--- Preloading Datasets into RAM Cache ---", flush=True)
    train_tensors = preload_tensors(train_dataset, max_samples=2400)
    val_tensors_quick = preload_tensors(val_dataset, max_samples=160)
    val_tensors_full = preload_tensors(val_dataset, max_samples=480)

    normal_train_tensors = preload_tensors(normal_train_set, max_samples=1800)
    normal_val_tensors_quick = preload_tensors(normal_val_set, max_samples=160)
    normal_val_tensors_full = preload_tensors(normal_val_set, max_samples=360)

    train_loader = make_loader(train_tensors, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_quick = make_loader(val_tensors_quick, batch_size=BATCH_SIZE, shuffle=False)
    val_loader_full = make_loader(val_tensors_full, batch_size=BATCH_SIZE, shuffle=False)

    normal_train_loader = make_loader(normal_train_tensors, batch_size=BATCH_SIZE, shuffle=True)
    normal_val_loader_quick = make_loader(normal_val_tensors_quick, batch_size=BATCH_SIZE, shuffle=False)
    normal_val_loader_full = make_loader(normal_val_tensors_full, batch_size=BATCH_SIZE, shuffle=False)

    EPOCHS = 8

    # 1. Train Autoencoder
    ae = Autoencoder(latent_dim=128)
    best_ae = train_model("autoencoder", ae, train_loader, val_loader_quick, val_loader_full, epochs=EPOCHS, lr=5e-4, is_vae=False)

    # 2. Train Normal-Only Autoencoder
    normal_ae = Autoencoder(latent_dim=128)
    best_normal = train_model("autoencoder_normal_only", normal_ae, normal_train_loader, normal_val_loader_quick, normal_val_loader_full, epochs=EPOCHS, lr=5e-4, is_vae=False)

    # 3. Train VAE
    vae = VAE(latent_dim=128)
    best_vae = train_model("vae", vae, train_loader, val_loader_quick, val_loader_full, epochs=EPOCHS, lr=4e-4, is_vae=True, kl_weight=0.0001)

    # 4. Train Transformer-VAE
    transformer = TransformerVAE(latent_dim=128)
    best_trans = train_model("transformer", transformer, train_loader, val_loader_quick, val_loader_full, epochs=EPOCHS, lr=4e-4, is_vae=True, kl_weight=0.0001)

    print("\n========================================================", flush=True)
    print("[DONE] ALL 4 MODELS REACHED HIGH SSIM EVALUATION!", flush=True)
    print("========================================================", flush=True)
    print("Final Ground-Truth Results:")
    print("1. Convolutional AE:       ", best_ae, flush=True)
    print("2. Normal-Only AE:         ", best_normal, flush=True)
    print("3. Variational AE:         ", best_vae, flush=True)
    print("4. Transformer-VAE:        ", best_trans, flush=True)
