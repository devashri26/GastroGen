import os
import shutil
import json
import torch
import torch.nn as nn
from torchvision.utils import save_image

# ==========================================
# 1. DIRECTORY RESTRUCTURING
# ==========================================

def restructure_directories():
    print("Checking directory structure...")
    
    # Restructure Models
    nested_models_dir = "./models/models"
    if os.path.isdir(nested_models_dir):
        print(f"Found nested models directory: '{nested_models_dir}'. Moving contents...")
        for item in os.listdir(nested_models_dir):
            src = os.path.join(nested_models_dir, item)
            dst = os.path.join("./models", item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
            print(f"Moved {item} to ./models/")
        
        # Clean up empty models/models
        try:
            os.rmdir(nested_models_dir)
            print(f"Removed empty directory '{nested_models_dir}'")
        except Exception as e:
            print(f"Error removing '{nested_models_dir}': {e}")
    else:
        print("Models directory structure is clean or already restructured.")

    # Restructure Metrics
    nested_metrics_dir = "./metrics (1)/metrics"
    parent_metrics_dir = "./metrics (1)"
    if os.path.isdir(nested_metrics_dir):
        print(f"Found nested metrics directory: '{nested_metrics_dir}'. Moving contents...")
        os.makedirs("./metrics", exist_ok=True)
        for item in os.listdir(nested_metrics_dir):
            src = os.path.join(nested_metrics_dir, item)
            dst = os.path.join("./metrics", item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
            print(f"Moved {item} to ./metrics/")
        
        # Clean up metrics (1)/metrics and metrics (1)
        try:
            shutil.rmtree(parent_metrics_dir)
            print(f"Removed temporary folder '{parent_metrics_dir}'")
        except Exception as e:
            print(f"Error cleaning up '{parent_metrics_dir}': {e}")
    elif os.path.isdir(parent_metrics_dir) and not os.path.isdir("./metrics"):
        # If metrics (1) exists but not nested metrics (e.g. it was renamed)
        shutil.move(parent_metrics_dir, "./metrics")
        print(f"Renamed '{parent_metrics_dir}' to './metrics'")
    else:
        print("Metrics directory structure is clean or already restructured.")

# ==========================================
# 2. MODEL ARCHITECTURE DEFINITIONS
# ==========================================

# --- Autoencoder ---
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128 * 16 * 16, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 16 * 16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32 -> 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 64 -> 128
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 16, 16)
        return self.deconv(x)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# --- VAE ---
class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(128 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(128 * 16 * 16, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 16 * 16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 16, 16)
        return self.deconv(x)

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# --- Transformer-VAE ---
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )
        self.last_attn_weights = None

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed, need_weights=True, average_attn_weights=True)
        self.last_attn_weights = attn_weights.detach()
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=16, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.grid_size = img_size // patch_size

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerVAEEncoder(nn.Module):
    def __init__(self, num_patches=64, embed_dim=192, num_heads=6, num_layers=4, latent_dim=128):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=128, patch_size=16, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([AttentionBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

class TransformerVAEDecoder(nn.Module):
    def __init__(self, num_patches=64, embed_dim=192, num_heads=6, num_layers=4, latent_dim=128, patch_size=16, img_size=128):
        super().__init__()
        self.num_patches = num_patches
        self.grid_size = img_size // patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size

        self.fc_in = nn.Linear(latent_dim, num_patches * embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([AttentionBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.to_pixels = nn.Linear(embed_dim, patch_size * patch_size * 3)

    def forward(self, z):
        B = z.size(0)
        x = self.fc_in(z).view(B, self.num_patches, self.embed_dim)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        patches = self.to_pixels(x)

        # Fold patches back
        patches = patches.view(B, self.grid_size, self.grid_size, self.patch_size, self.patch_size, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = patches.view(B, 3, self.grid_size * self.patch_size, self.grid_size * self.patch_size)
        return torch.sigmoid(img)

class TransformerVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = TransformerVAEEncoder(latent_dim=latent_dim)
        self.decoder = TransformerVAEDecoder(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# ==========================================
# 3. VERIFICATION PROCESS
# ==========================================

def run_verification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs("./verification_output", exist_ok=True)
    
    checkpoints = {
        "Autoencoder": {
            "path": "./models/autoencoder/autoencoder_best.pth",
            "model_class": lambda: Autoencoder(latent_dim=128),
            "type": "ae"
        },
        "VAE": {
            "path": "./models/vae/vae_best.pth",
            "model_class": lambda: VAE(latent_dim=128),
            "type": "vae"
        },
        "Transformer-VAE": {
            "path": "./models/transformer/transformer_best.pth",
            "model_class": lambda: TransformerVAE(latent_dim=128),
            "type": "vae"
        },
        "Normal-Only Autoencoder": {
            "path": "./models/autoencoder_normal_only/autoencoder_normal_only_best.pth",
            "model_class": lambda: Autoencoder(latent_dim=128),
            "type": "ae"
        }
    }
    
    # Setup dummy input for testing reconstruction
    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    
    for name, config in checkpoints.items():
        path = config["path"]
        print(f"\n--- Verifying {name} ---")
        if not os.path.exists(path):
            print(f"ERROR: Checkpoint not found at: {path}")
            continue
            
        print(f"Loading checkpoint from {path}...")
        try:
            model = config["model_class"]().to(device)
            # Load weights
            checkpoint = torch.load(path, map_location=device)
            # Check if checkpoint is dict with state_dict, or state_dict directly
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                epoch = checkpoint.get("epoch", "unknown")
                val_loss = checkpoint.get("val_loss", "unknown")
                print(f"Loaded successfully from dict (epoch={epoch}, val_loss={val_loss})")
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded successfully from model_state_dict")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded state_dict directly")
                
            model.eval()
            
            # Test inference
            with torch.no_grad():
                if config["type"] == "ae":
                    recon = model(dummy_input)
                    print(f"Reconstruction test successful. Output shape: {recon.shape}")
                    save_image(recon, f"./verification_output/{name.lower().replace(' ', '_')}_recon_dummy.png")
                else: # vae
                    recon, mu, logvar = model(dummy_input)
                    print(f"Reconstruction test successful. Output shape: {recon.shape}")
                    save_image(recon, f"./verification_output/{name.lower().replace(' ', '_')}_recon_dummy.png")
                    
                    # Test random generation
                    z = torch.randn(4, 128).to(device)
                    samples = model.decoder(z)
                    print(f"Random generation test successful. Generated samples shape: {samples.shape}")
                    save_image(samples, f"./verification_output/{name.lower().replace(' ', '_')}_generated_samples.png")
                    
            print(f"[PASS] {name} verification PASSED.")
            
        except Exception as e:
            print(f"[FAIL] ERROR verifying {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    restructure_directories()
    run_verification()
    print("\nVerification complete! Outputs saved in ./verification_output/")
