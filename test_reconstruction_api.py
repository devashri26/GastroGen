import app
from fastapi.testclient import TestClient

client = TestClient(app.app)
with open("data_dir/polyps/polyps/00072d5f-7cd8-434c-8a5a-1a0bb2c9711d.jpg", "rb") as f:
    img_bytes = f.read()

models = ["autoencoder", "normal_only_autoencoder", "vae", "transformer_vae"]
print("\n--- Testing /api/reconstruct with real endoscopy image ---")
for m in models:
    res = client.post("/api/reconstruct", files={"file": ("test.jpg", img_bytes, "image/jpeg")}, data={"model_name": m})
    data = res.json()
    print(f"Model: {m:<25} | Status: {res.status_code} | SSIM: {data.get('ssim'):.4f} | PSNR: {data.get('psnr'):.2f} dB | MSE: {data.get('mse'):.6f}")

print("\n--- Testing /api/screen endpoint ---")
res = client.post("/api/screen", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
data = res.json()
print(f"Screening API: Status: {res.status_code} | Anomaly: {data.get('is_anomaly')} | MSE: {data.get('mse'):.6f} | Score: {data.get('score_percentage'):.1f}%")
