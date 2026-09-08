export interface ReconstructionResult {
  original: string;
  reconstructed: string;
  ssim: number;
  psnr: number;
  mse: number;
  latency: number;
  attn_overlay?: string | null;
}

/**
 * Fetch a local public image (like /samples/polyps.png) and convert it to a File object
 */
export async function urlToFile(url: string, filename = 'sample.png'): Promise<File> {
  const response = await fetch(url);
  const blob = await response.blob();
  return new File([blob], filename, { type: blob.type || 'image/png' });
}

/**
 * Perform real GPU/PyTorch neural network inference via GastroGen backend
 */
export async function runReconstruction(
  fileOrUrl: File | string,
  modelName: 'autoencoder' | 'vae' | 'transformer_vae' | 'vit_mae' | 'sd_vae' = 'autoencoder',
  upscale = false
): Promise<ReconstructionResult> {
  const startTime = performance.now();

  let file: File;
  if (typeof fileOrUrl === 'string') {
    file = await urlToFile(fileOrUrl, 'selected_sample.png');
  } else {
    file = fileOrUrl;
  }

  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_name', modelName);
  formData.append('upscale', upscale ? 'true' : 'false');

  try {
    const res = await fetch('/api/reconstruct', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Server returned ${res.status}: ${res.statusText}`);
    }

    const data = await res.json();
    const endTime = performance.now();
    const latency = Math.round(endTime - startTime);

    return {
      original: data.original,
      reconstructed: data.reconstructed,
      ssim: Number((data.ssim || 0).toFixed(4)),
      psnr: Number((data.psnr || 0).toFixed(2)),
      mse: Number((data.mse || 0).toFixed(6)),
      latency,
      attn_overlay: data.attn_overlay || null,
    };
  } catch (error) {
    console.error('Error during reconstruction inference:', error);
    throw error;
  }
}
