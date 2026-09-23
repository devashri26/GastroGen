/**
 * GastroGen — Centralized API Client Service
 * Connects React frontend to the Hugging Face ZeroGPU FastAPI Backend
 */

const API_URL = ((import.meta as any).env?.VITE_API_URL || '').replace(/\/+$/, '');
const HF_TOKEN = ((import.meta as any).env?.VITE_HF_TOKEN || '').trim();

function getHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  if (HF_TOKEN) {
    headers['Authorization'] = `Bearer ${HF_TOKEN}`;
  }
  return headers;
}

export interface HealthResponse {
  status: string;
  service: string;
  zerogpu: boolean;
  models: {
    autoencoder: boolean;
    vae: boolean;
    transformer: boolean;
    normal_only_autoencoder: boolean;
  };
}

export interface ReconstructionMetrics {
  ssim: number;
  psnr: number;
  mse: number;
  latency: number;
}

export interface ReconstructionResponse {
  success: boolean;
  model: string;
  reconstruction: string;
  reconstructed: string;
  original: string;
  metrics: ReconstructionMetrics;
  ssim: number;
  psnr: number;
  mse: number;
  latency: number;
  attn_overlay?: string | null;
}

export interface ReconstructionResult {
  original: string;
  reconstructed: string;
  ssim: number;
  psnr: number;
  mse: number;
  latency: number;
  attn_overlay?: string | null;
}

export interface ClassificationResponse {
  success: boolean;
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  attention_overlay?: string | null;
  latency?: number;
}

export interface GenerationParams {
  model?: string;
  seed?: number;
  noise_scale?: number;
  condition?: string;
}

export interface GenerationResponse {
  success: boolean;
  model: string;
  image: string;
}

export interface AnomalyResponse {
  success: boolean;
  anomaly_score: number;
  threshold: number;
  is_anomaly: boolean;
  reconstruction: string;
  heatmap: string;
  overlay: string;
}

/**
 * Helper to fetch a static sample URL and convert it to a File object for inference
 */
export async function urlToFile(url: string, filename = 'sample.png'): Promise<File> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load preset sample from ${url}`);
  }
  const blob = await response.blob();
  return new File([blob], filename, { type: blob.type || 'image/png' });
}

async function resolveFile(fileOrUrl: File | string, defaultFilename = 'image.png'): Promise<File> {
  if (typeof fileOrUrl === 'string') {
    return await urlToFile(fileOrUrl, defaultFilename);
  }
  return fileOrUrl;
}

/**
 * Check backend status and ZeroGPU model readiness
 * GET /health
 */
export async function healthCheck(): Promise<HealthResponse> {
  const endpoint = `${API_URL}/health`;
  const res = await fetch(endpoint, {
    headers: getHeaders({ 'Accept': 'application/json' }),
  });
  if (!res.ok) {
    throw new Error(`Health probe failed: ${res.status} ${res.statusText}`);
  }
  return await res.json();
}

/**
 * Perform high-fidelity reconstruction using Autoencoder or Spatial VAE
 * POST /api/reconstruct
 */
export async function reconstructImage(
  fileOrUrl: File | string,
  modelName: 'autoencoder' | 'vae' | 'transformer_vae' | string = 'autoencoder',
  upscale = false
): Promise<ReconstructionResponse> {
  const startTime = performance.now();
  const file = await resolveFile(fileOrUrl, 'selected_sample.png');

  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', modelName);
  formData.append('model_name', modelName);
  formData.append('upscale', upscale ? 'true' : 'false');

  const endpoint = `${API_URL}/api/reconstruct`;
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: getHeaders(),
    body: formData,
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => null);
    throw new Error(errorBody?.error || `Server returned ${res.status}: ${res.statusText}`);
  }

  const data = await res.json();
  const totalLatency = Math.round(performance.now() - startTime);

  const metrics: ReconstructionMetrics = {
    ssim: Number((data.metrics?.ssim ?? data.ssim ?? 0).toFixed(4)),
    psnr: Number((data.metrics?.psnr ?? data.psnr ?? 0).toFixed(2)),
    mse: Number((data.metrics?.mse ?? data.mse ?? 0).toFixed(6)),
    latency: data.metrics?.latency ?? data.latency ?? totalLatency,
  };

  return {
    success: true,
    model: data.model || modelName,
    reconstruction: data.reconstruction || data.reconstructed,
    reconstructed: data.reconstructed || data.reconstruction,
    original: data.original,
    metrics,
    ssim: metrics.ssim,
    psnr: metrics.psnr,
    mse: metrics.mse,
    latency: metrics.latency,
    attn_overlay: data.attn_overlay || null,
  };
}

/**
 * Classify endoscopy images using Vision Transformer (12-head self-attention rollout)
 * POST /api/classify
 */
export async function classifyImage(fileOrUrl: File | string): Promise<ClassificationResponse> {
  const file = await resolveFile(fileOrUrl, 'classify_sample.png');

  const formData = new FormData();
  formData.append('file', file);

  const endpoint = `${API_URL}/api/classify`;
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: getHeaders(),
    body: formData,
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => null);
    throw new Error(errorBody?.error || `Server returned ${res.status}: ${res.statusText}`);
  }

  return await res.json();
}

/**
 * Generate synthetic endoscopy images using Spatial VAE manifold sampling
 * POST /api/generate
 */
export async function generateImage(params: GenerationParams = {}): Promise<GenerationResponse> {
  const formData = new FormData();
  formData.append('model', params.model || 'spatial_vae');
  if (params.seed !== undefined && params.seed !== null) {
    formData.append('seed', String(params.seed));
  }
  if (params.noise_scale !== undefined && params.noise_scale !== null) {
    formData.append('noise_scale', String(params.noise_scale));
  }
  if (params.condition) {
    formData.append('condition', params.condition);
  }

  const endpoint = `${API_URL}/api/generate`;
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: getHeaders(),
    body: formData,
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => null);
    throw new Error(errorBody?.error || `Server returned ${res.status}: ${res.statusText}`);
  }

  return await res.json();
}

/**
 * Anomaly screening using Normal-Only Autoencoder
 * POST /api/anomaly
 */
export async function detectAnomaly(fileOrUrl: File | string): Promise<AnomalyResponse> {
  const file = await resolveFile(fileOrUrl, 'anomaly_sample.png');

  const formData = new FormData();
  formData.append('file', file);

  const endpoint = `${API_URL}/api/anomaly`;
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: getHeaders(),
    body: formData,
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => null);
    throw new Error(errorBody?.error || `Server returned ${res.status}: ${res.statusText}`);
  }

  return await res.json();
}

/**
 * Backwards compatibility helper for existing page components
 */
export async function runReconstruction(
  fileOrUrl: File | string,
  modelName: 'autoencoder' | 'vae' | 'transformer_vae' | 'vit_mae' | 'sd_vae' = 'autoencoder',
  upscale = false
): Promise<ReconstructionResult> {
  const response = await reconstructImage(fileOrUrl, modelName, upscale);
  return {
    original: response.original,
    reconstructed: response.reconstructed,
    ssim: response.ssim,
    psnr: response.psnr,
    mse: response.mse,
    latency: response.latency,
    attn_overlay: response.attn_overlay,
  };
}
