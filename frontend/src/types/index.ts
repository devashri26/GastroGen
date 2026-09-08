export type PageType = 
  | 'autoencoder' 
  | 'vae' 
  | 'vit' 
  | 'comparison' 
  | 'about';

export type ViTViewMode = 
  | 'original' 
  | 'patches' 
  | 'heatmap' 
  | 'overlay';

export interface SampleCategory {
  id: string;
  name: string;
  type: 'Pathological' | 'Anatomical' | 'Therapeutic';
  image: string;
  reconstruction: {
    ae: string;
    vae: string;
    vit: string;
  };
  metrics: {
    ae: { ssim: number; psnr: number; mse: number; latency: number };
    vae: { ssim: number; psnr: number; mse: number; latency: number };
    vit: { ssim: number; psnr: number; mse: number; latency: number };
  };
  classification: {
    predictedClass: string;
    confidence: number;
    latency: number;
    probabilities: { label: string; percentage: number }[];
    topPatches: { id: string; score: number; region: string }[];
  };
}

export interface MetricCardData {
  title: string;
  value: string | number;
  unit?: string;
  subtext?: string;
  trend?: 'up' | 'down' | 'neutral';
  color?: 'purple' | 'teal' | 'pink' | 'blue' | 'emerald';
}

export interface LossPoint {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  reconLoss?: number;
  edgeLoss?: number;
  klLoss?: number;
  valSSIM?: number;
  valPSNR?: number;
}

export interface DatasetClassInfo {
  id: string;
  name: string;
  category: 'Pathological' | 'Therapeutic' | 'Anatomical';
  count: number;
  badgeText: string;
  description: string;
  location: string;
  clinicalRelevance: string;
  thumbnail: string;
}
