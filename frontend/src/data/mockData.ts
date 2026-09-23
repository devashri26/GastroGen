import { SampleCategory, LossPoint, DatasetClassInfo } from '../types';

export const SAMPLES_CATALOG: Record<string, SampleCategory> = {
  'normal-pylorus': {
    id: 'normal-pylorus',
    name: 'Normal Pylorus',
    type: 'Anatomical',
    image: '/samples/normal-pylorus.png',
    reconstruction: {
      ae: '/comparisons/autoencoder_normal_cecum_comparison.png',
      vae: '/comparisons/vae_normal_cecum_comparison.png',
      vit: '/comparisons/transformer_normal_cecum_comparison.png',
    },
    metrics: {
      ae: { ssim: 0.9885, psnr: 27.43, mse: 0.001806, latency: 841.2 },
      vae: { ssim: 0.9984, psnr: 35.03, mse: 0.000314, latency: 914.5 },
      vit: { ssim: 0.9813, psnr: 34.66, mse: 0.000416, latency: 1017.3 },
    },
    classification: {
      predictedClass: 'Normal Pylorus',
      confidence: 96.6,
      latency: 1017.3,
      probabilities: [
        { label: 'Normal Pylorus', percentage: 96.6 },
        { label: 'Dyed-Lifted Polyps', percentage: 0.6 },
        { label: 'Normal Z-line', percentage: 0.5 },
        { label: 'Ulcerative Colitis', percentage: 0.5 },
        { label: 'Normal Cecum', percentage: 0.5 },
        { label: 'Esophagitis', percentage: 0.5 },
        { label: 'Polyps', percentage: 0.4 },
        { label: 'Dyed-Resection Margins', percentage: 0.4 },
      ],
      topPatches: [
        { id: 'Patch (9, 1)', score: 0.428, region: 'Pyloric Ring Aperture' },
        { id: 'Patch (7, 5)', score: 0.312, region: 'Duodenal Mucosal Ridge' },
        { id: 'Patch (11, 8)', score: 0.184, region: 'Antral Peristaltic Fold' },
        { id: 'Patch (4, 3)', score: 0.076, region: 'Gastric Mucosa Floor' },
      ]
    }
  },
  'polyps': {
    id: 'polyps',
    name: 'Polyp',
    type: 'Pathological',
    image: '/samples/polyps.png',
    reconstruction: {
      ae: '/comparisons/autoencoder_polyps_comparison.png',
      vae: '/comparisons/vae_polyps_comparison.png',
      vit: '/comparisons/transformer_polyps_comparison.png',
    },
    metrics: {
      ae: { ssim: 0.9810, psnr: 32.42, mse: 0.000573, latency: 832.0 },
      vae: { ssim: 0.9681, psnr: 28.45, mse: 0.001429, latency: 890.1 },
      vit: { ssim: 0.9845, psnr: 36.50, mse: 0.000224, latency: 998.4 },
    },
    classification: {
      predictedClass: 'Polyps (Adenomatous)',
      confidence: 98.2,
      latency: 998.4,
      probabilities: [
        { label: 'Polyps', percentage: 98.2 },
        { label: 'Dyed-Lifted Polyps', percentage: 0.8 },
        { label: 'Ulcerative Colitis', percentage: 0.3 },
        { label: 'Esophagitis', percentage: 0.2 },
        { label: 'Normal Cecum', percentage: 0.2 },
        { label: 'Normal Pylorus', percentage: 0.1 },
        { label: 'Normal Z-line', percentage: 0.1 },
        { label: 'Dyed-Resection Margins', percentage: 0.1 },
      ],
      topPatches: [
        { id: 'Patch (6, 7)', score: 0.512, region: 'Elevated Polyp Margin' },
        { id: 'Patch (8, 6)', score: 0.289, region: 'Pit Pattern Microstructure' },
        { id: 'Patch (5, 9)', score: 0.124, region: 'Surrounding Colonic Wall' },
        { id: 'Patch (10, 4)', score: 0.075, region: 'Lumen Depth' },
      ]
    }
  },
  'normal-cecum': {
    id: 'normal-cecum',
    name: 'Normal Cecum',
    type: 'Anatomical',
    image: '/samples/normal-cecum.png',
    reconstruction: {
      ae: '/comparisons/autoencoder_normal_cecum_comparison.png',
      vae: '/comparisons/vae_normal_cecum_comparison.png',
      vit: '/comparisons/transformer_normal_cecum_comparison.png',
    },
    metrics: {
      ae: { ssim: 0.9968, psnr: 32.06, mse: 0.00062, latency: 815.0 },
      vae: { ssim: 0.9984, psnr: 35.03, mse: 0.00031, latency: 885.0 },
      vit: { ssim: 0.9850, psnr: 35.10, mse: 0.00031, latency: 1005.0 },
    },
    classification: {
      predictedClass: 'Normal Cecum',
      confidence: 97.4,
      latency: 1005.0,
      probabilities: [
        { label: 'Normal Cecum', percentage: 97.4 },
        { label: 'Normal Pylorus', percentage: 1.1 },
        { label: 'Normal Z-line', percentage: 0.6 },
        { label: 'Ulcerative Colitis', percentage: 0.3 },
        { label: 'Polyps', percentage: 0.2 },
        { label: 'Esophagitis', percentage: 0.2 },
        { label: 'Dyed-Lifted Polyps', percentage: 0.1 },
        { label: 'Dyed-Resection Margins', percentage: 0.1 },
      ],
      topPatches: [
        { id: 'Patch (8, 8)', score: 0.445, region: 'Appendiceal Orifice' },
        { id: 'Patch (4, 7)', score: 0.320, region: 'Ileocecal Valve Fold' },
        { id: 'Patch (9, 3)', score: 0.145, region: 'Cecal Tri-radiate Fold' },
        { id: 'Patch (6, 5)', score: 0.090, region: 'Mucosal Vascular Sheen' },
      ]
    }
  },
  'ulcerative-colitis': {
    id: 'ulcerative-colitis',
    name: 'Ulcerative Colitis',
    type: 'Pathological',
    image: '/samples/ulcerative-colitis.png',
    reconstruction: {
      ae: '/comparisons/autoencoder_ulcerative_colitis_comparison.png',
      vae: '/comparisons/vae_ulcerative_colitis_comparison.png',
      vit: '/comparisons/transformer_ulcerative_colitis_comparison.png',
    },
    metrics: {
      ae: { ssim: 0.9825, psnr: 31.90, mse: 0.00064, latency: 835.0 },
      vae: { ssim: 0.9760, psnr: 30.50, mse: 0.00089, latency: 902.0 },
      vit: { ssim: 0.9830, psnr: 34.80, mse: 0.00033, latency: 1012.0 },
    },
    classification: {
      predictedClass: 'Ulcerative Colitis (Grade 2)',
      confidence: 95.8,
      latency: 1012.0,
      probabilities: [
        { label: 'Ulcerative Colitis', percentage: 95.8 },
        { label: 'Esophagitis', percentage: 1.8 },
        { label: 'Polyps', percentage: 0.9 },
        { label: 'Dyed-Resection Margins', percentage: 0.5 },
        { label: 'Normal Cecum', percentage: 0.4 },
        { label: 'Normal Pylorus', percentage: 0.2 },
        { label: 'Normal Z-line', percentage: 0.2 },
        { label: 'Dyed-Lifted Polyps', percentage: 0.2 },
      ],
      topPatches: [
        { id: 'Patch (7, 6)', score: 0.485, region: 'Friable Mucosal Bleeding' },
        { id: 'Patch (5, 8)', score: 0.275, region: 'Erythematous Ulceration' },
        { id: 'Patch (10, 7)', score: 0.160, region: 'Loss of Vascular Pattern' },
        { id: 'Patch (3, 4)', score: 0.080, region: 'Exudative Fibrin Base' },
      ]
    }
  },
  'esophagitis': {
    id: 'esophagitis',
    name: 'Esophagitis',
    type: 'Pathological',
    image: '/samples/esophagitis.png',
    reconstruction: {
      ae: '/comparisons/autoencoder_esophagitis_comparison.png',
      vae: '/comparisons/vae_esophagitis_comparison.png',
      vit: '/comparisons/transformer_esophagitis_comparison.png',
    },
    metrics: {
      ae: { ssim: 0.9790, psnr: 31.40, mse: 0.00072, latency: 828.0 },
      vae: { ssim: 0.9710, psnr: 29.80, mse: 0.00104, latency: 895.0 },
      vit: { ssim: 0.9805, psnr: 34.20, mse: 0.00038, latency: 1008.0 },
    },
    classification: {
      predictedClass: 'Reflux Esophagitis (LA Class B)',
      confidence: 94.9,
      latency: 1008.0,
      probabilities: [
        { label: 'Esophagitis', percentage: 94.9 },
        { label: 'Normal Z-line', percentage: 2.3 },
        { label: 'Ulcerative Colitis', percentage: 1.2 },
        { label: 'Polyps', percentage: 0.6 },
        { label: 'Dyed-Lifted Polyps', percentage: 0.4 },
        { label: 'Normal Pylorus', percentage: 0.2 },
        { label: 'Normal Cecum', percentage: 0.2 },
        { label: 'Dyed-Resection Margins', percentage: 0.2 },
      ],
      topPatches: [
        { id: 'Patch (8, 4)', score: 0.490, region: 'Longitudinal Mucosal Break' },
        { id: 'Patch (6, 5)', score: 0.280, region: 'Squamocolumnar Junction' },
        { id: 'Patch (9, 6)', score: 0.145, region: 'Erythematous Streak' },
        { id: 'Patch (4, 8)', score: 0.085, region: 'Esophageal Mucosa' },
      ]
    }
  },
  'normal-z-line': {
    id: 'normal-z-line',
    name: 'Normal Z-line',
    type: 'Anatomical',
    image: '/samples/normal-z-line.png',
    reconstruction: {
      ae: '/comparisons/autoencoder_normal_only_normal_cecum_comparison.png',
      vae: '/comparisons/vae_normal_cecum_comparison.png',
      vit: '/comparisons/transformer_normal_cecum_comparison.png',
    },
    metrics: {
      ae: { ssim: 0.9850, psnr: 33.10, mse: 0.00049, latency: 820.0 },
      vae: { ssim: 0.9780, psnr: 32.40, mse: 0.00057, latency: 888.0 },
      vit: { ssim: 0.9820, psnr: 35.00, mse: 0.00032, latency: 1010.0 },
    },
    classification: {
      predictedClass: 'Normal Z-line (Gastroesophageal)',
      confidence: 96.1,
      latency: 1010.0,
      probabilities: [
        { label: 'Normal Z-line', percentage: 96.1 },
        { label: 'Esophagitis', percentage: 1.9 },
        { label: 'Normal Pylorus', percentage: 0.8 },
        { label: 'Normal Cecum', percentage: 0.4 },
        { label: 'Polyps', percentage: 0.3 },
        { label: 'Ulcerative Colitis', percentage: 0.2 },
        { label: 'Dyed-Lifted Polyps', percentage: 0.2 },
        { label: 'Dyed-Resection Margins', percentage: 0.1 },
      ],
      topPatches: [
        { id: 'Patch (7, 7)', score: 0.460, region: 'Squamo-columnar Demarcation' },
        { id: 'Patch (9, 5)', score: 0.310, region: 'Gastric Cardia Folds' },
        { id: 'Patch (5, 6)', score: 0.150, region: 'Tubular Esophageal Wall' },
        { id: 'Patch (3, 8)', score: 0.080, region: 'Diaphragmatic Pinch' },
      ]
    }
  },
  'dyed-lifted-polyps': {
    id: 'dyed-lifted-polyps',
    name: 'Dyed-Lifted Polyps',
    type: 'Therapeutic',
    image: '/samples/dyed-lifted-polyps.png',
    reconstruction: {
      ae: '/comparisons/autoencoder_polyps_comparison.png',
      vae: '/comparisons/vae_polyps_comparison.png',
      vit: '/comparisons/transformer_polyps_comparison.png',
    },
    metrics: {
      ae: { ssim: 0.9760, psnr: 30.80, mse: 0.00083, latency: 845.0 },
      vae: { ssim: 0.9690, psnr: 29.40, mse: 0.00115, latency: 910.0 },
      vit: { ssim: 0.9810, psnr: 34.50, mse: 0.00035, latency: 1020.0 },
    },
    classification: {
      predictedClass: 'Dyed-Lifted Polyps (EMR Saline Lift)',
      confidence: 97.8,
      latency: 1020.0,
      probabilities: [
        { label: 'Dyed-Lifted Polyps', percentage: 97.8 },
        { label: 'Dyed-Resection Margins', percentage: 1.1 },
        { label: 'Polyps', percentage: 0.5 },
        { label: 'Ulcerative Colitis', percentage: 0.2 },
        { label: 'Esophagitis', percentage: 0.1 },
        { label: 'Normal Cecum', percentage: 0.1 },
        { label: 'Normal Pylorus', percentage: 0.1 },
        { label: 'Normal Z-line', percentage: 0.1 },
      ],
      topPatches: [
        { id: 'Patch (7, 6)', score: 0.520, region: 'Indigo Carmine Submucosal Cushion' },
        { id: 'Patch (8, 8)', score: 0.260, region: 'Elevated Polyp Apex' },
        { id: 'Patch (4, 5)', score: 0.140, region: 'Submucosal Fluid Margin' },
        { id: 'Patch (10, 6)', score: 0.080, region: 'Surrounding Colonic Lumen' },
      ]
    }
  },
  'dyed-resection-margins': {
    id: 'dyed-resection-margins',
    name: 'Dyed Resection Margins',
    type: 'Therapeutic',
    image: '/samples/dyed-resection-margins.png',
    reconstruction: {
      ae: '/comparisons/autoencoder_polyps_comparison.png',
      vae: '/comparisons/vae_polyps_comparison.png',
      vit: '/comparisons/transformer_polyps_comparison.png',
    },
    metrics: {
      ae: { ssim: 0.9800, psnr: 31.50, mse: 0.00071, latency: 840.0 },
      vae: { ssim: 0.9720, psnr: 30.10, mse: 0.00098, latency: 905.0 },
      vit: { ssim: 0.9820, psnr: 34.70, mse: 0.00034, latency: 1015.0 },
    },
    classification: {
      predictedClass: 'Dyed Resection Margins (Post-Polypectomy)',
      confidence: 96.9,
      latency: 1015.0,
      probabilities: [
        { label: 'Dyed-Resection Margins', percentage: 96.9 },
        { label: 'Dyed-Lifted Polyps', percentage: 1.8 },
        { label: 'Ulcerative Colitis', percentage: 0.4 },
        { label: 'Polyps', percentage: 0.3 },
        { label: 'Esophagitis', percentage: 0.2 },
        { label: 'Normal Cecum', percentage: 0.2 },
        { label: 'Normal Pylorus', percentage: 0.1 },
        { label: 'Normal Z-line', percentage: 0.1 },
      ],
      topPatches: [
        { id: 'Patch (6, 6)', score: 0.470, region: 'Post-Polypectomy Ulcer Bed' },
        { id: 'Patch (7, 4)', score: 0.310, region: 'Peripheral Dyed Margin' },
        { id: 'Patch (9, 7)', score: 0.150, region: 'Coagulated Thermal Vessel' },
        { id: 'Patch (4, 8)', score: 0.070, region: 'Colonic Muscularis Layer' },
      ]
    }
  }
};

export const SAMPLE_KEYS = [
  'polyps',
  'normal-cecum',
  'ulcerative-colitis',
  'esophagitis',
  'normal-pylorus',
  'normal-z-line',
  'dyed-lifted-polyps',
  'dyed-resection-margins',
];

// 50 Epoch Loss History for Autoencoder v2
export const AE_LOSS_HISTORY_50: LossPoint[] = [
  { epoch: 1, trainLoss: 0.285, valLoss: 0.264, reconLoss: 0.285, valSSIM: 0.651, valPSNR: 19.8 },
  { epoch: 5, trainLoss: 0.182, valLoss: 0.168, reconLoss: 0.182, valSSIM: 0.764, valPSNR: 22.4 },
  { epoch: 10, trainLoss: 0.124, valLoss: 0.119, reconLoss: 0.124, valSSIM: 0.832, valPSNR: 24.1 },
  { epoch: 15, trainLoss: 0.098, valLoss: 0.095, reconLoss: 0.098, valSSIM: 0.887, valPSNR: 25.8 },
  { epoch: 20, trainLoss: 0.084, valLoss: 0.086, reconLoss: 0.084, valSSIM: 0.918, valPSNR: 26.9 },
  { epoch: 25, trainLoss: 0.076, valLoss: 0.081, reconLoss: 0.076, valSSIM: 0.941, valPSNR: 28.3 },
  { epoch: 30, trainLoss: 0.069, valLoss: 0.079, reconLoss: 0.069, valSSIM: 0.958, valPSNR: 29.5 },
  { epoch: 35, trainLoss: 0.063, valLoss: 0.078, reconLoss: 0.063, valSSIM: 0.969, valPSNR: 30.8 },
  { epoch: 40, trainLoss: 0.059, valLoss: 0.0774, reconLoss: 0.059, valSSIM: 0.976, valPSNR: 32.1 },
  { epoch: 45, trainLoss: 0.056, valLoss: 0.0772, reconLoss: 0.056, valSSIM: 0.980, valPSNR: 33.4 },
  { epoch: 50, trainLoss: 0.054, valLoss: 0.0770, reconLoss: 0.054, valSSIM: 0.9885, valPSNR: 34.14 },
];

// 50 Epoch Loss History for VAE v2 (Composite Recon + KL + Sobel Edge)
export const VAE_LOSS_HISTORY_50: LossPoint[] = [
  { epoch: 1, trainLoss: 0.342, valLoss: 0.315, reconLoss: 0.280, edgeLoss: 0.048, klLoss: 0.014, valSSIM: 0.612, valPSNR: 18.2 },
  { epoch: 5, trainLoss: 0.210, valLoss: 0.198, reconLoss: 0.170, edgeLoss: 0.031, klLoss: 0.009, valSSIM: 0.742, valPSNR: 21.6 },
  { epoch: 10, trainLoss: 0.145, valLoss: 0.138, reconLoss: 0.118, edgeLoss: 0.021, klLoss: 0.006, valSSIM: 0.814, valPSNR: 23.9 },
  { epoch: 15, trainLoss: 0.112, valLoss: 0.109, reconLoss: 0.092, edgeLoss: 0.016, klLoss: 0.004, valSSIM: 0.865, valPSNR: 25.4 },
  { epoch: 20, trainLoss: 0.092, valLoss: 0.094, reconLoss: 0.076, edgeLoss: 0.013, klLoss: 0.003, valSSIM: 0.902, valPSNR: 26.8 },
  { epoch: 25, trainLoss: 0.081, valLoss: 0.087, reconLoss: 0.067, edgeLoss: 0.011, klLoss: 0.003, valSSIM: 0.932, valPSNR: 28.1 },
  { epoch: 30, trainLoss: 0.072, valLoss: 0.083, reconLoss: 0.060, edgeLoss: 0.009, klLoss: 0.003, valSSIM: 0.951, valPSNR: 29.4 },
  { epoch: 35, trainLoss: 0.065, valLoss: 0.081, reconLoss: 0.054, edgeLoss: 0.008, klLoss: 0.003, valSSIM: 0.963, valPSNR: 30.6 },
  { epoch: 40, trainLoss: 0.060, valLoss: 0.079, reconLoss: 0.050, edgeLoss: 0.007, klLoss: 0.003, valSSIM: 0.970, valPSNR: 31.8 },
  { epoch: 45, trainLoss: 0.057, valLoss: 0.078, reconLoss: 0.048, edgeLoss: 0.006, klLoss: 0.003, valSSIM: 0.973, valPSNR: 32.8 },
  { epoch: 50, trainLoss: 0.055, valLoss: 0.0776, reconLoss: 0.046, edgeLoss: 0.006, klLoss: 0.003, valSSIM: 0.9745, valPSNR: 33.2 },
];

// 8 Dataset Category Cards for Dataset Info Page
export const DATASET_CLASSES: DatasetClassInfo[] = [
  {
    id: 'polyps',
    name: 'Polyps',
    category: 'Pathological',
    count: 1000,
    badgeText: 'High Clinical Priority',
    description: 'Elevated, pedunculated, or sessile precancerous mucosal neoplasms requiring endoscopic mucosal resection (EMR).',
    location: 'Colorectum',
    clinicalRelevance: 'Colorectal Cancer Screening & Prevention',
    thumbnail: '/samples/polyps.png',
  },
  {
    id: 'ulcerative-colitis',
    name: 'Ulcerative Colitis',
    category: 'Pathological',
    count: 1000,
    badgeText: 'Inflammatory',
    description: 'Chronic inflammatory bowel disease characterized by diffuse friability, mucosal erythema, exudate, and loss of vascular pattern.',
    location: 'Colon & Rectum',
    clinicalRelevance: 'Mayo Endoscopic Scoring & Disease Activity',
    thumbnail: '/samples/ulcerative-colitis.png',
  },
  {
    id: 'esophagitis',
    name: 'Esophagitis',
    category: 'Pathological',
    count: 1000,
    badgeText: 'Upper GI',
    description: 'Mucosal breaks and linear ulcerations in the distal esophagus caused by gastroesophageal reflux disease (GERD).',
    location: 'Lower Esophagus',
    clinicalRelevance: 'Los Angeles (LA) Classification Grade A-D',
    thumbnail: '/samples/esophagitis.png',
  },
  {
    id: 'dyed-lifted-polyps',
    name: 'Dyed-Lifted Polyps',
    category: 'Therapeutic',
    count: 1000,
    badgeText: 'Interventional',
    description: 'Polyps elevated via submucosal injection of saline tinted with indigo carmine or methylene blue dye prior to snare resection.',
    location: 'Colorectal Submucosa',
    clinicalRelevance: 'Complete En-Bloc Resection Assessment',
    thumbnail: '/samples/dyed-lifted-polyps.png',
  },
  {
    id: 'dyed-resection-margins',
    name: 'Dyed-Resection Margins',
    category: 'Therapeutic',
    count: 1000,
    badgeText: 'Post-Procedure',
    description: 'Post-polypectomy mucosal defect borders visualized with chromoendoscopy to ensure clean negative margins (R0 resection).',
    location: 'Post-EMR/ESD Wound',
    clinicalRelevance: 'Tumor Residual & Recurrence Surveillance',
    thumbnail: '/samples/dyed-resection-margins.png',
  },
  {
    id: 'normal-cecum',
    name: 'Normal Cecum',
    category: 'Anatomical',
    count: 1000,
    badgeText: 'Anatomical Landmark',
    description: 'The proximal pouch of the large intestine showcasing the appendiceal orifice and the tri-radiate mucosal fold pattern.',
    location: 'Proximal Right Colon',
    clinicalRelevance: 'Complete Colonoscopy Verification Target',
    thumbnail: '/samples/normal-cecum.png',
  },
  {
    id: 'normal-pylorus',
    name: 'Normal Pylorus',
    category: 'Anatomical',
    count: 1000,
    badgeText: 'Upper GI Valve',
    description: 'The smooth muscular sphincter valve that regulates gastric emptying into the duodenal bulb with healthy mucosal ring tone.',
    location: 'Gastric Outlet / Duodenum',
    clinicalRelevance: 'Upper Endoscopy Orientation & Patency',
    thumbnail: '/samples/normal-pylorus.png',
  },
  {
    id: 'normal-z-line',
    name: 'Normal Z-Line',
    category: 'Anatomical',
    count: 1000,
    badgeText: 'GE Junction',
    description: 'The sharp, regular squamocolumnar mucosal demarcation separating pale stratified squamous and pink columnar epithelium.',
    location: 'Gastroesophageal Junction',
    clinicalRelevance: 'Barrett’s Esophagus & Hiatal Hernia Screening',
    thumbnail: '/samples/normal-z-line.png',
  },
];
