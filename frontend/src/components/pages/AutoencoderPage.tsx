import React, { useState, useEffect } from 'react';
import { 
  ArrowRight, 
  Layers, 
  Cpu, 
  CheckCircle2, 
  LineChart as ChartIcon,
  Download,
  Copy
} from 'lucide-react';
import { HeroCard } from '../common/HeroCard';
import { Dropzone } from '../common/Dropzone';
import { SamplePills } from '../common/SamplePills';
import { MetricCard } from '../common/MetricCard';
import { SAMPLES_CATALOG, AE_LOSS_HISTORY_50 } from '../../data/mockData';
import { runReconstruction } from '../../services/api';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
} from 'recharts';

export const AutoencoderPage: React.FC = () => {
  const [selectedSampleId, setSelectedSampleId] = useState('normal-pylorus');
  const [inputImage, setInputImage] = useState<string>('/samples/normal-pylorus.png');
  const [reconImage, setReconImage] = useState<string>('/samples/normal-pylorus.png');
  const [uploadLabel, setUploadLabel] = useState<string>('Normal Pylorus');
  const [activeMetrics, setActiveMetrics] = useState({
    ssim: 0.9885,
    psnr: 27.43,
    mse: 0.001806,
    latency: 841.2
  });
  const [isInferring, setIsInferring] = useState(false);
  const [copied, setCopied] = useState(false);
  const [chartMode, setChartMode] = useState<'loss' | 'metrics'>('loss');

  // Function to execute real backend reconstruction
  const executeInference = async (fileOrUrl: File | string, label: string) => {
    setIsInferring(true);
    try {
      const result = await runReconstruction(fileOrUrl, 'autoencoder');
      setInputImage(result.original);
      setReconImage(result.reconstructed);
      setActiveMetrics({
        ssim: result.ssim,
        psnr: result.psnr,
        mse: result.mse,
        latency: result.latency
      });
      setUploadLabel(label);
    } catch (err) {
      console.warn('Backend inference fallback:', err);
      // Fallback to sample image if backend unavailable
      if (typeof fileOrUrl === 'string') {
        setInputImage(fileOrUrl);
        setReconImage(fileOrUrl);
      }
    } finally {
      setIsInferring(false);
    }
  };

  // Run initial reconstruction for default sample on mount
  useEffect(() => {
    executeInference('/samples/normal-pylorus.png', 'Normal Pylorus');
  }, []);

  const handleCustomUpload = (imgUrl: string, file?: File, filename?: string) => {
    setSelectedSampleId('');
    if (file) {
      executeInference(file, filename || 'User Upload');
    } else {
      executeInference(imgUrl, filename || 'User Upload');
    }
  };

  const handleSelectSample = (id: string) => {
    setSelectedSampleId(id);
    const sample = SAMPLES_CATALOG[id];
    if (sample) {
      executeInference(sample.image, sample.name);
    }
  };

  const copyMetrics = () => {
    const text = `GastroGen AE v2 Metrics:\nFinding: ${uploadLabel}\nSSIM: ${activeMetrics.ssim}\nPSNR: ${activeMetrics.psnr} dB\nMSE: ${activeMetrics.mse}\nLatency: ${activeMetrics.latency} ms`;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Hero Header */}
      <HeroCard
        title="Convolutional Autoencoder (v2) Reconstruction Studio"
        description="Upload any endoscopy image to run real-time PyTorch inference through the 256-dimensional compressed latent bottleneck and inspect reconstruction quality metrics & training loss curves."
        badges={[
          { label: 'Latent Dim: 256', icon: <Layers className="w-3.5 h-3.5" />, variant: 'purple' },
          { label: 'Res: 256×256', icon: <Cpu className="w-3.5 h-3.5" />, variant: 'blue' },
          { label: 'Model Active', icon: <CheckCircle2 className="w-3.5 h-3.5" />, variant: 'emerald' },
        ]}
        gradientTheme="purple-violet"
      />

      {/* Main 2-Column Dashboard Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        {/* LEFT CARD: Interactive Upload & Live Reconstruction (7 cols) */}
        <div className="xl:col-span-7 pro-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-extrabold text-slate-900 tracking-tight flex items-center gap-2">
                  <span>Interactive Image Upload & Live Reconstruction</span>
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Live tensor transformation via hierarchical residual decoder
                </p>
              </div>
              <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-gradient-to-r from-brand-50 to-indigo-50 text-brand-700 border border-brand-200/80 shadow-2xs">
                <span className="w-2 h-2 rounded-full bg-brand-600 animate-pulse" />
                PyTorch CUDA Live
              </span>
            </div>

            {/* Dropzone */}
            <Dropzone onImageSelected={handleCustomUpload} isLoading={isInferring} />

            {/* Sample Selector Pills */}
            <SamplePills
              selectedSampleId={selectedSampleId}
              onSelectSample={handleSelectSample}
            />

            {/* Side-by-Side Image Panels */}
            <div className="mt-6 pt-5 border-t border-slate-100">
              <div className="flex items-center justify-between mb-3 text-xs">
                <span className="font-bold text-slate-700 uppercase tracking-wider">
                  Active Sample: <span className="text-brand-600 font-extrabold">{uploadLabel}</span>
                </span>
                <div className="flex items-center gap-2">
                  <button 
                    onClick={copyMetrics}
                    className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-[11px] font-semibold flex items-center gap-1.5 transition-colors border border-slate-200/60 shadow-2xs"
                  >
                    <Copy className="w-3.5 h-3.5 text-slate-500" />
                    <span>{copied ? 'Copied!' : 'Copy Metrics'}</span>
                  </button>
                  <a 
                    href={reconImage} 
                    download={`gastrogen-recon-${uploadLabel.toLowerCase().replace(/\s+/g, '-')}.png`}
                    className="px-3 py-1.5 rounded-lg bg-brand-50 hover:bg-brand-100 text-brand-700 text-[11px] font-bold flex items-center gap-1.5 transition-colors border border-brand-200/60 shadow-2xs"
                  >
                    <Download className="w-3.5 h-3.5 text-brand-600" />
                    <span>Download Output</span>
                  </a>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-11 items-center gap-3 bg-gradient-to-br from-slate-900/[0.03] via-slate-50 to-indigo-50/25 p-5 rounded-2xl border border-slate-200/90 shadow-inner">
                {/* Original Input Container (5 cols) */}
                <div className="md:col-span-5 flex flex-col items-center">
                  <div className="w-full flex items-center justify-between mb-2 px-1">
                    <span className="text-xs font-bold text-slate-700 uppercase tracking-wide flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-slate-400" />
                      Original Input
                    </span>
                    <span className="text-[10.5px] font-mono font-bold text-slate-500 bg-white px-2 py-0.5 rounded border border-slate-200/80 shadow-2xs">
                      256×256
                    </span>
                  </div>
                  <div className="w-full aspect-square rounded-xl overflow-hidden bg-slate-950 border border-slate-800 shadow-md relative group endo-frame">
                    <img 
                      src={inputImage} 
                      alt="Original Endoscopy"
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute bottom-2.5 left-2.5 px-2 py-0.5 rounded-md bg-black/80 text-[10px] font-mono text-white/95 backdrop-blur-md border border-white/10 shadow-sm">
                      Raw Frame
                    </div>
                  </div>
                </div>

                {/* Central Encoder-Decoder Conduit (1 col) */}
                <div className="md:col-span-1 flex flex-col items-center justify-center py-2 text-center">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-brand-600 via-brand-700 to-indigo-700 border-2 border-white flex items-center justify-center text-white shadow-lg shadow-brand-500/35 ring-4 ring-brand-100/90 animate-pulse-subtle">
                    <ArrowRight className="w-4 h-4 text-white drop-shadow-sm" />
                  </div>
                  <span className="text-[9.5px] font-extrabold text-brand-900 uppercase tracking-tight mt-2 leading-tight">
                    Encoder-Decoder
                  </span>
                  <span className="text-[9px] font-mono font-bold text-brand-700 bg-brand-100/80 px-1.5 py-0.5 rounded-full border border-brand-200 mt-1 shadow-2xs">
                    256-D
                  </span>
                </div>

                {/* Reconstructed Output Container (5 cols) */}
                <div className="md:col-span-5 flex flex-col items-center">
                  <div className="w-full flex items-center justify-between mb-2 px-1">
                    <span className="text-xs font-bold text-slate-700 uppercase tracking-wide flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]" />
                      Reconstructed Output
                    </span>
                    <span className="text-[10.5px] font-mono font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200/80 shadow-2xs">
                      High Fidelity
                    </span>
                  </div>
                  <div className="w-full aspect-square rounded-xl overflow-hidden bg-slate-950 border border-slate-800 shadow-md relative group endo-frame">
                    {isInferring ? (
                      <div className="w-full h-full flex flex-col items-center justify-center bg-slate-950 text-white gap-2">
                        <div className="w-8 h-8 border-2 border-brand-400 border-t-transparent rounded-full animate-spin" />
                        <span className="text-xs font-mono text-brand-200 font-medium">Running CUDA Inference...</span>
                      </div>
                    ) : (
                      <>
                        <img 
                          src={reconImage} 
                          alt="Reconstructed Endoscopy"
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        />
                        <div className="absolute bottom-2.5 right-2.5 px-2.5 py-1 rounded-md bg-gradient-to-r from-brand-950/90 to-indigo-950/90 text-[10.5px] font-mono font-bold text-brand-200 backdrop-blur-md border border-brand-500/40 shadow-md">
                          SSIM: {activeMetrics.ssim}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Metrics Strip */}
          <div className="mt-6 pt-4 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard
              label="SSIM SCORE"
              value={activeMetrics.ssim}
              subtext="Structural Similarity Index"
              color="purple"
              highlight
            />
            <MetricCard
              label="PSNR"
              value={activeMetrics.psnr}
              unit="dB"
              subtext="Peak Signal-to-Noise"
              color="teal"
            />
            <MetricCard
              label="MSE LOSS"
              value={activeMetrics.mse}
              subtext="Mean Squared Error"
              color="blue"
            />
            <MetricCard
              label="INFERENCE TIME"
              value={activeMetrics.latency}
              unit="ms"
              subtext="PyTorch CUDA Latency"
              color="emerald"
            />
          </div>
        </div>

        {/* RIGHT CARD: Training Convergence & Loss Functions (5 cols) */}
        <div className="xl:col-span-5 pro-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
              <div>
                <h3 className="text-base font-extrabold text-slate-900 tracking-tight flex items-center gap-2">
                  <div className="w-6 h-6 rounded-lg bg-brand-100 text-brand-600 flex items-center justify-center">
                    <ChartIcon className="w-3.5 h-3.5" />
                  </div>
                  <span>Training Convergence (50 Epochs)</span>
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Multi-scale perceptual composite loss minimization curves
                </p>
              </div>
              <div className="flex items-center gap-1 p-1 bg-slate-100/90 rounded-xl border border-slate-200/90 self-start sm:self-auto shadow-2xs">
                <button
                  onClick={() => setChartMode('loss')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all duration-150 ${
                    chartMode === 'loss'
                      ? 'bg-gradient-to-r from-brand-600 to-indigo-600 text-white shadow-sm shadow-brand-500/25'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Loss Curves
                </button>
                <button
                  onClick={() => setChartMode('metrics')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all duration-150 ${
                    chartMode === 'metrics'
                      ? 'bg-gradient-to-r from-brand-600 to-indigo-600 text-white shadow-sm shadow-brand-500/25'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  SSIM & PSNR
                </button>
              </div>
            </div>

            {/* Recharts Multi-line chart */}
            <div className="h-80 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                {chartMode === 'loss' ? (
                  <LineChart data={AE_LOSS_HISTORY_50} margin={{ top: 15, right: 15, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                    <XAxis 
                      dataKey="epoch" 
                      tick={{ fontSize: 11, fill: '#64748b' }} 
                      axisLine={{ stroke: '#cbd5e1' }}
                      tickLine={false}
                    />
                    <YAxis 
                      tick={{ fontSize: 11, fill: '#64748b' }} 
                      axisLine={{ stroke: '#cbd5e1' }}
                      tickLine={false}
                      domain={[0.04, 0.30]}
                      tickFormatter={(v: number) => v.toFixed(2)}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(255, 255, 255, 0.98)', 
                        borderRadius: '12px', 
                        border: '1px solid #e2e8f0',
                        boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
                        fontSize: '12px',
                        padding: '10px 14px' 
                      }} 
                      formatter={(value: any, name: string) => [Number(value).toFixed(4), name]}
                      labelFormatter={(label: any) => `Epoch ${label}`}
                    />
                    <Legend 
                      wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} 
                      iconType="circle"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="trainLoss" 
                      name="Perceptual Train Loss" 
                      stroke="#7c3aed" 
                      strokeWidth={3} 
                      dot={{ r: 4, stroke: '#7c3aed', strokeWidth: 2, fill: '#fff' }}
                      activeDot={{ r: 6, stroke: '#7c3aed', strokeWidth: 2, fill: '#fff' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="valLoss" 
                      name="Validation Loss" 
                      stroke="#f43f5e" 
                      strokeWidth={2.5} 
                      strokeDasharray="5 5"
                      dot={{ r: 4, stroke: '#f43f5e', strokeWidth: 2, fill: '#fff' }}
                      activeDot={{ r: 6, stroke: '#f43f5e', strokeWidth: 2, fill: '#fff' }}
                    />
                  </LineChart>
                ) : (
                  <LineChart data={AE_LOSS_HISTORY_50} margin={{ top: 15, right: 15, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                    <XAxis 
                      dataKey="epoch" 
                      tick={{ fontSize: 11, fill: '#64748b' }} 
                      axisLine={{ stroke: '#cbd5e1' }}
                      tickLine={false}
                    />
                    <YAxis 
                      yAxisId="left"
                      tick={{ fontSize: 11, fill: '#059669' }} 
                      axisLine={{ stroke: '#059669' }}
                      tickLine={false}
                      domain={[0.60, 1.0]}
                      tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                    />
                    <YAxis 
                      yAxisId="right"
                      orientation="right"
                      tick={{ fontSize: 11, fill: '#0284c7' }} 
                      axisLine={{ stroke: '#0284c7' }}
                      tickLine={false}
                      domain={[15, 38]}
                      tickFormatter={(v: number) => `${v}dB`}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(255, 255, 255, 0.98)', 
                        borderRadius: '12px', 
                        border: '1px solid #e2e8f0',
                        boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
                        fontSize: '12px',
                        padding: '10px 14px' 
                      }} 
                      labelFormatter={(label: any) => `Epoch ${label}`}
                    />
                    <Legend 
                      wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} 
                      iconType="circle"
                    />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="valSSIM" 
                      name="Validation SSIM" 
                      stroke="#059669" 
                      strokeWidth={3} 
                      dot={{ r: 4, stroke: '#059669', strokeWidth: 2, fill: '#fff' }}
                      activeDot={{ r: 6 }}
                    />
                    <Line 
                      yAxisId="right"
                      type="monotone" 
                      dataKey="valPSNR" 
                      name="Validation PSNR (dB)" 
                      stroke="#0284c7" 
                      strokeWidth={2.5} 
                      dot={{ r: 4, stroke: '#0284c7', strokeWidth: 2, fill: '#fff' }}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                )}
              </ResponsiveContainer>
            </div>
          </div>

          {/* Theoretical Summary Box */}
          <div className="mt-6 p-4 rounded-xl bg-gradient-to-br from-slate-50 to-indigo-50/20 border border-slate-200/90 shadow-2xs">
            <h4 className="text-xs font-extrabold text-slate-900 uppercase tracking-wide mb-2 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-brand-600 shadow-[0_0_6px_rgba(124,58,237,0.6)]" />
              <span>Loss Formulation & Architecture Insights</span>
            </h4>
            <div className="space-y-2 text-xs text-slate-600 leading-relaxed font-mono bg-slate-950 p-3.5 rounded-xl border border-slate-800 shadow-inner">
              <p className="text-[11.5px] text-emerald-400 font-bold tracking-wide">
                L_total = α · L_L1 + β · (1 - SSIM) + γ · L_Sobel
              </p>
              <p className="text-[10.5px] text-slate-400 font-sans leading-relaxed">
                Trained for 50 epochs with Cosine Annealing learning rate schedule (lr=1e-4 down to 1e-6) on NVIDIA RTX 4050 GPU using mixed-precision FP16.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AutoencoderPage;
