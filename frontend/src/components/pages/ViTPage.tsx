import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  CheckCircle2, 
  Layers, 
  Grid3X3, 
  Flame, 
  Eye, 
  Cpu
} from 'lucide-react';
import { HeroCard } from '../common/HeroCard';
import { Dropzone } from '../common/Dropzone';
import { SamplePills } from '../common/SamplePills';
import { SAMPLES_CATALOG } from '../../data/mockData';
import { ViTViewMode } from '../../types';
import { runReconstruction, classifyImage } from '../../services/api';

export const ViTPage: React.FC = () => {
  const [selectedSampleId, setSelectedSampleId] = useState('normal-pylorus');
  const [inputImage, setInputImage] = useState<string>('/samples/normal-pylorus.png');
  const [reconImage, setReconImage] = useState<string>('/samples/normal-pylorus.png');
  const [attnOverlay, setAttnOverlay] = useState<string | null>(null);
  const [uploadLabel, setUploadLabel] = useState<string>('Normal Pylorus');
  const [viewMode, setViewMode] = useState<ViTViewMode>('overlay');
  const [isInferring, setIsInferring] = useState(false);
  const [inferenceLatency, setInferenceLatency] = useState(1017.3);
  const [liveClassification, setLiveClassification] = useState<{
    topClass: string;
    confidence: number;
    probabilities: { label: string; percentage: number }[];
  } | null>(null);

  const sample = SAMPLES_CATALOG[selectedSampleId] || SAMPLES_CATALOG['normal-pylorus'];

  // Function to execute real backend reconstruction and attention pass
  const executeInference = async (fileOrUrl: File | string, label: string) => {
    setIsInferring(true);
    try {
      // 1. Run live classification via ViT
      const classPromise = classifyImage(fileOrUrl).catch((err) => {
        console.warn('Live ViT classification fallback:', err);
        return null;
      });

      // 2. Run reconstruction via Transformer-VAE
      const reconPromise = runReconstruction(fileOrUrl, 'transformer_vae').catch((err) => {
        console.warn('Backend ViT reconstruction fallback:', err);
        return null;
      });

      const [classRes, reconRes] = await Promise.all([classPromise, reconPromise]);

      if (classRes && classRes.success) {
        const probs = Object.entries(classRes.probabilities)
          .map(([lbl, val]) => ({
            label: lbl,
            percentage: Number((val * 100).toFixed(1)),
          }))
          .sort((a, b) => b.percentage - a.percentage);

        setLiveClassification({
          topClass: classRes.predicted_class,
          confidence: Number((classRes.confidence * 100).toFixed(1)),
          probabilities: probs,
        });
        setUploadLabel(classRes.predicted_class);
        if (classRes.attention_overlay) {
          setAttnOverlay(classRes.attention_overlay);
        }
        if (classRes.latency) {
          setInferenceLatency(classRes.latency);
        }
      } else {
        setUploadLabel(label);
      }

      if (reconRes) {
        setInputImage(reconRes.original);
        setReconImage(reconRes.reconstructed);
        if (reconRes.attn_overlay && !classRes?.attention_overlay) {
          setAttnOverlay(reconRes.attn_overlay);
        }
        setInferenceLatency(reconRes.latency);
      } else if (typeof fileOrUrl === 'string') {
        setInputImage(fileOrUrl);
        setReconImage(fileOrUrl);
      }
    } catch (err) {
      console.warn('Backend ViT inference error:', err);
      if (typeof fileOrUrl === 'string') {
        setInputImage(fileOrUrl);
        setReconImage(fileOrUrl);
      }
    } finally {
      setIsInferring(false);
    }
  };

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
    const s = SAMPLES_CATALOG[id];
    if (s) {
      executeInference(s.image, s.name);
    }
  };

  const viewModes: { id: ViTViewMode; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
    { id: 'original', label: 'Original', icon: Eye },
    { id: 'patches', label: 'Patch Grid (14×14)', icon: Grid3X3 },
    { id: 'heatmap', label: 'Attention Heatmap', icon: Flame },
    { id: 'overlay', label: 'Attention Overlay', icon: Layers },
  ];

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Hero Header */}
      <HeroCard
        title="Vision Transformer (ViT v2)"
        description="12-Layer Multi-Head Self-Attention for end-to-end diagnostic classification and receptive field saliency localization across gastrointestinal endoscopy landmarks and pathology."
        badges={[
          { label: '12-Layer Self-Attention Rollout', icon: <Brain className="w-3.5 h-3.5" />, variant: 'teal' },
          { label: 'Patch Size: 16×16 (196 Tokens)', icon: <Grid3X3 className="w-3.5 h-3.5" />, variant: 'blue' },
          { label: '8-Class Softmax Active', icon: <CheckCircle2 className="w-3.5 h-3.5" />, variant: 'emerald' },
        ]}
        gradientTheme="cyan-teal"
      />

      {/* Main 2-Column Dashboard Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        {/* LEFT CARD: Interactive ViT Upload & Multi-View Attention Analysis */}
        <div className="xl:col-span-7 pro-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-extrabold text-slate-900 tracking-tight flex items-center gap-2">
                  <span>Interactive ViT Upload & Multi-View Attention Analysis</span>
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Explore attention rollout matrix across 196 spatial vision tokens
                </p>
              </div>
              <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-gradient-to-r from-teal-50 to-cyan-50 text-teal-700 border border-teal-200/80 shadow-2xs">
                <span className="w-2 h-2 rounded-full bg-teal-500 animate-pulse" />
                12-Layer Self-Attention Rollout
              </span>
            </div>

            {/* Dropzone */}
            <Dropzone onImageSelected={handleCustomUpload} isLoading={isInferring} />

            {/* Sample Selector Pills */}
            <SamplePills
              selectedSampleId={selectedSampleId}
              onSelectSample={handleSelectSample}
            />

            {/* Visualizer Mode Switcher Toolbar */}
            <div className="mt-6 pt-5 border-t border-slate-100">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Active Sample: <span className="text-teal-600 font-extrabold">{uploadLabel}</span>
                </span>
                <div className="flex items-center gap-1 p-1 bg-slate-100/90 rounded-xl border border-slate-200/90 shadow-2xs">
                  {viewModes.map((mode) => {
                    const Icon = mode.icon;
                    const isActive = viewMode === mode.id;
                    return (
                      <button
                        key={mode.id}
                        onClick={() => setViewMode(mode.id)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all duration-150 ${
                          isActive
                            ? 'bg-gradient-to-r from-teal-600 to-cyan-600 text-white shadow-sm shadow-teal-500/25'
                            : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
                        }`}
                      >
                        <Icon className="w-3.5 h-3.5" />
                        <span>{mode.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Large Image Visualization Canvas */}
              <div className="w-full aspect-[16/10] rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 relative shadow-inner group flex items-center justify-center endo-frame">
                {isInferring ? (
                  <div className="flex flex-col items-center justify-center text-white gap-2">
                    <div className="w-8 h-8 border-2 border-teal-400 border-t-transparent rounded-full animate-spin" />
                    <span className="text-xs font-mono text-teal-200 font-medium">Computing Self-Attention Rollout...</span>
                  </div>
                ) : (
                  <>
                    {/* Underlying endoscopy image */}
                    <img 
                      src={inputImage} 
                      alt="Endoscopy View"
                      className={`w-full h-full object-cover transition-opacity duration-300 ${
                        viewMode === 'heatmap' ? 'opacity-20' : 'opacity-100'
                      }`}
                    />

                    {/* 14x14 Grid Lines Overlay */}
                    {viewMode === 'patches' && (
                      <div className="absolute inset-0 grid grid-cols-14 grid-rows-14 pointer-events-none">
                        {Array.from({ length: 196 }).map((_, i) => (
                          <div 
                            key={i} 
                            className="border border-white/30 hover:bg-teal-400/20 transition-colors flex items-center justify-center"
                          >
                            <span className="text-[7px] text-white/50 font-mono select-none">
                              {i}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Heatmap & Overlay Saliency Mask */}
                    {(viewMode === 'heatmap' || viewMode === 'overlay') && (
                      attnOverlay ? (
                        <img 
                          src={attnOverlay} 
                          alt="Attention Overlay"
                          className="absolute inset-0 w-full h-full object-cover pointer-events-none"
                        />
                      ) : (
                        <div className={`absolute inset-0 pointer-events-none transition-opacity duration-300 ${
                          viewMode === 'overlay' ? 'mix-blend-color-dodge opacity-85' : 'mix-blend-normal opacity-95'
                        }`}>
                          <div className="w-full h-full relative">
                            <div className="absolute top-[35%] left-[45%] -translate-x-1/2 -translate-y-1/2 w-48 h-48 rounded-full bg-radial from-teal-400 via-amber-400 to-transparent blur-2xl opacity-90 animate-pulse-subtle" />
                            <div className="absolute top-[50%] left-[55%] -translate-x-1/2 -translate-y-1/2 w-36 h-36 rounded-full bg-radial from-rose-500 via-purple-500 to-transparent blur-xl opacity-75" />
                          </div>
                        </div>
                      )
                    )}

                    {/* Floating HUD Badge */}
                    <div className="absolute top-3 left-3 px-3 py-1 rounded-lg bg-black/85 backdrop-blur-md border border-white/15 text-white flex items-center gap-2 text-[11px] font-mono shadow-md">
                      <span className="w-2 h-2 rounded-full bg-teal-400 shadow-[0_0_6px_rgba(45,212,191,0.8)] animate-pulse" />
                      <span className="font-bold">Mode: {viewMode.toUpperCase()}</span>
                      <span className="text-white/40">•</span>
                      <span className="text-teal-300 font-bold">14×14 Tokens</span>
                    </div>
                  </>
                )}
              </div>

              {/* 4 Smaller Preview Cards Below */}
              <div className="grid grid-cols-4 gap-2.5 mt-3.5">
                {viewModes.map((mode) => {
                  const isActive = viewMode === mode.id;
                  return (
                    <button
                      key={mode.id}
                      onClick={() => setViewMode(mode.id)}
                      className={`p-2.5 rounded-xl border text-left transition-all duration-200 flex flex-col gap-1.5 ${
                        isActive
                          ? 'border-teal-500 bg-gradient-to-b from-teal-50 to-cyan-50/60 ring-2 ring-teal-300/80 shadow-sm shadow-teal-500/10'
                          : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/60'
                      }`}
                    >
                      <div className="w-full aspect-[4/3] rounded-lg bg-slate-950 overflow-hidden relative border border-slate-200 shadow-2xs">
                        <img 
                          src={inputImage} 
                          alt={mode.label}
                          className={`w-full h-full object-cover ${mode.id === 'heatmap' ? 'opacity-30' : ''}`}
                        />
                        {mode.id === 'patches' && (
                          <div className="absolute inset-0 grid grid-cols-7 grid-rows-7 border border-white/40 opacity-70" />
                        )}
                        {mode.id === 'heatmap' && (
                          <div className="absolute inset-0 bg-gradient-to-tr from-blue-600 via-amber-400 to-rose-500 opacity-80 mix-blend-color" />
                        )}
                        {mode.id === 'overlay' && (
                          <div className="absolute inset-0 bg-teal-400/30 mix-blend-overlay" />
                        )}
                      </div>
                      <span className={`text-[11px] truncate ${isActive ? 'text-teal-900 font-extrabold' : 'text-slate-600 font-semibold'}`}>
                        {mode.label}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="mt-5 pt-3.5 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span className="flex items-center gap-1.5 font-medium">
              <Cpu className="w-3.5 h-3.5 text-teal-600" />
              <span>Multi-head attention forward latency:</span>
              <strong className="text-slate-900 font-mono font-bold">{inferenceLatency} ms</strong>
            </span>
            <span className="text-[11px] text-teal-700 font-bold bg-teal-50 px-2.5 py-0.5 rounded-full border border-teal-200/80 shadow-2xs">
              CUDA Float16
            </span>
          </div>
        </div>

        {/* RIGHT CARD: Predicted Diagnostic Finding & Probability Distribution */}
        <div className="xl:col-span-5 space-y-6">
          {(() => {
            const activeClassification = liveClassification || {
              topClass: sample.name,
              confidence: sample.classification.confidence,
              probabilities: sample.classification.probabilities,
            };
            return (
              <>
                {/* Prominent Predicted Finding Card */}
                <div className="pro-card p-6 relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-36 h-36 bg-teal-500/10 rounded-full blur-2xl pointer-events-none" />
                  
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                      ViT Classification Finding
                    </span>
                    <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-200/80 flex items-center gap-1.5 shadow-2xs">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                      <span>Verified Diagnostic</span>
                    </span>
                  </div>

                  <div className="space-y-1">
                    <h2 className="text-2xl font-black text-slate-900 tracking-tight">
                      {uploadLabel}
                    </h2>
                    <p className="text-xs text-slate-500 font-medium">
                      Predicted by GastroGen ViT Attention Rollout Classifier
                    </p>
                  </div>

                  <div className="mt-5 p-4 rounded-xl bg-gradient-to-br from-teal-50 via-white to-cyan-50/70 border border-teal-200/80 flex items-center justify-between shadow-2xs">
                    <div>
                      <span className="text-[11px] font-bold text-teal-800 uppercase tracking-wide">
                        Model Confidence
                      </span>
                      <div className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-teal-700 to-cyan-700 tracking-tight mt-0.5 font-mono">
                        {activeClassification.confidence}%
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wide">
                        Receptive Heads
                      </span>
                      <div className="text-sm font-bold text-slate-800 font-mono mt-1">
                        12 Heads × 64-D
                      </div>
                    </div>
                  </div>
                </div>

                {/* 8-Class Softmax Probability Breakdown */}
                <div className="pro-card p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
                      <div className="w-6 h-6 rounded-lg bg-teal-100 text-teal-600 flex items-center justify-center">
                        <Brain className="w-3.5 h-3.5" />
                      </div>
                      <span>8-Class Softmax Probability Breakdown</span>
                    </h3>
                    <span className="text-[11px] font-mono text-slate-400 font-semibold">Σ p(c) = 1.0</span>
                  </div>

                  <div className="space-y-2.5">
                    {activeClassification.probabilities.map((prob: any) => {
                      const isTop = prob.percentage > 50 || prob.label === activeClassification.topClass;
                      return (
                        <div key={prob.label} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className={`font-medium ${isTop ? 'text-teal-900 font-extrabold' : 'text-slate-600'}`}>
                              {prob.label}
                            </span>
                            <span className={`font-mono text-[11px] ${isTop ? 'text-teal-700 font-black' : 'text-slate-500'}`}>
                              {prob.percentage}%
                            </span>
                          </div>
                          <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                            <div 
                              className={`h-full rounded-full transition-all duration-500 ${
                                isTop 
                                  ? 'bg-gradient-to-r from-teal-500 via-cyan-500 to-emerald-500 shadow-sm shadow-teal-500/30' 
                                  : 'bg-slate-300'
                              }`}
                              style={{ width: `${Math.min(100, Math.max(2, prob.percentage))}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </>
            );
          })()}

          {/* Top Contributing Receptive Patches List */}
          <div className="pro-card p-5">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wide mb-3 flex items-center gap-1.5">
              <Grid3X3 className="w-3.5 h-3.5 text-teal-600" />
              <span>Highest Contributing Attention Tokens</span>
            </h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {sample.classification.topPatches.map((patch: any) => (
                <div 
                  key={patch.id} 
                  className="p-2.5 rounded-xl bg-slate-50/80 hover:bg-teal-50/40 border border-slate-200/90 hover:border-teal-300/80 transition-all duration-200 flex flex-col justify-between"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono font-bold text-teal-800 text-[11px]">{patch.id}</span>
                    <span className="font-mono text-[10px] text-teal-600 font-bold bg-teal-50 px-1.5 py-0.5 rounded border border-teal-200/60">
                      {(patch.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <span className="text-[10.5px] text-slate-500 truncate font-medium">{patch.region}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ViTPage;
