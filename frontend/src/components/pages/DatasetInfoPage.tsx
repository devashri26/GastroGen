import React, { useState } from 'react';
import { 
  Database, 
  CheckCircle2, 
  Layers, 
  Sparkles, 
  Image as ImageIcon, 
  ZoomIn, 
  Activity, 
  FileCheck,
  Building2,
  X
} from 'lucide-react';
import { HeroCard } from '../common/HeroCard';
import { DATASET_CLASSES } from '../../data/mockData';

export const DatasetInfoPage: React.FC = () => {
  const [activeGalleryClass, setActiveGalleryClass] = useState('dyed-lifted-polyps');
  const [selectedPreviewImage, setSelectedPreviewImage] = useState<string | null>(null);

  // Gallery items for the 4-column endoscopy showcase
  const galleryItems = [
    { id: 1, title: 'Dyed-Lifted Polyp Frame #01', sub: 'Saline Indigo Carmine Lift', img: '/samples/dyed-lifted-polyps.png' },
    { id: 2, title: 'Pedunculated Polyp Snare #02', sub: 'Pre-resection Assessment', img: '/samples/polyps.png' },
    { id: 3, title: 'Resection Margin Margin #03', sub: 'Clean Peripheral Wound Bed', img: '/samples/dyed-resection-margins.png' },
    { id: 4, title: 'Ulcerative Colitis Mucosa #04', sub: 'Active Friable Erythema', img: '/samples/ulcerative-colitis.png' },
  ];

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Hero Header */}
      <HeroCard
        title="Kvasir-v2 Gastrointestinal Endoscopy Dataset Explorer"
        description="8,000 clinically validated endoscopic images across 8 balanced pathological and anatomical classes. Preprocessed to 256×256 RGB with zero-leakage patient-level split (80% Train, 10% Val, 10% Test)."
        badges={[
          { label: '8,000 Images', icon: <Database className="w-3.5 h-3.5" />, variant: 'purple' },
          { label: '8 Balanced Classes', icon: <Layers className="w-3.5 h-3.5" />, variant: 'teal' },
          { label: 'Preprocessed v2 (256×256)', icon: <CheckCircle2 className="w-3.5 h-3.5" />, variant: 'emerald' },
        ]}
        gradientTheme="purple-violet"
      />

      {/* Dataset Statistics Summary Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-soft">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">Total Dataset Volume</span>
          <span className="text-2xl font-black text-slate-900 font-mono">8,000</span>
          <span className="text-xs text-slate-500 font-medium block mt-0.5">High-Res Clinical Frames</span>
        </div>
        <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-soft">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">Clinical Classes</span>
          <span className="text-2xl font-black text-brand-700 font-mono">8 Balanced</span>
          <span className="text-xs text-slate-500 font-medium block mt-0.5">1,000 Images / Category</span>
        </div>
        <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-soft">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">Split Distribution</span>
          <span className="text-2xl font-black text-teal-700 font-mono">80 / 10 / 10</span>
          <span className="text-xs text-slate-500 font-medium block mt-0.5">Train / Val / Ground Truth</span>
        </div>
        <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-soft">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">Clinical Verification</span>
          <span className="text-2xl font-black text-emerald-700 font-mono">100%</span>
          <span className="text-xs text-slate-500 font-medium block mt-0.5">Expert GI Endoscopist Consensus</span>
        </div>
      </div>

      {/* 8-Card Dataset Category Grid */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-slate-900 tracking-tight">
              8 Clinically Validated Endoscopy Categories (1,000 Images Each)
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Balanced taxonomy covering upper and lower GI pathology, therapeutics, and key orientation landmarks
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md border border-slate-200">
            Kvasir-v2 Taxonomy
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {DATASET_CLASSES.map((cls) => {
            const categoryBadgeStyle = {
              Pathological: 'bg-rose-50 text-rose-700 border-rose-200/70',
              Therapeutic: 'bg-teal-50 text-teal-700 border-teal-200/70',
              Anatomical: 'bg-blue-50 text-blue-700 border-blue-200/70',
            }[cls.category];

            return (
              <div 
                key={cls.id}
                className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-soft hover:shadow-card transition-all duration-200 flex flex-col justify-between group"
              >
                <div>
                  {/* Top Thumbnail & Category Badge */}
                  <div className="w-full aspect-[16/10] rounded-xl overflow-hidden bg-slate-900 border border-slate-200 mb-3 relative group-hover:scale-[1.01] transition-transform">
                    <img 
                      src={cls.thumbnail} 
                      alt={cls.name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute top-2 left-2">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border shadow-xs ${categoryBadgeStyle}`}>
                        {cls.category}
                      </span>
                    </div>
                    <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-black/75 text-[10px] font-mono text-white backdrop-blur-sm">
                      {cls.count.toLocaleString()} imgs
                    </div>
                  </div>

                  {/* Header info */}
                  <div className="flex items-start justify-between gap-1 mb-1">
                    <h4 className="font-bold text-slate-900 text-sm group-hover:text-brand-700 transition-colors">
                      {cls.name}
                    </h4>
                    <span className="text-[10px] font-semibold text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded shrink-0">
                      {cls.location}
                    </span>
                  </div>

                  <span className="inline-block text-[11px] font-semibold text-brand-600 mb-2">
                    {cls.badgeText}
                  </span>

                  <p className="text-xs text-slate-500 leading-relaxed line-clamp-2">
                    {cls.description}
                  </p>
                </div>

                {/* Footer Clinical Relevance */}
                <div className="mt-3.5 pt-2.5 border-t border-slate-100 flex items-center justify-between text-[11px]">
                  <span className="text-slate-400 font-medium">Relevance:</span>
                  <span className="text-slate-700 font-semibold truncate max-w-[170px]">{cls.clinicalRelevance}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Representative Endoscopy Samples Gallery */}
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-soft">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5 pb-4 border-b border-slate-100">
          <div>
            <h3 className="text-base font-bold text-slate-900 tracking-tight">
              Representative Endoscopy Samples — Dyed-Lifted Polyps (1,000 images)
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              High-resolution clinical frames demonstrating mucosal vascular patterns and lesion boundaries
            </p>
          </div>
          <span className="text-xs font-semibold text-brand-600 bg-brand-50 px-3 py-1 rounded-full border border-brand-200/60">
            Kvasir-v2 Verified Samples
          </span>
        </div>

        {/* 4-Column Image Gallery */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {galleryItems.map((item) => (
            <div 
              key={item.id}
              onClick={() => setSelectedPreviewImage(item.img)}
              className="group cursor-pointer rounded-2xl border border-slate-200/80 overflow-hidden bg-slate-50/50 hover:bg-white hover:border-brand-300 hover:shadow-card transition-all duration-200 p-3"
            >
              <div className="w-full aspect-square rounded-xl overflow-hidden bg-slate-900 relative mb-3">
                <img 
                  src={item.img} 
                  alt={item.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity w-9 h-9 rounded-full bg-white/90 text-slate-900 flex items-center justify-center shadow-md">
                    <ZoomIn className="w-4 h-4" />
                  </div>
                </div>
              </div>
              <h5 className="font-bold text-xs text-slate-900 group-hover:text-brand-700 transition-colors">
                {item.title}
              </h5>
              <span className="text-[11px] text-slate-500 font-medium block mt-0.5">
                {item.sub}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Modal Zoom Preview */}
      {selectedPreviewImage && (
        <div 
          onClick={() => setSelectedPreviewImage(null)}
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-6 animate-in fade-in duration-150"
        >
          <div className="relative max-w-2xl w-full bg-white rounded-2xl overflow-hidden shadow-2xl p-4 border border-white/20" onClick={(e) => e.stopPropagation()}>
            <button 
              onClick={() => setSelectedPreviewImage(null)}
              className="absolute top-6 right-6 w-9 h-9 rounded-full bg-black/75 text-white flex items-center justify-center hover:bg-black transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
            <div className="aspect-square w-full rounded-xl overflow-hidden bg-slate-950">
              <img src={selectedPreviewImage} alt="Zoomed Endoscopy" className="w-full h-full object-contain" />
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-slate-600">
              <span className="font-semibold text-slate-800">256×256 Clinical Endoscopic Ground Truth Frame</span>
              <span className="font-mono text-brand-600">Kvasir-v2 Dataset</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
