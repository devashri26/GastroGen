import React from 'react';
import { 
  Layers, 
  Share2, 
  Brain, 
  BarChart3, 
  Info, 
  Sparkles,
  HeartPulse
} from 'lucide-react';
import { PageType } from '../../types';

interface SidebarProps {
  currentPage: PageType;
  onSelectPage: (page: PageType) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentPage, onSelectPage }) => {
  const navItems: { id: PageType; label: string; sublabel: string; icon: React.ComponentType<{ className?: string }> }[] = [
    { id: 'autoencoder', label: 'Autoencoder', sublabel: 'AE v2', icon: Layers },
    { id: 'vae', label: 'Variational AE', sublabel: 'VAE v2', icon: Share2 },
    { id: 'vit', label: 'Vision Transformer', sublabel: 'ViT v2', icon: Brain },
    { id: 'comparison', label: 'Model Comparison', sublabel: 'Benchmark', icon: BarChart3 },
    { id: 'about', label: 'About', sublabel: 'Architecture', icon: Info },
  ];

  return (
    <aside className="w-[260px] h-screen fixed left-0 top-0 bg-white/95 backdrop-blur-xl border-r border-slate-200/90 flex flex-col justify-between z-30 select-none shadow-[2px_0_16px_rgba(15,23,42,0.03)]">
      {/* Top Branding */}
      <div>
        <div className="h-[75px] px-6 border-b border-slate-100/90 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 via-brand-700 to-indigo-700 flex items-center justify-center text-white shadow-md shadow-brand-600/30 ring-2 ring-brand-100/80">
            <HeartPulse className="w-5 h-5 text-white" />
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-1.5">
              <span className="font-extrabold text-lg text-slate-900 tracking-tight leading-none">GastroGen</span>
              <span className="px-1.5 py-0.5 text-[10px] font-bold bg-gradient-to-r from-brand-100 to-indigo-100 text-brand-700 rounded-full border border-brand-200/60">v2.0</span>
            </div>
            <span className="text-[11px] font-medium text-slate-500 mt-1 leading-tight line-clamp-1">
              Generative AI for GI Health
            </span>
          </div>
        </div>

        {/* Navigation Items */}
        <div className="p-3.5 space-y-1">
          <div className="px-3 pt-2 pb-1.5 text-[10.5px] font-bold text-slate-400 uppercase tracking-wider">
            Models & Benchmark
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentPage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectPage(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 group relative ${
                  isActive
                    ? 'bg-gradient-to-r from-brand-50/90 via-indigo-50/50 to-white text-brand-700 shadow-xs border border-brand-200/90 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50/80 border border-transparent'
                }`}
              >
                {isActive && (
                  <span className="absolute left-0 top-2 bottom-2 w-1 bg-gradient-to-b from-brand-500 to-indigo-600 rounded-r-full shadow-[0_0_8px_rgba(124,58,237,0.5)]" />
                )}
                <div className="flex items-center gap-3">
                  <div className={`p-1.5 rounded-lg transition-colors ${
                    isActive ? 'bg-brand-100/80 text-brand-600 shadow-2xs' : 'text-slate-400 group-hover:text-brand-500 group-hover:bg-slate-100'
                  }`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="text-[13.5px]">{item.label}</span>
                </div>
                <span className={`text-[11px] px-2 py-0.5 rounded-md font-mono ${
                  isActive ? 'bg-brand-100/90 text-brand-800 font-bold border border-brand-200/60' : 'text-slate-400 bg-slate-100 group-hover:bg-slate-200/60'
                }`}>
                  {item.sublabel}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Bottom AI Branding Box */}
      <div className="p-4 border-t border-slate-100/90 bg-gradient-to-b from-white to-slate-50/70">
        <div className="p-4 rounded-2xl bg-gradient-to-br from-[#0f172a] via-[#1e1b4b] to-[#2e1065] text-white relative overflow-hidden shadow-lg shadow-indigo-950/20 border border-white/10">
          <div className="absolute top-0 right-0 w-28 h-28 bg-brand-500/20 rounded-full blur-2xl pointer-events-none" />
          <div className="flex items-center gap-2 mb-2">
            <div className="w-6 h-6 rounded-full bg-brand-500/30 flex items-center justify-center border border-brand-400/40 shadow-inner">
              <Sparkles className="w-3.5 h-3.5 text-brand-300 animate-pulse" />
            </div>
            <span className="text-[11px] font-bold text-brand-200 tracking-wider uppercase">Deep Health AI</span>
          </div>
          <p className="text-xs font-semibold text-white/90 leading-tight">
            AI for a
          </p>
          <p className="text-sm font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-brand-200 via-white to-cyan-200">
            Healthier Tomorrow
          </p>
          <div className="mt-3 pt-2.5 border-t border-white/10 flex items-center justify-between text-[10.5px] text-white/60">
            <span className="font-mono">PyTorch 2.14 • CUDA</span>
            <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)] animate-ping" />
              Active
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
};
