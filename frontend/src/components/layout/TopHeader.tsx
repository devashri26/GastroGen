import React, { useState } from 'react';
import { 
  ChevronDown, 
  Activity, 
  ShieldCheck, 
  Cpu, 
  User, 
  Bell,
  Sparkles,
  ExternalLink
} from 'lucide-react';

interface TopHeaderProps {
  onOpenInfoModal?: () => void;
}

export const TopHeader: React.FC<TopHeaderProps> = () => {
  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <header className="h-[75px] fixed top-0 left-[260px] right-0 bg-white/85 backdrop-blur-xl border-b border-slate-200/90 z-20 px-8 flex items-center justify-between shadow-[0_4px_20px_rgba(15,23,42,0.03)]">
      {/* Left side: Sub-branding / Current Pipeline Status */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-700 text-xs font-semibold shadow-[0_0_14px_rgba(16,185,129,0.16)]">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span>CUDA RTX 4050 Active</span>
        </div>
        <div className="hidden lg:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-100/90 border border-slate-200/90 text-slate-700 text-xs font-medium shadow-2xs">
          <Cpu className="w-3.5 h-3.5 text-brand-600" />
          <span>PyTorch v2.14 • 256×256 Pipeline</span>
        </div>
      </div>

      {/* Center: Multi-Model Framework Title & Subtitle */}
      <div className="hidden md:flex flex-col items-center text-center px-4">
        <h1 className="text-[15px] font-bold text-slate-900 tracking-tight flex items-center gap-2">
          <span className="bg-gradient-to-r from-slate-900 via-slate-800 to-brand-950 bg-clip-text text-transparent">
            A Multi-Model Generative AI Framework
          </span>
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 text-[10.5px] font-semibold bg-gradient-to-r from-brand-600 to-indigo-600 text-white rounded-full shadow-sm shadow-brand-500/25">
            <Sparkles className="w-3 h-3 text-brand-200" />
            Research v2
          </span>
        </h1>
        <p className="text-[11.5px] font-medium text-slate-500 max-w-xl truncate mt-0.5">
          for Synthetic Gastrointestinal Data Augmentation and Disease Reconstruction
        </p>
      </div>

      {/* Right side: User Profile & Quick Actions */}
      <div className="flex items-center gap-3">
        <button 
          title="System notifications"
          className="w-9 h-9 rounded-full bg-slate-50/90 hover:bg-slate-100 border border-slate-200/90 flex items-center justify-center text-slate-600 hover:text-slate-900 transition-all relative shadow-2xs hover:shadow-xs"
        >
          <Bell className="w-4 h-4" />
          <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-brand-600 ring-2 ring-white" />
        </button>

        {/* User Dropdown */}
        <div className="relative">
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex items-center gap-2.5 pl-2 pr-3 py-1.5 rounded-full bg-white hover:bg-slate-50 border border-slate-200/90 hover:border-brand-300 transition-all duration-200 group shadow-2xs hover:shadow-sm"
          >
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-brand-600 via-brand-700 to-indigo-700 text-white flex items-center justify-center font-bold text-xs shadow-sm shadow-brand-600/30 ring-2 ring-brand-200/80">
              D
            </div>
            <div className="flex flex-col text-left">
              <span className="text-xs font-bold text-slate-800 leading-tight group-hover:text-brand-700 transition-colors">Welcome, Devashri</span>
              <span className="text-[10px] text-slate-400 font-medium leading-none">Clinical AI Researcher</span>
            </div>
            <ChevronDown className={`w-3.5 h-3.5 text-slate-400 group-hover:text-slate-600 transition-transform duration-200 ${dropdownOpen ? 'rotate-180' : ''}`} />
          </button>

          {dropdownOpen && (
            <div className="absolute right-0 mt-2 w-56 bg-white rounded-2xl shadow-xl border border-slate-200/80 p-2 z-50 animate-in fade-in slide-in-from-top-2 duration-150">
              <div className="px-3 py-2 border-b border-slate-100">
                <p className="text-xs font-bold text-slate-900">Devashri</p>
                <p className="text-[11px] text-slate-500">GI Endoscopy & AI Lab</p>
              </div>
              <div className="p-1 space-y-0.5 text-xs text-slate-600">
                <div className="px-3 py-2 rounded-xl bg-brand-50/60 text-brand-800 font-medium flex items-center justify-between">
                  <span>Role: Lead Clinician</span>
                  <ShieldCheck className="w-4 h-4 text-brand-600" />
                </div>
                <button 
                  onClick={() => setDropdownOpen(false)}
                  className="w-full text-left px-3 py-2 rounded-xl hover:bg-slate-100 text-slate-700 flex items-center gap-2"
                >
                  <Activity className="w-3.5 h-3.5 text-slate-500" />
                  <span>Pipeline Telemetry</span>
                </button>
                <button 
                  onClick={() => setDropdownOpen(false)}
                  className="w-full text-left px-3 py-2 rounded-xl hover:bg-slate-100 text-slate-700 flex items-center gap-2"
                >
                  <User className="w-3.5 h-3.5 text-slate-500" />
                  <span>Researcher Profile</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
