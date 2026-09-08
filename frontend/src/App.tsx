import React, { useState } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { TopHeader } from './components/layout/TopHeader';
import { AutoencoderPage } from './components/pages/AutoencoderPage';
import { VAEPage } from './components/pages/VAEPage';
import { ViTPage } from './components/pages/ViTPage';
import { ModelComparisonPage } from './components/pages/ModelComparisonPage';
import { AboutPage } from './components/pages/AboutPage';
import { PageType } from './types';
import { HeartPulse, ExternalLink } from 'lucide-react';

export const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<PageType>('autoencoder');

  const renderActivePage = () => {
    switch (currentPage) {
      case 'autoencoder':
        return <AutoencoderPage />;
      case 'vae':
        return <VAEPage />;
      case 'vit':
        return <ViTPage />;
      case 'comparison':
        return <ModelComparisonPage />;
      case 'about':
        return <AboutPage />;
      default:
        return <AutoencoderPage />;
    }
  };

  return (
    <div className="min-h-screen bg-transparent text-slate-800 font-sans antialiased flex selection:bg-brand-500 selection:text-white relative">
      {/* Persistent Fixed Navigation Sidebar */}
      <Sidebar currentPage={currentPage} onSelectPage={setCurrentPage} />

      {/* Main Layout Container */}
      <div className="flex-1 ml-[260px] flex flex-col min-h-screen">
        {/* Persistent Fixed Header */}
        <TopHeader />

        {/* Dynamic Page Content View */}
        <main className="flex-1 pt-[75px] px-8 py-8 lg:px-10 lg:py-10 max-w-7xl w-full mx-auto animate-in fade-in duration-200">
          {renderActivePage()}
        </main>

        {/* Clinical Platform Research Footer */}
        <footer className="ml-0 border-t border-slate-200/90 bg-white/90 backdrop-blur-md py-6 px-8 lg:px-10 mt-12 text-xs text-slate-500 shadow-[0_-2px_12px_rgba(0,0,0,0.02)]">
          <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-brand-600 to-indigo-600 flex items-center justify-center text-white shadow-sm shadow-brand-500/30">
                <HeartPulse className="w-4 h-4" />
              </div>
              <div>
                <span className="font-bold text-slate-800 tracking-tight">GastroGen Research Platform</span>
                <span className="text-slate-300 mx-2">•</span>
                <span className="text-slate-500 font-medium">Kvasir-v2 Medical Endoscopy Benchmark Suite</span>
              </div>
            </div>

            <div className="flex items-center gap-6 text-[11.5px]">
              <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200/70 text-emerald-700 font-semibold shadow-2xs">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>PyTorch 2.14 GPU Connected</span>
              </div>
              <span className="text-slate-300">|</span>
              <button 
                onClick={() => setCurrentPage('about')}
                className="hover:text-brand-600 font-semibold text-slate-600 transition-colors flex items-center gap-1.5"
              >
                <span>Research Documentation</span>
                <ExternalLink className="w-3 h-3 text-slate-400" />
              </button>
              <span className="text-slate-300">|</span>
              <span className="text-slate-400 font-medium">© 2026 AI for Gastrointestinal Health</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default App;
