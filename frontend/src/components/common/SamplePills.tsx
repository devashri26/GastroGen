import React from 'react';
import { SAMPLE_KEYS, SAMPLES_CATALOG } from '../../data/mockData';

interface SamplePillsProps {
  selectedSampleId: string;
  onSelectSample: (sampleId: string) => void;
  title?: string;
}

export const SamplePills: React.FC<SamplePillsProps> = ({
  selectedSampleId,
  onSelectSample,
  title = "Or pick sample:"
}) => {
  return (
    <div className="space-y-2 mt-4">
      {title && (
        <span className="text-xs font-bold text-slate-500 flex items-center gap-1.5">
          <span>{title}</span>
        </span>
      )}
      <div className="flex flex-wrap gap-2">
        {SAMPLE_KEYS.map((key: string) => {
          const sample = (SAMPLES_CATALOG as Record<string, any>)[key];
          const isSelected = selectedSampleId === key;
          return (
            <button
              key={key}
              onClick={() => onSelectSample(key)}
              className={`px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all duration-200 flex items-center gap-2 ${
                isSelected
                  ? 'bg-gradient-to-r from-brand-600 via-indigo-600 to-violet-700 text-white shadow-md shadow-brand-600/30 ring-2 ring-brand-300/80 scale-[1.02]'
                  : 'bg-white hover:bg-slate-50 text-slate-700 hover:text-brand-700 border border-slate-200/90 hover:border-brand-300 shadow-2xs hover:shadow-xs'
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${
                isSelected 
                  ? 'bg-white shadow-[0_0_6px_rgba(255,255,255,0.8)]' 
                  : sample.type === 'Pathological' 
                    ? 'bg-rose-500 shadow-[0_0_6px_rgba(244,63,94,0.4)]' 
                    : sample.type === 'Therapeutic' 
                      ? 'bg-teal-500 shadow-[0_0_6px_rgba(13,148,136,0.4)]' 
                      : 'bg-blue-500 shadow-[0_0_6px_rgba(59,130,246,0.4)]'
              }`} />
              <span>{sample.name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
};
