import React from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  subtext?: string;
  color?: 'purple' | 'teal' | 'pink' | 'emerald' | 'amber' | 'blue';
  highlight?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  subtext,
  color = 'purple',
  highlight = false,
}) => {
  const colorStyles = {
    purple: {
      bar: 'bg-gradient-to-r from-brand-500 to-indigo-500',
      text: 'text-brand-700',
      badge: 'bg-brand-50 text-brand-700 border-brand-200/60',
    },
    teal: {
      bar: 'bg-gradient-to-r from-teal-500 to-emerald-500',
      text: 'text-teal-700',
      badge: 'bg-teal-50 text-teal-700 border-teal-200/60',
    },
    pink: {
      bar: 'bg-gradient-to-r from-pink-500 to-rose-500',
      text: 'text-pink-700',
      badge: 'bg-pink-50 text-pink-700 border-pink-200/60',
    },
    emerald: {
      bar: 'bg-gradient-to-r from-emerald-500 to-teal-500',
      text: 'text-emerald-700',
      badge: 'bg-emerald-50 text-emerald-700 border-emerald-200/60',
    },
    amber: {
      bar: 'bg-gradient-to-r from-amber-500 to-orange-500',
      text: 'text-amber-700',
      badge: 'bg-amber-50 text-amber-700 border-amber-200/60',
    },
    blue: {
      bar: 'bg-gradient-to-r from-blue-500 to-indigo-500',
      text: 'text-blue-700',
      badge: 'bg-blue-50 text-blue-700 border-blue-200/60',
    },
  }[color];

  return (
    <div className={`p-4 rounded-xl border bg-white shadow-soft transition-all duration-200 relative overflow-hidden group hover:shadow-card hover:-translate-y-0.5 ${
      highlight 
        ? 'ring-2 ring-brand-500/25 border-brand-300 shadow-md shadow-brand-500/10' 
        : 'border-slate-200/90 hover:border-slate-300'
    }`}>
      {/* Top accent gradient bar */}
      <div className={`absolute top-0 left-0 right-0 h-1 ${colorStyles.bar}`} />

      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[10.5px] font-bold uppercase tracking-wider text-slate-400 block">
          {label}
        </span>
        {highlight && (
          <span className="px-1.5 py-0.5 text-[9px] font-extrabold uppercase rounded bg-brand-100 text-brand-700 border border-brand-200/70">
            Optimal
          </span>
        )}
      </div>

      <div className="flex items-baseline gap-1.5">
        <span className="text-2xl font-black text-slate-900 tracking-tight font-mono">
          {value}
        </span>
        {unit && (
          <span className="text-xs font-bold text-slate-500">
            {unit}
          </span>
        )}
      </div>
      {subtext && (
        <span className="text-[11px] text-slate-400 font-medium mt-1.5 block leading-tight">
          {subtext}
        </span>
      )}
    </div>
  );
};
