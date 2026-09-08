import React from 'react';

interface HeroCardProps {
  title: string;
  description: string;
  badges: { label: string; icon?: React.ReactNode; variant?: 'purple' | 'teal' | 'pink' | 'emerald' | 'blue' }[];
  gradientTheme?: 'purple-violet' | 'purple-pink' | 'cyan-teal' | 'blue-purple';
}

export const HeroCard: React.FC<HeroCardProps> = ({
  title,
  description,
  badges,
  gradientTheme = 'purple-violet',
}) => {
  const gradientClasses = {
    'purple-violet': 'from-[#1e1b4b] via-[#312e81] to-[#4c1d95] border-indigo-500/30',
    'purple-pink': 'from-[#2e1065] via-[#4a044e] to-[#831843] border-pink-500/30',
    'cyan-teal': 'from-[#042f2e] via-[#115e59] to-[#0e7490] border-teal-500/30',
    'blue-purple': 'from-[#0f172a] via-[#1e1b4b] to-[#1e40af] border-blue-500/30',
  }[gradientTheme];

  return (
    <div className={`w-full rounded-2xl bg-gradient-to-r ${gradientClasses} p-7 text-white relative overflow-hidden shadow-[inset_0_1px_0_rgba(255,255,255,0.2),0_10px_30px_-5px_rgba(15,23,42,0.15)] border mb-6`}>
      {/* Decorative background glow circles */}
      <div className="absolute -top-24 -right-24 w-80 h-80 bg-white/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-24 -left-24 w-80 h-80 bg-brand-400/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.05)_0,transparent_70%)] pointer-events-none" />

      <div className="relative z-10 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-5">
        <div className="max-w-3xl space-y-2">
          <h2 className="text-2xl font-extrabold tracking-tight text-white flex items-center gap-2.5 drop-shadow-sm">
            {title}
          </h2>
          <p className="text-sm text-slate-200/95 leading-relaxed font-normal">
            {description}
          </p>
        </div>

        {/* Right Badges */}
        <div className="flex flex-wrap items-center gap-2 shrink-0">
          {badges.map((badge, idx) => {
            const variantClasses = {
              purple: 'bg-brand-500/25 border-brand-300/40 text-white shadow-brand-500/15',
              teal: 'bg-teal-500/25 border-teal-300/40 text-white shadow-teal-500/15',
              pink: 'bg-pink-500/25 border-pink-300/40 text-white shadow-pink-500/15',
              emerald: 'bg-emerald-500/25 border-emerald-300/40 text-white shadow-emerald-500/15',
              blue: 'bg-blue-500/25 border-blue-300/40 text-white shadow-blue-500/15',
            }[badge.variant || 'purple'];

            return (
              <span
                key={idx}
                className={`inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold backdrop-blur-md border ${variantClasses} shadow-sm transition-all hover:scale-[1.02]`}
              >
                {badge.icon}
                <span>{badge.label}</span>
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
};
