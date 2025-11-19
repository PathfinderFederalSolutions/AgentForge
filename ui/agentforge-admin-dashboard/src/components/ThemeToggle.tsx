'use client';

import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';

export function ThemeToggle() {
  const snap = useSnapshot(store);
  const isNight = snap.theme === 'night';

  return (
    <div className="flex items-center gap-3">
      <span className="label">{isNight ? 'NIGHT OPS' : 'DAY MODE'}</span>
      <button
        aria-label="Toggle theme"
        className="relative h-7 w-14 rounded-full border border-white/10 bg-white/5
                   dark:border-red-900/50 dark:bg-black/40"
        onClick={() => store.toggleTheme()}
      >
        <div
          className={`absolute top-0.5 h-6 w-6 rounded-full transition-transform 
                      ${isNight ? 'translate-x-7 bg-red-600' : 'translate-x-1 bg-day-accent'}`}
        />
      </button>
    </div>
  );
}
