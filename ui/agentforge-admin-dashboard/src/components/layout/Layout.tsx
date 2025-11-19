'use client';

import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import TopoStage from '../TopoStage';

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const snap = useSnapshot(store);

  return (
    <div data-theme={snap.theme} className="relative min-h-screen overflow-hidden" suppressHydrationWarning>
      {/* 3D Background */}
      <TopoStage />

      {/* Neon scan bar (day mode) */}
      {snap.theme === 'day' && (
        <div className="pointer-events-none fixed inset-x-0 top-24 h-px bg-gradient-to-r from-transparent via-day-accent/70 to-transparent animate-scan" />
      )}

      {/* Grid overlay */}
      <div className="pointer-events-none fixed inset-0 z-10 mix-blend-screen opacity-20 [mask-image:linear-gradient(to_bottom,transparent,black_20%,black_80%,transparent)]">
        <svg className="h-full w-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="grid" width="64" height="64" patternUnits="userSpaceOnUse">
              <path d="M 64 0 L 0 0 0 64" fill="none" stroke="currentColor" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" className="text-day-lines dark:text-night-grid"/>
        </svg>
      </div>

      {/* Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="ml-0 lg:ml-80 transition-all duration-300">
        {/* Header */}
        <Header />

        {/* Page Content */}
        <main className="relative z-20 pt-16 min-h-screen">
          <div className="p-4 lg:p-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
