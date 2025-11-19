'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Home, 
  Bot, 
  Briefcase, 
  Activity, 
  Settings, 
  Users, 
  Database, 
  Shield, 
  BarChart3,
  Terminal,
  Zap,
  Brain,
  Network,
  BookOpen,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { clsx } from 'clsx';

const navigation = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'AI Capabilities', href: '/ai-capabilities', icon: Brain, highlight: true },
  { name: 'Instructions', href: '/instructions', icon: BookOpen },
  { name: 'Agents', href: '/agents', icon: Bot },
  { name: 'Jobs', href: '/jobs', icon: Briefcase },
  { name: 'Swarm Control', href: '/swarm', icon: Users },
  { name: 'Neural Mesh', href: '/neural-mesh', icon: Network },
  { name: 'Quantum Scheduler', href: '/quantum', icon: Zap },
  { name: 'Monitoring', href: '/monitoring', icon: Activity },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Terminal', href: '/terminal', icon: Terminal },
  { name: 'Data', href: '/data', icon: Database },
  { name: 'Security', href: '/security', icon: Shield },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = usePathname();

  return (
    <motion.div
      initial={false}
      animate={{ width: collapsed ? 80 : 280 }}
      className="fixed left-0 top-0 h-full z-40 hud-card border-r border-white/10 dark:border-red-900/40 hidden lg:block"
    >
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/10 dark:border-red-900/40">
          <AnimatePresence>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="flex items-center gap-3"
              >
                <div className="w-8 h-8 rounded-lg bg-day-accent dark:bg-night-text flex items-center justify-center">
                  <Bot className="w-5 h-5 text-white dark:text-black" />
                </div>
                <div>
                  <div className="font-semibold text-sm">AgentForge</div>
                  <div className="text-xs opacity-60">Swarm Ops</div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1.5 rounded-lg hover:bg-white/10 dark:hover:bg-red-900/10 transition-colors"
          >
            {collapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <ChevronLeft className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          {navigation.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={clsx(
                  'flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group relative',
                  isActive
                    ? 'bg-day-accent/20 text-day-accent dark:bg-night-text/20 dark:text-night-text'
                    : 'hover:bg-white/10 dark:hover:bg-red-900/10 text-day-text/80 dark:text-night-text/80 hover:text-day-text dark:hover:text-night-text'
                )}
              >
                <item.icon className="w-5 h-5 flex-shrink-0" />
                <AnimatePresence>
                  {!collapsed && (
                    <motion.span
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -10 }}
                      className="font-medium text-sm"
                    >
                      {item.name}
                    </motion.span>
                  )}
                </AnimatePresence>
                
                {/* Active indicator */}
                {isActive && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute right-2 w-2 h-2 rounded-full bg-day-accent dark:bg-night-text"
                  />
                )}

                {/* Tooltip for collapsed state */}
                {collapsed && (
                  <div className="absolute left-full ml-2 px-2 py-1 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50">
                    {item.name}
                  </div>
                )}
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-white/10 dark:border-red-900/40">
          <div className={clsx(
            'flex items-center gap-3 px-3 py-2',
            collapsed ? 'justify-center' : ''
          )}>
            <div className="w-8 h-8 rounded-full bg-day-accent/20 dark:bg-night-text/20 flex items-center justify-center">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            </div>
            <AnimatePresence>
              {!collapsed && (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  className="flex-1"
                >
                  <div className="text-xs font-medium">System Online</div>
                  <div className="text-xs opacity-60">All systems operational</div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
