'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  Target,
  FileText,
  Crosshair,
  Activity,
  Shield,
  Radio,
  Folder,
  Clock,
  TrendingUp
} from 'lucide-react';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onExecute: (command: string) => void;
  theme: any;
}

export default function IntelligenceCommandPalette({ isOpen, onClose, onExecute, theme }: CommandPaletteProps) {
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState(0);

  const commands = [
    {
      id: 'threat-analysis',
      icon: Target,
      label: 'Analyze Threat',
      description: 'Run quick threat analysis',
      shortcut: 'Ctrl+I',
      category: 'Intelligence'
    },
    {
      id: 'planning',
      icon: FileText,
      label: 'Generate Plan',
      description: 'Create strategic plan',
      shortcut: 'Ctrl+P',
      category: 'Planning'
    },
    {
      id: 'coa-options',
      icon: Crosshair,
      label: 'Generate COAs',
      description: 'Create course of action options',
      shortcut: 'Ctrl+C',
      category: 'Planning'
    },
    {
      id: 'wargame',
      icon: Activity,
      label: 'Run Wargame',
      description: 'Simulate operations',
      shortcut: 'Ctrl+W',
      category: 'Simulation'
    },
    {
      id: 'intel-dashboard',
      icon: Shield,
      label: 'Intelligence Dashboard',
      description: 'Open intelligence monitoring',
      shortcut: 'Ctrl+M',
      category: 'Intelligence'
    },
    {
      id: 'monitoring',
      icon: Radio,
      label: 'Start Continuous Monitoring',
      description: 'Enable real-time intelligence',
      shortcut: 'Ctrl+R',
      category: 'Intelligence'
    },
    {
      id: 'projects',
      icon: Folder,
      label: 'Project Manager',
      description: 'Manage projects and missions',
      shortcut: 'Ctrl+Shift+P',
      category: 'Management'
    },
    {
      id: 'timeline',
      icon: Clock,
      label: 'Timeline Visualization',
      description: 'View intelligence timeline',
      shortcut: 'Ctrl+T',
      category: 'Visualization'
    },
    {
      id: 'realtime-feed',
      icon: TrendingUp,
      label: 'Real-Time Intel Feed',
      description: 'Live intelligence events',
      shortcut: 'Ctrl+F',
      category: 'Intelligence'
    }
  ];

  const filteredCommands = commands.filter(cmd =>
    cmd.label.toLowerCase().includes(search.toLowerCase()) ||
    cmd.description.toLowerCase().includes(search.toLowerCase()) ||
    cmd.category.toLowerCase().includes(search.toLowerCase())
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelected(prev => (prev + 1) % filteredCommands.length);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelected(prev => (prev - 1 + filteredCommands.length) % filteredCommands.length);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredCommands[selected]) {
          onExecute(filteredCommands[selected].id);
          onClose();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, selected, filteredCommands, onExecute, onClose]);

  useEffect(() => {
    if (isOpen) {
      setSearch('');
      setSelected(0);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-start justify-center p-4 pt-32"
        style={{
          background: 'rgba(0,0,0,0.8)',
          backdropFilter: 'blur(8px)'
        }}
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0, y: -20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.9, opacity: 0, y: -20 }}
          className="w-full max-w-2xl rounded-xl overflow-hidden"
          style={{
            background: theme.bg,
            border: `1px solid ${theme.border}`,
            boxShadow: `0 0 60px ${theme.accent}30`
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Search Input */}
          <div className="p-4 border-b" style={{ borderColor: theme.border }}>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5" style={{ color: theme.textSecondary }} />
              <input
                type="text"
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setSelected(0);
                }}
                placeholder="Type a command or search..."
                autoFocus
                className="w-full pl-12 pr-4 py-3 rounded-lg text-lg"
                style={{
                  background: theme.cardBg,
                  border: `1px solid ${theme.border}`,
                  color: theme.text,
                  outline: 'none'
                }}
              />
            </div>
          </div>

          {/* Commands List */}
          <div className="max-h-96 overflow-y-auto p-2">
            {filteredCommands.length === 0 ? (
              <div className="text-center py-8" style={{ color: theme.textSecondary }}>
                No commands found
              </div>
            ) : (
              <div className="space-y-1">
                {filteredCommands.map((command, idx) => (
                  <motion.button
                    key={command.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: idx * 0.02 }}
                    onClick={() => {
                      onExecute(command.id);
                      onClose();
                    }}
                    className="w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all text-left"
                    style={{
                      background: selected === idx ? `${theme.accent}20` : 'transparent',
                      border: `1px solid ${selected === idx ? theme.accent : 'transparent'}`
                    }}
                    onMouseEnter={() => setSelected(idx)}
                  >
                    <div 
                      className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                      style={{ background: `${theme.accent}20` }}
                    >
                      <command.icon className="w-5 h-5" style={{ color: theme.accent }} />
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold" style={{ color: theme.text }}>
                          {command.label}
                        </span>
                        <span 
                          className="text-xs px-2 py-0.5 rounded"
                          style={{
                            background: `${theme.accent}15`,
                            color: theme.accent
                          }}
                        >
                          {command.category}
                        </span>
                      </div>
                      <p className="text-sm truncate" style={{ color: theme.textSecondary }}>
                        {command.description}
                      </p>
                    </div>

                    {command.shortcut && (
                      <div 
                        className="px-2 py-1 rounded text-xs font-mono flex-shrink-0"
                        style={{
                          background: theme.cardBg,
                          color: theme.textSecondary,
                          border: `1px solid ${theme.border}`
                        }}
                      >
                        {command.shortcut}
                      </div>
                    )}
                  </motion.button>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div 
            className="p-3 border-t text-xs flex items-center justify-between"
            style={{ borderColor: theme.border, color: theme.textSecondary }}
          >
            <div className="flex items-center gap-4">
              <span>↑↓ Navigate</span>
              <span>↵ Select</span>
              <span>Esc Close</span>
            </div>
            <span>Ctrl+K to toggle</span>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

