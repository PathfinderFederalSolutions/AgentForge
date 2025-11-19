'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  Target,
  FileText,
  Swords,
  Gamepad2,
  Shield,
  Radio,
  Upload,
  Database,
  Settings,
  HelpCircle,
  Zap,
  Brain,
  Activity
} from 'lucide-react';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/store';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onCommand: (commandId: string) => void;
}

interface Command {
  id: string;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string; style?: any }>;
  shortcut?: string;
  category: 'intelligence' | 'planning' | 'data' | 'navigation' | 'system';
}

const COMMANDS: Command[] = [
  // Intelligence
  {
    id: 'intelligence-dashboard',
    label: 'Open Intelligence Dashboard',
    description: 'View real-time threat monitoring and analysis',
    icon: Shield,
    shortcut: 'Ctrl+M',
    category: 'intelligence'
  },
  {
    id: 'intel-feed',
    label: 'Real-Time Intel Feed',
    description: 'Live intelligence event stream',
    icon: Radio,
    category: 'intelligence'
  },
  {
    id: 'analyze-threat',
    label: 'Analyze Threat',
    description: 'Quick threat analysis with intelligence agents',
    icon: Target,
    shortcut: 'Ctrl+I',
    category: 'intelligence'
  },
  
  // Planning
  {
    id: 'generate-plan',
    label: 'Generate Plan',
    description: 'Create strategic plan with goal planning',
    icon: FileText,
    shortcut: 'Ctrl+P',
    category: 'planning'
  },
  {
    id: 'coa-options',
    label: 'Generate COAs',
    description: 'Create courses of action',
    icon: Swords,
    shortcut: 'Ctrl+C',
    category: 'planning'
  },
  {
    id: 'run-wargame',
    label: 'Run Wargame',
    description: 'Simulate tactical scenarios',
    icon: Gamepad2,
    shortcut: 'Ctrl+W',
    category: 'planning'
  },
  
  // Data
  {
    id: 'quick-upload',
    label: 'Upload Data',
    description: 'Add files or connect data sources',
    icon: Upload,
    shortcut: 'Ctrl+U',
    category: 'data'
  },
  {
    id: 'data-sources',
    label: 'Manage Data Sources',
    description: 'View and manage connected data',
    icon: Database,
    category: 'data'
  },
  
  // Navigation
  {
    id: 'projects',
    label: 'Project Management',
    description: 'Manage missions and campaigns',
    icon: FileText,
    category: 'navigation'
  },
  {
    id: 'analytics',
    label: 'Advanced Analytics',
    description: 'View detailed system analytics',
    icon: Activity,
    category: 'navigation'
  },
  {
    id: 'capabilities',
    label: 'Platform Capabilities',
    description: 'Explore all platform features',
    icon: Brain,
    category: 'navigation'
  },
  
  // System
  {
    id: 'settings',
    label: 'Settings',
    description: 'Configure system preferences',
    icon: Settings,
    category: 'system'
  },
  {
    id: 'help',
    label: 'Help & Documentation',
    description: 'View help and guides',
    icon: HelpCircle,
    category: 'system'
  }
];

const CATEGORY_LABELS = {
  intelligence: '🛡️ Intelligence',
  planning: '📋 Planning & Operations',
  data: '📊 Data Management',
  navigation: '🧭 Navigation',
  system: '⚙️ System'
};

export default function CommandPalette({ isOpen, onClose, onCommand }: CommandPaletteProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const snap = useSnapshot(store);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) {
        // Global shortcuts
        if (e.ctrlKey || e.metaKey) {
          switch (e.key.toLowerCase()) {
            case 'k':
              e.preventDefault();
              // Open command palette (handled by parent)
              break;
            case 'i':
              e.preventDefault();
              onCommand('analyze-threat');
              break;
            case 'p':
              e.preventDefault();
              onCommand('generate-plan');
              break;
            case 'c':
              e.preventDefault();
              onCommand('coa-options');
              break;
            case 'w':
              e.preventDefault();
              onCommand('run-wargame');
              break;
            case 'm':
              e.preventDefault();
              onCommand('intelligence-dashboard');
              break;
            case 'u':
              e.preventDefault();
              onCommand('quick-upload');
              break;
          }
        }
      } else {
        // Palette navigation
        switch (e.key) {
          case 'ArrowDown':
            e.preventDefault();
            setSelectedIndex(prev => Math.min(prev + 1, filteredCommands.length - 1));
            break;
          case 'ArrowUp':
            e.preventDefault();
            setSelectedIndex(prev => Math.max(prev - 1, 0));
            break;
          case 'Enter':
            e.preventDefault();
            if (filteredCommands[selectedIndex]) {
              handleCommandSelect(filteredCommands[selectedIndex].id);
            }
            break;
          case 'Escape':
            e.preventDefault();
            onClose();
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, selectedIndex, searchQuery]);

  const filteredCommands = COMMANDS.filter(cmd => 
    cmd.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
    cmd.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
    cmd.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const groupedCommands = filteredCommands.reduce((acc, cmd) => {
    if (!acc[cmd.category]) {
      acc[cmd.category] = [];
    }
    acc[cmd.category].push(cmd);
    return acc;
  }, {} as Record<string, Command[]>);

  const handleCommandSelect = (commandId: string) => {
    onCommand(commandId);
    setSearchQuery('');
    setSelectedIndex(0);
    onClose();
  };

  const theme = snap.theme === 'day' ? {
    bg: '#05080D',
    text: '#D6E2F0',
    textSecondary: '#B8C5D1', 
    accent: '#00A39B',
    border: 'rgba(255,255,255,0.2)',
    cardBg: 'rgba(255,255,255,0.08)',
  } : {
    bg: '#000000',
    text: '#FF2B2B',
    textSecondary: '#891616',
    accent: '#FF2B2B',
    border: 'rgba(137,22,22,0.6)',
    cardBg: 'rgba(137,22,22,0.15)',
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-[100] flex items-start justify-center pt-24 p-4"
        style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)' }}
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0, y: -20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.95, opacity: 0, y: -20 }}
          className="w-full max-w-2xl rounded-2xl overflow-hidden"
          style={{
            background: theme.bg,
            border: `1px solid ${theme.border}`,
            boxShadow: `0 0 60px ${theme.accent}40`
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Search Input */}
          <div className="p-4 border-b" style={{ borderColor: theme.border }}>
            <div className="relative">
              <Search 
                className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5" 
                style={{ color: theme.accent }} 
              />
              <input
                ref={inputRef}
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setSelectedIndex(0);
                }}
                placeholder="Search commands... (use arrow keys to navigate)"
                className="w-full pl-12 pr-4 py-3 rounded-lg text-base"
                style={{
                  background: theme.cardBg,
                  color: theme.text,
                  border: `1px solid ${theme.border}`,
                  outline: 'none'
                }}
              />
            </div>
          </div>

          {/* Commands List */}
          <div className="max-h-96 overflow-y-auto p-2">
            {Object.entries(groupedCommands).map(([category, commands]) => (
              <div key={category} className="mb-4">
                <div 
                  className="px-3 py-2 text-xs font-bold uppercase"
                  style={{ color: theme.textSecondary }}
                >
                  {CATEGORY_LABELS[category as keyof typeof CATEGORY_LABELS]}
                </div>
                <div className="space-y-1">
                  {commands.map((cmd, idx) => {
                    const Icon = cmd.icon;
                    const globalIndex = filteredCommands.indexOf(cmd);
                    const isSelected = globalIndex === selectedIndex;
                    
                    return (
                      <motion.button
                        key={cmd.id}
                        onClick={() => handleCommandSelect(cmd.id)}
                        className="w-full p-3 rounded-lg flex items-center gap-3 text-left transition-all"
                        style={{
                          background: isSelected ? `${theme.accent}20` : 'transparent',
                          border: `1px solid ${isSelected ? theme.accent : 'transparent'}`
                        }}
                        whileHover={{ 
                          background: `${theme.accent}15`,
                          x: 4
                        }}
                      >
                        <div 
                          className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                          style={{ background: `${theme.accent}20` }}
                        >
                          <Icon className="w-5 h-5" style={{ color: theme.accent }} />
                        </div>
                        
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between mb-1">
                            <h4 className="font-semibold text-sm" style={{ color: theme.text }}>
                              {cmd.label}
                            </h4>
                            {cmd.shortcut && (
                              <div 
                                className="px-2 py-1 rounded text-xs font-mono"
                                style={{ 
                                  background: theme.cardBg,
                                  color: theme.textSecondary,
                                  border: `1px solid ${theme.border}`
                                }}
                              >
                                {cmd.shortcut}
                              </div>
                            )}
                          </div>
                          <p className="text-xs line-clamp-1" style={{ color: theme.textSecondary }}>
                            {cmd.description}
                          </p>
                        </div>
                      </motion.button>
                    );
                  })}
                </div>
              </div>
            ))}

            {filteredCommands.length === 0 && (
              <div className="text-center py-12">
                <Search className="w-12 h-12 mx-auto mb-3 opacity-50" style={{ color: theme.textSecondary }} />
                <p style={{ color: theme.textSecondary }}>No commands found</p>
              </div>
            )}
          </div>

          {/* Footer */}
          <div 
            className="p-3 border-t flex items-center justify-between text-xs"
            style={{ borderColor: theme.border, color: theme.textSecondary }}
          >
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <kbd 
                  className="px-2 py-1 rounded"
                  style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                >
                  ↑↓
                </kbd>
                <span>Navigate</span>
              </div>
              <div className="flex items-center gap-1">
                <kbd 
                  className="px-2 py-1 rounded"
                  style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                >
                  Enter
                </kbd>
                <span>Select</span>
              </div>
              <div className="flex items-center gap-1">
                <kbd 
                  className="px-2 py-1 rounded"
                  style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                >
                  Esc
                </kbd>
                <span>Close</span>
              </div>
            </div>
            <div>
              <span>Press </span>
              <kbd 
                className="px-2 py-1 rounded"
                style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
              >
                Ctrl+K
              </kbd>
              <span> to reopen</span>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

