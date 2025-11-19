'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Zap,
  Target,
  Shield,
  Activity,
  Folder,
  Database,
  Bot,
  Search,
  Upload,
  Play,
  BarChart3,
  Settings,
  ChevronUp,
  ChevronDown
} from 'lucide-react';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/store';

interface QuickAction {
  id: string;
  label: string;
  icon: React.ComponentType<{className?: string}>;
  description: string;
  color: string;
  action: () => void;
  category: 'analysis' | 'planning' | 'intelligence' | 'monitoring' | 'management' | 'data';
}

interface QuickActionToolbarProps {
  onOpenUploadModal: () => void;
  onOpenCapabilityShowcase: () => void;
  onOpenAdvancedAnalytics: () => void;
  onOpenProjectsSidebar: () => void;
  onOpenCOAWarGamePanel: () => void;
  onOpenRealTimeIntelPanel: () => void;
  onOpenIntelDashboard: () => void;
}

export default function QuickActionToolbar({
  onOpenUploadModal,
  onOpenCapabilityShowcase,
  onOpenAdvancedAnalytics,
  onOpenProjectsSidebar,
  onOpenCOAWarGamePanel,
  onOpenRealTimeIntelPanel,
  onOpenIntelDashboard
}: QuickActionToolbarProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const snap = useSnapshot(store);

  const theme = snap.theme === 'day' ? {
    bg: '#05080D',
    text: '#D6E2F0',
    textSecondary: '#B8C5D1',
    accent: '#00A39B',
    neon: '#CCFF00',
    border: 'rgba(255,255,255,0.2)',
    cardBg: 'rgba(255,255,255,0.08)',
    grid: '#0E1622',
    lines: '#0F2237'
  } : {
    bg: '#000000',
    text: '#FF2B2B',
    textSecondary: '#891616',
    accent: '#FF2B2B',
    border: 'rgba(137,22,22,0.6)',
    cardBg: 'rgba(137,22,22,0.15)',
    grid: '#1a0000',
    lines: '#891616'
  };

  const quickActions: QuickAction[] = [
    {
      id: 'analyze-threat',
      label: 'Analyze Threat',
      icon: Search,
      description: 'Deep threat analysis with AI swarm processing',
      color: '#FF6B6B',
      category: 'analysis',
      action: () => {
        store.sendMessage('Analyze current threats with maximum intelligence amplification', {
          include_planning: true,
          generate_coas: true,
          run_wargaming: true,
          intelligence_analysis: true,
          intelligence_domain: 'MULTI_DOMAIN'
        });
      }
    },
    {
      id: 'generate-plan',
      label: 'Generate Plan',
      icon: Target,
      description: 'Strategic planning with COA development',
      color: '#FFD700',
      category: 'planning',
      action: () => {
        store.sendMessage('Generate comprehensive strategic plan with multiple courses of action', {
          include_planning: true,
          generate_coas: true,
          run_wargaming: false,
          intelligence_analysis: true
        });
      }
    },
    {
      id: 'coa-options',
      label: 'COA Options',
      icon: Zap,
      description: 'Compare and analyze courses of action',
      color: '#00A39B',
      category: 'planning',
      action: () => onOpenCOAWarGamePanel()
    },
    {
      id: 'run-wargame',
      label: 'Run Wargame',
      icon: Play,
      description: 'Execute wargaming simulation',
      color: '#9C27B0',
      category: 'analysis',
      action: () => {
        store.sendMessage('Execute comprehensive wargaming simulation for current scenario', {
          include_planning: true,
          generate_coas: true,
          run_wargaming: true,
          intelligence_analysis: true
        });
      }
    },
    {
      id: 'intelligence-dashboard',
      label: 'Intelligence Dashboard',
      icon: Shield,
      description: 'Real-time threat monitoring and metrics',
      color: '#4CAF50',
      category: 'intelligence',
      action: () => onOpenIntelDashboard()
    },
    {
      id: 'real-time-intel',
      label: 'Live Intel Feed',
      icon: Activity,
      description: 'Real-time intelligence event stream',
      color: '#FF9800',
      category: 'intelligence',
      action: () => onOpenRealTimeIntelPanel()
    },
    {
      id: 'start-monitoring',
      label: 'Start Monitoring',
      icon: Database,
      description: 'Continuous intelligence monitoring',
      color: '#2196F3',
      category: 'monitoring',
      action: () => {
        store.sendMessage('Initiate continuous intelligence monitoring and analysis', {
          include_planning: false,
          generate_coas: false,
          run_wargaming: false,
          intelligence_analysis: true,
          continuous_monitoring: true
        });
      }
    },
    {
      id: 'upload-data',
      label: 'Upload Intelligence',
      icon: Upload,
      description: 'Add data sources with intelligence metadata',
      color: '#607D8B',
      category: 'data',
      action: () => onOpenUploadModal()
    },
    {
      id: 'project-management',
      label: 'Project Management',
      icon: Folder,
      description: 'Multi-project workflow management',
      color: '#795548',
      category: 'management',
      action: () => onOpenProjectsSidebar()
    },
    {
      id: 'advanced-analytics',
      label: 'Advanced Analytics',
      icon: BarChart3,
      description: 'Deep analytics and performance insights',
      color: '#E91E63',
      category: 'analysis',
      action: () => onOpenAdvancedAnalytics()
    },
    {
      id: 'ai-capabilities',
      label: 'AI Capabilities',
      icon: Bot,
      description: 'Explore all available AI capabilities',
      color: '#673AB7',
      category: 'management',
      action: () => onOpenCapabilityShowcase()
    }
  ];

  const categories = [
    { id: 'all', label: 'All Actions', count: quickActions.length },
    { id: 'analysis', label: 'Analysis', count: quickActions.filter(a => a.category === 'analysis').length },
    { id: 'planning', label: 'Planning', count: quickActions.filter(a => a.category === 'planning').length },
    { id: 'intelligence', label: 'Intelligence', count: quickActions.filter(a => a.category === 'intelligence').length },
    { id: 'monitoring', label: 'Monitoring', count: quickActions.filter(a => a.category === 'monitoring').length },
    { id: 'data', label: 'Data', count: quickActions.filter(a => a.category === 'data').length },
    { id: 'management', label: 'Management', count: quickActions.filter(a => a.category === 'management').length }
  ];

  const filteredActions = selectedCategory === 'all'
    ? quickActions
    : quickActions.filter(action => action.category === selectedCategory);

  return (
    <div className="fixed top-20 left-1/2 transform -translate-x-1/2 z-40">
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: -10 }}
            className="mb-4 p-6 rounded-xl shadow-2xl"
            style={{
              background: theme.cardBg,
              border: `1px solid ${theme.border}`,
              backdropFilter: 'blur(20px)',
              minWidth: '600px',
              maxWidth: '800px'
            }}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-lg font-bold" style={{ color: theme.text }}>
                  Quick Actions
                </h3>
                <p className="text-sm" style={{ color: theme.textSecondary }}>
                  Access all AgentForge capabilities instantly
                </p>
              </div>
              <button
                onClick={() => setIsExpanded(false)}
                className="p-2 rounded-lg btn-hover"
                style={{ color: theme.text }}
              >
                <ChevronUp className="w-5 h-5" />
              </button>
            </div>

            {/* Category Filter */}
            <div className="flex gap-2 mb-6 overflow-x-auto">
              {categories.map((category) => (
                <button
                  key={category.id}
                  onClick={() => setSelectedCategory(category.id)}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all btn-hover whitespace-nowrap"
                  style={{
                    background: selectedCategory === category.id ? theme.accent : theme.cardBg,
                    color: selectedCategory === category.id ? 'white' : theme.text,
                    border: `1px solid ${selectedCategory === category.id ? theme.accent : theme.border}`
                  }}
                >
                  {category.label}
                  <span className="px-2 py-1 rounded-full text-xs"
                        style={{
                          background: selectedCategory === category.id ? 'rgba(255,255,255,0.2)' : `${theme.accent}20`,
                          color: selectedCategory === category.id ? 'white' : theme.accent
                        }}>
                    {category.count}
                  </span>
                </button>
              ))}
            </div>

            {/* Action Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {filteredActions.map((action) => {
                const IconComponent = action.icon;
                return (
                  <motion.button
                    key={action.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: Math.random() * 0.2 }}
                    onClick={() => {
                      action.action();
                      setIsExpanded(false);
                    }}
                    className="p-4 rounded-lg text-left transition-all btn-hover group"
                    style={{
                      background: theme.bg,
                      border: `1px solid ${theme.border}`,
                      minHeight: '120px'
                    }}
                  >
                    <div className="flex flex-col h-full">
                      <div className="flex items-center gap-3 mb-3">
                        <div
                          className="w-10 h-10 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform"
                          style={{ background: action.color + '20' }}
                        >
                          <IconComponent
                            className="w-5 h-5"
                            style={{ color: action.color }}
                          />
                        </div>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-semibold text-sm truncate" style={{ color: theme.text }}>
                            {action.label}
                          </h4>
                          <div className="text-xs px-2 py-1 rounded mt-1 inline-block"
                               style={{
                                 background: action.color + '15',
                                 color: action.color,
                                 border: `1px solid ${action.color}30`
                               }}>
                            {action.category}
                          </div>
                        </div>
                      </div>
                      <p className="text-xs leading-relaxed flex-1" style={{ color: theme.textSecondary }}>
                        {action.description}
                      </p>
                    </div>
                  </motion.button>
                );
              })}
            </div>

            {/* Quick Stats */}
            <div className="mt-6 pt-4 border-t" style={{ borderColor: theme.border }}>
              <div className="grid grid-cols-4 gap-4 text-center">
                <div>
                  <div className="text-lg font-mono font-bold" style={{ color: theme.accent }}>
                    {snap.activeJobs.length}
                  </div>
                  <div className="text-xs" style={{ color: theme.textSecondary }}>Active Jobs</div>
                </div>
                <div>
                  <div className="text-lg font-mono font-bold" style={{ color: theme.accent }}>
                    {snap.dataSources.length}
                  </div>
                  <div className="text-xs" style={{ color: theme.textSecondary }}>Data Sources</div>
                </div>
                <div>
                  <div className="text-lg font-mono font-bold" style={{ color: theme.accent }}>
                    {snap.swarmActivity.length}
                  </div>
                  <div className="text-xs" style={{ color: theme.textSecondary }}>Active Agents</div>
                </div>
                <div>
                  <div className="text-lg font-mono font-bold" style={{ color: theme.accent }}>
                    {store.realAgentMetrics?.totalAgentsDeployed || 0}
                  </div>
                  <div className="text-xs" style={{ color: theme.textSecondary }}>Total Deployed</div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle Button */}
      <motion.button
        onClick={() => setIsExpanded(!isExpanded)}
        className="px-6 py-3 rounded-full shadow-lg hover:scale-105 transition-all font-medium"
        style={{
          background: theme.accent,
          color: 'white',
          border: `2px solid ${theme.neon || theme.accent}`,
          boxShadow: `0 4px 20px ${theme.accent}40`
        }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4" />
          <span>Quick Actions</span>
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </motion.button>
    </div>
  );
}
