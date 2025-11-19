'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Target,
  Zap,
  TrendingUp,
  TrendingDown,
  Play,
  Pause,
  RotateCcw,
  BarChart3,
  Map,
  Clock,
  Users,
  Shield,
  AlertTriangle,
  CheckCircle,
  X,
  ChevronRight,
  ChevronDown
} from 'lucide-react';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/store';

export type COA = {
  id: string;
  name: string;
  description: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  successProbability: number;
  resourceRequirements: {
    personnel: number;
    equipment: string[];
    time: string;
  };
  objectives: string[];
  assumptions: string[];
  branches: COABranch[];
  status: 'draft' | 'ready' | 'executing' | 'completed';
};

export type COABranch = {
  id: string;
  name: string;
  probability: number;
  outcomes: string[];
  followOnActions: string[];
};

export type WargameResult = {
  coaId: string;
  scenario: string;
  iterations: number;
  successRate: number;
  averageDuration: string;
  keyFindings: string[];
  riskFactors: string[];
  recommendedCOA: string;
  timeline: WargameEvent[];
};

export type WargameEvent = {
  time: string;
  event: string;
  outcome: string;
  probability: number;
  coaImpacted: string[];
};

export default function COAWarGamePanel({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<'coa' | 'wargame' | 'comparison'>('coa');
  const [selectedCOA, setSelectedCOA] = useState<string | null>(null);
  const [isWargameRunning, setIsWargameRunning] = useState(false);
  const [wargameProgress, setWargameProgress] = useState(0);
  const snap = useSnapshot(store);

  // Mock COA data
  const [coas] = useState<COA[]>([
    {
      id: 'coa-1',
      name: 'Direct Engagement',
      description: 'Immediate interdiction of hostile submarine activity',
      riskLevel: 'high',
      successProbability: 0.65,
      resourceRequirements: {
        personnel: 45,
        equipment: ['ASW aircraft', 'Surface vessels', 'Sonar buoys'],
        time: '48 hours'
      },
      objectives: [
        'Locate and identify submarine',
        'Establish tracking network',
        'Execute interdiction if authorized'
      ],
      assumptions: [
        'Submarine is within detection range',
        'Weather conditions favorable',
        'Allied assets available'
      ],
      branches: [
        {
          id: 'branch-1',
          name: 'Successful Detection',
          probability: 0.7,
          outcomes: ['Submarine located', 'Tracking established'],
          followOnActions: ['Deploy additional sensors', 'Request authorization']
        },
        {
          id: 'branch-2',
          name: 'Detection Failure',
          probability: 0.3,
          outcomes: ['Submarine evades detection', 'Search area expanded'],
          followOnActions: ['Increase search radius', 'Deploy backup assets']
        }
      ],
      status: 'ready'
    },
    {
      id: 'coa-2',
      name: 'Passive Monitoring',
      description: 'Establish surveillance network without immediate engagement',
      riskLevel: 'medium',
      successProbability: 0.85,
      resourceRequirements: {
        personnel: 25,
        equipment: ['Sonar arrays', 'Satellite coverage', 'ISR assets'],
        time: '72 hours'
      },
      objectives: [
        'Establish passive surveillance',
        'Monitor submarine patterns',
        'Gather intelligence'
      ],
      assumptions: [
        'Submarine follows predictable patterns',
        'Satellite coverage available',
        'No immediate threat to assets'
      ],
      branches: [
        {
          id: 'branch-3',
          name: 'Pattern Established',
          probability: 0.8,
          outcomes: ['Movement patterns identified', 'Intelligence gathered'],
          followOnActions: ['Update threat assessment', 'Plan engagement timing']
        }
      ],
      status: 'ready'
    },
    {
      id: 'coa-3',
      name: 'Coordinated Multi-Asset Response',
      description: 'Deploy comprehensive multi-domain response',
      riskLevel: 'critical',
      successProbability: 0.45,
      resourceRequirements: {
        personnel: 120,
        equipment: ['Air assets', 'Surface vessels', 'Submarines', 'Cyber warfare units'],
        time: '96 hours'
      },
      objectives: [
        'Multi-domain coverage',
        'Cyber disruption',
        'Physical interdiction'
      ],
      assumptions: [
        'All assets available and coordinated',
        'Cyber warfare authorization granted',
        'Multi-national cooperation'
      ],
      branches: [
        {
          id: 'branch-4',
          name: 'Full Success',
          probability: 0.3,
          outcomes: ['Submarine neutralized', 'All domains coordinated'],
          followOnActions: ['Conduct BDA', 'Update ROE']
        },
        {
          id: 'branch-5',
          name: 'Partial Success',
          probability: 0.4,
          outcomes: ['Some objectives achieved', 'Cooperation maintained'],
          followOnActions: ['Reassess situation', 'Adjust strategy']
        }
      ],
      status: 'draft'
    }
  ]);

  // Mock wargame results
  const [wargameResults] = useState<WargameResult[]>([
    {
      coaId: 'coa-1',
      scenario: 'Submarine interdiction in contested waters',
      iterations: 1000,
      successRate: 0.65,
      averageDuration: '36 hours',
      keyFindings: [
        'Early detection critical for success',
        'Weather conditions significantly impact outcome',
        'Coordination between air and surface assets essential'
      ],
      riskFactors: [
        'Hostile countermeasures',
        'Poor weather conditions',
        'Asset availability delays'
      ],
      recommendedCOA: 'coa-1',
      timeline: [
        { time: 'T+0', event: 'Initial detection', outcome: 'Submarine located', probability: 0.8, coaImpacted: ['coa-1', 'coa-2'] },
        { time: 'T+6h', event: 'Asset deployment', outcome: 'ASW assets en route', probability: 0.9, coaImpacted: ['coa-1'] },
        { time: 'T+12h', event: 'First contact', outcome: 'Sonar contact established', probability: 0.6, coaImpacted: ['coa-1'] },
        { time: 'T+24h', event: 'Engagement decision', outcome: 'Interdiction authorized', probability: 0.7, coaImpacted: ['coa-1'] },
        { time: 'T+36h', event: 'Mission complete', outcome: 'Submarine neutralized', probability: 0.65, coaImpacted: ['coa-1'] }
      ]
    }
  ]);

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

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return '#4CAF50';
      case 'medium': return '#FFD700';
      case 'high': return '#FF8C00';
      case 'critical': return '#FF2B2B';
      default: return theme.accent;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return '#4CAF50';
      case 'executing': return '#FFD700';
      case 'completed': return '#00A39B';
      case 'draft': return '#888888';
      default: return theme.accent;
    }
  };

  const runWargame = async (coaId: string) => {
    setIsWargameRunning(true);
    setWargameProgress(0);

    // Simulate wargame execution
    for (let i = 0; i <= 100; i += 10) {
      await new Promise(resolve => setTimeout(resolve, 200));
      setWargameProgress(i);
    }

    setIsWargameRunning(false);
    setWargameProgress(100);
    setActiveTab('wargame');
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        style={{ background: 'rgba(0, 0, 0, 0.8)' }}
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="relative w-full max-w-6xl h-[90vh] rounded-lg overflow-hidden"
          style={{
            background: theme.cardBg,
            border: `1px solid ${theme.border}`
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b" style={{ borderColor: theme.border }}>
            <div className="flex items-center gap-3">
              <Target className="w-6 h-6" style={{ color: theme.accent }} />
              <h2 className="text-xl font-bold" style={{ color: theme.text }}>
                COA & Wargaming Analysis
              </h2>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg btn-hover"
              style={{ color: theme.text }}
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Tab Navigation */}
          <div className="flex border-b" style={{ borderColor: theme.border }}>
            {[
              { id: 'coa', label: 'Courses of Action', icon: Target },
              { id: 'wargame', label: 'Wargame Results', icon: Zap },
              { id: 'comparison', label: 'COA Comparison', icon: BarChart3 }
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as any)}
                className="flex-1 flex items-center justify-center gap-2 px-6 py-4 transition-all btn-hover"
                style={{
                  background: activeTab === id ? theme.accent : 'transparent',
                  color: activeTab === id ? 'white' : theme.text,
                  borderBottom: activeTab === id ? `2px solid ${theme.neon || theme.accent}` : 'none'
                }}
              >
                <Icon className="w-4 h-4" />
                <span className="font-medium">{label}</span>
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6">
            {activeTab === 'coa' && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {coas.map((coa) => (
                    <motion.div
                      key={coa.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="p-4 rounded-lg cursor-pointer transition-all btn-hover"
                      style={{
                        background: theme.cardBg,
                        border: `1px solid ${theme.border}`,
                        borderLeft: `4px solid ${getRiskColor(coa.riskLevel)}`
                      }}
                      onClick={() => setSelectedCOA(selectedCOA === coa.id ? null : coa.id)}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <h3 className="font-semibold mb-1" style={{ color: theme.text }}>
                            {coa.name}
                          </h3>
                          <div className="flex items-center gap-2 mb-2">
                            <span className="px-2 py-1 rounded text-xs font-medium"
                                  style={{ background: getRiskColor(coa.riskLevel), color: 'white' }}>
                              {coa.riskLevel.toUpperCase()} RISK
                            </span>
                            <span className="px-2 py-1 rounded text-xs font-medium"
                                  style={{ background: getStatusColor(coa.status), color: 'white' }}>
                              {coa.status.toUpperCase()}
                            </span>
                          </div>
                        </div>
                        {selectedCOA === coa.id ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                      </div>

                      <p className="text-sm mb-3" style={{ color: theme.textSecondary }}>
                        {coa.description}
                      </p>

                      <div className="flex items-center justify-between text-xs mb-3">
                        <span style={{ color: theme.textSecondary }}>Success Probability:</span>
                        <span className="font-mono font-bold" style={{ color: theme.accent }}>
                          {(coa.successProbability * 100).toFixed(0)}%
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-xs">
                        <span style={{ color: theme.textSecondary }}>Resources:</span>
                        <span style={{ color: theme.text }}>
                          {coa.resourceRequirements.personnel} personnel
                        </span>
                      </div>

                      {selectedCOA === coa.id && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          className="mt-4 pt-4 border-t"
                          style={{ borderColor: theme.border }}
                        >
                          <div className="space-y-4">
                            <div>
                              <h4 className="font-medium text-sm mb-2" style={{ color: theme.text }}>Objectives</h4>
                              <ul className="space-y-1">
                                {coa.objectives.map((objective, index) => (
                                  <li key={index} className="flex items-start gap-2 text-xs" style={{ color: theme.textSecondary }}>
                                    <CheckCircle className="w-3 h-3 mt-0.5 flex-shrink-0" style={{ color: theme.accent }} />
                                    {objective}
                                  </li>
                                ))}
                              </ul>
                            </div>

                            <div>
                              <h4 className="font-medium text-sm mb-2" style={{ color: theme.text }}>Branches & Sequels</h4>
                              <div className="space-y-2">
                                {coa.branches.map((branch) => (
                                  <div key={branch.id} className="p-3 rounded" style={{ background: theme.bg, border: `1px solid ${theme.border}` }}>
                                    <div className="flex items-center justify-between mb-2">
                                      <span className="font-medium text-sm" style={{ color: theme.text }}>{branch.name}</span>
                                      <span className="text-xs font-mono" style={{ color: theme.accent }}>
                                        {(branch.probability * 100).toFixed(0)}%
                                      </span>
                                    </div>
                                    <div className="space-y-1">
                                      <div className="text-xs" style={{ color: theme.textSecondary }}>
                                        <strong>Outcomes:</strong> {branch.outcomes.join(', ')}
                                      </div>
                                      <div className="text-xs" style={{ color: theme.textSecondary }}>
                                        <strong>Follow-on:</strong> {branch.followOnActions.join(', ')}
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>

                            <div className="flex gap-2 pt-3">
                              <button
                                onClick={() => runWargame(coa.id)}
                                disabled={isWargameRunning}
                                className="flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all btn-hover disabled:opacity-50"
                                style={{
                                  background: theme.accent,
                                  color: 'white'
                                }}
                              >
                                {isWargameRunning ? (
                                  <div className="flex items-center gap-2">
                                    <Zap className="w-4 h-4 animate-pulse" />
                                    Running Wargame...
                                  </div>
                                ) : (
                                  <div className="flex items-center gap-2">
                                    <Play className="w-4 h-4" />
                                    Run Wargame
                                  </div>
                                )}
                              </button>
                              {coa.status === 'ready' && (
                                <button
                                  className="px-4 py-2 rounded-lg text-sm font-medium transition-all btn-hover"
                                  style={{
                                    background: '#4CAF50',
                                    color: 'white'
                                  }}
                                >
                                  Execute COA
                                </button>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'wargame' && (
              <div className="space-y-6">
                {wargameResults.map((result) => (
                  <motion.div
                    key={result.coaId}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-6 rounded-lg"
                    style={{
                      background: theme.cardBg,
                      border: `1px solid ${theme.border}`
                    }}
                  >
                    <div className="flex items-center justify-between mb-6">
                      <div>
                        <h3 className="text-lg font-bold mb-1" style={{ color: theme.text }}>
                          Wargame Results: {coas.find(c => c.id === result.coaId)?.name}
                        </h3>
                        <p className="text-sm" style={{ color: theme.textSecondary }}>
                          {result.scenario}
                        </p>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-mono font-bold mb-1" style={{ color: theme.accent }}>
                          {(result.successRate * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs" style={{ color: theme.textSecondary }}>Success Rate</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                      <div className="p-4 rounded-lg text-center" style={{ background: theme.bg, border: `1px solid ${theme.border}` }}>
                        <div className="text-lg font-mono font-bold mb-1" style={{ color: theme.text }}>
                          {result.iterations.toLocaleString()}
                        </div>
                        <div className="text-xs" style={{ color: theme.textSecondary }}>Iterations</div>
                      </div>
                      <div className="p-4 rounded-lg text-center" style={{ background: theme.bg, border: `1px solid ${theme.border}` }}>
                        <div className="text-lg font-mono font-bold mb-1" style={{ color: theme.text }}>
                          {result.averageDuration}
                        </div>
                        <div className="text-xs" style={{ color: theme.textSecondary }}>Avg Duration</div>
                      </div>
                      <div className="p-4 rounded-lg text-center" style={{ background: theme.bg, border: `1px solid ${theme.border}` }}>
                        <div className="text-lg font-mono font-bold mb-1" style={{ color: theme.accent }}>
                          {result.recommendedCOA === result.coaId ? 'RECOMMENDED' : 'ALTERNATIVE'}
                        </div>
                        <div className="text-xs" style={{ color: theme.textSecondary }}>Assessment</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                      <div>
                        <h4 className="font-semibold mb-3" style={{ color: theme.text }}>Key Findings</h4>
                        <ul className="space-y-2">
                          {result.keyFindings.map((finding, index) => (
                            <li key={index} className="flex items-start gap-2 text-sm" style={{ color: theme.textSecondary }}>
                              <CheckCircle className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: '#4CAF50' }} />
                              {finding}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-semibold mb-3" style={{ color: theme.text }}>Risk Factors</h4>
                        <ul className="space-y-2">
                          {result.riskFactors.map((risk, index) => (
                            <li key={index} className="flex items-start gap-2 text-sm" style={{ color: theme.textSecondary }}>
                              <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: '#FF8C00' }} />
                              {risk}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-3" style={{ color: theme.text }}>Timeline Analysis</h4>
                      <div className="space-y-2">
                        {result.timeline.map((event, index) => (
                          <div key={index} className="flex items-center gap-4 p-3 rounded-lg" style={{ background: theme.bg, border: `1px solid ${theme.border}` }}>
                            <div className="font-mono text-sm font-bold min-w-12" style={{ color: theme.accent }}>
                              {event.time}
                            </div>
                            <div className="flex-1">
                              <div className="font-medium text-sm" style={{ color: theme.text }}>{event.event}</div>
                              <div className="text-xs" style={{ color: theme.textSecondary }}>{event.outcome}</div>
                            </div>
                            <div className="text-right">
                              <div className="text-sm font-mono font-bold" style={{ color: theme.neon || theme.accent }}>
                                {(event.probability * 100).toFixed(0)}%
                              </div>
                              <div className="text-xs" style={{ color: theme.textSecondary }}>Probability</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                ))}

                {wargameResults.length === 0 && (
                  <div className="text-center py-12">
                    <Zap className="w-16 h-16 mx-auto mb-4" style={{ color: theme.textSecondary, opacity: 0.5 }} />
                    <h3 className="text-lg font-medium mb-2" style={{ color: theme.text }}>No Wargame Results</h3>
                    <p className="text-sm" style={{ color: theme.textSecondary }}>
                      Run a wargame analysis on a COA to see results here.
                    </p>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'comparison' && (
              <div className="space-y-6">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${theme.border}` }}>
                        <th className="text-left p-3 font-semibold" style={{ color: theme.text }}>COA</th>
                        <th className="text-center p-3 font-semibold" style={{ color: theme.text }}>Success Rate</th>
                        <th className="text-center p-3 font-semibold" style={{ color: theme.text }}>Risk Level</th>
                        <th className="text-center p-3 font-semibold" style={{ color: theme.text }}>Resources</th>
                        <th className="text-center p-3 font-semibold" style={{ color: theme.text }}>Time</th>
                        <th className="text-center p-3 font-semibold" style={{ color: theme.text }}>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {coas.map((coa) => (
                        <tr key={coa.id} style={{ borderBottom: `1px solid ${theme.border}` }}>
                          <td className="p-3">
                            <div>
                              <div className="font-medium" style={{ color: theme.text }}>{coa.name}</div>
                              <div className="text-xs" style={{ color: theme.textSecondary }}>{coa.description}</div>
                            </div>
                          </td>
                          <td className="p-3 text-center">
                            <div className="font-mono font-bold text-lg" style={{ color: theme.accent }}>
                              {(coa.successProbability * 100).toFixed(0)}%
                            </div>
                          </td>
                          <td className="p-3 text-center">
                            <span className="px-2 py-1 rounded text-xs font-medium"
                                  style={{ background: getRiskColor(coa.riskLevel), color: 'white' }}>
                              {coa.riskLevel.toUpperCase()}
                            </span>
                          </td>
                          <td className="p-3 text-center">
                            <div className="text-sm" style={{ color: theme.text }}>
                              {coa.resourceRequirements.personnel}
                            </div>
                            <div className="text-xs" style={{ color: theme.textSecondary }}>
                              personnel
                            </div>
                          </td>
                          <td className="p-3 text-center">
                            <div className="text-sm" style={{ color: theme.text }}>
                              {coa.resourceRequirements.time}
                            </div>
                          </td>
                          <td className="p-3 text-center">
                            <div className="flex gap-1 justify-center">
                              <button
                                onClick={() => runWargame(coa.id)}
                                className="px-3 py-1 rounded text-xs font-medium transition-all btn-hover"
                                style={{
                                  background: theme.accent,
                                  color: 'white'
                                }}
                              >
                                Wargame
                              </button>
                              {coa.status === 'ready' && (
                                <button
                                  className="px-3 py-1 rounded text-xs font-medium transition-all btn-hover"
                                  style={{
                                    background: '#4CAF50',
                                    color: 'white'
                                  }}
                                >
                                  Execute
                                </button>
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Risk/Benefit Visualization */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="p-6 rounded-lg" style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}>
                    <h4 className="font-semibold mb-4" style={{ color: theme.text }}>Risk vs Success Probability</h4>
                    <div className="space-y-3">
                      {coas.map((coa) => (
                        <div key={coa.id} className="flex items-center gap-4">
                          <div className="w-24 text-sm" style={{ color: theme.text }}>{coa.name}</div>
                          <div className="flex-1 relative h-8">
                            <div
                              className="absolute top-0 left-0 h-full rounded"
                              style={{
                                width: `${coa.successProbability * 100}%`,
                                background: getRiskColor(coa.riskLevel),
                                opacity: 0.7
                              }}
                            />
                            <div className="absolute inset-0 flex items-center justify-center">
                              <span className="text-xs font-mono font-bold" style={{ color: theme.text }}>
                                {(coa.successProbability * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="p-6 rounded-lg" style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}>
                    <h4 className="font-semibold mb-4" style={{ color: theme.text }}>Resource Requirements</h4>
                    <div className="space-y-4">
                      {coas.map((coa) => (
                        <div key={coa.id} className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium" style={{ color: theme.text }}>{coa.name}</span>
                            <span className="text-sm font-mono" style={{ color: theme.accent }}>
                              {coa.resourceRequirements.personnel}
                            </span>
                          </div>
                          <div className="w-full bg-gray-700 rounded-full h-2">
                            <div
                              className="h-2 rounded-full"
                              style={{
                                width: `${(coa.resourceRequirements.personnel / 120) * 100}%`,
                                background: theme.accent
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Wargame Progress Overlay */}
          <AnimatePresence>
            {isWargameRunning && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 flex items-center justify-center"
                style={{
                  background: 'rgba(0, 0, 0, 0.8)',
                  backdropFilter: 'blur(4px)'
                }}
              >
                <div className="p-8 rounded-lg text-center" style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}>
                  <Zap className="w-12 h-12 mx-auto mb-4 animate-pulse" style={{ color: theme.accent }} />
                  <h3 className="text-lg font-bold mb-2" style={{ color: theme.text }}>Running Wargame Simulation</h3>
                  <p className="text-sm mb-4" style={{ color: theme.textSecondary }}>
                    Analyzing {coas.find(c => c.id === selectedCOA)?.name} across multiple scenarios...
                  </p>
                  <div className="w-64 mx-auto mb-4">
                    <div className="w-full rounded-full h-3" style={{ background: 'rgba(255,255,255,0.1)' }}>
                      <div
                        className="h-3 rounded-full transition-all duration-300"
                        style={{
                          width: `${wargameProgress}%`,
                          background: theme.accent
                        }}
                      />
                    </div>
                  </div>
                  <div className="text-sm font-mono" style={{ color: theme.accent }}>
                    {wargameProgress}% Complete
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
