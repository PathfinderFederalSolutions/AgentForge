"use client";

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Shield,
  AlertTriangle,
  TrendingUp,
  Activity,
  Target,
  Zap,
  Globe,
  Radar,
  X,
  ChevronRight,
  Info
} from 'lucide-react';

interface IntelligenceDashboardProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ThreatData {
  id: string;
  name: string;
  level: 'CRITICAL' | 'HIGH' | 'ELEVATED' | 'MODERATE' | 'LOW';
  confidence: number;
  domain: string;
  firstDetected: number;
  lastUpdated: number;
  indicators: string[];
}

interface IntelligenceMetrics {
  totalInjects: number;
  totalFusions: number;
  ttpDetections: number;
  campaignsDetected: number;
  activeStreams: number;
  processingRate: number;
  avgLatency: number;
  overallConfidence: number;
}

export default function IntelligenceDashboard({ isOpen, onClose }: IntelligenceDashboardProps) {
  const [activeThreats, setActiveThreats] = useState<ThreatData[]>([]);
  const [metrics, setMetrics] = useState<IntelligenceMetrics>({
    totalInjects: 0,
    totalFusions: 0,
    ttpDetections: 0,
    campaignsDetected: 0,
    activeStreams: 0,
    processingRate: 0,
    avgLatency: 0,
    overallConfidence: 0
  });
  const [selectedThreat, setSelectedThreat] = useState<ThreatData | null>(null);

  useEffect(() => {
    if (!isOpen) return;

    // Fetch active threats and metrics
    const fetchIntelligence = async () => {
      try {
        // Fetch active threats
        const threatsResponse = await fetch('http://localhost:8001/v1/intelligence/continuous/threats/active');
        if (threatsResponse.ok) {
          const threatsData = await threatsResponse.json();
          setActiveThreats(threatsData.threats || []);
        }

        // Fetch continuous state metrics
        const metricsResponse = await fetch('http://localhost:8001/v1/intelligence/continuous/state');
        if (metricsResponse.ok) {
          const metricsData = await metricsResponse.json();
          setMetrics({
            totalInjects: metricsData.total_injects_processed || 0,
            totalFusions: metricsData.total_fusions || 0,
            ttpDetections: metricsData.total_ttp_detections || 0,
            campaignsDetected: metricsData.total_campaigns_detected || 0,
            activeStreams: metricsData.active_streams || 0,
            processingRate: metricsData.processing_rate || 0,
            avgLatency: metricsData.avg_latency || 0,
            overallConfidence: 0.87 // Would come from latest analysis
          });
        }
      } catch (error) {
        console.error('Failed to fetch intelligence data:', error);
      }
    };

    fetchIntelligence();
    const interval = setInterval(fetchIntelligence, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [isOpen]);

  const getThreatColor = (level: string) => {
    switch (level) {
      case 'CRITICAL': return '#FF2B2B';
      case 'HIGH': return '#FF8C00';
      case 'ELEVATED': return '#FFD700';
      case 'MODERATE': return '#00A39B';
      case 'LOW': return '#4CAF50';
      default: return '#888';
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        style={{
          background: 'rgba(0,0,0,0.85)',
          backdropFilter: 'blur(8px)'
        }}
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="relative w-full max-w-6xl max-h-[90vh] overflow-hidden rounded-2xl"
          style={{
            background: '#05080D',
            border: '1px solid rgba(0, 163, 155, 0.3)',
            boxShadow: '0 0 40px rgba(0, 163, 155, 0.2)'
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div 
            className="flex items-center justify-between p-6 border-b"
            style={{ borderColor: 'rgba(0, 163, 155, 0.2)' }}
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-[#00A39B] to-[#CCFF00] flex items-center justify-center">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">Intelligence Dashboard</h2>
                <p className="text-sm text-gray-400">Real-Time Threat Monitoring & Analysis</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/10 transition-colors"
            >
              <X className="w-6 h-6 text-gray-400" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto" style={{ maxHeight: 'calc(90vh - 100px)' }}>
            {/* Metrics Grid */}
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Total Injects</span>
                  <Activity className="w-4 h-4 text-[#00A39B]" />
                </div>
                <div className="text-2xl font-bold text-white">{metrics.totalInjects}</div>
                <div className="text-xs text-gray-500 mt-1">
                  {metrics.processingRate.toFixed(1)}/s
                </div>
              </div>

              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">TTP Detections</span>
                  <Target className="w-4 h-4 text-orange-500" />
                </div>
                <div className="text-2xl font-bold text-white">{metrics.ttpDetections}</div>
                <div className="text-xs text-gray-500 mt-1">
                  {metrics.campaignsDetected} campaigns
                </div>
              </div>

              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Confidence</span>
                  <TrendingUp className="w-4 h-4 text-green-500" />
                </div>
                <div className="text-2xl font-bold text-white">
                  {(metrics.overallConfidence * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {metrics.avgLatency.toFixed(2)}s latency
                </div>
              </div>

              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Active Streams</span>
                  <Zap className="w-4 h-4 text-yellow-500" />
                </div>
                <div className="text-2xl font-bold text-white">{metrics.activeStreams}</div>
                <div className="text-xs text-gray-500 mt-1">
                  {metrics.totalFusions} fusions
                </div>
              </div>
            </div>

            {/* Active Threats */}
            <div className="mb-6">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-orange-500" />
                Active Threats ({activeThreats.length})
              </h3>

              {activeThreats.length === 0 ? (
                <div className="bg-white/5 rounded-lg p-6 border border-white/10 text-center">
                  <Shield className="w-12 h-12 text-green-500 mx-auto mb-3" />
                  <p className="text-gray-400">No active threats detected</p>
                  <p className="text-sm text-gray-500 mt-1">System monitoring all intelligence streams</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {activeThreats.map((threat) => (
                    <motion.div
                      key={threat.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="bg-white/5 rounded-lg p-4 border cursor-pointer hover:bg-white/10 transition-colors"
                      style={{ borderColor: `${getThreatColor(threat.level)}40` }}
                      onClick={() => setSelectedThreat(threat)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <div 
                              className="px-2 py-1 rounded text-xs font-bold"
                              style={{ 
                                background: `${getThreatColor(threat.level)}20`,
                                color: getThreatColor(threat.level)
                              }}
                            >
                              {threat.level}
                            </div>
                            <h4 className="text-white font-semibold">{threat.name}</h4>
                          </div>
                          <div className="flex items-center gap-4 text-sm">
                            <div className="flex items-center gap-1">
                              <Globe className="w-4 h-4 text-gray-400" />
                              <span className="text-gray-400">{threat.domain}</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Target className="w-4 h-4 text-gray-400" />
                              <span className="text-gray-400">{(threat.confidence * 100).toFixed(0)}% confidence</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Radar className="w-4 h-4 text-gray-400" />
                              <span className="text-gray-400">{threat.indicators?.length || 0} indicators</span>
                            </div>
                          </div>
                        </div>
                        <ChevronRight className="w-5 h-5 text-gray-400" />
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>

            {/* System Status */}
            <div>
              <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-[#00A39B]" />
                System Status
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm text-gray-400">Intelligence Processing</span>
                    <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Agent Specialization</span>
                      <span className="text-green-400">Operational</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Multi-Domain Fusion</span>
                      <span className="text-green-400">Active</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">TTP Recognition</span>
                      <span className="text-green-400">27 patterns</span>
                    </div>
                  </div>
                </div>

                <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm text-gray-400">Advanced Features</span>
                    <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">COA Generation</span>
                      <span className="text-green-400">Ready</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Wargaming</span>
                      <span className="text-green-400">Ready</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Self-Healing</span>
                      <span className="text-green-400">Active</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Threat Detail Sidebar */}
          <AnimatePresence>
            {selectedThreat && (
              <motion.div
                initial={{ x: '100%' }}
                animate={{ x: 0 }}
                exit={{ x: '100%' }}
                className="absolute top-0 right-0 w-1/3 h-full bg-[#0A0F1A] border-l"
                style={{ borderColor: 'rgba(0, 163, 155, 0.3)' }}
              >
                <div className="p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-bold text-white">Threat Details</h3>
                    <button
                      onClick={() => setSelectedThreat(null)}
                      className="p-2 rounded-lg hover:bg-white/10"
                    >
                      <X className="w-5 h-5 text-gray-400" />
                    </button>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <label className="text-sm text-gray-400 mb-1 block">Threat Name</label>
                      <p className="text-white font-semibold">{selectedThreat.name}</p>
                    </div>

                    <div>
                      <label className="text-sm text-gray-400 mb-1 block">Threat Level</label>
                      <div 
                        className="inline-block px-3 py-1 rounded text-sm font-bold"
                        style={{ 
                          background: `${getThreatColor(selectedThreat.level)}20`,
                          color: getThreatColor(selectedThreat.level)
                        }}
                      >
                        {selectedThreat.level}
                      </div>
                    </div>

                    <div>
                      <label className="text-sm text-gray-400 mb-1 block">Confidence</label>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-white/10 rounded-full h-2">
                          <div 
                            className="h-full rounded-full bg-gradient-to-r from-[#00A39B] to-[#CCFF00]"
                            style={{ width: `${selectedThreat.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-white font-semibold">
                          {(selectedThreat.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>

                    <div>
                      <label className="text-sm text-gray-400 mb-1 block">Domain</label>
                      <p className="text-white">{selectedThreat.domain}</p>
                    </div>

                    <div>
                      <label className="text-sm text-gray-400 mb-1 block">Indicators</label>
                      <div className="space-y-1">
                        {selectedThreat.indicators?.map((indicator, idx) => (
                          <div 
                            key={idx}
                            className="text-sm px-2 py-1 rounded bg-white/5 text-gray-300"
                          >
                            • {indicator}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="pt-4 border-t" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>
                      <button className="w-full bg-gradient-to-r from-[#00A39B] to-[#00857F] text-white px-4 py-2 rounded-lg font-semibold hover:opacity-90 transition-opacity">
                        Generate Response COAs
                      </button>
                    </div>
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

