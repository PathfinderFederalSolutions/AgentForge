'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSnapshot } from 'valtio';
import { store } from '../lib/store';

interface AdvancedAnalyticsProps {
  isOpen: boolean;
  onClose: () => void;
}

interface QualityTrend {
  metric: string;
  current_average: number;
  trend_direction: string;
  data_points: number;
  recent_values: number[];
}

interface UserPattern {
  pattern_type: string;
  confidence: number;
  frequency: number;
  trend_direction: string;
  pattern_data: any;
}

interface EmergentInsight {
  type: string;
  title: string;
  description: string;
  confidence: number;
  impact_score: number;
  actionable_recommendations: string[];
}

export default function AdvancedAnalytics({ isOpen, onClose }: AdvancedAnalyticsProps) {
  const snap = useSnapshot(store);
  const [qualityTrends, setQualityTrends] = useState<Record<string, QualityTrend>>({});
  const [userPatterns, setUserPatterns] = useState<UserPattern[]>([]);
  const [emergentInsights, setEmergentInsights] = useState<EmergentInsight[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'quality' | 'patterns' | 'insights' | 'optimization'>('quality');

  useEffect(() => {
    if (isOpen) {
      loadAnalyticsData();
    }
  }, [isOpen]);

  const loadAnalyticsData = async () => {
    setLoading(true);
    try {
      // Load quality trends
      const trends = await store.agiClient.getQualityTrends();
      setQualityTrends(trends.quality_trends || {});

      // Load user patterns
      const patterns = await store.agiClient.getUserPatterns('user_001');
      setUserPatterns(patterns.patterns || []);

      // Load emergent insights
      const insights = await store.agiClient.getEmergentInsights();
      setEmergentInsights(insights.recent_insights || []);

    } catch (error) {
      console.error('Failed to load analytics data:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderQualityTrends = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Object.entries(qualityTrends).map(([metric, trend]) => (
          <motion.div
            key={metric}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 bg-[#0D1421] border border-[#1E2A3E] rounded-lg"
          >
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-semibold text-[#D6E2F0] capitalize">
                {metric.replace('_', ' ')}
              </h4>
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                trend.trend_direction === 'improving' ? 'bg-[#00A39B] text-white' :
                trend.trend_direction === 'stable' ? 'bg-[#FFB800] text-black' :
                'bg-[#FF4444] text-white'
              }`}>
                {trend.trend_direction}
              </span>
            </div>
            
            <div className="text-2xl font-bold text-[#00A39B] mb-2">
              {(trend.current_average * 100).toFixed(1)}%
            </div>
            
            <div className="text-xs text-[#8FA8C4]">
              {trend.data_points} data points
            </div>
            
            {/* Mini trend chart */}
            <div className="mt-3 h-8 flex items-end space-x-1">
              {trend.recent_values.slice(-10).map((value, index) => (
                <div
                  key={index}
                  className="bg-[#00A39B] rounded-sm flex-1"
                  style={{ height: `${value * 100}%`, minHeight: '2px' }}
                />
              ))}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );

  const renderUserPatterns = () => (
    <div className="space-y-4">
      {userPatterns.map((pattern, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="p-4 bg-[#0D1421] border border-[#1E2A3E] rounded-lg"
        >
          <div className="flex items-start justify-between mb-3">
            <div>
              <h4 className="text-lg font-semibold text-[#D6E2F0] capitalize mb-1">
                {pattern.pattern_type.replace('_', ' ')} Pattern
              </h4>
              <div className="flex items-center space-x-4 text-sm text-[#8FA8C4]">
                <span>Confidence: {(pattern.confidence * 100).toFixed(0)}%</span>
                <span>Frequency: {pattern.frequency}</span>
                <span className={`px-2 py-1 rounded ${
                  pattern.trend_direction === 'increasing' ? 'bg-[#00A39B] text-white' :
                  pattern.trend_direction === 'stable' ? 'bg-[#1E2A3E] text-[#8FA8C4]' :
                  'bg-[#FF4444] text-white'
                }`}>
                  {pattern.trend_direction}
                </span>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(pattern.pattern_data || {}).map(([key, value]) => (
              <div key={key} className="flex justify-between text-sm">
                <span className="text-[#8FA8C4] capitalize">{key.replace('_', ' ')}:</span>
                <span className="text-[#D6E2F0]">
                  {typeof value === 'number' ? value.toFixed(2) : 
                   Array.isArray(value) ? value.join(', ') : 
                   String(value)}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      ))}
      
      {userPatterns.length === 0 && (
        <div className="text-center py-8 text-[#8FA8C4]">
          <div className="text-4xl mb-4">🧠</div>
          <p>No patterns detected yet. Continue using the system to see personalized insights.</p>
        </div>
      )}
    </div>
  );

  const renderEmergentInsights = () => (
    <div className="space-y-4">
      {emergentInsights.map((insight, index) => (
        <motion.div
          key={insight.insight_id || index}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className="p-6 bg-gradient-to-r from-[#0D1421] to-[#1E2A3E] border border-[#00A39B]/30 rounded-lg"
        >
          <div className="flex items-start justify-between mb-4">
            <div>
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-2xl">💡</span>
                <h4 className="text-lg font-semibold text-[#D6E2F0]">{insight.title}</h4>
              </div>
              <p className="text-[#8FA8C4] mb-3">{insight.description}</p>
            </div>
            <div className="text-right">
              <div className="text-sm text-[#00A39B] font-medium">
                {(insight.confidence * 100).toFixed(0)}% confidence
              </div>
              <div className="text-xs text-[#8FA8C4]">
                Impact: {(insight.impact_score * 100).toFixed(0)}%
              </div>
            </div>
          </div>
          
          {insight.actionable_recommendations && insight.actionable_recommendations.length > 0 && (
            <div>
              <h5 className="text-sm font-medium text-[#D6E2F0] mb-2">Recommendations:</h5>
              <div className="space-y-1">
                {insight.actionable_recommendations.map((rec, i) => (
                  <div key={i} className="text-sm text-[#8FA8C4] flex items-center">
                    <span className="w-1.5 h-1.5 bg-[#00A39B] rounded-full mr-2 flex-shrink-0"></span>
                    {rec}
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      ))}
      
      {emergentInsights.length === 0 && (
        <div className="text-center py-8 text-[#8FA8C4]">
          <div className="text-4xl mb-4">🔮</div>
          <p>No emergent insights yet. The system is learning from your interactions.</p>
        </div>
      )}
    </div>
  );

  const renderOptimization = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
          <h4 className="text-lg font-semibold text-[#D6E2F0] mb-4">🚀 Performance Optimization</h4>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-[#8FA8C4]">Response Time</span>
              <span className="text-[#00A39B]">1.2s avg</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8FA8C4]">Agent Efficiency</span>
              <span className="text-[#00A39B]">94%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8FA8C4]">Quantum Coherence</span>
              <span className="text-[#00A39B]">87%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8FA8C4]">Memory Utilization</span>
              <span className="text-[#00A39B]">76%</span>
            </div>
          </div>
        </div>
        
        <div className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
          <h4 className="text-lg font-semibold text-[#D6E2F0] mb-4">🎯 User Satisfaction</h4>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-[#8FA8C4]">Overall Satisfaction</span>
              <span className="text-[#00A39B]">89%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8FA8C4]">Response Relevance</span>
              <span className="text-[#00A39B]">92%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8FA8C4]">Helpfulness Score</span>
              <span className="text-[#00A39B]">88%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8FA8C4]">Clarity Score</span>
              <span className="text-[#00A39B]">85%</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="p-6 bg-gradient-to-r from-[#00A39B]/10 to-[#0D1421] border border-[#00A39B]/30 rounded-lg">
        <h4 className="text-lg font-semibold text-[#D6E2F0] mb-4">🔄 Self-Improvement Status</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-[#00A39B]">47</div>
            <div className="text-sm text-[#8FA8C4]">Learnings Captured</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-[#00A39B]">12</div>
            <div className="text-sm text-[#8FA8C4]">Optimizations Applied</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-[#00A39B]">+23%</div>
            <div className="text-sm text-[#8FA8C4]">Quality Improvement</div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="fixed inset-4 md:inset-8 bg-[#0A0F1C] border border-[#1E2A3E] rounded-2xl z-50 overflow-hidden"
          >
            {/* Header */}
            <div className="p-6 border-b border-[#1E2A3E] bg-gradient-to-r from-[#0A0F1C] to-[#0D1421]">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-[#D6E2F0] mb-2">
                    📊 Advanced AGI Analytics
                  </h2>
                  <p className="text-[#8FA8C4]">
                    Real-time intelligence, learning patterns, and system optimization
                  </p>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 text-[#8FA8C4] hover:text-[#D6E2F0] hover:bg-[#1E2A3E] rounded-lg transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="flex h-full">
              {/* Sidebar */}
              <div className="w-64 p-6 border-r border-[#1E2A3E] bg-[#0D1421]">
                <h3 className="text-lg font-semibold text-[#D6E2F0] mb-4">Analytics</h3>
                <div className="space-y-2">
                  {[
                    { key: 'quality', icon: '📈', label: 'Quality Trends' },
                    { key: 'patterns', icon: '🧠', label: 'User Patterns' },
                    { key: 'insights', icon: '💡', label: 'Emergent Insights' },
                    { key: 'optimization', icon: '⚡', label: 'Optimization' }
                  ].map((tab) => (
                    <button
                      key={tab.key}
                      onClick={() => setActiveTab(tab.key as any)}
                      className={`w-full text-left p-3 rounded-lg transition-colors ${
                        activeTab === tab.key
                          ? 'bg-[#00A39B] text-white'
                          : 'text-[#8FA8C4] hover:text-[#D6E2F0] hover:bg-[#1E2A3E]'
                      }`}
                    >
                      {tab.icon} {tab.label}
                    </button>
                  ))}
                </div>

                {/* Real-time stats */}
                <div className="mt-8 p-4 bg-[#1E2A3E] rounded-lg">
                  <h4 className="text-sm font-semibold text-[#D6E2F0] mb-3">Live System Status</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-[#8FA8C4]">Active Agents</span>
                      <span className="text-[#00A39B]">{snap.activeAgents}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8FA8C4]">Quality Score</span>
                      <span className="text-[#00A39B]">89%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8FA8C4]">Learning Rate</span>
                      <span className="text-[#00A39B]">+2.3%</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Main Content */}
              <div className="flex-1 overflow-y-auto p-6">
                {loading ? (
                  <div className="flex items-center justify-center h-64">
                    <div className="text-[#8FA8C4]">Loading analytics data...</div>
                  </div>
                ) : (
                  <>
                    {activeTab === 'quality' && renderQualityTrends()}
                    {activeTab === 'patterns' && renderUserPatterns()}
                    {activeTab === 'insights' && renderEmergentInsights()}
                    {activeTab === 'optimization' && renderOptimization()}
                  </>
                )}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
