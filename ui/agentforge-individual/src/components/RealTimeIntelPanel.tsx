'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  Shield,
  Eye,
  TrendingUp,
  TrendingDown,
  Zap,
  MapPin,
  Clock,
  Users,
  Target,
  Bell,
  X,
  Filter,
  Play,
  Pause,
  RefreshCw
} from 'lucide-react';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/store';

export type IntelligenceEvent = {
  id: string;
  timestamp: Date;
  type: 'threat' | 'pattern' | 'campaign' | 'anomaly' | 'cascade';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  source: string;
  location?: {
    lat: number;
    lng: number;
    name: string;
  };
  intelligenceDomain: 'SIGINT' | 'CYBINT' | 'HUMINT' | 'GEOINT' | 'OSINT' | 'TECHINT';
  confidence: number;
  tags: string[];
  relatedEvents: string[];
  actions: IntelligenceAction[];
};

export type IntelligenceAction = {
  id: string;
  type: 'investigate' | 'alert' | 'respond' | 'monitor' | 'escalate';
  label: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  automated: boolean;
};

export type IntelligencePattern = {
  id: string;
  name: string;
  description: string;
  confidence: number;
  events: string[];
  trend: 'increasing' | 'stable' | 'decreasing';
  lastDetected: Date;
  severity: 'low' | 'medium' | 'high' | 'critical';
};

export default function RealTimeIntelPanel({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<'feed' | 'patterns' | 'campaigns' | 'alerts'>('feed');
  const [isLive, setIsLive] = useState(true);
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [filterDomain, setFilterDomain] = useState<string>('all');
  const snap = useSnapshot(store);

  // Mock intelligence events data
  const [intelligenceEvents] = useState<IntelligenceEvent[]>([
    {
      id: 'event-1',
      timestamp: new Date(Date.now() - 5 * 60 * 1000), // 5 minutes ago
      type: 'threat',
      severity: 'high',
      title: 'Submarine Detection - Pacific Sector',
      description: 'SIGINT sensors detected anomalous acoustic signatures consistent with submarine propulsion systems in designated patrol area.',
      source: 'Underwater Sensor Network Alpha',
      location: { lat: 35.6762, lng: 139.6503, name: 'Pacific Ocean Sector 7' },
      intelligenceDomain: 'SIGINT',
      confidence: 0.85,
      tags: ['submarine', 'acoustic', 'threat', 'patrol'],
      relatedEvents: ['event-3'],
      actions: [
        { id: 'action-1', type: 'investigate', label: 'Deploy ASW assets', priority: 'high', automated: false },
        { id: 'action-2', type: 'alert', label: 'Notify command', priority: 'high', automated: true },
        { id: 'action-3', type: 'monitor', label: 'Increase surveillance', priority: 'medium', automated: true }
      ]
    },
    {
      id: 'event-2',
      timestamp: new Date(Date.now() - 12 * 60 * 1000), // 12 minutes ago
      type: 'pattern',
      severity: 'medium',
      title: 'Cyber Reconnaissance Pattern Detected',
      description: 'Pattern recognition identified coordinated scanning activity targeting defense contractor networks.',
      source: 'Network IDS Cluster Beta',
      intelligenceDomain: 'CYBINT',
      confidence: 0.72,
      tags: ['cyber', 'reconnaissance', 'scanning', 'coordinated'],
      relatedEvents: [],
      actions: [
        { id: 'action-4', type: 'monitor', label: 'Track IP ranges', priority: 'medium', automated: true },
        { id: 'action-5', type: 'investigate', label: 'Analyze malware samples', priority: 'medium', automated: false }
      ]
    },
    {
      id: 'event-3',
      timestamp: new Date(Date.now() - 18 * 60 * 1000), // 18 minutes ago
      type: 'campaign',
      severity: 'critical',
      title: 'Multi-Domain Campaign Indicators',
      description: 'Correlated intelligence indicates coordinated multi-domain campaign involving maritime and cyber operations.',
      source: 'Intelligence Fusion Engine',
      intelligenceDomain: 'MULTI_DOMAIN',
      confidence: 0.91,
      tags: ['campaign', 'multi-domain', 'coordinated', 'maritime', 'cyber'],
      relatedEvents: ['event-1', 'event-2'],
      actions: [
        { id: 'action-6', type: 'escalate', label: 'Activate defense protocols', priority: 'critical', automated: true },
        { id: 'action-7', type: 'respond', label: 'Deploy countermeasures', priority: 'critical', automated: false }
      ]
    },
    {
      id: 'event-4',
      timestamp: new Date(Date.now() - 25 * 60 * 1000), // 25 minutes ago
      type: 'anomaly',
      severity: 'low',
      title: 'Unusual Satellite Activity',
      description: 'GEOINT analysis detected unusual satellite positioning patterns over restricted airspace.',
      source: 'Satellite Imagery Processor',
      location: { lat: 38.9072, lng: -77.0369, name: 'Washington DC Area' },
      intelligenceDomain: 'GEOINT',
      confidence: 0.68,
      tags: ['satellite', 'geoint', 'anomaly', 'airspace'],
      relatedEvents: [],
      actions: [
        { id: 'action-8', type: 'monitor', label: 'Track satellite path', priority: 'low', automated: true }
      ]
    }
  ]);

  // Mock patterns data
  const [patterns] = useState<IntelligencePattern[]>([
    {
      id: 'pattern-1',
      name: 'Submarine Patrol Pattern Alpha',
      description: 'Recurring submarine patrol pattern in Pacific sector with 72-hour cycle',
      confidence: 0.88,
      events: ['event-1', 'event-3'],
      trend: 'stable',
      lastDetected: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
      severity: 'high'
    },
    {
      id: 'pattern-2',
      name: 'Cyber Reconnaissance Campaign',
      description: 'Coordinated cyber reconnaissance targeting defense infrastructure',
      confidence: 0.76,
      events: ['event-2'],
      trend: 'increasing',
      lastDetected: new Date(Date.now() - 45 * 60 * 1000), // 45 minutes ago
      severity: 'medium'
    }
  ]);

  // Mock campaigns data
  const [campaigns] = useState<any[]>([
    {
      id: 'campaign-1',
      name: 'Operation Shadow Fleet',
      description: 'Coordinated submarine and cyber operations targeting naval assets',
      startDate: new Date('2024-10-15'),
      status: 'active',
      severity: 'critical',
      eventsCount: 15,
      domains: ['SIGINT', 'CYBINT', 'GEOINT'],
      lastActivity: new Date(Date.now() - 5 * 60 * 1000)
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

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#FF2B2B';
      case 'high': return '#FF8C00';
      case 'medium': return '#FFD700';
      case 'low': return '#4CAF50';
      default: return theme.accent;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'threat': return AlertTriangle;
      case 'pattern': return TrendingUp;
      case 'campaign': return Target;
      case 'anomaly': return Eye;
      case 'cascade': return Zap;
      default: return Activity;
    }
  };

  const getDomainColor = (domain: string) => {
    switch (domain) {
      case 'SIGINT': return '#00A39B';
      case 'CYBINT': return '#FF6B6B';
      case 'HUMINT': return '#FFD700';
      case 'GEOINT': return '#4CAF50';
      case 'OSINT': return '#9C27B0';
      case 'TECHINT': return '#FF9800';
      case 'MULTI_DOMAIN': return '#E91E63';
      default: return theme.accent;
    }
  };

  const filteredEvents = intelligenceEvents.filter(event => {
    if (filterSeverity !== 'all' && event.severity !== filterSeverity) return false;
    if (filterDomain !== 'all' && event.intelligenceDomain !== filterDomain) return false;
    return true;
  });

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: '100%' }}
        animate={{ x: 0 }}
        exit={{ x: '100%' }}
        transition={{ type: 'tween', duration: 0.3 }}
        style={{
          position: 'fixed',
          right: '0',
          top: '70px',
          width: '420px',
          background: theme.cardBg,
          backdropFilter: 'blur(20px)',
          borderLeft: `1px solid ${theme.border}`,
          height: 'calc(100vh - 70px)',
          zIndex: 45,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          boxShadow: `-8px 0 40px rgba(0,0,0,0.3)`
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b" style={{ borderColor: theme.border }}>
          <div className="flex items-center gap-3">
            <Activity className="w-5 h-5" style={{ color: theme.accent }} />
            <h2 className="font-bold text-lg" style={{ color: theme.text }}>
              Real-Time Intelligence
            </h2>
            <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`} />
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsLive(!isLive)}
              className="p-2 rounded-lg btn-hover"
              style={{ color: theme.text }}
              title={isLive ? 'Pause live feed' : 'Resume live feed'}
            >
              {isLive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg btn-hover"
              style={{ color: theme.text }}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b" style={{ borderColor: theme.border }}>
          {[
            { id: 'feed', label: 'Live Feed', count: filteredEvents.length },
            { id: 'patterns', label: 'Patterns', count: patterns.length },
            { id: 'campaigns', label: 'Campaigns', count: campaigns.length },
            { id: 'alerts', label: 'Alerts', count: intelligenceEvents.filter(e => e.severity === 'critical').length }
          ].map(({ id, label, count }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-3 text-sm font-medium transition-all btn-hover"
              style={{
                background: activeTab === id ? theme.accent : 'transparent',
                color: activeTab === id ? 'white' : theme.text,
                borderBottom: activeTab === id ? `2px solid ${theme.neon || theme.accent}` : 'none'
              }}
            >
              {label}
              {count > 0 && (
                <span className="px-2 py-1 rounded-full text-xs"
                      style={{
                        background: activeTab === id ? 'rgba(255,255,255,0.2)' : getSeverityColor('medium'),
                        color: activeTab === id ? 'white' : 'white'
                      }}>
                  {count}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Filters */}
        {activeTab === 'feed' && (
          <div className="p-3 border-b" style={{ borderColor: theme.border }}>
            <div className="flex gap-2">
              <select
                value={filterSeverity}
                onChange={(e) => setFilterSeverity(e.target.value)}
                className="flex-1 px-3 py-2 rounded-lg text-sm"
                style={{
                  background: theme.bg,
                  color: theme.text,
                  border: `1px solid ${theme.border}`
                }}
              >
                <option value="all">All Severities</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
              <select
                value={filterDomain}
                onChange={(e) => setFilterDomain(e.target.value)}
                className="flex-1 px-3 py-2 rounded-lg text-sm"
                style={{
                  background: theme.bg,
                  color: theme.text,
                  border: `1px solid ${theme.border}`
                }}
              >
                <option value="all">All Domains</option>
                <option value="SIGINT">SIGINT</option>
                <option value="CYBINT">CYBINT</option>
                <option value="HUMINT">HUMINT</option>
                <option value="GEOINT">GEOINT</option>
                <option value="OSINT">OSINT</option>
                <option value="TECHINT">TECHINT</option>
                <option value="MULTI_DOMAIN">Multi-Domain</option>
              </select>
            </div>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {activeTab === 'feed' && (
            <div className="p-4 space-y-3">
              {filteredEvents.map((event) => {
                const TypeIcon = getTypeIcon(event.type);
                return (
                  <motion.div
                    key={event.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-lg cursor-pointer transition-all btn-hover"
                    style={{
                      background: theme.cardBg,
                      border: `1px solid ${theme.border}`,
                      borderLeft: `4px solid ${getSeverityColor(event.severity)}`
                    }}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <TypeIcon className="w-4 h-4" style={{ color: getSeverityColor(event.severity) }} />
                        <span className="text-sm font-medium" style={{ color: theme.text }}>
                          {event.title}
                        </span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="px-2 py-1 rounded text-xs font-medium"
                              style={{
                                background: getDomainColor(event.intelligenceDomain),
                                color: 'white'
                              }}>
                          {event.intelligenceDomain}
                        </span>
                      </div>
                    </div>

                    <p className="text-sm mb-3" style={{ color: theme.textSecondary, lineHeight: '1.4' }}>
                      {event.description}
                    </p>

                    <div className="flex items-center justify-between text-xs mb-3">
                      <div className="flex items-center gap-3">
                        <span style={{ color: theme.textSecondary }}>
                          {event.source}
                        </span>
                        {event.location && (
                          <div className="flex items-center gap-1">
                            <MapPin className="w-3 h-3" style={{ color: theme.accent }} />
                            <span style={{ color: theme.accent }}>{event.location.name}</span>
                          </div>
                        )}
                      </div>
                      <span style={{ color: theme.textSecondary }}>
                        {formatTimeAgo(event.timestamp)}
                      </span>
                    </div>

                    {/* Tags */}
                    {event.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mb-3">
                        {event.tags.map((tag) => (
                          <span
                            key={tag}
                            className="px-2 py-1 rounded text-xs"
                            style={{
                              background: `${theme.accent}15`,
                              color: theme.accent,
                              border: `1px solid ${theme.accent}30`
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Confidence and Actions */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-xs" style={{ color: theme.textSecondary }}>Confidence:</span>
                        <span className="text-xs font-mono font-bold" style={{ color: theme.accent }}>
                          {(event.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex gap-1">
                        {event.actions.slice(0, 2).map((action) => (
                          <button
                            key={action.id}
                            className="px-2 py-1 rounded text-xs font-medium transition-all btn-hover"
                            style={{
                              background: getSeverityColor(action.priority),
                              color: 'white'
                            }}
                          >
                            {action.label}
                          </button>
                        ))}
                        {event.actions.length > 2 && (
                          <span className="px-2 py-1 rounded text-xs" style={{ color: theme.textSecondary }}>
                            +{event.actions.length - 2}
                          </span>
                        )}
                      </div>
                    </div>
                  </motion.div>
                );
              })}

              {filteredEvents.length === 0 && (
                <div className="text-center py-8">
                  <Activity className="w-12 h-12 mx-auto mb-3" style={{ color: theme.textSecondary, opacity: 0.5 }} />
                  <p className="text-sm" style={{ color: theme.textSecondary }}>
                    No intelligence events match current filters
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'patterns' && (
            <div className="p-4 space-y-3">
              {patterns.map((pattern) => (
                <motion.div
                  key={pattern.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 rounded-lg"
                  style={{
                    background: theme.cardBg,
                    border: `1px solid ${theme.border}`,
                    borderLeft: `4px solid ${getSeverityColor(pattern.severity)}`
                  }}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <h4 className="font-medium text-sm mb-1" style={{ color: theme.text }}>
                        {pattern.name}
                      </h4>
                      <p className="text-xs" style={{ color: theme.textSecondary }}>
                        {pattern.description}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      {pattern.trend === 'increasing' && <TrendingUp className="w-4 h-4 text-red-400" />}
                      {pattern.trend === 'decreasing' && <TrendingDown className="w-4 h-4 text-green-400" />}
                      {pattern.trend === 'stable' && <div className="w-4 h-1 bg-yellow-400 rounded" />}
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-xs mt-3">
                    <div className="flex items-center gap-3">
                      <span style={{ color: theme.textSecondary }}>
                        {pattern.events.length} events
                      </span>
                      <span style={{ color: theme.textSecondary }}>
                        {formatTimeAgo(pattern.lastDetected)}
                      </span>
                    </div>
                    <span className="font-mono font-bold" style={{ color: theme.accent }}>
                      {(pattern.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </motion.div>
              ))}

              {patterns.length === 0 && (
                <div className="text-center py-8">
                  <TrendingUp className="w-12 h-12 mx-auto mb-3" style={{ color: theme.textSecondary, opacity: 0.5 }} />
                  <p className="text-sm" style={{ color: theme.textSecondary }}>
                    No patterns detected
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'campaigns' && (
            <div className="p-4 space-y-3">
              {campaigns.map((campaign) => (
                <motion.div
                  key={campaign.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 rounded-lg"
                  style={{
                    background: theme.cardBg,
                    border: `1px solid ${theme.border}`,
                    borderLeft: `4px solid ${getSeverityColor(campaign.severity)}`
                  }}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <h4 className="font-medium text-sm mb-1" style={{ color: theme.text }}>
                        {campaign.name}
                      </h4>
                      <p className="text-xs" style={{ color: theme.textSecondary }}>
                        {campaign.description}
                      </p>
                    </div>
                    <span className="px-2 py-1 rounded text-xs font-medium"
                          style={{
                            background: campaign.status === 'active' ? '#4CAF50' : '#888888',
                            color: 'white'
                          }}>
                      {campaign.status}
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-xs mt-3">
                    <div className="flex items-center gap-3">
                      <span style={{ color: theme.textSecondary }}>
                        {campaign.eventsCount} events
                      </span>
                      <span style={{ color: theme.textSecondary }}>
                        {campaign.domains.join(', ')}
                      </span>
                    </div>
                    <span style={{ color: theme.textSecondary }}>
                      {formatTimeAgo(campaign.lastActivity)}
                    </span>
                  </div>
                </motion.div>
              ))}

              {campaigns.length === 0 && (
                <div className="text-center py-8">
                  <Target className="w-12 h-12 mx-auto mb-3" style={{ color: theme.textSecondary, opacity: 0.5 }} />
                  <p className="text-sm" style={{ color: theme.textSecondary }}>
                    No active campaigns
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'alerts' && (
            <div className="p-4 space-y-3">
              {intelligenceEvents.filter(e => e.severity === 'critical').map((event) => (
                <motion.div
                  key={event.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 rounded-lg border-2"
                  style={{
                    background: 'rgba(255, 43, 43, 0.1)',
                    borderColor: '#FF2B2B',
                    boxShadow: '0 0 20px rgba(255, 43, 43, 0.3)'
                  }}
                >
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-red-400 mt-1 flex-shrink-0" />
                    <div className="flex-1">
                      <h4 className="font-bold text-sm mb-1" style={{ color: theme.text }}>
                        CRITICAL ALERT: {event.title}
                      </h4>
                      <p className="text-sm mb-2" style={{ color: theme.textSecondary }}>
                        {event.description}
                      </p>
                      <div className="flex items-center gap-4 text-xs">
                        <span style={{ color: theme.textSecondary }}>
                          {event.source}
                        </span>
                        <span style={{ color: theme.textSecondary }}>
                          {formatTimeAgo(event.timestamp)}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-2 mt-3">
                    {event.actions.map((action) => (
                      <button
                        key={action.id}
                        className="px-3 py-2 rounded text-xs font-medium transition-all btn-hover"
                        style={{
                          background: getSeverityColor(action.priority),
                          color: 'white'
                        }}
                      >
                        {action.automated && <Zap className="w-3 h-3 inline mr-1" />}
                        {action.label}
                      </button>
                    ))}
                  </div>
                </motion.div>
              ))}

              {intelligenceEvents.filter(e => e.severity === 'critical').length === 0 && (
                <div className="text-center py-8">
                  <Shield className="w-12 h-12 mx-auto mb-3" style={{ color: theme.textSecondary, opacity: 0.5 }} />
                  <p className="text-sm" style={{ color: theme.textSecondary }}>
                    No critical alerts
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
