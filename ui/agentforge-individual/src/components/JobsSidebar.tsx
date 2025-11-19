'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Clock,
  CheckCircle,
  Play,
  Pause,
  Eye,
  Archive,
  Activity,
  Zap
} from 'lucide-react';
import { useSnapshot } from 'valtio';
import { store, ActiveJob, ArchivedJob } from '@/lib/store';

// No mock jobs - use only real jobs from store

export default function JobsSidebar() {
  const [activeTab, setActiveTab] = useState<'active' | 'archived'>('active');
  const [intelMetrics, setIntelMetrics] = useState<{ processingRate: number; activeStreams: number; ttpDetections: number } | null>(null);
  const snap = useSnapshot(store);

  useEffect(() => {
    let cancelled = false;
    const apiBase = typeof window !== 'undefined' ? (process.env.NEXT_PUBLIC_API_BASE || '//localhost:8001') : (process.env.NEXT_PUBLIC_API_BASE || '//localhost:8001');
    const fetchMetrics = async () => {
      try {
        const res = await fetch(`${apiBase}/v1/intelligence/continuous/state`);
        if (res.ok) {
          const data = await res.json();
          if (!cancelled) {
            setIntelMetrics({
              processingRate: data.processing_rate || 0,
              activeStreams: data.active_streams || 0,
              ttpDetections: data.total_ttp_detections || 0
            });
          }
        }
      } catch (e) {
        // ignore errors for sidebar
      }
    };
    fetchMetrics();
    const id = setInterval(fetchMetrics, 10000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

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


  return (
    <div 
      style={{
        position: 'fixed',
        right: '0',
        top: '70px',
        width: '320px',
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
      {/* Sidebar Header */}
      <div 
        className="p-4 border-b" 
        style={{ 
          borderColor: theme.border,
          background: theme.cardBg,
          backdropFilter: 'blur(20px)'
        }}
      >
        <h2 className="font-bold text-lg mb-3" style={{ color: theme.text }}>
          Job Management
        </h2>

        {/* Intelligence Metrics */}
        {intelMetrics && (
          <div className="mb-3 p-3 rounded-lg border" style={{ borderColor: theme.border }}>
            <div className="flex items-center justify-between text-xs" style={{ color: theme.text }}>
              <span className="flex items-center gap-1"><Activity className="w-3 h-3" /> {intelMetrics.activeStreams} streams</span>
              <span className="flex items-center gap-1"><Zap className="w-3 h-3" /> {intelMetrics.processingRate.toFixed(1)}/s</span>
              <span>{intelMetrics.ttpDetections} TTPs</span>
            </div>
          </div>
        )}
        
        {/* Tab Selector */}
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab('active')}
            className="flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all btn-hover"
            style={{
              background: activeTab === 'active' ? theme.accent : theme.cardBg,
              color: activeTab === 'active' ? 'white' : theme.text,
              border: `1px solid ${activeTab === 'active' ? theme.accent : theme.border}`
            }}
          >
            Active ({snap.activeJobs.length})
          </button>
          <button
            onClick={() => setActiveTab('archived')}
            className="flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all btn-hover"
            style={{
              background: activeTab === 'archived' ? theme.accent : theme.cardBg,
              color: activeTab === 'archived' ? 'white' : theme.text,
              border: `1px solid ${activeTab === 'archived' ? theme.accent : theme.border}`
            }}
          >
            Archived ({snap.archivedJobs.length})
          </button>
        </div>
      </div>

      {/* Job Lists */}
      <div className="flex-1 overflow-y-auto p-4">
        <AnimatePresence mode="wait">
          {activeTab === 'active' && (
            <motion.div
              key="active"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="space-y-3"
            >
              {snap.activeJobs.map((job, index) => (
                <div key={job.id} style={{ marginBottom: '1rem' }}>
                  <ActiveJobCard 
                    job={job} 
                    index={index} 
                    theme={theme} 
                    onClick={() => store.loadJobConversation(job.id)}
                  />
                </div>
              ))}
            </motion.div>
          )}

          {activeTab === 'archived' && (
            <motion.div
              key="archived"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-3"
            >
              {snap.archivedJobs.map((job, index) => (
                <div key={job.id} style={{ marginBottom: '1rem' }}>
                  <ArchivedJobCard 
                    job={job} 
                    index={index} 
                    theme={theme} 
                    onClick={() => store.loadJobConversation(job.id)}
                  />
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Quick Actions */}
      <div 
        className="p-4 border-t" 
        style={{ 
          borderColor: theme.border,
          background: theme.cardBg,
          backdropFilter: 'blur(20px)'
        }}
      >
        <div className="grid grid-cols-2 gap-2 mb-3">
          <button 
            className="btn-secondary p-3 rounded-lg text-sm font-medium"
            title="Start a new conversation"
            onClick={store.startNewChat}
          >
            <span className="text-lg mb-1 block">+</span>
            New Job
          </button>
          <button 
            className="btn-secondary p-3 rounded-lg text-sm font-medium"
            title="View detailed job management"
          >
            <Eye className="w-4 h-4 mx-auto mb-1" />
            View All
          </button>
        </div>
        
        {/* Audit Notice */}
        <div className="mt-3 p-2 rounded-lg text-xs" style={{ 
          background: `${theme.accent}15`, 
          border: `1px solid ${theme.accent}30`,
          color: theme.text,
          opacity: 0.8
        }}>
          🔒 All jobs are preserved for audit compliance
        </div>
      </div>
    </div>
  );
}

function ActiveJobCard({ job, index, theme, onClick }: { 
  job: ActiveJob; 
  index: number; 
  theme: { text: string; textSecondary: string; accent: string; border: string; cardBg: string }; 
  onClick: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className="hud-card p-4 cursor-pointer btn-hover"
      onClick={onClick}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="font-medium text-sm mb-2" style={{ color: theme.text }}>
            {job.title}
          </h3>
          <p className="text-xs mb-3" style={{ color: theme.textSecondary, opacity: 0.8, lineHeight: '1.4' }}>
            {job.description}
          </p>
        </div>
        <div className="flex items-center gap-1 ml-2">
          {job.status === 'running' && (
            <button
              onClick={(e) => {
                e.stopPropagation(); // Prevent triggering the card click
                store.pauseJob(job.id);
              }}
              className="p-1 rounded job-control-btn opacity-50 hover:opacity-80"
              title="Pause job"
              style={{ 
                background: 'rgba(255, 193, 7, 0.1)',
                border: `1px solid rgba(255, 193, 7, 0.3)`,
                borderRadius: '0.5rem'
              }}
            >
              <Pause className="w-3 h-3" style={{ color: '#FFC107' }} />
            </button>
          )}
          {job.status === 'paused' && (
            <button
              onClick={(e) => {
                e.stopPropagation(); // Prevent triggering the card click
                store.resumeJob(job.id);
              }}
              className="p-1 rounded job-control-btn opacity-50 hover:opacity-80"
              title="Resume job"
              style={{ 
                background: 'rgba(34, 197, 94, 0.1)',
                border: `1px solid rgba(34, 197, 94, 0.3)`,
                borderRadius: '0.5rem'
              }}
            >
              <Play className="w-3 h-3" style={{ color: '#22C55E' }} />
            </button>
          )}
          {job.type === 'task' && job.progress && job.progress >= 100 && (
            <button
              onClick={(e) => {
                e.stopPropagation(); // Prevent triggering the card click
                store.archiveJob(job.id);
              }}
              className="p-2 rounded-lg job-control-btn"
              title="Archive completed job"
              style={{ 
                background: 'rgba(99, 102, 241, 0.2)',
                border: `2px solid #6366F1`,
                borderRadius: '0.75rem'
              }}
            >
              <Archive className="w-4 h-4" style={{ color: '#6366F1' }} />
            </button>
          )}
        </div>
      </div>

      {/* Real-time Agent Status */}
      <div className="mb-4">
        <div className="flex justify-between text-xs mb-2" style={{ color: theme.text, opacity: 0.7 }}>
          <span>Agent Status</span>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: theme.accent }} />
            <span className="font-medium">Active</span>
          </div>
        </div>
        <div className="flex justify-between text-xs" style={{ color: theme.text, opacity: 0.6 }}>
          <span>Active Agents: {job.realAgentMetrics?.totalAgentsDeployed || 0}</span>
          <span>Runtime: {job.runtime || 'Unknown'}</span>
        </div>
      </div>

      {/* Job Stats */}
      <div className="grid grid-cols-3 gap-3 text-xs mb-3">
        <div className="text-center p-2 rounded-lg" style={{ background: `${theme.accent}10` }}>
          <div className="font-bold text-sm" style={{ color: theme.accent }}>{job.agentsAssigned}</div>
          <div style={{ color: theme.text, opacity: 0.7 }}>Agents</div>
        </div>
        <div className="text-center p-2 rounded-lg" style={{ background: `${theme.accent}10` }}>
          <div className="font-bold text-sm" style={{ color: theme.accent }}>{job.dataStreams.length}</div>
          <div style={{ color: theme.text, opacity: 0.7 }}>Streams</div>
        </div>
        <div className="text-center p-2 rounded-lg" style={{ background: `${theme.accent}10` }}>
          <div className="font-bold text-sm" style={{ color: theme.accent }}>{job.alertsGenerated}</div>
          <div style={{ color: theme.text, opacity: 0.7 }}>Alerts</div>
        </div>
      </div>

      {/* Runtime */}
      <div className="mt-2 text-xs" style={{ color: theme.textSecondary, opacity: 0.7 }}>
        Running for {(() => {
          const now = new Date();
          const diff = now.getTime() - job.startTime.getTime();
          const hours = Math.floor(diff / (1000 * 60 * 60));
          const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
          return `${hours}h ${minutes}m`;
        })()}
      </div>
    </motion.div>
  );
}

function ArchivedJobCard({ job, index, theme, onClick }: { 
  job: ArchivedJob; 
  index: number; 
  theme: { text: string; textSecondary: string; accent: string; border: string; cardBg: string }; 
  onClick: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className="hud-card p-4 cursor-pointer btn-hover"
      onClick={onClick}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="font-medium text-sm mb-2" style={{ color: theme.text }}>
            {job.title}
          </h3>
          <p className="text-xs mb-3" style={{ color: theme.textSecondary, opacity: 0.8, lineHeight: '1.4' }}>
            {job.description}
          </p>
        </div>
        <CheckCircle className="w-4 h-4 text-green-400 ml-2" />
      </div>

      {/* Job Results */}
      <div className="grid grid-cols-2 gap-2 text-xs mb-2">
        <div>
          <div className="font-medium" style={{ color: theme.accent }}>{job.duration}</div>
          <div style={{ color: theme.text, opacity: 0.6 }}>Duration</div>
        </div>
        <div>
          <div className="font-medium" style={{ color: theme.accent }}>{Math.round(job.confidence * 100)}%</div>
          <div style={{ color: theme.text, opacity: 0.6 }}>Confidence</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <div className="font-medium" style={{ color: theme.accent }}>{job.agentsUsed}</div>
          <div style={{ color: theme.text, opacity: 0.6 }}>Agents Used</div>
        </div>
        <div>
          <div className="font-medium" style={{ color: theme.accent }}>{job.outputSize}</div>
          <div style={{ color: theme.text, opacity: 0.6 }}>Output</div>
        </div>
      </div>

      {/* Completed Time */}
      <div className="mt-2 text-xs" style={{ color: theme.textSecondary, opacity: 0.7 }}>
        Completed {job.completedAt.toLocaleDateString()}
      </div>
    </motion.div>
  );
}

function formatDuration(startTime: Date): string {
  const now = new Date();
  const diff = now.getTime() - startTime.getTime();
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  return `${hours}h ${minutes}m`;
}
