'use client';

import { motion } from 'framer-motion';
import { X, Clock, Target, Activity, Zap } from 'lucide-react';

interface TimelineVisualizationProps {
  isOpen: boolean;
  onClose: () => void;
  theme: any;
}

export default function TimelineVisualization({ isOpen, onClose, theme }: TimelineVisualizationProps) {
  if (!isOpen) return null;

  const timelineEvents = [
    { time: '14:32', type: 'TTP', title: 'Submarine Detection', severity: 'CRITICAL' },
    { time: '14:45', type: 'FUSION', title: 'Multi-Domain Correlation', severity: 'HIGH' },
    { time: '15:10', type: 'PATTERN', title: 'Behavioral Anomaly', severity: 'MEDIUM' },
    { time: '15:30', type: 'CAMPAIGN', title: 'Campaign Link Identified', severity: 'HIGH' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)' }}
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9 }}
        animate={{ scale: 1 }}
        className="relative w-full max-w-5xl rounded-2xl p-6"
        style={{ background: theme.bg, border: `1px solid ${theme.border}` }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold" style={{ color: theme.text }}>
            Intelligence Timeline
          </h2>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10">
            <X className="w-6 h-6" style={{ color: theme.textSecondary }} />
          </button>
        </div>

        <div className="relative">
          <div className="absolute left-8 top-0 bottom-0 w-px" style={{ background: theme.border }} />
          
          <div className="space-y-6">
            {timelineEvents.map((event, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="relative pl-20"
              >
                <div 
                  className="absolute left-6 w-4 h-4 rounded-full"
                  style={{ background: theme.accent }}
                />
                <div 
                  className="p-4 rounded-lg"
                  style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <Clock className="w-4 h-4" style={{ color: theme.accent }} />
                      <span className="text-sm font-bold" style={{ color: theme.text }}>
                        {event.time}
                      </span>
                      <span className="text-xs px-2 py-1 rounded" style={{ background: `${theme.accent}20`, color: theme.accent }}>
                        {event.type}
                      </span>
                    </div>
                    <span className="text-xs px-2 py-1 rounded" style={{ 
                      background: event.severity === 'CRITICAL' ? '#FF2B2B20' : event.severity === 'HIGH' ? '#FF8C0020' : '#FFD70020',
                      color: event.severity === 'CRITICAL' ? '#FF2B2B' : event.severity === 'HIGH' ? '#FF8C00' : '#FFD700'
                    }}>
                      {event.severity}
                    </span>
                  </div>
                  <h4 className="font-semibold" style={{ color: theme.text }}>
                    {event.title}
                  </h4>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

