'use client';

import { motion } from 'framer-motion';
import { X, MapPin, Target, Shield } from 'lucide-react';

interface TacticalMapViewProps {
  isOpen: boolean;
  onClose: () => void;
  theme: any;
}

export default function TacticalMapView({ isOpen, onClose, theme }: TacticalMapViewProps) {
  if (!isOpen) return null;

  const threats = [
    { x: 30, y: 40, type: 'HOSTILE', name: 'Submarine Contact' },
    { x: 60, y: 30, type: 'FRIENDLY', name: 'Naval Asset' },
    { x: 45, y: 70, type: 'HOSTILE', name: 'Surface Vessel' },
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
        className="relative w-full max-w-6xl rounded-2xl p-6"
        style={{ background: theme.bg, border: `1px solid ${theme.border}` }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold" style={{ color: theme.text }}>
            Tactical Situation Map
          </h2>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10">
            <X className="w-6 h-6" style={{ color: theme.textSecondary }} />
          </button>
        </div>

        <div 
          className="relative w-full h-96 rounded-lg"
          style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
        >
          {/* Grid */}
          <div className="absolute inset-0 opacity-20">
            {[...Array(10)].map((_, i) => (
              <div key={`h-${i}`} className="absolute w-full h-px" style={{ top: `${i * 10}%`, background: theme.border }} />
            ))}
            {[...Array(10)].map((_, i) => (
              <div key={`v-${i}`} className="absolute h-full w-px" style={{ left: `${i * 10}%`, background: theme.border }} />
            ))}
          </div>

          {/* Threat Markers */}
          {threats.map((threat, idx) => (
            <motion.div
              key={idx}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: idx * 0.2 }}
              className="absolute group cursor-pointer"
              style={{ left: `${threat.x}%`, top: `${threat.y}%`, transform: 'translate(-50%, -50%)' }}
            >
              <div 
                className="w-6 h-6 rounded-full flex items-center justify-center"
                style={{
                  background: threat.type === 'HOSTILE' ? '#FF2B2B' : '#4CAF50',
                  boxShadow: `0 0 20px ${threat.type === 'HOSTILE' ? '#FF2B2B' : '#4CAF50'}`
                }}
              >
                {threat.type === 'HOSTILE' ? (
                  <Target className="w-4 h-4 text-white" />
                ) : (
                  <Shield className="w-4 h-4 text-white" />
                )}
              </div>
              <div 
                className="absolute left-8 top-1/2 transform -translate-y-1/2 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity px-2 py-1 rounded text-xs font-semibold"
                style={{ background: theme.cardBg, border: `1px solid ${theme.border}`, color: theme.text }}
              >
                {threat.name}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Legend */}
        <div className="mt-4 flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full" style={{ background: '#FF2B2B' }} />
            <span className="text-sm" style={{ color: theme.text }}>Hostile</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full" style={{ background: '#4CAF50' }} />
            <span className="text-sm" style={{ color: theme.text }}>Friendly</span>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

