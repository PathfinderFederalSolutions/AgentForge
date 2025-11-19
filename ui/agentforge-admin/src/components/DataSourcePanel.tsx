'use client';

import { motion, AnimatePresence } from 'framer-motion';
import {
  Database,
  Wifi,
  File,
  Trash2,
  CheckCircle,
  AlertTriangle,
  Clock,
  Upload,
  Link
} from 'lucide-react';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/store';

interface DataSourcePanelProps {
  onUpload: () => void;
  onStream: () => void;
}

export default function DataSourcePanel({ onUpload, onStream }: DataSourcePanelProps) {
  const snap = useSnapshot(store);

  const theme = snap.theme === 'day' ? {
    bg: '#05080D',
    text: '#D6E2F0',
    accent: '#00A39B',
    border: 'rgba(255,255,255,0.2)',
    cardBg: 'rgba(255,255,255,0.1)'
  } : {
    bg: '#000000',
    text: '#FF2B2B',
    accent: '#FF2B2B',
    border: 'rgba(137,22,22,0.4)',
    cardBg: 'rgba(0,0,0,0.3)'
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
      case 'ready':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'processing':
        return <Clock className="w-4 h-4 text-yellow-400 animate-spin" />;
      case 'error':
        return <AlertTriangle className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4" style={{ color: theme.text }} />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'file':
        return <File className="w-4 h-4" />;
      case 'stream':
        return <Wifi className="w-4 h-4" />;
      case 'database':
        return <Database className="w-4 h-4" />;
      default:
        return <File className="w-4 h-4" />;
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Quick Actions */}
      <div className="p-4 space-y-3">
        <button
          onClick={onUpload}
          className="w-full flex items-center gap-3 p-3 rounded-lg transition-all btn-hover"
          style={{
            background: theme.cardBg,
            border: `1px solid ${theme.border}`,
            color: theme.text
          }}
        >
          <Upload className="w-5 h-5" style={{ color: theme.accent }} />
          <span className="text-sm font-medium">Upload Files</span>
        </button>
        
        <button
          onClick={onStream}
          className="w-full flex items-center gap-3 p-3 rounded-lg transition-all btn-hover"
          style={{
            background: theme.cardBg,
            border: `1px solid ${theme.border}`,
            color: theme.text
          }}
        >
          <Link className="w-5 h-5" style={{ color: theme.accent }} />
          <span className="text-sm font-medium">Connect Stream</span>
        </button>
      </div>

      {/* Data Sources List */}
      <div className="flex-1 overflow-y-auto p-4">
        {snap.dataSources.length === 0 ? (
          <div className="text-center py-8">
            <Database className="w-12 h-12 mx-auto mb-4" style={{ color: theme.text, opacity: 0.5 }} />
            <p className="text-sm" style={{ color: theme.text, opacity: 0.7 }}>
              No data sources connected
            </p>
            <p className="text-xs mt-1" style={{ color: theme.text, opacity: 0.5 }}>
              Upload files or connect streams to get started
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            <h3 className="text-sm font-medium mb-3" style={{ color: theme.text }}>
              Connected Sources ({snap.dataSources.length})
            </h3>
            
            <AnimatePresence>
              {snap.dataSources.map((source) => (
                <motion.div
                  key={source.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="p-3 rounded-lg"
                  style={{
                    background: theme.cardBg,
                    border: `1px solid ${theme.border}`
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      <div style={{ color: theme.accent }}>
                        {getTypeIcon(source.type)}
                      </div>
                      <div className="flex-1">
                        <h4 className="text-sm font-medium" style={{ color: theme.text }}>
                          {source.name}
                        </h4>
                        <div className="flex items-center gap-2 mt-1">
                          {getStatusIcon(source.status)}
                          <span className="text-xs capitalize" style={{ color: theme.text, opacity: 0.7 }}>
                            {source.status}
                          </span>
                        </div>
                        {source.size && (
                          <p className="text-xs mt-1" style={{ color: theme.text, opacity: 0.6 }}>
                            {source.size}
                            {source.recordCount && ` • ${source.recordCount.toLocaleString()} records`}
                          </p>
                        )}
                      </div>
                    </div>
                    
                    <button
                      onClick={() => store.removeDataSource(source.id)}
                      className="p-1 rounded hover:bg-red-500/20 transition-colors"
                    >
                      <Trash2 className="w-4 h-4 text-red-400" />
                    </button>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Usage Info */}
      <div className="p-4 border-t" style={{ borderColor: theme.border }}>
        <div 
          className="p-3 rounded-lg"
          style={{ 
            background: `${theme.accent}15`,
            border: `1px solid ${theme.accent}30`
          }}
        >
          <div className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 mt-0.5" style={{ color: theme.accent }} />
            <div>
              <p className="text-xs font-medium" style={{ color: theme.text }}>
                Smart Agent Deployment
              </p>
              <p className="text-xs mt-1" style={{ color: theme.text, opacity: 0.8 }}>
                Agents automatically adapt to your data type and processing needs
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
