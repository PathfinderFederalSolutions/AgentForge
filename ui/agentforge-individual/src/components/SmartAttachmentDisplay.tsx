'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Database,
  File,
  X,
  ChevronDown,
  ChevronUp,
  Trash2,
  CheckCircle,
  AlertTriangle,
  Clock,
  Paperclip
} from 'lucide-react';

interface DataSource {
  id: string;
  name: string;
  type: string;
  status: string;
  size?: string;
  recordCount?: number;
}

interface SmartAttachmentDisplayProps {
  dataSources: any[];
  theme: any;
  onRemoveSource: (id: string) => void;
}

export default function SmartAttachmentDisplay({ dataSources, theme, onRemoveSource }: SmartAttachmentDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showAll, setShowAll] = useState(false);
  
  // Intelligent display logic based on number of attachments
  const maxVisible = dataSources.length > 20 ? 5 : dataSources.length > 10 ? 4 : 3;
  const visibleSources = showAll ? dataSources : dataSources.slice(0, maxVisible);
  const hasMore = dataSources.length > maxVisible;

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
      case 'ready':
        return <CheckCircle className="w-3 h-3 text-green-400" />;
      case 'processing':
        return <Clock className="w-3 h-3 text-yellow-400 animate-spin" />;
      case 'error':
        return <AlertTriangle className="w-3 h-3 text-red-400" />;
      default:
        return <Clock className="w-3 h-3" style={{ color: theme.text }} />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'file':
        return <File className="w-3 h-3" />;
      case 'database':
        return <Database className="w-3 h-3" />;
      default:
        return <File className="w-3 h-3" />;
    }
  };

  return (
    <div className="mb-3">
      {/* Compact Summary Bar */}
      <div 
        className="flex items-center justify-between p-2 rounded-lg cursor-pointer transition-all hover:opacity-80"
        style={{
          background: `${theme.accent}15`,
          border: `1px solid ${theme.accent}30`
        }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Paperclip className="w-4 h-4" style={{ color: theme.accent }} />
          <span className="text-sm font-medium" style={{ color: theme.text }}>
            {dataSources.length} attachment{dataSources.length !== 1 ? 's' : ''} connected
          </span>
          
          {/* Quick status indicators */}
          <div className="flex items-center gap-1 ml-2">
            {dataSources.filter(s => s.status === 'ready').length > 0 && (
              <div className="flex items-center gap-1">
                <CheckCircle className="w-3 h-3 text-green-400" />
                <span className="text-xs" style={{ color: theme.text, opacity: 0.7 }}>
                  {dataSources.filter(s => s.status === 'ready').length}
                </span>
              </div>
            )}
            {dataSources.filter(s => s.status === 'processing').length > 0 && (
              <div className="flex items-center gap-1">
                <Clock className="w-3 h-3 text-yellow-400" />
                <span className="text-xs" style={{ color: theme.text, opacity: 0.7 }}>
                  {dataSources.filter(s => s.status === 'processing').length}
                </span>
              </div>
            )}
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Clear All Button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              dataSources.forEach(source => onRemoveSource(source.id));
            }}
            className="p-1 rounded hover:bg-red-500/20 transition-colors"
            title="Remove all attachments"
          >
            <Trash2 className="w-3 h-3 text-red-400" />
          </button>
          
          {/* Expand/Collapse Button */}
          {isExpanded ? (
            <ChevronUp className="w-4 h-4" style={{ color: theme.text }} />
          ) : (
            <ChevronDown className="w-4 h-4" style={{ color: theme.text }} />
          )}
        </div>
      </div>

      {/* Expandable Attachment List */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2 space-y-2 max-h-40 overflow-y-auto"
            style={{
              background: theme.cardBg,
              border: `1px solid ${theme.border}`,
              borderRadius: '8px',
              padding: '8px'
            }}
          >
            {/* Show All Toggle */}
            {hasMore && (
              <div className="flex justify-between items-center pb-2 border-b" style={{ borderColor: theme.border }}>
                <span className="text-xs" style={{ color: theme.text, opacity: 0.7 }}>
                  Showing {visibleSources.length} of {dataSources.length}
                </span>
                <button
                  onClick={() => setShowAll(!showAll)}
                  className="text-xs px-2 py-1 rounded transition-colors"
                  style={{ 
                    color: theme.accent,
                    background: `${theme.accent}10`
                  }}
                >
                  {showAll ? 'Show Less' : 'Show All'}
                </button>
              </div>
            )}

            {/* Attachment List - Compact or Grid based on count */}
            <div className={dataSources.length > 10 ? "grid grid-cols-2 gap-1" : "space-y-1"}>
              {visibleSources.map((source) => (
                <motion.div
                  key={source.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 10 }}
                  className="flex items-center justify-between p-2 rounded transition-colors hover:bg-opacity-50 group"
                  style={{ background: `${theme.accent}05` }}
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <div style={{ color: theme.accent }}>
                      {getTypeIcon(source.type)}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1">
                        <span 
                          className="text-xs font-medium truncate" 
                          style={{ color: theme.text }}
                          title={source.name}
                        >
                          {dataSources.length > 10 
                            ? (source.name.length > 8 ? `${source.name.substring(0, 8)}...` : source.name)
                            : (source.name.length > 15 ? `${source.name.substring(0, 15)}...` : source.name)
                          }
                        </span>
                        {getStatusIcon(source.status)}
                      </div>
                      
                      {source.size && dataSources.length <= 10 && (
                        <div className="text-xs mt-1" style={{ color: theme.text, opacity: 0.6 }}>
                          {source.size}
                          {source.recordCount && ` • ${source.recordCount.toLocaleString()}`}
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* Individual Delete Button */}
                  <button
                    onClick={() => onRemoveSource(source.id)}
                    className="p-1 rounded hover:bg-red-500/20 transition-colors ml-1 opacity-0 group-hover:opacity-100"
                    title={`Remove ${source.name}`}
                  >
                    <X className="w-3 h-3 text-red-400" />
                  </button>
                </motion.div>
              ))}
            </div>

            {/* Quick Actions and Stats */}
            <div className="pt-2 border-t" style={{ borderColor: theme.border }}>
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <span className="text-xs" style={{ color: theme.text, opacity: 0.7 }}>
                    {dataSources.reduce((acc, s) => acc + (s.recordCount || 0), 0).toLocaleString()} records
                  </span>
                  
                  {/* Status Summary */}
                  <div className="flex items-center gap-2">
                    {dataSources.filter(s => s.status === 'ready').length > 0 && (
                      <div className="flex items-center gap-1">
                        <CheckCircle className="w-3 h-3 text-green-400" />
                        <span className="text-xs" style={{ color: theme.text, opacity: 0.6 }}>
                          {dataSources.filter(s => s.status === 'ready').length} ready
                        </span>
                      </div>
                    )}
                    {dataSources.filter(s => s.status === 'processing').length > 0 && (
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3 text-yellow-400" />
                        <span className="text-xs" style={{ color: theme.text, opacity: 0.6 }}>
                          {dataSources.filter(s => s.status === 'processing').length} processing
                        </span>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="flex gap-2">
                  {dataSources.length > 5 && (
                    <button
                      onClick={() => {
                        // Remove all error/failed attachments
                        dataSources
                          .filter(s => s.status === 'error')
                          .forEach(s => onRemoveSource(s.id));
                      }}
                      className="text-xs px-2 py-1 rounded transition-colors"
                      style={{ 
                        color: '#EF4444',
                        background: '#EF444410'
                      }}
                      title="Remove failed attachments"
                    >
                      Clear Failed
                    </button>
                  )}
                  
                  <button
                    onClick={() => setIsExpanded(false)}
                    className="text-xs px-2 py-1 rounded transition-colors"
                    style={{ 
                      color: theme.text,
                      background: `${theme.accent}10`
                    }}
                  >
                    Collapse
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
