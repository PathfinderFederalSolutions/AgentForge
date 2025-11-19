'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Download,
  FileText,
  Image as ImageIcon,
  FileJson,
  FileSpreadsheet,
  CheckCircle,
  Loader2
} from 'lucide-react';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/store';

interface ExportReportingProps {
  isOpen: boolean;
  onClose: () => void;
}

const EXPORT_FORMATS = [
  {
    id: 'pdf',
    label: 'PDF Report',
    description: 'Professional decision brief with analysis',
    icon: FileText,
    color: '#FF2B2B'
  },
  {
    id: 'json',
    label: 'JSON Data',
    description: 'Raw intelligence data export',
    icon: FileJson,
    color: '#FFD700'
  },
  {
    id: 'csv',
    label: 'CSV Export',
    description: 'Tabular data for analysis',
    icon: FileSpreadsheet,
    color: '#4CAF50'
  },
  {
    id: 'slides',
    label: 'Presentation',
    description: 'Executive briefing slides',
    icon: ImageIcon,
    color: '#2196F3'
  }
];

export default function ExportReporting({ isOpen, onClose }: ExportReportingProps) {
  const [selectedFormat, setSelectedFormat] = useState<string>('pdf');
  const [isExporting, setIsExporting] = useState(false);
  const [exportComplete, setExportComplete] = useState(false);
  const [includeAnalysis, setIncludeAnalysis] = useState(true);
  const [includeCOAs, setIncludeCOAs] = useState(true);
  const [includeWargaming, setIncludeWargaming] = useState(true);
  const [includeIntelligence, setIncludeIntelligence] = useState(true);
  
  const snap = useSnapshot(store);

  const handleExport = async () => {
    setIsExporting(true);
    setExportComplete(false);

    try {
      const exportData = {
        format: selectedFormat,
        timestamp: new Date().toISOString(),
        sections: {
          analysis: includeAnalysis,
          coas: includeCOAs,
          wargaming: includeWargaming,
          intelligence: includeIntelligence
        },
        data: {
          messages: snap.messages,
          dataSources: snap.dataSources,
          activeJobs: snap.activeJobs
        }
      };

      const response = await fetch('http://localhost:8001/v1/export/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(exportData)
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `agentforge-report-${Date.now()}.${selectedFormat}`;
        a.click();
        URL.revokeObjectURL(url);
        
        setExportComplete(true);
        setTimeout(() => {
          onClose();
          setExportComplete(false);
        }, 2000);
      }
    } catch (error) {
      console.error('Export failed:', error);
      // Fallback: export as JSON
      const exportData = {
        timestamp: new Date().toISOString(),
        messages: snap.messages,
        dataSources: snap.dataSources,
        activeJobs: snap.activeJobs
      };
      
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `agentforge-export-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      
      setExportComplete(true);
      setTimeout(() => {
        onClose();
        setExportComplete(false);
      }, 2000);
    }

    setIsExporting(false);
  };

  const theme = snap.theme === 'day' ? {
    bg: '#05080D',
    text: '#D6E2F0',
    textSecondary: '#B8C5D1', 
    accent: '#00A39B',
    border: 'rgba(255,255,255,0.2)',
    cardBg: 'rgba(255,255,255,0.08)',
  } : {
    bg: '#000000',
    text: '#FF2B2B',
    textSecondary: '#891616',
    accent: '#FF2B2B',
    border: 'rgba(137,22,22,0.6)',
    cardBg: 'rgba(137,22,22,0.15)',
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)' }}
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="relative w-full max-w-2xl rounded-2xl overflow-hidden"
          style={{
            background: theme.bg,
            border: `1px solid ${theme.border}`,
            boxShadow: `0 0 40px ${theme.accent}30`
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div 
            className="flex items-center justify-between p-6 border-b"
            style={{ borderColor: theme.border }}
          >
            <div className="flex items-center gap-3">
              <div 
                className="w-10 h-10 rounded-lg flex items-center justify-center"
                style={{ background: `linear-gradient(to right, ${theme.accent}, ${theme.accent})` }}
              >
                <Download className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold" style={{ color: theme.text }}>
                  Export & Reporting
                </h2>
                <p className="text-sm" style={{ color: theme.textSecondary }}>
                  Generate comprehensive intelligence reports
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/10 transition-colors"
            >
              <X className="w-6 h-6" style={{ color: theme.textSecondary }} />
            </button>
          </div>

          {/* Content */}
          <div className="p-6">
            {/* Format Selection */}
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-4" style={{ color: theme.text }}>
                Export Format
              </h3>
              <div className="grid grid-cols-2 gap-3">
                {EXPORT_FORMATS.map(format => {
                  const Icon = format.icon;
                  return (
                    <button
                      key={format.id}
                      onClick={() => setSelectedFormat(format.id)}
                      className="p-4 rounded-lg text-left transition-all"
                      style={{
                        background: selectedFormat === format.id ? `${format.color}20` : theme.cardBg,
                        border: `1px solid ${selectedFormat === format.id ? format.color : theme.border}`
                      }}
                    >
                      <Icon 
                        className="w-8 h-8 mb-2" 
                        style={{ color: format.color }} 
                      />
                      <p className="font-semibold text-sm mb-1" style={{ color: theme.text }}>
                        {format.label}
                      </p>
                      <p className="text-xs" style={{ color: theme.textSecondary }}>
                        {format.description}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Content Selection */}
            <div className="mb-6">
              <h3 className="text-sm font-bold mb-4" style={{ color: theme.text }}>
                Include in Report
              </h3>
              <div className="space-y-3">
                {[
                  { id: 'analysis', label: 'Intelligence Analysis', value: includeAnalysis, setValue: setIncludeAnalysis },
                  { id: 'coas', label: 'Courses of Action', value: includeCOAs, setValue: setIncludeCOAs },
                  { id: 'wargaming', label: 'Wargaming Results', value: includeWargaming, setValue: setIncludeWargaming },
                  { id: 'intelligence', label: 'Raw Intelligence Data', value: includeIntelligence, setValue: setIncludeIntelligence }
                ].map(option => (
                  <label
                    key={option.id}
                    className="flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all hover:bg-white/5"
                    style={{
                      background: theme.cardBg,
                      border: `1px solid ${theme.border}`
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={option.value}
                      onChange={(e) => option.setValue(e.target.checked)}
                      className="w-5 h-5"
                      style={{ accentColor: theme.accent }}
                    />
                    <span className="text-sm font-medium" style={{ color: theme.text }}>
                      {option.label}
                    </span>
                  </label>
                ))}
              </div>
            </div>

            {/* Export Button */}
            <button
              onClick={handleExport}
              disabled={isExporting || exportComplete}
              className="w-full py-4 rounded-lg font-medium transition-all disabled:opacity-50"
              style={{ background: theme.accent, color: 'white' }}
            >
              {isExporting ? (
                <div className="flex items-center justify-center gap-2">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Generating Report...</span>
                </div>
              ) : exportComplete ? (
                <div className="flex items-center justify-center gap-2">
                  <CheckCircle className="w-5 h-5" />
                  <span>Export Complete!</span>
                </div>
              ) : (
                <div className="flex items-center justify-center gap-2">
                  <Download className="w-5 h-5" />
                  <span>Export Report</span>
                </div>
              )}
            </button>

            {/* Info */}
            <div 
              className="mt-4 p-3 rounded-lg text-xs"
              style={{ background: `${theme.accent}15`, color: theme.textSecondary }}
            >
              <p>
                💡 <strong>Tip:</strong> PDF reports include visualizations and executive summaries. 
                JSON exports preserve all data for further analysis.
              </p>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

