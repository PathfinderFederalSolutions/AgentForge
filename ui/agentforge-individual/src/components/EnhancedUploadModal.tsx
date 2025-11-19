'use client';

import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  X,
  File,
  Database,
  Wifi,
  Globe,
  Loader2,
  Shield,
  Radio,
  Clock,
  CheckCircle,
  AlertCircle,
  Trash2
} from 'lucide-react';
import { store } from '@/lib/store';
import { useSnapshot } from 'valtio';

interface EnhancedUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface UploadFile {
  file: File;
  id: string;
  status: 'pending' | 'uploading' | 'processing' | 'complete' | 'error';
  progress: number;
  intelligence?: {
    domain: string;
    credibility: number;
    recordCount: number;
  };
}

const INTELLIGENCE_DOMAINS = [
  { id: 'SIGINT', label: 'Signals Intelligence', color: '#00A39B' },
  { id: 'HUMINT', label: 'Human Intelligence', color: '#FFD700' },
  { id: 'CYBINT', label: 'Cyber Intelligence', color: '#FF8C00' },
  { id: 'OSINT', label: 'Open Source Intelligence', color: '#4CAF50' },
  { id: 'GEOINT', label: 'Geospatial Intelligence', color: '#2196F3' },
  { id: 'MASINT', label: 'Measurement & Signature Intelligence', color: '#9C27B0' },
  { id: 'FININT', label: 'Financial Intelligence', color: '#FF5722' }
];

const PROCESSING_MODES = [
  { id: 'realtime', label: 'Real-Time', description: 'Immediate processing (<1s latency)', icon: Radio },
  { id: 'near-realtime', label: 'Near Real-Time', description: 'Fast processing (1-5s latency)', icon: Clock },
  { id: 'batch', label: 'Batch', description: 'Optimized throughput (>5s)', icon: Database }
];

export default function EnhancedUploadModal({ isOpen, onClose }: EnhancedUploadModalProps) {
  const [dragActive, setDragActive] = useState(false);
  const [uploadType, setUploadType] = useState<'file' | 'stream' | 'database'>('file');
  const [uploadFiles, setUploadFiles] = useState<UploadFile[]>([]);
  
  // Intelligence settings
  const [selectedDomain, setSelectedDomain] = useState<string>('OSINT');
  const [credibilityLevel, setCredibilityLevel] = useState<number>(0.7);
  const [processingMode, setProcessingMode] = useState<string>('near-realtime');
  const [enableStreamRegistration, setEnableStreamRegistration] = useState(false);
  const [streamName, setStreamName] = useState('');
  
  // Stream settings
  const [streamUrl, setStreamUrl] = useState('');
  const [selectedStreamType, setSelectedStreamType] = useState<'websocket' | 'rest' | null>(null);
  
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const snap = useSnapshot(store);

  const resetUploadState = () => {
    setUploadFiles([]);
    setIsUploading(false);
    setSelectedDomain('OSINT');
    setCredibilityLevel(0.7);
    setProcessingMode('near-realtime');
    setEnableStreamRegistration(false);
    setStreamName('');
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFiles = async (files: FileList) => {
    const fileArray = Array.from(files);
    const newUploadFiles: UploadFile[] = fileArray.map(file => ({
      file,
      id: `${Date.now()}-${Math.random()}`,
      status: 'pending',
      progress: 0
    }));
    
    setUploadFiles(prev => [...prev, ...newUploadFiles]);
  };

  const processUploadQueue = async () => {
    if (uploadFiles.length === 0) return;
    
    setIsUploading(true);

    for (const uploadFile of uploadFiles) {
      if (uploadFile.status !== 'pending') continue;

      try {
        // Update status to uploading
        setUploadFiles(prev => prev.map(uf => 
          uf.id === uploadFile.id ? { ...uf, status: 'uploading', progress: 10 } : uf
        ));

        // Upload to backend with intelligence context
        const formData = new FormData();
        formData.append('file', uploadFile.file);
        formData.append('intelligence_domain', selectedDomain);
        formData.append('credibility', credibilityLevel.toString());
        formData.append('processing_mode', processingMode);
        
        if (enableStreamRegistration && streamName) {
          formData.append('register_stream', 'true');
          formData.append('stream_name', streamName);
        }

        const response = await fetch('http://localhost:8001/v1/intelligence/ingest', {
          method: 'POST',
          body: formData
        });

        setUploadFiles(prev => prev.map(uf => 
          uf.id === uploadFile.id ? { ...uf, progress: 50, status: 'processing' } : uf
        ));

        if (!response.ok) throw new Error('Upload failed');

        const result = await response.json();

        // Update with intelligence data
        setUploadFiles(prev => prev.map(uf => 
          uf.id === uploadFile.id ? {
            ...uf,
            status: 'complete',
            progress: 100,
            intelligence: {
              domain: selectedDomain,
              credibility: credibilityLevel,
              recordCount: result.record_count || 1
            }
          } : uf
        ));

        // Add to store
        await store.addDataSource({
          name: uploadFile.file.name,
          type: 'file',
          status: 'ready',
          size: formatFileSize(uploadFile.file.size),
          recordCount: result.record_count || 1
        });

      } catch (error) {
        console.error('Upload error:', error);
        setUploadFiles(prev => prev.map(uf => 
          uf.id === uploadFile.id ? { ...uf, status: 'error', progress: 0 } : uf
        ));
      }
    }

    setIsUploading(false);
  };

  const handleStreamConnect = async () => {
    if (!streamUrl.trim()) return;

    setIsUploading(true);
    
    try {
      const response = await fetch('http://localhost:8001/v1/intelligence/continuous/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          stream_name: streamName || `stream_${Date.now()}`,
          stream_url: streamUrl,
          stream_type: selectedStreamType || 'websocket',
          intelligence_domain: selectedDomain,
          credibility_level: credibilityLevel,
          processing_mode: processingMode
        })
      });

      if (response.ok) {
        const result = await response.json();
        await store.addDataSource({
          name: streamName || result.stream_id,
          type: 'stream',
          status: 'connected'
        });
        
        setStreamUrl('');
        setStreamName('');
      }
    } catch (error) {
      console.error('Stream connection failed:', error);
    }
    
    setIsUploading(false);
  };

  const removeFile = (id: string) => {
    setUploadFiles(prev => prev.filter(uf => uf.id !== id));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const theme = snap.theme === 'day' ? {
    bg: '#05080D',
    text: '#D6E2F0',
    textSecondary: '#B8C5D1', 
    accent: '#00A39B',
    neon: '#CCFF00',
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

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0"
            style={{ background: 'rgba(0, 0, 0, 0.85)', backdropFilter: 'blur(8px)' }}
            onClick={() => {
              resetUploadState();
              onClose();
            }}
          />
          
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="relative rounded-2xl overflow-hidden"
            style={{ 
              width: '90%', 
              maxWidth: '900px',
              maxHeight: '90vh',
              background: theme.bg,
              border: `1px solid ${theme.border}`,
              boxShadow: `0 0 40px ${theme.accent}30`
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div 
              className="flex items-center justify-between p-6"
              style={{ borderBottom: `1px solid ${theme.border}` }}
            >
              <div className="flex items-center gap-3">
                <div 
                  className="w-10 h-10 rounded-lg flex items-center justify-center"
                  style={{ background: `linear-gradient(to right, ${theme.accent}, ${theme.accent})` }}
                >
                  <Shield className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold" style={{ color: theme.text }}>
                    Intelligence Data Ingestion
                  </h2>
                  <p className="text-sm" style={{ color: theme.textSecondary }}>
                    Upload files or connect live intelligence streams
                  </p>
                </div>
              </div>
              <button
                onClick={() => {
                  resetUploadState();
                  onClose();
                }}
                className="p-2 rounded-lg hover:bg-white/10 transition-colors"
              >
                <X className="w-6 h-6" style={{ color: theme.textSecondary }} />
              </button>
            </div>
            
            {/* Content */}
            <div className="p-6 overflow-y-auto" style={{ maxHeight: 'calc(90vh - 100px)' }}>
              {/* Upload Type Selector */}
              <div className="grid grid-cols-3 gap-3 mb-6">
                {[
                  { id: 'file', label: 'Files', icon: File },
                  { id: 'stream', label: 'Live Stream', icon: Wifi },
                  { id: 'database', label: 'Database', icon: Database }
                ].map(({ id, label, icon: Icon }) => (
                  <button
                    key={id}
                    onClick={() => setUploadType(id as any)}
                    className="flex flex-col items-center gap-2 p-4 rounded-lg transition-all"
                    style={{
                      background: uploadType === id ? theme.accent : theme.cardBg,
                      color: uploadType === id ? 'white' : theme.text,
                      border: `1px solid ${uploadType === id ? theme.accent : theme.border}`,
                      minHeight: '80px'
                    }}
                  >
                    <Icon className="w-6 h-6" />
                    <span className="text-sm font-medium">{label}</span>
                  </button>
                ))}
              </div>

              {/* Intelligence Settings */}
              <div className="mb-6 p-4 rounded-lg" style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}>
                <h3 className="text-sm font-bold mb-4" style={{ color: theme.text }}>
                  Intelligence Classification
                </h3>
                
                {/* Domain Selection */}
                <div className="mb-4">
                  <label className="block text-xs font-medium mb-2" style={{ color: theme.textSecondary }}>
                    Intelligence Domain
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {INTELLIGENCE_DOMAINS.map(domain => (
                      <button
                        key={domain.id}
                        onClick={() => setSelectedDomain(domain.id)}
                        className="px-3 py-2 rounded text-xs font-medium transition-all"
                        style={{
                          background: selectedDomain === domain.id ? domain.color : theme.cardBg,
                          color: selectedDomain === domain.id ? 'white' : theme.text,
                          border: `1px solid ${selectedDomain === domain.id ? domain.color : theme.border}`
                        }}
                      >
                        {domain.id}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Credibility Level */}
                <div className="mb-4">
                  <label className="block text-xs font-medium mb-2" style={{ color: theme.textSecondary }}>
                    Credibility Level: {(credibilityLevel * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={credibilityLevel}
                    onChange={(e) => setCredibilityLevel(parseFloat(e.target.value))}
                    className="w-full"
                    style={{ accentColor: theme.accent }}
                  />
                  <div className="flex justify-between text-xs mt-1" style={{ color: theme.textSecondary }}>
                    <span>Low</span>
                    <span>Medium</span>
                    <span>High</span>
                  </div>
                </div>

                {/* Processing Mode */}
                <div>
                  <label className="block text-xs font-medium mb-2" style={{ color: theme.textSecondary }}>
                    Processing Mode
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {PROCESSING_MODES.map(mode => {
                      const Icon = mode.icon;
                      return (
                        <button
                          key={mode.id}
                          onClick={() => setProcessingMode(mode.id)}
                          className="p-3 rounded-lg text-left transition-all"
                          style={{
                            background: processingMode === mode.id ? `${theme.accent}20` : theme.cardBg,
                            border: `1px solid ${processingMode === mode.id ? theme.accent : theme.border}`,
                            color: theme.text
                          }}
                        >
                          <Icon className="w-4 h-4 mb-1" style={{ color: theme.accent }} />
                          <p className="text-xs font-semibold">{mode.label}</p>
                          <p className="text-xs opacity-70">{mode.description}</p>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* File Upload */}
              {uploadType === 'file' && (
                <div>
                  <div
                    className="mb-4"
                    style={{
                      border: `2px dashed ${dragActive ? theme.accent : theme.border}`,
                      background: dragActive ? `${theme.accent}15` : theme.cardBg,
                      padding: '2rem',
                      borderRadius: '1rem',
                      textAlign: 'center'
                    }}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                  >
                    <Upload className="w-12 h-12 mb-4 mx-auto" style={{ color: theme.accent }} />
                    <h3 className="text-lg font-semibold mb-2" style={{ color: theme.text }}>
                      {dragActive ? 'Drop files here' : 'Drag & Drop Intelligence Files'}
                    </h3>
                    <p className="text-sm mb-4" style={{ color: theme.textSecondary }}>
                      Support for all document types, data files, images, and media
                    </p>
                    
                    <div className="flex gap-4 justify-center">
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="px-6 py-3 rounded-lg font-medium"
                        style={{ background: theme.accent, color: 'white' }}
                      >
                        Choose Files
                      </button>
                      <button
                        onClick={() => folderInputRef.current?.click()}
                        className="px-6 py-3 rounded-lg font-medium"
                        style={{ background: theme.cardBg, color: theme.text, border: `1px solid ${theme.border}` }}
                      >
                        Choose Folder
                      </button>
                    </div>

                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      className="hidden"
                      onChange={(e) => e.target.files && handleFiles(e.target.files)}
                    />
                    <input
                      ref={folderInputRef}
                      type="file"
                      {...({ webkitdirectory: '' } as any)}
                      className="hidden"
                      onChange={(e) => e.target.files && handleFiles(e.target.files)}
                    />
                  </div>

                  {/* Stream Registration Option */}
                  <div className="mb-4 p-4 rounded-lg" style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}>
                    <label className="flex items-center gap-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={enableStreamRegistration}
                        onChange={(e) => setEnableStreamRegistration(e.target.checked)}
                        className="w-4 h-4"
                        style={{ accentColor: theme.accent }}
                      />
                      <div>
                        <p className="text-sm font-medium" style={{ color: theme.text }}>
                          Register as Continuous Stream
                        </p>
                        <p className="text-xs" style={{ color: theme.textSecondary }}>
                          Monitor this data source for ongoing intelligence updates
                        </p>
                      </div>
                    </label>
                    {enableStreamRegistration && (
                      <input
                        type="text"
                        value={streamName}
                        onChange={(e) => setStreamName(e.target.value)}
                        placeholder="Stream name (e.g., 'submarine-tracking')"
                        className="w-full mt-3 px-3 py-2 rounded-lg text-sm"
                        style={{ 
                          background: theme.bg, 
                          color: theme.text, 
                          border: `1px solid ${theme.border}` 
                        }}
                      />
                    )}
                  </div>

                  {/* Upload Queue */}
                  {uploadFiles.length > 0 && (
                    <div className="mb-4">
                      <h4 className="text-sm font-bold mb-3" style={{ color: theme.text }}>
                        Upload Queue ({uploadFiles.length} files)
                      </h4>
                      <div className="space-y-2 max-h-60 overflow-y-auto">
                        {uploadFiles.map(uf => (
                          <div
                            key={uf.id}
                            className="p-3 rounded-lg flex items-center justify-between"
                            style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                          >
                            <div className="flex items-center gap-3 flex-1">
                              {uf.status === 'complete' && <CheckCircle className="w-4 h-4 text-green-400" />}
                              {uf.status === 'error' && <AlertCircle className="w-4 h-4 text-red-400" />}
                              {(uf.status === 'uploading' || uf.status === 'processing') && (
                                <Loader2 className="w-4 h-4 animate-spin" style={{ color: theme.accent }} />
                              )}
                              {uf.status === 'pending' && <File className="w-4 h-4" style={{ color: theme.textSecondary }} />}
                              
                              <div className="flex-1 min-w-0">
                                <p className="text-sm truncate" style={{ color: theme.text }}>
                                  {uf.file.name}
                                </p>
                                <p className="text-xs" style={{ color: theme.textSecondary }}>
                                  {formatFileSize(uf.file.size)} • {uf.status}
                                </p>
                              </div>
                            </div>
                            
                            {uf.progress > 0 && uf.progress < 100 && (
                              <div className="w-16 text-xs text-right font-mono" style={{ color: theme.accent }}>
                                {uf.progress}%
                              </div>
                            )}
                            
                            {uf.status === 'pending' && (
                              <button
                                onClick={() => removeFile(uf.id)}
                                className="p-1 rounded hover:bg-red-500/20"
                              >
                                <Trash2 className="w-4 h-4 text-red-400" />
                              </button>
                            )}
                          </div>
                        ))}
                      </div>
                      
                      {!isUploading && uploadFiles.some(uf => uf.status === 'pending') && (
                        <button
                          onClick={processUploadQueue}
                          className="w-full mt-3 py-3 rounded-lg font-medium"
                          style={{ background: theme.accent, color: 'white' }}
                        >
                          Process {uploadFiles.filter(uf => uf.status === 'pending').length} Files
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Stream Connection */}
              {uploadType === 'stream' && (
                <div>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-2" style={{ color: theme.text }}>
                        Stream Name
                      </label>
                      <input
                        type="text"
                        value={streamName}
                        onChange={(e) => setStreamName(e.target.value)}
                        placeholder="e.g., 'cyber-threats-feed'"
                        className="w-full px-3 py-2 rounded-lg"
                        style={{ 
                          background: theme.cardBg, 
                          color: theme.text, 
                          border: `1px solid ${theme.border}` 
                        }}
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-2" style={{ color: theme.text }}>
                        Stream URL
                      </label>
                      <input
                        type="url"
                        value={streamUrl}
                        onChange={(e) => setStreamUrl(e.target.value)}
                        placeholder="wss://api.example.com/stream or https://api.example.com/data"
                        className="w-full px-3 py-2 rounded-lg"
                        style={{ 
                          background: theme.cardBg, 
                          color: theme.text, 
                          border: `1px solid ${theme.border}` 
                        }}
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <button
                        onClick={() => setSelectedStreamType('websocket')}
                        className="p-4 rounded-lg text-left transition-all"
                        style={{ 
                          background: selectedStreamType === 'websocket' ? `${theme.accent}20` : theme.cardBg,
                          border: `1px solid ${selectedStreamType === 'websocket' ? theme.accent : theme.border}`,
                          color: theme.text
                        }}
                      >
                        <Wifi className="w-5 h-5 mb-2" style={{ color: theme.accent }} />
                        <p className="font-medium text-sm">WebSocket</p>
                        <p className="text-xs opacity-70">Real-time bidirectional</p>
                      </button>
                      
                      <button
                        onClick={() => setSelectedStreamType('rest')}
                        className="p-4 rounded-lg text-left transition-all"
                        style={{ 
                          background: selectedStreamType === 'rest' ? `${theme.accent}20` : theme.cardBg,
                          border: `1px solid ${selectedStreamType === 'rest' ? theme.accent : theme.border}`,
                          color: theme.text
                        }}
                      >
                        <Globe className="w-5 h-5 mb-2" style={{ color: theme.accent }} />
                        <p className="font-medium text-sm">REST API</p>
                        <p className="text-xs opacity-70">HTTP polling</p>
                      </button>
                    </div>

                    <button
                      onClick={handleStreamConnect}
                      disabled={!streamUrl.trim() || !streamName.trim() || isUploading}
                      className="w-full py-3 rounded-lg font-medium disabled:opacity-50"
                      style={{ background: theme.accent, color: 'white' }}
                    >
                      {isUploading ? (
                        <div className="flex items-center justify-center gap-2">
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Connecting...</span>
                        </div>
                      ) : (
                        'Register Intelligence Stream'
                      )}
                    </button>
                  </div>
                </div>
              )}

              {/* Database Connection */}
              {uploadType === 'database' && (
                <div className="text-center py-12" style={{ color: theme.textSecondary }}>
                  <Database className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>Database integration coming soon</p>
                  <p className="text-sm mt-2">Contact your administrator for database connectivity</p>
                </div>
              )}
            </div>

            {/* Footer */}
            <div 
              className="p-4 flex justify-between items-center"
              style={{ borderTop: `1px solid ${theme.border}` }}
            >
              <div className="text-sm" style={{ color: theme.textSecondary }}>
                {snap.dataSources.length > 0 && (
                  <span style={{ color: theme.accent }}>
                    ✓ {snap.dataSources.length} sources connected
                  </span>
                )}
              </div>
              <button
                onClick={() => {
                  resetUploadState();
                  onClose();
                }}
                className="px-6 py-2 rounded-lg font-medium"
                style={{ background: theme.accent, color: 'white' }}
              >
                Done
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}

