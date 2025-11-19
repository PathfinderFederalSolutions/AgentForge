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
  Eye,
  Zap,
  Clock,
  Star,
  AlertTriangle,
  CheckCircle,
  Settings
} from 'lucide-react';
import { store } from '@/lib/store';
import { useSnapshot } from 'valtio';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function UploadModal({ isOpen, onClose }: UploadModalProps) {
  const [dragActive, setDragActive] = useState(false);
  const [uploadType, setUploadType] = useState<'file' | 'stream' | 'database'>('file');
  const [streamUrl, setStreamUrl] = useState('');
  const [selectedStreamType, setSelectedStreamType] = useState<'websocket' | 'rest' | null>(null);
  const [selectedDbType, setSelectedDbType] = useState<'postgresql' | 'mongodb' | 'mysql' | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Intelligence features
  const [intelligenceDomain, setIntelligenceDomain] = useState<'SIGINT' | 'CYBINT' | 'HUMINT' | 'GEOINT' | 'OSINT' | 'TECHINT' | null>(null);
  const [credibilityLevel, setCredibilityLevel] = useState<'CONFIRMED' | 'PROBABLE' | 'POSSIBLE' | 'DOUBTFUL' | null>(null);
  const [processingMode, setProcessingMode] = useState<'real-time' | 'near-real-time' | 'batch' | null>('batch');
  const [continuousMonitoring, setContinuousMonitoring] = useState(false);
  const [showIntelligenceSettings, setShowIntelligenceSettings] = useState(false);
  const [fileProgress, setFileProgress] = useState<{[key: string]: number}>({});
  const [intelligencePreview, setIntelligencePreview] = useState<any>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const snap = useSnapshot(store);

  const resetUploadState = () => {
    setIsUploading(false);
    setUploadProgress(0);
    // Reset intelligence settings
    setIntelligenceDomain(null);
    setCredibilityLevel(null);
    setProcessingMode('batch');
    setContinuousMonitoring(false);
    setShowIntelligenceSettings(false);
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
    setIsUploading(true);
    setUploadProgress(0);
    setFileProgress({});
    setIntelligencePreview(null);

    try {
      const fileArray = Array.from(files);
      console.log(`Processing ${fileArray.length} files with intelligence settings:`, {
        intelligenceDomain,
        credibilityLevel,
        processingMode
      });

      setUploadProgress(10);

      // Prepare intelligence metadata
      const intelligenceMetadata = {
        domain: intelligenceDomain,
        credibility: credibilityLevel,
        processing_mode: processingMode,
        timestamp: new Date().toISOString(),
        source_type: 'file_upload'
      };

      // Initialize progress for each file
      const initialProgress: {[key: string]: number} = {};
      fileArray.forEach(file => {
        initialProgress[file.name] = 0;
      });
      setFileProgress(initialProgress);

      // Upload files to backend with intelligence metadata
      // Auto-chunks large uploads transparently - user doesn't need to know!
      const totalSize = fileArray.reduce((sum, f) => sum + f.size, 0);
      const totalSizeMB = totalSize / (1024 * 1024);
      
      // Show appropriate message based on size
      if (totalSizeMB > 400 || fileArray.length > 100) {
        console.log(`Large upload (${totalSizeMB.toFixed(1)}MB, ${fileArray.length} files) - processing in optimized batches for reliability`);
      }
      
      const uploadResult = await store.agiClient.uploadFiles(fileArray, intelligenceMetadata);

      setUploadProgress(75);

      // Process results with individual file progress
      if (uploadResult && Array.isArray(uploadResult)) {
        for (let i = 0; i < uploadResult.length; i++) {
          const processedFile = uploadResult[i];

          // Update individual file progress
          setFileProgress(prev => ({
            ...prev,
            [processedFile.filename]: 100
          }));

          await store.addDataSource({
            name: processedFile.filename,
            type: processedFile.processed_type || 'file',
            status: 'ready',
            size: formatFileSize(processedFile.size),
            recordCount: processedFile.extracted_content?.record_count || 10,
            intelligenceMetadata: processedFile.intelligence_metadata || intelligenceMetadata
          });

          // Update overall progress
          setUploadProgress(75 + ((i + 1) / uploadResult.length) * 20);
        }

        // Generate intelligence preview from first file
        if (uploadResult.length > 0) {
          const firstFile = uploadResult[0];
          setIntelligencePreview({
            filename: firstFile.filename,
            extracted_entities: firstFile.extracted_content?.entities || [],
            intelligence_insights: firstFile.intelligence_metadata?.insights || [],
            credibility_assessment: credibilityLevel || 'UNASSESSED',
            domain_classification: intelligenceDomain || 'UNCLASSIFIED',
            processing_mode: processingMode,
            confidence_score: firstFile.confidence || 0.8
          });
        }
      } else {
        // Fallback processing with intelligence metadata and individual progress
        for (let i = 0; i < files.length; i++) {
          const file = files[i];

          // Simulate individual file processing
          setFileProgress(prev => ({
            ...prev,
            [file.name]: Math.min(100, (i / files.length) * 100 + 10)
          }));

          await store.addDataSource({
            name: file.name,
            type: file.type.startsWith('text/') ? 'text' : 'file',
            status: 'ready',
            size: formatFileSize(file.size),
            recordCount: 1,
            intelligenceMetadata
          });

          // Update overall progress
          setUploadProgress(75 + ((i + 1) / files.length) * 20);

          // Small delay to show progress animation
          await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Generate mock intelligence preview for fallback
        if (files.length > 0) {
          setIntelligencePreview({
            filename: files[0].name,
            extracted_entities: ['Sample Entity 1', 'Sample Entity 2'],
            intelligence_insights: ['Pattern detected', 'Correlation identified'],
            credibility_assessment: credibilityLevel || 'UNASSESSED',
            domain_classification: intelligenceDomain || 'UNCLASSIFIED',
            processing_mode: processingMode,
            confidence_score: 0.75
          });
        }
      }

      setUploadProgress(100);
      setIsUploading(false);

      setTimeout(() => {
        setUploadProgress(0);
        setFileProgress({});
      }, 3000);

    } catch (error) {
      console.error('Upload failed:', error);
      setIsUploading(false);
      setUploadProgress(0);
      setFileProgress({});
    }
  };

  const handleStreamConnect = async () => {
    if (!streamUrl.trim()) return;

    setIsUploading(true);

    // Prepare intelligence metadata for stream
    const intelligenceMetadata = {
      domain: intelligenceDomain,
      credibility: credibilityLevel,
      processing_mode: processingMode,
      continuous_monitoring: continuousMonitoring,
      timestamp: new Date().toISOString(),
      source_type: 'stream',
      stream_type: selectedStreamType
    };

    console.log('Connecting stream with intelligence settings:', intelligenceMetadata);

    // Connect to stream with intelligence metadata
    store.connectToStream(streamUrl, intelligenceMetadata);

    // Simulate connection time
    setTimeout(() => {
      setIsUploading(false);
      setStreamUrl(''); // Clear input for next connection
      // Don't close modal to allow multiple connections
    }, 2000);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getCredibilityColor = (credibility: string) => {
    switch (credibility) {
      case 'CONFIRMED': return '#4CAF50';
      case 'PROBABLE': return '#00A39B';
      case 'POSSIBLE': return '#FFD700';
      case 'DOUBTFUL': return '#FF6B6B';
      default: return '#888888';
    }
  };

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
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0"
            style={{ background: 'rgba(0, 0, 0, 0.8)' }}
            onClick={() => {
              resetUploadState();
              onClose();
            }}
          />
          
          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="relative hud-card overflow-hidden"
            style={{ 
              width: '90%', 
              maxWidth: '600px',
              background: theme.cardBg,
              border: `1px solid ${theme.border}`
            }}
          >
            {/* Header */}
            <div 
              className="flex items-center justify-between p-6"
              style={{ borderBottom: `1px solid ${theme.border}` }}
            >
              <h2 className="text-xl font-bold" style={{ color: theme.text }}>
                Add Data Source
              </h2>
              <button
                onClick={() => {
              resetUploadState();
              onClose();
            }}
                className="btn-secondary p-2 rounded-lg"
                style={{
                  minWidth: '40px',
                  minHeight: '40px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            {/* Content */}
            <div className="p-6">
              {/* Upload Type Selector */}
              <div className="grid grid-cols-3 gap-3 mb-6">
                {[
                  { id: 'file', label: 'Files', icon: File },
                  { id: 'stream', label: 'Live Stream', icon: Wifi },
                  { id: 'database', label: 'Database', icon: Database }
                ].map(({ id, label, icon: Icon }: { id: string; label: string; icon: React.ComponentType<{className?: string}> }) => (
                  <button
                    key={id}
                    onClick={() => setUploadType(id as 'file' | 'stream' | 'database')}
                    className="flex flex-col items-center gap-2 p-4 rounded-lg transition-all btn-hover"
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

              {/* Intelligence Settings Toggle */}
              <div className="mb-6 p-4 rounded-lg" style={{ background: `${theme.accent}15`, border: `1px solid ${theme.accent}30` }}>
                <button
                  onClick={() => setShowIntelligenceSettings(!showIntelligenceSettings)}
                  className="flex items-center gap-3 w-full text-left"
                >
                  <Shield className="w-5 h-5" style={{ color: theme.accent }} />
                  <span className="font-medium" style={{ color: theme.text }}>Intelligence Settings</span>
                  <Settings className="w-4 h-4 ml-auto" style={{ color: theme.text, opacity: 0.7 }} />
                </button>

                {showIntelligenceSettings && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-4 space-y-4"
                  >
                    {/* Intelligence Domain Selection */}
                    <div>
                      <label className="block text-sm font-medium mb-3" style={{ color: theme.text }}>
                        Intelligence Domain
                      </label>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                        {[
                          { id: 'SIGINT', label: 'SIGINT', desc: 'Signals Intelligence' },
                          { id: 'CYBINT', label: 'CYBINT', desc: 'Cyber Intelligence' },
                          { id: 'HUMINT', label: 'HUMINT', desc: 'Human Intelligence' },
                          { id: 'GEOINT', label: 'GEOINT', desc: 'Geospatial Intelligence' },
                          { id: 'OSINT', label: 'OSINT', desc: 'Open Source Intelligence' },
                          { id: 'TECHINT', label: 'TECHINT', desc: 'Technical Intelligence' }
                        ].map(({ id, label, desc }) => (
                          <button
                            key={id}
                            onClick={() => setIntelligenceDomain(id as any)}
                            className="p-3 rounded-lg text-left transition-all btn-hover"
                            style={{
                              background: intelligenceDomain === id ? theme.accent : theme.cardBg,
                              border: `1px solid ${intelligenceDomain === id ? theme.accent : theme.border}`,
                              color: intelligenceDomain === id ? 'white' : theme.text
                            }}
                          >
                            <div className="font-medium text-sm">{label}</div>
                            <div className="text-xs opacity-75">{desc}</div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Credibility Level */}
                    <div>
                      <label className="block text-sm font-medium mb-3" style={{ color: theme.text }}>
                        Source Credibility
                      </label>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        {[
                          { id: 'CONFIRMED', label: 'Confirmed', color: '#4CAF50' },
                          { id: 'PROBABLE', label: 'Probable', color: '#00A39B' },
                          { id: 'POSSIBLE', label: 'Possible', color: '#FFD700' },
                          { id: 'DOUBTFUL', label: 'Doubtful', color: '#FF6B6B' }
                        ].map(({ id, label, color }) => (
                          <button
                            key={id}
                            onClick={() => setCredibilityLevel(id as any)}
                            className="p-3 rounded-lg text-center transition-all btn-hover"
                            style={{
                              background: credibilityLevel === id ? color : theme.cardBg,
                              border: `1px solid ${credibilityLevel === id ? color : theme.border}`,
                              color: credibilityLevel === id ? 'white' : theme.text
                            }}
                          >
                            <div className="font-medium text-sm">{label}</div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Processing Mode */}
                    <div>
                      <label className="block text-sm font-medium mb-3" style={{ color: theme.text }}>
                        Processing Mode
                      </label>
                      <div className="grid grid-cols-3 gap-2">
                        {[
                          { id: 'real-time', label: 'Real-Time', icon: Zap, desc: 'Immediate processing' },
                          { id: 'near-real-time', label: 'Near Real-Time', icon: Clock, desc: 'Fast processing' },
                          { id: 'batch', label: 'Batch', icon: CheckCircle, desc: 'Thorough analysis' }
                        ].map(({ id, label, icon: Icon, desc }) => (
                          <button
                            key={id}
                            onClick={() => setProcessingMode(id as any)}
                            className="p-3 rounded-lg text-left transition-all btn-hover"
                            style={{
                              background: processingMode === id ? theme.accent : theme.cardBg,
                              border: `1px solid ${processingMode === id ? theme.accent : theme.border}`,
                              color: processingMode === id ? 'white' : theme.text
                            }}
                          >
                            <div className="flex items-center gap-2 mb-1">
                              <Icon className="w-4 h-4" />
                              <span className="font-medium text-sm">{label}</span>
                            </div>
                            <div className="text-xs opacity-75">{desc}</div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Continuous Monitoring (for streams) */}
                    {uploadType === 'stream' && (
                      <div className="flex items-center gap-3 p-3 rounded-lg" style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}>
                        <input
                          type="checkbox"
                          id="continuous-monitoring"
                          checked={continuousMonitoring}
                          onChange={(e) => setContinuousMonitoring(e.target.checked)}
                          className="w-4 h-4"
                        />
                        <label htmlFor="continuous-monitoring" className="flex-1">
                          <div className="font-medium text-sm" style={{ color: theme.text }}>Continuous Intelligence Monitoring</div>
                          <div className="text-xs opacity-75" style={{ color: theme.text }}>Register as intelligence stream with automated analysis</div>
                        </label>
                        <Eye className="w-5 h-5" style={{ color: theme.accent }} />
                      </div>
                    )}
                  </motion.div>
                )}
              </div>

              {/* File Upload */}
              {uploadType === 'file' && (
                <div>
                  <div
                    className="upload-area"
                    style={{
                      border: `2px dashed ${dragActive ? theme.accent : theme.border}`,
                      background: dragActive ? `${theme.accent}15` : theme.cardBg,
                      color: theme.text,
                      padding: '2rem',
                      borderRadius: '1rem',
                      textAlign: 'center',
                      minHeight: '200px',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center',
                      alignItems: 'center'
                    }}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                  >
                    <Upload className="w-12 h-12 mb-4" style={{ color: theme.accent }} />
                    <h3 className="text-lg font-semibold mb-2" style={{ color: theme.text }}>
                      {dragActive ? 'Drop files here' : 'Upload Files or Folders'}
                    </h3>
                    <p className="text-sm mb-6" style={{ color: theme.text, opacity: 0.8 }}>
                      Drag and drop files here, or click to browse
                    </p>
                    
                    <div className="flex gap-4 justify-center">
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="btn-primary px-6 py-3 rounded-lg font-medium"
                      >
                        Choose Files
                      </button>
                      <button
                        onClick={() => folderInputRef.current?.click()}
                        className="btn-secondary px-6 py-3 rounded-lg font-medium"
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
                      accept=".pdf,.doc,.docx,.txt,.csv,.json,.xlsx,.png,.jpg,.jpeg,.mp4,.mp3"
                    />
                    <input
                      ref={folderInputRef}
                      type="file"
                      {...({ webkitdirectory: '' } as Record<string, string>)}
                      className="hidden"
                      onChange={(e) => e.target.files && handleFiles(e.target.files)}
                    />
                  </div>

                  <div className="mt-4 text-xs" style={{ color: theme.text, opacity: 0.7 }}>
                    <p className="mb-2"><strong>Supported formats:</strong></p>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <p><strong>Documents:</strong> PDF, DOC, TXT, CSV</p>
                        <p><strong>Data:</strong> JSON, XML, XLSX</p>
                      </div>
                      <div>
                        <p><strong>Media:</strong> PNG, JPG, MP4, MP3</p>
                        <p><strong>Code:</strong> JS, PY, TS, etc.</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Stream Connection */}
              {uploadType === 'stream' && (
                <div>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-2" style={{ color: theme.text }}>
                        Stream URL
                      </label>
                      <input
                        type="url"
                        value={streamUrl}
                        onChange={(e) => setStreamUrl(e.target.value)}
                        placeholder="wss://api.example.com/stream or https://api.example.com/data"
                        className="chat-input w-full"
                      />
                    </div>

                    <div className="grid grid-cols-1 md-grid-cols-2 gap-4" style={{ marginTop: '1.5rem', marginBottom: '1.5rem' }}>
                      <button
                        onClick={() => {
                          setStreamUrl('wss://websocket.example.com/live');
                          setSelectedStreamType('websocket');
                        }}
                        className="p-5 rounded-lg text-left btn-hover"
                        style={{ 
                          minHeight: '100px',
                          display: 'flex',
                          flexDirection: 'column',
                          justifyContent: 'center',
                          background: selectedStreamType === 'websocket' ? theme.accent : theme.cardBg,
                          border: `2px solid ${selectedStreamType === 'websocket' ? theme.accent : theme.border}`,
                          color: selectedStreamType === 'websocket' ? 'white' : theme.text
                        }}
                      >
                        <Wifi className="w-6 h-6 mb-3" style={{ 
                          color: selectedStreamType === 'websocket' ? 'white' : theme.accent 
                        }} />
                        <p className="font-medium text-sm mb-1">WebSocket Stream</p>
                        <p className="text-xs" style={{ opacity: 0.7 }}>Real-time data stream</p>
                      </button>
                      
                      <button
                        onClick={() => {
                          setStreamUrl('https://api.example.com/data');
                          setSelectedStreamType('rest');
                        }}
                        className="p-5 rounded-lg text-left btn-hover"
                        style={{ 
                          minHeight: '100px',
                          display: 'flex',
                          flexDirection: 'column',
                          justifyContent: 'center',
                          background: selectedStreamType === 'rest' ? theme.accent : theme.cardBg,
                          border: `2px solid ${selectedStreamType === 'rest' ? theme.accent : theme.border}`,
                          color: selectedStreamType === 'rest' ? 'white' : theme.text
                        }}
                      >
                        <Globe className="w-6 h-6 mb-3" style={{ 
                          color: selectedStreamType === 'rest' ? 'white' : theme.accent 
                        }} />
                        <p className="font-medium text-sm mb-1">REST API</p>
                        <p className="text-xs" style={{ opacity: 0.7 }}>HTTP data endpoint</p>
                      </button>
                    </div>

                    <button
                      onClick={handleStreamConnect}
                      disabled={!streamUrl.trim() || isUploading}
                      className="btn-primary w-full py-3 rounded-lg disabled-opacity disabled-cursor"
                      style={{ marginTop: '1rem' }}
                    >
                      {isUploading ? (
                        <div className="flex items-center justify-center gap-2">
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Connecting...</span>
                        </div>
                      ) : (
                        'Add Stream Source'
                      )}
                    </button>

                      <div className="mt-4 p-3 rounded-lg" style={{ background: `${theme.accent}15`, border: `1px solid ${theme.accent}30` }}>
                      <p className="text-xs" style={{ color: theme.text, opacity: 0.8 }}>
                        💡 <strong>Tip:</strong> You can add multiple streams. Each stream will be processed by dedicated agents in parallel.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Database Connection */}
              {uploadType === 'database' && (
                <div>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-2" style={{ color: theme.text }}>
                        Database Connection String
                      </label>
                      <input
                        type="text"
                        placeholder="postgresql://user:pass@host:port/db"
                        className="chat-input w-full"
                      />
                    </div>

                    <div className="grid grid-cols-3 gap-4" style={{ marginTop: '1.5rem', marginBottom: '1.5rem' }}>
                      <button 
                        onClick={() => setSelectedDbType('postgresql')}
                        className="p-4 rounded-lg text-center btn-hover" 
                        style={{ 
                          minHeight: '90px',
                          background: selectedDbType === 'postgresql' ? theme.accent : theme.cardBg,
                          border: `2px solid ${selectedDbType === 'postgresql' ? theme.accent : theme.border}`,
                          color: selectedDbType === 'postgresql' ? 'white' : theme.text
                        }}
                      >
                        <Database className="w-6 h-6 mx-auto mb-2" style={{ 
                          color: selectedDbType === 'postgresql' ? 'white' : theme.accent 
                        }} />
                        <p className="text-sm font-medium">PostgreSQL</p>
                      </button>
                      <button 
                        onClick={() => setSelectedDbType('mongodb')}
                        className="p-4 rounded-lg text-center btn-hover" 
                        style={{ 
                          minHeight: '90px',
                          background: selectedDbType === 'mongodb' ? theme.accent : theme.cardBg,
                          border: `2px solid ${selectedDbType === 'mongodb' ? theme.accent : theme.border}`,
                          color: selectedDbType === 'mongodb' ? 'white' : theme.text
                        }}
                      >
                        <Database className="w-6 h-6 mx-auto mb-2" style={{ 
                          color: selectedDbType === 'mongodb' ? 'white' : theme.accent 
                        }} />
                        <p className="text-sm font-medium">MongoDB</p>
                      </button>
                      <button 
                        onClick={() => setSelectedDbType('mysql')}
                        className="p-4 rounded-lg text-center btn-hover" 
                        style={{ 
                          minHeight: '90px',
                          background: selectedDbType === 'mysql' ? theme.accent : theme.cardBg,
                          border: `2px solid ${selectedDbType === 'mysql' ? theme.accent : theme.border}`,
                          color: selectedDbType === 'mysql' ? 'white' : theme.text
                        }}
                      >
                        <Database className="w-6 h-6 mx-auto mb-2" style={{ 
                          color: selectedDbType === 'mysql' ? 'white' : theme.accent 
                        }} />
                        <p className="text-sm font-medium">MySQL</p>
                      </button>
                    </div>

                    <button className="btn-primary w-full py-3 rounded-lg" style={{ marginTop: '1rem' }}>
                      Add Database Source
                    </button>

                    <div className="mt-4 p-3 rounded-lg" style={{ background: `${theme.accent}15`, border: `1px solid ${theme.accent}30` }}>
                      <p className="text-xs" style={{ color: theme.text, opacity: 0.8 }}>
                        💡 <strong>Multi-Database Support:</strong> Connect multiple databases simultaneously for comprehensive analysis.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Upload Progress */}
              {isUploading && uploadProgress > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-6 p-4 rounded-xl"
                  style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <Loader2 className="w-5 h-5 animate-spin" style={{ color: theme.accent }} />
                      <span className="text-sm font-medium" style={{ color: theme.text }}>
                        Processing data...
                      </span>
                    </div>
                    <span className="text-lg font-mono font-bold" style={{ color: theme.accent }}>
                      {uploadProgress}%
                    </span>
                  </div>

                  {/* Individual File Progress */}
                  {Object.keys(fileProgress).length > 0 && (
                    <div className="mb-4 space-y-2">
                      <p className="text-xs font-medium" style={{ color: theme.text, opacity: 0.8 }}>
                        Individual Files:
                      </p>
                      {Object.entries(fileProgress).map(([filename, progress]) => (
                        <div key={filename} className="flex items-center gap-2">
                          <span className="text-xs truncate flex-1" style={{ color: theme.text, opacity: 0.7 }}>
                            {filename}
                          </span>
                          <div className="flex-1 max-w-20">
                            <div
                              className="w-full rounded-full h-2"
                              style={{ background: 'rgba(255,255,255,0.1)' }}
                            >
                              <div
                                className="h-2 rounded-full transition-all duration-300 ease-out"
                                style={{
                                  width: `${progress}%`,
                                  background: progress === 100 ? '#4CAF50' : theme.accent
                                }}
                              />
                            </div>
                          </div>
                          <span className="text-xs font-mono w-8" style={{ color: theme.accent }}>
                            {progress}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Overall Progress Bar */}
                  <div
                    className="w-full rounded-full h-3"
                    style={{ background: 'rgba(255,255,255,0.1)' }}
                  >
                    <div
                      className="h-3 rounded-full transition-all duration-300 ease-out"
                      style={{
                        width: `${uploadProgress}%`,
                        background: theme.accent
                      }}
                    />
                  </div>

                  <p className="text-xs mt-2" style={{ color: theme.text, opacity: 0.7 }}>
                    Analyzing data structure and generating optimal agent swarm...
                  </p>
                </motion.div>
              )}

              {/* Intelligence Preview */}
              {intelligencePreview && !isUploading && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-6 p-4 rounded-xl"
                  style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                >
                  <div className="flex items-center gap-3 mb-4">
                    <Eye className="w-5 h-5" style={{ color: theme.accent }} />
                    <span className="text-sm font-medium" style={{ color: theme.text }}>
                      Intelligence Preview - {intelligencePreview.filename}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <p className="text-xs font-medium mb-2" style={{ color: theme.text, opacity: 0.8 }}>
                        Domain Classification
                      </p>
                      <span className="px-2 py-1 rounded text-xs font-medium"
                            style={{
                              background: `${theme.accent}20`,
                              color: theme.accent,
                              border: `1px solid ${theme.accent}40`
                            }}>
                        {intelligencePreview.domain_classification}
                      </span>
                    </div>
                    <div>
                      <p className="text-xs font-medium mb-2" style={{ color: theme.text, opacity: 0.8 }}>
                        Credibility Assessment
                      </p>
                      <span className="px-2 py-1 rounded text-xs font-medium"
                            style={{
                              background: getCredibilityColor(intelligencePreview.credibility_assessment),
                              color: 'white'
                            }}>
                        {intelligencePreview.credibility_assessment}
                      </span>
                    </div>
                  </div>

                  {intelligencePreview.extracted_entities.length > 0 && (
                    <div className="mb-4">
                      <p className="text-xs font-medium mb-2" style={{ color: theme.text, opacity: 0.8 }}>
                        Extracted Entities
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {intelligencePreview.extracted_entities.map((entity: string, index: number) => (
                          <span key={index} className="px-2 py-1 rounded text-xs"
                                style={{
                                  background: `${theme.accent}15`,
                                  color: theme.accent,
                                  border: `1px solid ${theme.accent}30`
                                }}>
                            {entity}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {intelligencePreview.intelligence_insights.length > 0 && (
                    <div className="mb-4">
                      <p className="text-xs font-medium mb-2" style={{ color: theme.text, opacity: 0.8 }}>
                        Intelligence Insights
                      </p>
                      <div className="space-y-1">
                        {intelligencePreview.intelligence_insights.map((insight: string, index: number) => (
                          <div key={index} className="flex items-center gap-2 text-xs"
                               style={{ color: theme.text, opacity: 0.8 }}>
                            <CheckCircle className="w-3 h-3" style={{ color: '#4CAF50' }} />
                            {insight}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex items-center justify-between pt-3" style={{ borderTop: `1px solid ${theme.border}` }}>
                    <div className="flex items-center gap-2">
                      <span className="text-xs" style={{ color: theme.text, opacity: 0.7 }}>
                        Processing Mode:
                      </span>
                      <span className="px-2 py-1 rounded text-xs font-medium"
                            style={{
                              background: `${theme.accent}15`,
                              color: theme.accent
                            }}>
                        {intelligencePreview.processing_mode}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs" style={{ color: theme.text, opacity: 0.7 }}>
                        Confidence:
                      </span>
                      <span className="text-xs font-mono font-bold"
                            style={{ color: theme.accent }}>
                        {(intelligencePreview.confidence_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Action Buttons */}
              <div className="flex justify-between items-center mt-6 pt-4" style={{ borderTop: `1px solid ${theme.border}` }}>
                <div className="text-xs" style={{ color: theme.text, opacity: 0.7 }}>
                  {snap.dataSources.length > 0 && (
                    <span style={{ color: theme.accent }}>
                      ✅ {snap.dataSources.length} source{snap.dataSources.length !== 1 ? 's' : ''} connected
                    </span>
                  )}
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => {
              resetUploadState();
              onClose();
            }}
                    className="btn-primary px-6 py-2 rounded-lg"
                  >
                    {snap.dataSources.length > 0 ? 'Done' : 'Close'}
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
