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
  Loader2
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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const snap = useSnapshot(store);

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

    try {
      // Convert FileList to File array
      const fileArray = Array.from(files);
      
      // Upload files to backend
      const uploadResult = await store.agiClient.uploadFiles(fileArray);
      
      setUploadProgress(50);

      // Process upload results and add to data sources
      if (uploadResult && Array.isArray(uploadResult)) {
        for (const processedFile of uploadResult) {
          await store.addDataSource({
            name: processedFile.filename,
            type: processedFile.processed_type || 'file',
            status: 'ready',
            size: formatFileSize(processedFile.size),
            recordCount: processedFile.extracted_content?.record_count || 
                        processedFile.extracted_content?.word_count ||
                        processedFile.metadata?.rows
          });
        }
      } else {
        // Fallback to local processing if backend fails
        for (let i = 0; i < files.length; i++) {
          const file = files[i];
          
          await store.addDataSource({
            name: file.name,
            type: 'file',
            status: 'ready',
            size: formatFileSize(file.size),
            recordCount: file.type.includes('json') ? Math.floor(Math.random() * 10000) + 1000 : undefined
          });
        }
      }

      setUploadProgress(100);
      setIsUploading(false);
      
      // Show success message
      setTimeout(() => {
        setUploadProgress(0);
      }, 2000);

    } catch (error) {
      console.error('File upload failed:', error);
      
      // Fallback to local processing
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        await store.addDataSource({
          name: file.name,
          type: 'file',
          status: 'error',
          size: formatFileSize(file.size)
        });
      }
      
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleStreamConnect = async () => {
    if (!streamUrl.trim()) return;

    setIsUploading(true);
    store.connectToStream(streamUrl);
    
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
            onClick={onClose}
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
                onClick={onClose}
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
                  <div className="flex items-center gap-3 mb-3">
                    <Loader2 className="w-5 h-5 animate-spin" style={{ color: theme.accent }} />
                    <span className="text-sm font-medium" style={{ color: theme.text }}>
                      Processing data...
                    </span>
                  </div>
                  
                  <div 
                    className="w-full rounded-full h-2"
                    style={{ background: 'rgba(255,255,255,0.1)' }}
                  >
                    <motion.div
                      className="h-2 rounded-full"
                      style={{ background: theme.accent }}
                      initial={{ width: 0 }}
                      animate={{ width: `${uploadProgress}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                  
                  <p className="text-xs mt-2" style={{ color: theme.text, opacity: 0.7 }}>
                    Analyzing data structure and generating optimal agent swarm...
                  </p>
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
                    onClick={onClose}
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
