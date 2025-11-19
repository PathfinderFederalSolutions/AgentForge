'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Plus,
  Folder,
  Clock,
  Users,
  FileText,
  Download,
  Share2,
  Archive,
  Search,
  Filter,
  Calendar,
  Target,
  CheckCircle,
  AlertTriangle,
  TrendingUp
} from 'lucide-react';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/store';

interface ProjectManagementPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

interface Project {
  id: string;
  name: string;
  description: string;
  type: 'threat-analysis' | 'cyber-incident' | 'infrastructure' | 'maritime' | 'custom';
  status: 'active' | 'paused' | 'completed';
  createdAt: Date;
  lastUpdated: Date;
  analysisCount: number;
  dataSources: string[];
  participants: number;
  insights: number;
  confidence: number;
}

const PROJECT_TEMPLATES = [
  {
    id: 'submarine-threat',
    name: 'Submarine Threat Analysis',
    description: 'Track and analyze submarine movements, capabilities, and potential threats',
    type: 'maritime' as const,
    icon: '🚢',
    defaultDataSources: ['SIGINT', 'GEOINT', 'OSINT']
  },
  {
    id: 'cyber-incident',
    name: 'Cyber Incident Response',
    description: 'Investigate and respond to cyber security incidents and breaches',
    type: 'cyber-incident' as const,
    icon: '🔐',
    defaultDataSources: ['CYBINT', 'OSINT', 'SIGINT']
  },
  {
    id: 'infrastructure',
    name: 'Critical Infrastructure Protection',
    description: 'Monitor and protect critical infrastructure from threats',
    type: 'infrastructure' as const,
    icon: '🏭',
    defaultDataSources: ['GEOINT', 'OSINT', 'HUMINT']
  },
  {
    id: 'threat-campaign',
    name: 'Threat Campaign Tracking',
    description: 'Monitor and analyze ongoing threat campaigns and TTPs',
    type: 'threat-analysis' as const,
    icon: '🎯',
    defaultDataSources: ['CYBINT', 'OSINT', 'FININT']
  }
];

export default function ProjectManagementPanel({ isOpen, onClose }: ProjectManagementPanelProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showTemplates, setShowTemplates] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState<'all' | 'active' | 'paused' | 'completed'>('all');
  
  const snap = useSnapshot(store);

  useEffect(() => {
    if (isOpen) {
      loadProjects();
    }
  }, [isOpen]);

  const loadProjects = async () => {
    try {
      const response = await fetch('http://localhost:8001/v1/projects');
      if (response.ok) {
        const data = await response.json();
        setProjects(data.projects || []);
      }
    } catch (error) {
      console.error('Failed to load projects:', error);
      // Load mock data for demo
      setProjects([
        {
          id: '1',
          name: 'Arctic Submarine Monitoring',
          description: 'Track submarine activity in Arctic regions',
          type: 'maritime',
          status: 'active',
          createdAt: new Date('2024-01-15'),
          lastUpdated: new Date(),
          analysisCount: 24,
          dataSources: ['SIGINT', 'GEOINT', 'OSINT'],
          participants: 5,
          insights: 47,
          confidence: 0.87
        },
        {
          id: '2',
          name: 'APT Threat Campaign Analysis',
          description: 'Ongoing analysis of APT28 activities',
          type: 'cyber-incident',
          status: 'active',
          createdAt: new Date('2024-02-01'),
          lastUpdated: new Date(),
          analysisCount: 18,
          dataSources: ['CYBINT', 'OSINT'],
          participants: 3,
          insights: 32,
          confidence: 0.92
        }
      ]);
    }
  };

  const createProject = async (template?: typeof PROJECT_TEMPLATES[0]) => {
    const newProject: Project = {
      id: `project-${Date.now()}`,
      name: template?.name || 'New Project',
      description: template?.description || '',
      type: template?.type || 'custom',
      status: 'active',
      createdAt: new Date(),
      lastUpdated: new Date(),
      analysisCount: 0,
      dataSources: template?.defaultDataSources || [],
      participants: 1,
      insights: 0,
      confidence: 0
    };

    setProjects(prev => [newProject, ...prev]);
    setSelectedProject(newProject);
    setShowCreateModal(false);
    setShowTemplates(false);
  };

  const exportProject = async (projectId: string) => {
    // Export project data
    console.log('Exporting project:', projectId);
  };

  const shareProject = async (projectId: string) => {
    // Share project with team
    console.log('Sharing project:', projectId);
  };

  const archiveProject = async (projectId: string) => {
    setProjects(prev => prev.map(p => 
      p.id === projectId ? { ...p, status: 'completed' as const } : p
    ));
  };

  const filteredProjects = projects.filter(p => {
    const matchesSearch = p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         p.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filterStatus === 'all' || p.status === filterStatus;
    return matchesSearch && matchesFilter;
  });

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
          className="relative w-full max-w-7xl max-h-[90vh] overflow-hidden rounded-2xl"
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
                <Folder className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold" style={{ color: theme.text }}>
                  Project Management
                </h2>
                <p className="text-sm" style={{ color: theme.textSecondary }}>
                  Organize analyses into missions and campaigns
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
          <div className="flex h-[calc(90vh-100px)]">
            {/* Sidebar - Project List */}
            <div 
              className="w-1/3 border-r overflow-y-auto p-4"
              style={{ borderColor: theme.border }}
            >
              {/* Search & Filter */}
              <div className="mb-4 space-y-3">
                <div className="relative">
                  <Search 
                    className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4" 
                    style={{ color: theme.textSecondary }} 
                  />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search projects..."
                    className="w-full pl-10 pr-4 py-2 rounded-lg text-sm"
                    style={{
                      background: theme.cardBg,
                      color: theme.text,
                      border: `1px solid ${theme.border}`
                    }}
                  />
                </div>

                <div className="flex gap-2">
                  {(['all', 'active', 'paused', 'completed'] as const).map(status => (
                    <button
                      key={status}
                      onClick={() => setFilterStatus(status)}
                      className="px-3 py-1 rounded-lg text-xs font-medium capitalize transition-all"
                      style={{
                        background: filterStatus === status ? theme.accent : theme.cardBg,
                        color: filterStatus === status ? 'white' : theme.text,
                        border: `1px solid ${filterStatus === status ? theme.accent : theme.border}`
                      }}
                    >
                      {status}
                    </button>
                  ))}
                </div>
              </div>

              {/* New Project Button */}
              <button
                onClick={() => setShowTemplates(true)}
                className="w-full mb-4 py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-all hover:opacity-90"
                style={{ background: theme.accent, color: 'white' }}
              >
                <Plus className="w-5 h-5" />
                New Project
              </button>

              {/* Project List */}
              <div className="space-y-2">
                {filteredProjects.map(project => (
                  <motion.button
                    key={project.id}
                    onClick={() => setSelectedProject(project)}
                    className="w-full p-4 rounded-lg text-left transition-all"
                    style={{
                      background: selectedProject?.id === project.id ? `${theme.accent}20` : theme.cardBg,
                      border: `1px solid ${selectedProject?.id === project.id ? theme.accent : theme.border}`
                    }}
                    whileHover={{ scale: 1.02 }}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="font-semibold text-sm" style={{ color: theme.text }}>
                        {project.name}
                      </h3>
                      <div 
                        className="px-2 py-0.5 rounded text-xs font-medium"
                        style={{
                          background: project.status === 'active' ? '#4CAF5020' : 
                                     project.status === 'paused' ? '#FFD70020' : '#88888820',
                          color: project.status === 'active' ? '#4CAF50' : 
                                 project.status === 'paused' ? '#FFD700' : '#888888'
                        }}
                      >
                        {project.status}
                      </div>
                    </div>
                    
                    <p className="text-xs mb-3 line-clamp-2" style={{ color: theme.textSecondary }}>
                      {project.description}
                    </p>
                    
                    <div className="flex items-center gap-4 text-xs" style={{ color: theme.textSecondary }}>
                      <div className="flex items-center gap-1">
                        <FileText className="w-3 h-3" />
                        <span>{project.analysisCount}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Target className="w-3 h-3" />
                        <span>{project.insights}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <TrendingUp className="w-3 h-3" />
                        <span>{(project.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Main Content - Project Details */}
            <div className="flex-1 overflow-y-auto p-6">
              {selectedProject ? (
                <div>
                  {/* Project Header */}
                  <div className="mb-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h2 className="text-2xl font-bold mb-2" style={{ color: theme.text }}>
                          {selectedProject.name}
                        </h2>
                        <p className="text-sm" style={{ color: theme.textSecondary }}>
                          {selectedProject.description}
                        </p>
                      </div>
                      
                      <div className="flex gap-2">
                        <button
                          onClick={() => shareProject(selectedProject.id)}
                          className="p-2 rounded-lg transition-all hover:bg-white/10"
                          title="Share project"
                        >
                          <Share2 className="w-5 h-5" style={{ color: theme.accent }} />
                        </button>
                        <button
                          onClick={() => exportProject(selectedProject.id)}
                          className="p-2 rounded-lg transition-all hover:bg-white/10"
                          title="Export project"
                        >
                          <Download className="w-5 h-5" style={{ color: theme.accent }} />
                        </button>
                        <button
                          onClick={() => archiveProject(selectedProject.id)}
                          className="p-2 rounded-lg transition-all hover:bg-white/10"
                          title="Archive project"
                        >
                          <Archive className="w-5 h-5" style={{ color: theme.accent }} />
                        </button>
                      </div>
                    </div>

                    {/* Project Stats */}
                    <div className="grid grid-cols-4 gap-4">
                      <div 
                        className="p-4 rounded-lg"
                        style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs" style={{ color: theme.textSecondary }}>Analyses</span>
                          <FileText className="w-4 h-4" style={{ color: theme.accent }} />
                        </div>
                        <div className="text-2xl font-bold" style={{ color: theme.text }}>
                          {selectedProject.analysisCount}
                        </div>
                      </div>

                      <div 
                        className="p-4 rounded-lg"
                        style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs" style={{ color: theme.textSecondary }}>Data Sources</span>
                          <FileText className="w-4 h-4" style={{ color: theme.accent }} />
                        </div>
                        <div className="text-2xl font-bold" style={{ color: theme.text }}>
                          {selectedProject.dataSources.length}
                        </div>
                      </div>

                      <div 
                        className="p-4 rounded-lg"
                        style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs" style={{ color: theme.textSecondary }}>Insights</span>
                          <Target className="w-4 h-4" style={{ color: theme.accent }} />
                        </div>
                        <div className="text-2xl font-bold" style={{ color: theme.text }}>
                          {selectedProject.insights}
                        </div>
                      </div>

                      <div 
                        className="p-4 rounded-lg"
                        style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs" style={{ color: theme.textSecondary }}>Confidence</span>
                          <TrendingUp className="w-4 h-4" style={{ color: theme.accent }} />
                        </div>
                        <div className="text-2xl font-bold" style={{ color: theme.text }}>
                          {(selectedProject.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Project Timeline */}
                  <div className="mb-6">
                    <h3 className="text-lg font-bold mb-4" style={{ color: theme.text }}>
                      Project Timeline
                    </h3>
                    <div 
                      className="p-4 rounded-lg"
                      style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                    >
                      <div className="flex items-center justify-between text-sm">
                        <div>
                          <p style={{ color: theme.textSecondary }}>Created</p>
                          <p style={{ color: theme.text }}>{selectedProject.createdAt.toLocaleDateString()}</p>
                        </div>
                        <div>
                          <p style={{ color: theme.textSecondary }}>Last Updated</p>
                          <p style={{ color: theme.text }}>{selectedProject.lastUpdated.toLocaleDateString()}</p>
                        </div>
                        <div>
                          <p style={{ color: theme.textSecondary }}>Duration</p>
                          <p style={{ color: theme.text }}>
                            {Math.ceil((new Date().getTime() - selectedProject.createdAt.getTime()) / (1000 * 60 * 60 * 24))} days
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Data Sources */}
                  <div className="mb-6">
                    <h3 className="text-lg font-bold mb-4" style={{ color: theme.text }}>
                      Data Sources
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {selectedProject.dataSources.map((source, idx) => (
                        <div
                          key={idx}
                          className="px-3 py-2 rounded-lg text-sm font-medium"
                          style={{
                            background: `${theme.accent}20`,
                            color: theme.accent,
                            border: `1px solid ${theme.accent}40`
                          }}
                        >
                          {source}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Recent Activity */}
                  <div>
                    <h3 className="text-lg font-bold mb-4" style={{ color: theme.text }}>
                      Recent Activity
                    </h3>
                    <div className="space-y-2">
                      {[1, 2, 3].map((_, idx) => (
                        <div
                          key={idx}
                          className="p-4 rounded-lg flex items-center gap-3"
                          style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                        >
                          <CheckCircle className="w-5 h-5 text-green-400" />
                          <div className="flex-1">
                            <p className="text-sm font-medium" style={{ color: theme.text }}>
                              Analysis completed
                            </p>
                            <p className="text-xs" style={{ color: theme.textSecondary }}>
                              2 hours ago
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <Folder className="w-16 h-16 mx-auto mb-4 opacity-50" style={{ color: theme.textSecondary }} />
                    <p style={{ color: theme.textSecondary }}>Select a project to view details</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Templates Modal */}
          <AnimatePresence>
            {showTemplates && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 flex items-center justify-center p-6"
                style={{ background: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(8px)' }}
                onClick={() => setShowTemplates(false)}
              >
                <motion.div
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.9, opacity: 0 }}
                  className="w-full max-w-4xl rounded-2xl p-6"
                  style={{ background: theme.bg, border: `1px solid ${theme.border}` }}
                  onClick={(e) => e.stopPropagation()}
                >
                  <h3 className="text-xl font-bold mb-6" style={{ color: theme.text }}>
                    Choose a Project Template
                  </h3>
                  
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    {PROJECT_TEMPLATES.map(template => (
                      <button
                        key={template.id}
                        onClick={() => createProject(template)}
                        className="p-6 rounded-lg text-left transition-all hover:scale-105"
                        style={{
                          background: theme.cardBg,
                          border: `1px solid ${theme.border}`
                        }}
                      >
                        <div className="text-4xl mb-3">{template.icon}</div>
                        <h4 className="font-bold mb-2" style={{ color: theme.text }}>
                          {template.name}
                        </h4>
                        <p className="text-sm mb-4" style={{ color: theme.textSecondary }}>
                          {template.description}
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {template.defaultDataSources.map((source, idx) => (
                            <span
                              key={idx}
                              className="px-2 py-1 rounded text-xs"
                              style={{
                                background: `${theme.accent}20`,
                                color: theme.accent
                              }}
                            >
                              {source}
                            </span>
                          ))}
                        </div>
                      </button>
                    ))}
                  </div>

                  <button
                    onClick={() => createProject()}
                    className="w-full py-3 rounded-lg font-medium"
                    style={{
                      background: theme.cardBg,
                      color: theme.text,
                      border: `1px solid ${theme.border}`
                    }}
                  >
                    Start from Scratch
                  </button>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

