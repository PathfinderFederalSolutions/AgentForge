'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Plus,
  Folder,
  FileText,
  Calendar,
  Users,
  Download,
  Share2,
  Archive,
  Play,
  Pause,
  Trash2,
  Edit,
  Clock,
  CheckCircle2,
  AlertCircle,
  TrendingUp
} from 'lucide-react';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/store';

interface ProjectManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

interface Project {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'paused' | 'completed' | 'archived';
  createdAt: Date;
  updatedAt: Date;
  dataSources: string[];
  analyses: number;
  results: number;
  teamMembers: string[];
  template?: string;
}

export default function ProjectManager({ isOpen, onClose }: ProjectManagerProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDescription, setNewProjectDescription] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const snap = useSnapshot(store);

  const theme = snap.theme === 'day' ? {
    bg: '#05080D',
    text: '#D6E2F0',
    textSecondary: '#B8C5D1',
    accent: '#00A39B',
    neon: '#CCFF00',
    border: 'rgba(255,255,255,0.2)',
    cardBg: 'rgba(255,255,255,0.08)'
  } : {
    bg: '#000000',
    text: '#FF2B2B',
    textSecondary: '#891616',
    accent: '#FF2B2B',
    border: 'rgba(137,22,22,0.6)',
    cardBg: 'rgba(137,22,22,0.15)'
  };

  const projectTemplates = [
    {
      id: 'submarine-threat',
      name: 'Submarine Threat Analysis',
      description: 'Maritime threat detection and tracking',
      domains: ['GEOINT', 'SIGINT'],
      defaultSources: ['sonar_data', 'satellite_imagery']
    },
    {
      id: 'cyber-incident',
      name: 'Cyber Incident Response',
      description: 'Cybersecurity threat analysis and response',
      domains: ['CYBINT'],
      defaultSources: ['network_logs', 'threat_intel']
    },
    {
      id: 'infrastructure-protection',
      name: 'Infrastructure Protection',
      description: 'Critical infrastructure monitoring',
      domains: ['GEOINT', 'OSINT'],
      defaultSources: ['satellite_data', 'social_media']
    },
    {
      id: 'custom',
      name: 'Custom Project',
      description: 'Start from scratch with custom configuration',
      domains: [],
      defaultSources: []
    }
  ];

  useEffect(() => {
    // Load projects from localStorage or backend
    const savedProjects = localStorage.getItem('agentforge_projects');
    if (savedProjects) {
      const parsed = JSON.parse(savedProjects);
      setProjects(parsed.map((p: any) => ({
        ...p,
        createdAt: new Date(p.createdAt),
        updatedAt: new Date(p.updatedAt)
      })));
    }
  }, [isOpen]);

  const createProject = () => {
    if (!newProjectName.trim()) return;

    const template = projectTemplates.find(t => t.id === selectedTemplate);
    const newProject: Project = {
      id: `project-${Date.now()}`,
      name: newProjectName,
      description: newProjectDescription,
      status: 'active',
      createdAt: new Date(),
      updatedAt: new Date(),
      dataSources: template?.defaultSources || [],
      analyses: 0,
      results: 0,
      teamMembers: ['John Doe'],
      template: selectedTemplate
    };

    const updated = [...projects, newProject];
    setProjects(updated);
    localStorage.setItem('agentforge_projects', JSON.stringify(updated));

    // Reset form
    setNewProjectName('');
    setNewProjectDescription('');
    setSelectedTemplate('');
    setShowCreateModal(false);
  };

  const updateProjectStatus = (projectId: string, status: 'active' | 'paused' | 'completed' | 'archived') => {
    const updated = projects.map(p =>
      p.id === projectId ? { ...p, status, updatedAt: new Date() } : p
    );
    setProjects(updated);
    localStorage.setItem('agentforge_projects', JSON.stringify(updated));
  };

  const deleteProject = (projectId: string) => {
    const updated = projects.filter(p => p.id !== projectId);
    setProjects(updated);
    localStorage.setItem('agentforge_projects', JSON.stringify(updated));
    if (selectedProject?.id === projectId) {
      setSelectedProject(null);
    }
  };

  const exportProject = (project: Project) => {
    const exportData = {
      project,
      exportDate: new Date().toISOString(),
      version: '1.0'
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${project.name.replace(/\s+/g, '_')}_export.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <Play className="w-4 h-4 text-green-400" />;
      case 'paused': return <Pause className="w-4 h-4 text-yellow-400" />;
      case 'completed': return <CheckCircle2 className="w-4 h-4 text-blue-400" />;
      case 'archived': return <Archive className="w-4 h-4 text-gray-400" />;
      default: return <AlertCircle className="w-4 h-4 text-gray-400" />;
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        style={{
          background: 'rgba(0,0,0,0.85)',
          backdropFilter: 'blur(8px)'
        }}
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="relative w-full max-w-6xl max-h-[90vh] overflow-hidden rounded-2xl"
          style={{
            background: theme.bg,
            border: `1px solid ${theme.border}`,
            boxShadow: `0 0 40px ${theme.accent}20`
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div 
            className="flex items-center justify-between p-6 border-b"
            style={{ borderColor: theme.border }}
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-[#00A39B] to-[#CCFF00] flex items-center justify-center">
                <Folder className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold" style={{ color: theme.text }}>
                  Project Manager
                </h2>
                <p className="text-sm" style={{ color: theme.textSecondary }}>
                  Organize analyses into projects and missions
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowCreateModal(true)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg transition-colors"
                style={{
                  background: theme.accent,
                  color: 'white'
                }}
              >
                <Plus className="w-4 h-4" />
                <span className="text-sm font-semibold">New Project</span>
              </button>
              <button
                onClick={onClose}
                className="p-2 rounded-lg hover:bg-white/10 transition-colors"
              >
                <X className="w-6 h-6" style={{ color: theme.textSecondary }} />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="flex h-[calc(90vh-100px)]">
            {/* Project List */}
            <div 
              className="w-1/3 overflow-y-auto p-4 border-r"
              style={{ borderColor: theme.border }}
            >
              <h3 className="text-sm font-semibold mb-3 uppercase tracking-wide" style={{ color: theme.textSecondary }}>
                Projects ({projects.length})
              </h3>
              
              {projects.length === 0 ? (
                <div className="text-center py-12">
                  <Folder className="w-12 h-12 mx-auto mb-3" style={{ color: theme.textSecondary, opacity: 0.5 }} />
                  <p className="text-sm" style={{ color: theme.textSecondary }}>
                    No projects yet
                  </p>
                  <button
                    onClick={() => setShowCreateModal(true)}
                    className="mt-4 px-4 py-2 rounded-lg text-sm font-medium"
                    style={{
                      background: `${theme.accent}20`,
                      color: theme.accent
                    }}
                  >
                    Create your first project
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  {projects.map(project => (
                    <motion.div
                      key={project.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="p-3 rounded-lg cursor-pointer transition-all"
                      style={{
                        background: selectedProject?.id === project.id ? `${theme.accent}20` : theme.cardBg,
                        border: `1px solid ${selectedProject?.id === project.id ? theme.accent : theme.border}`
                      }}
                      onClick={() => setSelectedProject(project)}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          {getStatusIcon(project.status)}
                          <h4 className="font-semibold text-sm truncate" style={{ color: theme.text }}>
                            {project.name}
                          </h4>
                        </div>
                        <span 
                          className="text-xs px-2 py-0.5 rounded"
                          style={{
                            background: `${theme.accent}20`,
                            color: theme.accent
                          }}
                        >
                          {project.status}
                        </span>
                      </div>
                      <p className="text-xs mb-2 line-clamp-2" style={{ color: theme.textSecondary }}>
                        {project.description}
                      </p>
                      <div className="flex items-center gap-3 text-xs" style={{ color: theme.textSecondary }}>
                        <div className="flex items-center gap-1">
                          <FileText className="w-3 h-3" />
                          <span>{project.analyses}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <TrendingUp className="w-3 h-3" />
                          <span>{project.results}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          <span>{project.updatedAt.toLocaleDateString()}</span>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>

            {/* Project Details */}
            <div className="flex-1 overflow-y-auto p-6">
              {selectedProject ? (
                <div>
                  {/* Project Header */}
                  <div className="flex items-start justify-between mb-6">
                    <div className="flex-1">
                      <h3 className="text-2xl font-bold mb-2" style={{ color: theme.text }}>
                        {selectedProject.name}
                      </h3>
                      <p className="text-sm mb-4" style={{ color: theme.textSecondary }}>
                        {selectedProject.description}
                      </p>
                      <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2">
                          <Calendar className="w-4 h-4" style={{ color: theme.accent }} />
                          <span className="text-xs" style={{ color: theme.textSecondary }}>
                            Created {selectedProject.createdAt.toLocaleDateString()}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Users className="w-4 h-4" style={{ color: theme.accent }} />
                          <span className="text-xs" style={{ color: theme.textSecondary }}>
                            {selectedProject.teamMembers.length} members
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      {selectedProject.status === 'active' && (
                        <button
                          onClick={() => updateProjectStatus(selectedProject.id, 'paused')}
                          className="p-2 rounded-lg hover:bg-white/10"
                          title="Pause"
                        >
                          <Pause className="w-4 h-4" style={{ color: theme.text }} />
                        </button>
                      )}
                      {selectedProject.status === 'paused' && (
                        <button
                          onClick={() => updateProjectStatus(selectedProject.id, 'active')}
                          className="p-2 rounded-lg hover:bg-white/10"
                          title="Resume"
                        >
                          <Play className="w-4 h-4" style={{ color: theme.text }} />
                        </button>
                      )}
                      {(selectedProject.status === 'active' || selectedProject.status === 'paused') && (
                        <button
                          onClick={() => updateProjectStatus(selectedProject.id, 'completed')}
                          className="p-2 rounded-lg hover:bg-white/10"
                          title="Mark Complete"
                        >
                          <CheckCircle2 className="w-4 h-4" style={{ color: theme.text }} />
                        </button>
                      )}
                      <button
                        onClick={() => exportProject(selectedProject)}
                        className="p-2 rounded-lg hover:bg-white/10"
                        title="Export"
                      >
                        <Download className="w-4 h-4" style={{ color: theme.text }} />
                      </button>
                      <button
                        className="p-2 rounded-lg hover:bg-white/10"
                        title="Share"
                      >
                        <Share2 className="w-4 h-4" style={{ color: theme.text }} />
                      </button>
                      <button
                        onClick={() => {
                          if (confirm('Are you sure you want to delete this project?')) {
                            deleteProject(selectedProject.id);
                          }
                        }}
                        className="p-2 rounded-lg hover:bg-white/10"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4 text-red-400" />
                      </button>
                    </div>
                  </div>

                  {/* Project Stats */}
                  <div className="grid grid-cols-4 gap-4 mb-6">
                    <div className="p-4 rounded-lg" style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}>
                      <div className="text-2xl font-bold mb-1" style={{ color: theme.accent }}>
                        {selectedProject.dataSources.length}
                      </div>
                      <div className="text-xs" style={{ color: theme.textSecondary }}>
                        Data Sources
                      </div>
                    </div>
                    <div className="p-4 rounded-lg" style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}>
                      <div className="text-2xl font-bold mb-1" style={{ color: theme.accent }}>
                        {selectedProject.analyses}
                      </div>
                      <div className="text-xs" style={{ color: theme.textSecondary }}>
                        Analyses
                      </div>
                    </div>
                    <div className="p-4 rounded-lg" style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}>
                      <div className="text-2xl font-bold mb-1" style={{ color: theme.accent }}>
                        {selectedProject.results}
                      </div>
                      <div className="text-xs" style={{ color: theme.textSecondary }}>
                        Results
                      </div>
                    </div>
                    <div className="p-4 rounded-lg" style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}>
                      <div className="text-2xl font-bold mb-1" style={{ color: theme.accent }}>
                        {selectedProject.teamMembers.length}
                      </div>
                      <div className="text-xs" style={{ color: theme.textSecondary }}>
                        Team Members
                      </div>
                    </div>
                  </div>

                  {/* Project Timeline */}
                  <div className="mb-6">
                    <h4 className="text-sm font-semibold mb-3" style={{ color: theme.text }}>
                      Timeline
                    </h4>
                    <div className="space-y-3">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: `${theme.accent}20` }}>
                          <CheckCircle2 className="w-4 h-4" style={{ color: theme.accent }} />
                        </div>
                        <div className="flex-1">
                          <div className="text-sm font-medium" style={{ color: theme.text }}>
                            Project Created
                          </div>
                          <div className="text-xs" style={{ color: theme.textSecondary }}>
                            {selectedProject.createdAt.toLocaleString()}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: `${theme.accent}20` }}>
                          <Edit className="w-4 h-4" style={{ color: theme.accent }} />
                        </div>
                        <div className="flex-1">
                          <div className="text-sm font-medium" style={{ color: theme.text }}>
                            Last Updated
                          </div>
                          <div className="text-xs" style={{ color: theme.textSecondary }}>
                            {selectedProject.updatedAt.toLocaleString()}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Data Sources */}
                  <div>
                    <h4 className="text-sm font-semibold mb-3" style={{ color: theme.text }}>
                      Data Sources
                    </h4>
                    {selectedProject.dataSources.length === 0 ? (
                      <div className="text-center py-8 rounded-lg" style={{ background: theme.cardBg }}>
                        <FileText className="w-8 h-8 mx-auto mb-2" style={{ color: theme.textSecondary, opacity: 0.5 }} />
                        <p className="text-sm" style={{ color: theme.textSecondary }}>
                          No data sources attached
                        </p>
                      </div>
                    ) : (
                      <div className="grid grid-cols-2 gap-3">
                        {selectedProject.dataSources.map((source, idx) => (
                          <div
                            key={idx}
                            className="p-3 rounded-lg flex items-center gap-2"
                            style={{ background: theme.cardBg, border: `1px solid ${theme.border}` }}
                          >
                            <FileText className="w-4 h-4" style={{ color: theme.accent }} />
                            <span className="text-sm" style={{ color: theme.text }}>
                              {source}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <Folder className="w-16 h-16 mx-auto mb-4" style={{ color: theme.textSecondary, opacity: 0.3 }} />
                    <p style={{ color: theme.textSecondary }}>
                      Select a project to view details
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </motion.div>

        {/* Create Project Modal */}
        <AnimatePresence>
          {showCreateModal && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex items-center justify-center p-4"
              style={{ background: 'rgba(0,0,0,0.5)' }}
              onClick={() => setShowCreateModal(false)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="w-full max-w-lg rounded-xl p-6"
                style={{
                  background: theme.bg,
                  border: `1px solid ${theme.border}`
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <h3 className="text-xl font-bold mb-4" style={{ color: theme.text }}>
                  Create New Project
                </h3>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2" style={{ color: theme.text }}>
                      Project Name
                    </label>
                    <input
                      type="text"
                      value={newProjectName}
                      onChange={(e) => setNewProjectName(e.target.value)}
                      placeholder="Enter project name"
                      className="w-full px-4 py-2 rounded-lg"
                      style={{
                        background: theme.cardBg,
                        border: `1px solid ${theme.border}`,
                        color: theme.text
                      }}
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2" style={{ color: theme.text }}>
                      Description
                    </label>
                    <textarea
                      value={newProjectDescription}
                      onChange={(e) => setNewProjectDescription(e.target.value)}
                      placeholder="Enter project description"
                      rows={3}
                      className="w-full px-4 py-2 rounded-lg resize-none"
                      style={{
                        background: theme.cardBg,
                        border: `1px solid ${theme.border}`,
                        color: theme.text
                      }}
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2" style={{ color: theme.text }}>
                      Template
                    </label>
                    <div className="grid grid-cols-2 gap-3">
                      {projectTemplates.map(template => (
                        <button
                          key={template.id}
                          onClick={() => setSelectedTemplate(template.id)}
                          className="p-3 rounded-lg text-left transition-all"
                          style={{
                            background: selectedTemplate === template.id ? `${theme.accent}20` : theme.cardBg,
                            border: `1px solid ${selectedTemplate === template.id ? theme.accent : theme.border}`
                          }}
                        >
                          <div className="font-semibold text-sm mb-1" style={{ color: theme.text }}>
                            {template.name}
                          </div>
                          <div className="text-xs" style={{ color: theme.textSecondary }}>
                            {template.description}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="flex gap-3 pt-4">
                    <button
                      onClick={() => setShowCreateModal(false)}
                      className="flex-1 px-4 py-2 rounded-lg"
                      style={{
                        background: theme.cardBg,
                        color: theme.text
                      }}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={createProject}
                      disabled={!newProjectName.trim()}
                      className="flex-1 px-4 py-2 rounded-lg font-semibold"
                      style={{
                        background: theme.accent,
                        color: 'white',
                        opacity: !newProjectName.trim() ? 0.5 : 1
                      }}
                    >
                      Create Project
                    </button>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </AnimatePresence>
  );
}

