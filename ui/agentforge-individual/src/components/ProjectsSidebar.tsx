'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Folder,
  Plus,
  Search,
  Calendar,
  Users,
  Share,
  Download,
  Eye,
  Edit,
  Archive,
  Target,
  Shield,
  Zap,
  Globe
} from 'lucide-react';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/store';

export type Project = {
  id: string;
  name: string;
  description: string;
  type: 'intelligence' | 'planning' | 'wargaming' | 'analysis';
  status: 'active' | 'completed' | 'archived';
  createdAt: Date;
  updatedAt: Date;
  dataSources: string[];
  collaborators: string[];
  intelligenceDomain?: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  tags: string[];
  timeline: {
    startDate?: Date;
    endDate?: Date;
    milestones: Array<{
      id: string;
      title: string;
      completed: boolean;
      dueDate?: Date;
    }>;
  };
};

const projectTemplates = [
  {
    name: 'Submarine Threat Analysis',
    type: 'intelligence' as const,
    description: 'Track and analyze submarine threats in designated area',
    intelligenceDomain: 'SIGINT',
    tags: ['naval', 'threat', 'submarine', 'tracking']
  },
  {
    name: 'Cyber Incident Response',
    type: 'planning' as const,
    description: 'Coordinate response to cyber security incidents',
    intelligenceDomain: 'CYBINT',
    tags: ['cyber', 'security', 'incident', 'response']
  },
  {
    name: 'Infrastructure Protection',
    type: 'wargaming' as const,
    description: 'Simulate and plan critical infrastructure defense',
    intelligenceDomain: 'GEOINT',
    tags: ['infrastructure', 'defense', 'critical', 'protection']
  },
  {
    name: 'Multi-Domain Intelligence Fusion',
    type: 'analysis' as const,
    description: 'Fuse intelligence from multiple domains for comprehensive analysis',
    intelligenceDomain: 'MULTI_DOMAIN',
    tags: ['fusion', 'multi-domain', 'comprehensive', 'analysis']
  }
];

export default function ProjectsSidebar({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<'active' | 'templates' | 'archived'>('active');
  const [searchTerm, setSearchTerm] = useState('');
  const [showCreateProject, setShowCreateProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectType, setNewProjectType] = useState<Project['type']>('intelligence');
  const [selectedTemplate, setSelectedTemplate] = useState<typeof projectTemplates[0] | null>(null);
  const snap = useSnapshot(store);

  // Mock projects data (in production, this would come from backend)
  const [projects, setProjects] = useState<Project[]>([
    {
      id: 'proj-1',
      name: 'Pacific Submarine Tracking',
      description: 'Monitor and track submarine activity in Pacific theater',
      type: 'intelligence',
      status: 'active',
      createdAt: new Date('2024-11-01'),
      updatedAt: new Date(),
      dataSources: ['sonar-feeds', 'satellite-intel', 'naval-reports'],
      collaborators: ['john.doe', 'jane.smith'],
      intelligenceDomain: 'SIGINT',
      priority: 'high',
      tags: ['naval', 'tracking', 'submarine'],
      timeline: {
        startDate: new Date('2024-11-01'),
        milestones: [
          { id: 'm1', title: 'Initial threat assessment', completed: true, dueDate: new Date('2024-11-05') },
          { id: 'm2', title: 'Pattern analysis complete', completed: false, dueDate: new Date('2024-11-15') }
        ]
      }
    },
    {
      id: 'proj-2',
      name: 'Cyber Defense Planning',
      description: 'Develop comprehensive cyber defense strategy',
      type: 'planning',
      status: 'active',
      createdAt: new Date('2024-10-15'),
      updatedAt: new Date(),
      dataSources: ['threat-intel', 'vulnerability-scans', 'incident-reports'],
      collaborators: ['bob.wilson'],
      intelligenceDomain: 'CYBINT',
      priority: 'critical',
      tags: ['cyber', 'defense', 'strategy'],
      timeline: {
        startDate: new Date('2024-10-15'),
        endDate: new Date('2024-12-15'),
        milestones: [
          { id: 'm3', title: 'COAs generated', completed: true, dueDate: new Date('2024-11-01') },
          { id: 'm4', title: 'Strategy finalized', completed: false, dueDate: new Date('2024-11-30') }
        ]
      }
    }
  ]);

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

  const getPriorityColor = (priority: Project['priority']) => {
    switch (priority) {
      case 'critical': return '#FF2B2B';
      case 'high': return '#FF8C00';
      case 'medium': return '#FFD700';
      case 'low': return '#4CAF50';
      default: return theme.accent;
    }
  };

  const getTypeIcon = (type: Project['type']) => {
    switch (type) {
      case 'intelligence': return Shield;
      case 'planning': return Target;
      case 'wargaming': return Zap;
      case 'analysis': return Globe;
      default: return Folder;
    }
  };

  const createProject = () => {
    if (!newProjectName.trim()) return;

    const newProject: Project = {
      id: `proj-${Date.now()}`,
      name: newProjectName,
      description: selectedTemplate?.description || 'New intelligence project',
      type: selectedTemplate?.type || newProjectType,
      status: 'active',
      createdAt: new Date(),
      updatedAt: new Date(),
      dataSources: [],
      collaborators: ['current_user'],
      intelligenceDomain: selectedTemplate?.intelligenceDomain,
      priority: 'medium',
      tags: selectedTemplate?.tags || [],
      timeline: {
        milestones: []
      }
    };

    setProjects(prev => [newProject, ...prev]);
    setNewProjectName('');
    setSelectedTemplate(null);
    setShowCreateProject(false);
  };

  const filteredProjects = projects.filter(project =>
    project.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    project.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
    project.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: '100%' }}
        animate={{ x: 0 }}
        exit={{ x: '100%' }}
        transition={{ type: 'tween', duration: 0.3 }}
        style={{
          position: 'fixed',
          right: '0',
          top: '70px',
          width: '380px',
          background: theme.cardBg,
          backdropFilter: 'blur(20px)',
          borderLeft: `1px solid ${theme.border}`,
          height: 'calc(100vh - 70px)',
          zIndex: 45,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          boxShadow: `-8px 0 40px rgba(0,0,0,0.3)`
        }}
      >
        {/* Sidebar Header */}
        <div
          className="p-4 border-b"
          style={{
            borderColor: theme.border,
            background: theme.cardBg,
            backdropFilter: 'blur(20px)'
          }}
        >
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-bold text-lg" style={{ color: theme.text }}>
              Project Management
            </h2>
            <button
              onClick={onClose}
              className="p-2 rounded-lg btn-hover"
              style={{
                color: theme.text,
                opacity: 0.7
              }}
            >
              ×
            </button>
          </div>

          {/* Tab Selector */}
          <div className="flex gap-1 mb-3">
            {[
              { id: 'active', label: 'Active', count: projects.filter(p => p.status === 'active').length },
              { id: 'templates', label: 'Templates', count: projectTemplates.length },
              { id: 'archived', label: 'Archived', count: projects.filter(p => p.status === 'archived').length }
            ].map(({ id, label, count }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as any)}
                className="flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-all btn-hover"
                style={{
                  background: activeTab === id ? theme.accent : theme.cardBg,
                  color: activeTab === id ? 'white' : theme.text,
                  border: `1px solid ${activeTab === id ? theme.accent : theme.border}`
                }}
              >
                {label} ({count})
              </button>
            ))}
          </div>

          {/* Search and Create */}
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4" style={{ color: theme.textSecondary }} />
              <input
                type="text"
                placeholder="Search projects..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-3 py-2 rounded-lg text-sm"
                style={{
                  background: theme.bg,
                  color: theme.text,
                  border: `1px solid ${theme.border}`
                }}
              />
            </div>
            <button
              onClick={() => setShowCreateProject(true)}
              className="p-2 rounded-lg btn-hover"
              style={{
                background: theme.accent,
                color: 'white'
              }}
              title="Create new project"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-4">
          {activeTab === 'active' && (
            <div className="space-y-3">
              {filteredProjects.filter(p => p.status === 'active').map((project) => {
                const TypeIcon = getTypeIcon(project.type);
                return (
                  <motion.div
                    key={project.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-lg cursor-pointer transition-all btn-hover"
                    style={{
                      background: theme.cardBg,
                      border: `1px solid ${theme.border}`,
                      borderLeft: `3px solid ${getPriorityColor(project.priority)}`
                    }}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <TypeIcon className="w-4 h-4" style={{ color: theme.accent }} />
                        <h3 className="font-medium text-sm" style={{ color: theme.text }}>
                          {project.name}
                        </h3>
                      </div>
                      <div className="flex items-center gap-1">
                        <div
                          className="w-2 h-2 rounded-full"
                          style={{ background: getPriorityColor(project.priority) }}
                          title={`Priority: ${project.priority}`}
                        />
                      </div>
                    </div>

                    <p className="text-xs mb-3" style={{ color: theme.textSecondary, lineHeight: '1.4' }}>
                      {project.description}
                    </p>

                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-3">
                        <span style={{ color: theme.textSecondary }}>
                          {project.dataSources.length} sources
                        </span>
                        <span style={{ color: theme.textSecondary }}>
                          {project.collaborators.length} collaborators
                        </span>
                      </div>
                      <span style={{ color: theme.accent }}>
                        {project.intelligenceDomain}
                      </span>
                    </div>

                    {/* Tags */}
                    {project.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {project.tags.slice(0, 3).map((tag) => (
                          <span
                            key={tag}
                            className="px-2 py-1 rounded text-xs"
                            style={{
                              background: `${theme.accent}20`,
                              color: theme.accent,
                              border: `1px solid ${theme.accent}40`
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                        {project.tags.length > 3 && (
                          <span className="text-xs" style={{ color: theme.textSecondary }}>
                            +{project.tags.length - 3}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Milestones Progress */}
                    <div className="mt-3">
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span style={{ color: theme.textSecondary }}>Progress</span>
                        <span style={{ color: theme.accent }}>
                          {project.timeline.milestones.filter(m => m.completed).length}/{project.timeline.milestones.length}
                        </span>
                      </div>
                      <div
                        className="w-full rounded-full h-1"
                        style={{ background: 'rgba(255,255,255,0.1)' }}
                      >
                        <div
                          className="h-1 rounded-full transition-all duration-300"
                          style={{
                            width: `${project.timeline.milestones.length > 0 ?
                              (project.timeline.milestones.filter(m => m.completed).length / project.timeline.milestones.length) * 100 : 0}%`,
                            background: theme.accent
                          }}
                        />
                      </div>
                    </div>
                  </motion.div>
                );
              })}

              {filteredProjects.filter(p => p.status === 'active').length === 0 && (
                <div className="text-center py-8">
                  <Folder className="w-12 h-12 mx-auto mb-3" style={{ color: theme.textSecondary, opacity: 0.5 }} />
                  <p className="text-sm" style={{ color: theme.textSecondary }}>
                    No active projects found
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'templates' && (
            <div className="space-y-3">
              {projectTemplates.map((template, index) => {
                const TypeIcon = getTypeIcon(template.type);
                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="p-4 rounded-lg cursor-pointer transition-all btn-hover"
                    style={{
                      background: theme.cardBg,
                      border: `1px solid ${theme.border}`
                    }}
                    onClick={() => {
                      setSelectedTemplate(template);
                      setNewProjectName(template.name);
                      setNewProjectType(template.type);
                      setShowCreateProject(true);
                    }}
                  >
                    <div className="flex items-center gap-3 mb-2">
                      <TypeIcon className="w-5 h-5" style={{ color: theme.accent }} />
                      <h3 className="font-medium" style={{ color: theme.text }}>
                        {template.name}
                      </h3>
                    </div>

                    <p className="text-sm mb-3" style={{ color: theme.textSecondary }}>
                      {template.description}
                    </p>

                    <div className="flex items-center justify-between">
                      <div className="flex flex-wrap gap-1">
                        {template.tags.map((tag) => (
                          <span
                            key={tag}
                            className="px-2 py-1 rounded text-xs"
                            style={{
                              background: `${theme.accent}15`,
                              color: theme.accent,
                              border: `1px solid ${theme.accent}30`
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                      <span
                        className="px-2 py-1 rounded text-xs font-medium"
                        style={{
                          background: `${theme.accent}20`,
                          color: theme.accent
                        }}
                      >
                        {template.intelligenceDomain}
                      </span>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          )}

          {activeTab === 'archived' && (
            <div className="space-y-3">
              {projects.filter(p => p.status === 'archived').map((project) => (
                <motion.div
                  key={project.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 rounded-lg opacity-75"
                  style={{
                    background: theme.cardBg,
                    border: `1px solid ${theme.border}`
                  }}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Archive className="w-4 h-4" style={{ color: theme.textSecondary }} />
                    <h3 className="font-medium text-sm line-through" style={{ color: theme.textSecondary }}>
                      {project.name}
                    </h3>
                  </div>
                  <p className="text-xs" style={{ color: theme.textSecondary }}>
                    Archived on {project.updatedAt.toLocaleDateString()}
                  </p>
                </motion.div>
              ))}

              {projects.filter(p => p.status === 'archived').length === 0 && (
                <div className="text-center py-8">
                  <Archive className="w-12 h-12 mx-auto mb-3" style={{ color: theme.textSecondary, opacity: 0.5 }} />
                  <p className="text-sm" style={{ color: theme.textSecondary }}>
                    No archived projects
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Create Project Modal */}
        <AnimatePresence>
          {showCreateProject && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center p-4"
              style={{ background: 'rgba(0, 0, 0, 0.8)' }}
              onClick={() => setShowCreateProject(false)}
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.95, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: 20 }}
                className="relative p-6 rounded-lg"
                style={{
                  width: '400px',
                  background: theme.cardBg,
                  border: `1px solid ${theme.border}`
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <h3 className="font-bold text-lg mb-4" style={{ color: theme.text }}>
                  {selectedTemplate ? 'Create Project from Template' : 'Create New Project'}
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
                      placeholder="Enter project name..."
                      className="w-full px-3 py-2 rounded-lg"
                      style={{
                        background: theme.bg,
                        color: theme.text,
                        border: `1px solid ${theme.border}`
                      }}
                    />
                  </div>

                  {!selectedTemplate && (
                    <div>
                      <label className="block text-sm font-medium mb-2" style={{ color: theme.text }}>
                        Project Type
                      </label>
                      <div className="grid grid-cols-2 gap-2">
                        {[
                          { id: 'intelligence', label: 'Intelligence', icon: Shield },
                          { id: 'planning', label: 'Planning', icon: Target },
                          { id: 'wargaming', label: 'Wargaming', icon: Zap },
                          { id: 'analysis', label: 'Analysis', icon: Globe }
                        ].map(({ id, label, icon: Icon }) => (
                          <button
                            key={id}
                            onClick={() => setNewProjectType(id as Project['type'])}
                            className="p-3 rounded-lg text-left transition-all btn-hover"
                            style={{
                              background: newProjectType === id ? theme.accent : theme.cardBg,
                              border: `1px solid ${newProjectType === id ? theme.accent : theme.border}`,
                              color: newProjectType === id ? 'white' : theme.text
                            }}
                          >
                            <Icon className="w-4 h-4 mb-1" />
                            <div className="text-sm font-medium">{label}</div>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {selectedTemplate && (
                    <div className="p-3 rounded-lg" style={{ background: `${theme.accent}15`, border: `1px solid ${theme.accent}30` }}>
                      <p className="text-sm" style={{ color: theme.text }}>
                        <strong>Template:</strong> {selectedTemplate.name}
                      </p>
                      <p className="text-xs mt-1" style={{ color: theme.textSecondary }}>
                        {selectedTemplate.description}
                      </p>
                    </div>
                  )}
                </div>

                <div className="flex gap-3 mt-6">
                  <button
                    onClick={() => setShowCreateProject(false)}
                    className="flex-1 px-4 py-2 rounded-lg btn-hover"
                    style={{
                      background: theme.cardBg,
                      color: theme.text,
                      border: `1px solid ${theme.border}`
                    }}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={createProject}
                    disabled={!newProjectName.trim()}
                    className="flex-1 px-4 py-2 rounded-lg btn-primary disabled-opacity disabled-cursor"
                  >
                    Create Project
                  </button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </AnimatePresence>
  );
}
