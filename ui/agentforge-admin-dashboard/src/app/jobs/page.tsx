'use client';

import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Briefcase,
  Plus,
  Play,
  Pause,
  Square,
  RotateCcw,
  Trash2,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Filter,
  Download,
  Eye,
  MoreVertical
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Modal } from '@/components/ui/Modal';
import { Layout } from '@/components/layout/Layout';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';

// Extended mock job data
const mockJobs = [
  {
    id: 'job-001',
    title: 'Document Analysis Pipeline',
    description: 'Analyze 10,000 legal documents for key insights and compliance issues',
    status: 'running',
    priority: 'high',
    progress: 67,
    createdAt: new Date('2024-01-15T10:30:00Z'),
    startedAt: new Date('2024-01-15T10:32:00Z'),
    estimatedCompletion: new Date('2024-01-15T12:45:00Z'),
    assignedAgent: 'Neural Processor Alpha',
    tags: ['legal', 'compliance', 'nlp'],
    resourceUsage: { cpu: 78, memory: 65, gpu: 89 },
    logs: ['Started document ingestion', 'Processing batch 1/15', 'NLP analysis in progress']
  },
  {
    id: 'job-002',
    title: 'Quantum Optimization Task',
    description: 'Optimize neural network hyperparameters using quantum computing algorithms',
    status: 'queued',
    priority: 'medium',
    progress: 0,
    createdAt: new Date('2024-01-15T11:15:00Z'),
    estimatedCompletion: new Date('2024-01-15T14:30:00Z'),
    assignedAgent: null,
    tags: ['optimization', 'quantum', 'ml'],
    resourceUsage: { cpu: 0, memory: 0, gpu: 0 },
    logs: ['Job queued for processing']
  },
  {
    id: 'job-003',
    title: 'Real-time Data Processing',
    description: 'Process streaming sensor data from IoT devices in manufacturing plant',
    status: 'completed',
    priority: 'high',
    progress: 100,
    createdAt: new Date('2024-01-15T09:00:00Z'),
    startedAt: new Date('2024-01-15T09:02:00Z'),
    completedAt: new Date('2024-01-15T10:15:00Z'),
    assignedAgent: 'Data Processor Gamma',
    tags: ['iot', 'streaming', 'manufacturing'],
    resourceUsage: { cpu: 0, memory: 0, gpu: 0 },
    logs: ['Processing completed successfully', 'Generated 1,247 insights', 'Data archived']
  },
  {
    id: 'job-004',
    title: 'Security Threat Analysis',
    description: 'Analyze network traffic patterns for potential security threats',
    status: 'failed',
    priority: 'critical',
    progress: 23,
    createdAt: new Date('2024-01-15T08:45:00Z'),
    startedAt: new Date('2024-01-15T08:47:00Z'),
    failedAt: new Date('2024-01-15T09:12:00Z'),
    assignedAgent: 'Security Monitor Delta',
    tags: ['security', 'network', 'threat-detection'],
    resourceUsage: { cpu: 0, memory: 0, gpu: 0 },
    logs: ['Started network analysis', 'Detected anomaly in traffic', 'Error: Memory overflow', 'Job terminated']
  },
  {
    id: 'job-005',
    title: 'Image Classification Training',
    description: 'Train deep learning model on medical imaging dataset',
    status: 'paused',
    priority: 'low',
    progress: 45,
    createdAt: new Date('2024-01-15T07:30:00Z'),
    startedAt: new Date('2024-01-15T07:35:00Z'),
    pausedAt: new Date('2024-01-15T09:20:00Z'),
    assignedAgent: 'ML Trainer Epsilon',
    tags: ['ml', 'medical', 'computer-vision'],
    resourceUsage: { cpu: 0, memory: 0, gpu: 0 },
    logs: ['Training started', 'Epoch 1/10 completed', 'Training paused by user']
  }
];

export default function JobsPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [priorityFilter, setPriorityFilter] = useState('all');
  const [selectedJob, setSelectedJob] = useState<any | null>(null);
  const [sourceFilter, setSourceFilter] = useState('all'); // Add source filter
  const [showCreateModal, setShowCreateModal] = useState(false);
  const snap = useSnapshot(store);

  // Combine mock jobs with real jobs from store
  const allJobs = useMemo(() => {
    const realJobs = snap.jobs.map(job => ({
      id: job.id,
      title: (job as any).title || `Task ${job.id}`,
      description: (job as any).description || 'Real-time job from WebSocket',
      status: job.status,
      priority: (job as any).priority || 'medium' as const,
      progress: job.status === 'completed' ? 100 : job.status === 'running' ? 50 : 0,
      createdAt: new Date(job.updatedAt),
      startedAt: new Date(job.updatedAt),
      assignedAgent: job.owner || 'Unknown',
      tags: (job as any).source === 'user_interface' ? ['user-interface', 'real-time'] : ['system', 'real-time'],
      source: (job as any).source || 'system',
      userId: (job as any).userId,
      sessionId: (job as any).sessionId,
      resourceUsage: { cpu: 0, memory: 0, gpu: 0 },
      logs: [`Status: ${job.status}`]
    }));
    
    // Add source info to mock jobs
    const mockJobsWithSource = mockJobs.map(job => ({
      ...job,
      source: 'system',
      tags: [...job.tags, 'system']
    }));
    
    return [...mockJobsWithSource, ...realJobs];
  }, [snap.jobs]);

  const filteredJobs = useMemo(() => {
    return allJobs.filter(job => {
      const matchesSearch = job.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           job.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           job.id.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus = statusFilter === 'all' || job.status === statusFilter;
      const matchesPriority = priorityFilter === 'all' || job.priority === priorityFilter;
      const matchesSource = sourceFilter === 'all' || job.source === sourceFilter;
      return matchesSearch && matchesStatus && matchesPriority && matchesSource;
    });
  }, [allJobs, searchTerm, statusFilter, priorityFilter, sourceFilter]);

  const jobStats = useMemo(() => {
    const stats = allJobs.reduce((acc, job) => {
      acc[job.status] = (acc[job.status] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    return {
      total: allJobs.length,
      running: stats.running || 0,
      queued: stats.queued || 0,
      completed: stats.completed || 0,
      failed: stats.failed || 0,
      paused: stats.paused || 0
    };
  }, [allJobs]);

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Job Management</h1>
            <p className="text-sm opacity-70 mt-1">
              Monitor and control job execution across your agent swarm
            </p>
          </div>
          <Button
            variant="primary"
            icon={<Plus className="h-4 w-4" />}
            onClick={() => setShowCreateModal(true)}
          >
            Create Job
          </Button>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          <JobStatCard
            title="Total"
            value={jobStats.total}
            icon={<Briefcase className="h-5 w-5" />}
            color="blue"
          />
          <JobStatCard
            title="Running"
            value={jobStats.running}
            icon={<Play className="h-5 w-5" />}
            color="green"
          />
          <JobStatCard
            title="Queued"
            value={jobStats.queued}
            icon={<Clock className="h-5 w-5" />}
            color="yellow"
          />
          <JobStatCard
            title="Completed"
            value={jobStats.completed}
            icon={<CheckCircle className="h-5 w-5" />}
            color="green"
          />
          <JobStatCard
            title="Failed"
            value={jobStats.failed}
            icon={<XCircle className="h-5 w-5" />}
            color="red"
          />
          <JobStatCard
            title="Paused"
            value={jobStats.paused}
            icon={<Pause className="h-5 w-5" />}
            color="gray"
          />
        </div>

        {/* Filters */}
        <Card>
          <CardContent className="p-4">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex-1 min-w-64">
                <Input
                  variant="search"
                  placeholder="Search jobs..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
              
              <select
                className="input w-auto min-w-32"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <option value="all">All Status</option>
                <option value="running">Running</option>
                <option value="queued">Queued</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
                <option value="paused">Paused</option>
              </select>

              <select
                className="input w-auto min-w-32"
                value={priorityFilter}
                onChange={(e) => setPriorityFilter(e.target.value)}
              >
                <option value="all">All Priority</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>

              <select
                className="input w-auto min-w-32"
                value={sourceFilter}
                onChange={(e) => setSourceFilter(e.target.value)}
              >
                <option value="all">All Sources</option>
                <option value="user_interface">User Interface</option>
                <option value="system">System</option>
              </select>

              <Button variant="ghost" icon={<Filter className="h-4 w-4" />}>
                Advanced
              </Button>
              <Button variant="ghost" icon={<Download className="h-4 w-4" />}>
                Export
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Jobs Table */}
        <Card>
          <CardHeader>
            <CardTitle>Jobs ({filteredJobs.length})</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="border-b border-white/10 dark:border-red-900/40">
                  <tr className="text-left">
                    <th className="px-6 py-3 label text-xs">Job</th>
                    <th className="px-6 py-3 label text-xs">Status</th>
                    <th className="px-6 py-3 label text-xs">Priority</th>
                    <th className="px-6 py-3 label text-xs">Progress</th>
                    <th className="px-6 py-3 label text-xs">Agent</th>
                    <th className="px-6 py-3 label text-xs">Created</th>
                    <th className="px-6 py-3 label text-xs">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  <AnimatePresence>
                    {filteredJobs.map((job, index) => (
                      <JobRow
                        key={job.id}
                        job={job}
                        index={index}
                        onSelect={() => setSelectedJob(job)}
                      />
                    ))}
                  </AnimatePresence>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Job Details Modal */}
        <Modal
          isOpen={!!selectedJob}
          onClose={() => setSelectedJob(null)}
          title={selectedJob?.title}
          size="xl"
        >
          {selectedJob && <JobDetails job={selectedJob} />}
        </Modal>

        {/* Create Job Modal */}
        <Modal
          isOpen={showCreateModal}
          onClose={() => setShowCreateModal(false)}
          title="Create New Job"
          size="lg"
        >
          <CreateJobForm onClose={() => setShowCreateModal(false)} />
        </Modal>
      </div>
    </Layout>
  );
}

function JobStatCard({ title, value, icon, color }: {
  title: string;
  value: number;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'yellow' | 'red' | 'gray';
}) {
  const colorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    yellow: 'text-yellow-400',
    red: 'text-red-400',
    gray: 'text-gray-400'
  };

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="label text-xs">{title}</p>
            <p className="text-xl font-bold mt-1">{value}</p>
          </div>
          <div className={colorClasses[color]}>
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function JobRow({ job, index, onSelect }: {
  job: any; // Allow flexible job types
  index: number;
  onSelect: () => void;
}) {
  const statusConfig = {
    running: { badge: 'info', icon: Play, color: 'text-blue-400' },
    queued: { badge: 'warning', icon: Clock, color: 'text-yellow-400' },
    completed: { badge: 'success', icon: CheckCircle, color: 'text-green-400' },
    failed: { badge: 'danger', icon: XCircle, color: 'text-red-400' },
    paused: { badge: 'info', icon: Pause, color: 'text-gray-400' }
  };

  const priorityConfig = {
    critical: { badge: 'danger', color: 'text-red-400' },
    high: { badge: 'warning', color: 'text-orange-400' },
    medium: { badge: 'info', color: 'text-blue-400' },
    low: { badge: 'info', color: 'text-gray-400' }
  };

  const statusConf = statusConfig[job.status as keyof typeof statusConfig];
  const priorityConf = priorityConfig[job.priority as keyof typeof priorityConfig];

  return (
    <motion.tr
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.2, delay: index * 0.05 }}
      className="border-b border-white/5 dark:border-red-900/20 hover:bg-white/5 dark:hover:bg-red-900/5 cursor-pointer"
      onClick={onSelect}
    >
      <td className="px-6 py-4">
        <div>
          <p className="font-medium text-sm">{job.title}</p>
          <p className="text-xs opacity-60 mt-1 line-clamp-1">{job.description}</p>
          <div className="flex gap-1 mt-2">
            {job.tags?.map((tag: string) => (
              <Badge key={tag} size="sm" className="text-xs">
                {tag}
              </Badge>
            ))}
          </div>
        </div>
      </td>
      <td className="px-6 py-4">
        <Badge variant={statusConf?.badge as any} size="sm">
          {job.status}
        </Badge>
      </td>
      <td className="px-6 py-4">
        <Badge variant={priorityConf?.badge as any} size="sm">
          {job.priority}
        </Badge>
      </td>
      <td className="px-6 py-4">
        <div className="flex items-center gap-2">
          <div className="flex-1 bg-white/10 dark:bg-black/20 rounded-full h-2">
            <motion.div
              className="bg-day-accent dark:bg-night-text h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${job.progress}%` }}
              transition={{ duration: 1, ease: 'easeOut' }}
            />
          </div>
          <span className="text-xs font-mono w-10">{job.progress}%</span>
        </div>
      </td>
      <td className="px-6 py-4">
        <span className="text-sm">{job.assignedAgent || '—'}</span>
      </td>
      <td className="px-6 py-4">
        <span className="text-sm">{job.createdAt.toLocaleDateString()}</span>
      </td>
      <td className="px-6 py-4">
        <div className="flex items-center gap-2">
          <Button size="sm" variant="ghost" icon={<Eye className="h-3 w-3" />} />
          <Button size="sm" variant="ghost" icon={<MoreVertical className="h-3 w-3" />} />
        </div>
      </td>
    </motion.tr>
  );
}

function JobDetails({ job }: { job: any }) {
  return (
    <div className="space-y-6">
      {/* Job Overview */}
      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="label text-sm mb-3">Job Information</h3>
          <div className="space-y-3">
            <div>
              <p className="label text-xs">Job ID</p>
              <p className="font-mono text-sm">{job.id}</p>
            </div>
            <div>
              <p className="label text-xs">Description</p>
              <p className="text-sm">{job.description}</p>
            </div>
            <div>
              <p className="label text-xs">Priority</p>
              <Badge variant="info" size="sm">{job.priority}</Badge>
            </div>
          </div>
        </div>

        <div>
          <h3 className="label text-sm mb-3">Execution Details</h3>
          <div className="space-y-3">
            <div>
              <p className="label text-xs">Status</p>
              <Badge variant="success" size="sm">{job.status}</Badge>
            </div>
            <div>
              <p className="label text-xs">Assigned Agent</p>
              <p className="text-sm">{job.assignedAgent || 'Unassigned'}</p>
            </div>
            <div>
              <p className="label text-xs">Progress</p>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex-1 bg-white/10 dark:bg-black/20 rounded-full h-2">
                  <div
                    className="bg-day-accent dark:bg-night-text h-2 rounded-full transition-all duration-300"
                    style={{ width: `${job.progress}%` }}
                  />
                </div>
                <span className="text-sm font-mono">{job.progress}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Resource Usage */}
      <div>
        <h3 className="label text-sm mb-3">Resource Usage</h3>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <p className="label text-xs mb-1">CPU</p>
            <p className="text-lg font-bold">{job.resourceUsage.cpu}%</p>
          </div>
          <div>
            <p className="label text-xs mb-1">Memory</p>
            <p className="text-lg font-bold">{job.resourceUsage.memory}%</p>
          </div>
          <div>
            <p className="label text-xs mb-1">GPU</p>
            <p className="text-lg font-bold">{job.resourceUsage.gpu}%</p>
          </div>
        </div>
      </div>

      {/* Logs */}
      <div>
        <h3 className="label text-sm mb-3">Recent Logs</h3>
        <div className="bg-black/20 rounded-lg p-4 font-mono text-sm max-h-40 overflow-y-auto">
          {job.logs?.map((log: string, index: number) => (
            <div key={index} className="opacity-80 mb-1">
              [{typeof window !== 'undefined' ? new Date().toLocaleTimeString() : 'Loading...'}] {log}
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <Button variant="primary" icon={<Play className="h-4 w-4" />}>
          Resume
        </Button>
        <Button variant="secondary" icon={<Pause className="h-4 w-4" />}>
          Pause
        </Button>
        <Button variant="secondary" icon={<RotateCcw className="h-4 w-4" />}>
          Restart
        </Button>
        <Button variant="danger" icon={<Trash2 className="h-4 w-4" />}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

function CreateJobForm({ onClose }: { onClose: () => void }) {
  return (
    <div className="space-y-4">
      <Input label="Job Title" placeholder="Enter job title..." />
      
      <div>
        <label className="label block mb-2">Description</label>
        <textarea 
          className="input w-full h-24 resize-none" 
          placeholder="Describe what this job should do..."
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="label block mb-2">Priority</label>
          <select className="input w-full">
            <option>Low</option>
            <option>Medium</option>
            <option>High</option>
            <option>Critical</option>
          </select>
        </div>

        <div>
          <label className="label block mb-2">Agent Type</label>
          <select className="input w-full">
            <option>Any Available</option>
            <option>Neural Mesh</option>
            <option>Quantum Scheduler</option>
            <option>Universal I/O</option>
            <option>Security</option>
          </select>
        </div>
      </div>

      <Input label="Tags" placeholder="machine-learning, data-processing, etc." />

      <div className="flex gap-3 pt-4">
        <Button variant="primary" className="flex-1">
          Create Job
        </Button>
        <Button variant="ghost" onClick={onClose}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
