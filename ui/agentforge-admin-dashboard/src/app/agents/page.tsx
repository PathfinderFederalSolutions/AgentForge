'use client';

import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bot,
  Plus,
  Search,
  Filter,
  MoreVertical,
  Play,
  Pause,
  Square,
  Trash2,
  Settings,
  Activity,
  Cpu,
  HardDrive,
  Zap,
  Clock,
  CheckCircle,
  AlertTriangle,
  XCircle
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Modal } from '@/components/ui/Modal';
import { Layout } from '@/components/layout/Layout';

// Mock agent data
const mockAgents = [
  {
    id: 'agent-001',
    name: 'Neural Processor Alpha',
    type: 'neural-mesh',
    status: 'active',
    cpu: 45,
    memory: 62,
    gpu: 89,
    uptime: '2d 14h 32m',
    tasksCompleted: 1247,
    currentTask: 'Document Analysis Pipeline',
    location: 'us-east-1a',
    version: '2.1.4'
  },
  {
    id: 'agent-002',
    name: 'Quantum Scheduler Beta',
    type: 'quantum-scheduler',
    status: 'active',
    cpu: 78,
    memory: 34,
    gpu: 12,
    uptime: '1d 8h 15m',
    tasksCompleted: 892,
    currentTask: 'Task Distribution Optimization',
    location: 'us-east-1b',
    version: '1.9.2'
  },
  {
    id: 'agent-003',
    name: 'Data Processor Gamma',
    type: 'universal-io',
    status: 'idle',
    cpu: 12,
    memory: 28,
    gpu: 0,
    uptime: '4d 2h 7m',
    tasksCompleted: 2156,
    currentTask: null,
    location: 'us-west-2a',
    version: '3.0.1'
  },
  {
    id: 'agent-004',
    name: 'Security Monitor Delta',
    type: 'security',
    status: 'warning',
    cpu: 91,
    memory: 87,
    gpu: 45,
    uptime: '12h 45m',
    tasksCompleted: 445,
    currentTask: 'Threat Detection Scan',
    location: 'eu-west-1a',
    version: '1.5.8'
  },
  {
    id: 'agent-005',
    name: 'Analytics Engine Epsilon',
    type: 'analytics',
    status: 'error',
    cpu: 0,
    memory: 0,
    gpu: 0,
    uptime: '0m',
    tasksCompleted: 0,
    currentTask: null,
    location: 'us-east-1c',
    version: '2.3.1'
  }
];

export default function AgentsPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [selectedAgent, setSelectedAgent] = useState<typeof mockAgents[0] | null>(null);
  const [showDeployModal, setShowDeployModal] = useState(false);

  const filteredAgents = useMemo(() => {
    return mockAgents.filter(agent => {
      const matchesSearch = agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           agent.id.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus = statusFilter === 'all' || agent.status === statusFilter;
      const matchesType = typeFilter === 'all' || agent.type === typeFilter;
      return matchesSearch && matchesStatus && matchesType;
    });
  }, [searchTerm, statusFilter, typeFilter]);

  const agentStats = useMemo(() => {
    const stats = mockAgents.reduce((acc, agent) => {
      acc[agent.status] = (acc[agent.status] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    return {
      total: mockAgents.length,
      active: stats.active || 0,
      idle: stats.idle || 0,
      warning: stats.warning || 0,
      error: stats.error || 0
    };
  }, []);

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Agent Management</h1>
            <p className="text-sm opacity-70 mt-1">
              Monitor and control your AI agent swarm
            </p>
          </div>
          <Button
            variant="primary"
            icon={<Plus className="h-4 w-4" />}
            onClick={() => setShowDeployModal(true)}
          >
            Deploy Agent
          </Button>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <StatCard
            title="Total Agents"
            value={agentStats.total}
            icon={<Bot className="h-5 w-5" />}
            color="blue"
          />
          <StatCard
            title="Active"
            value={agentStats.active}
            icon={<CheckCircle className="h-5 w-5" />}
            color="green"
          />
          <StatCard
            title="Idle"
            value={agentStats.idle}
            icon={<Clock className="h-5 w-5" />}
            color="gray"
          />
          <StatCard
            title="Warning"
            value={agentStats.warning}
            icon={<AlertTriangle className="h-5 w-5" />}
            color="yellow"
          />
          <StatCard
            title="Error"
            value={agentStats.error}
            icon={<XCircle className="h-5 w-5" />}
            color="red"
          />
        </div>

        {/* Filters */}
        <Card>
          <CardContent className="p-4">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex-1 min-w-64">
                <Input
                  variant="search"
                  placeholder="Search agents..."
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
                <option value="active">Active</option>
                <option value="idle">Idle</option>
                <option value="warning">Warning</option>
                <option value="error">Error</option>
              </select>

              <select
                className="input w-auto min-w-32"
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
              >
                <option value="all">All Types</option>
                <option value="neural-mesh">Neural Mesh</option>
                <option value="quantum-scheduler">Quantum Scheduler</option>
                <option value="universal-io">Universal I/O</option>
                <option value="security">Security</option>
                <option value="analytics">Analytics</option>
              </select>

              <Button variant="ghost" icon={<Filter className="h-4 w-4" />}>
                More Filters
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Agents Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          <AnimatePresence>
            {filteredAgents.map((agent) => (
              <AgentCard
                key={agent.id}
                agent={agent}
                onSelect={() => setSelectedAgent(agent)}
              />
            ))}
          </AnimatePresence>
        </div>

        {/* Agent Details Modal */}
        <Modal
          isOpen={!!selectedAgent}
          onClose={() => setSelectedAgent(null)}
          title={selectedAgent?.name}
          size="lg"
        >
          {selectedAgent && <AgentDetails agent={selectedAgent} />}
        </Modal>

        {/* Deploy Agent Modal */}
        <Modal
          isOpen={showDeployModal}
          onClose={() => setShowDeployModal(false)}
          title="Deploy New Agent"
          size="md"
        >
          <DeployAgentForm onClose={() => setShowDeployModal(false)} />
        </Modal>
      </div>
    </Layout>
  );
}

function StatCard({ title, value, icon, color }: {
  title: string;
  value: number;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'gray' | 'yellow' | 'red';
}) {
  const colorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    gray: 'text-gray-400',
    yellow: 'text-yellow-400',
    red: 'text-red-400'
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

function AgentCard({ agent, onSelect }: {
  agent: typeof mockAgents[0];
  onSelect: () => void;
}) {
  const statusConfig = {
    active: { badge: 'success', icon: CheckCircle, color: 'text-green-400' },
    idle: { badge: 'info', icon: Clock, color: 'text-gray-400' },
    warning: { badge: 'warning', icon: AlertTriangle, color: 'text-yellow-400' },
    error: { badge: 'danger', icon: XCircle, color: 'text-red-400' }
  };

  const config = statusConfig[agent.status as keyof typeof statusConfig];

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.2 }}
    >
      <Card hover className="cursor-pointer" onClick={onSelect}>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-day-accent/20 dark:bg-night-text/20 flex items-center justify-center">
                <Bot className="w-5 h-5 text-day-accent dark:text-night-text" />
              </div>
              <div>
                <CardTitle className="text-sm">{agent.name}</CardTitle>
                <p className="text-xs opacity-60">{agent.id}</p>
              </div>
            </div>
            <Badge variant={config.badge as any} size="sm">
              {agent.status}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="pt-0">
          <div className="space-y-3">
            {/* Current Task */}
            {agent.currentTask && (
              <div>
                <p className="label text-xs mb-1">Current Task</p>
                <p className="text-xs opacity-80">{agent.currentTask}</p>
              </div>
            )}

            {/* Resource Usage */}
            <div className="grid grid-cols-3 gap-3">
              <ResourceMetric
                label="CPU"
                value={agent.cpu}
                icon={<Cpu className="h-3 w-3" />}
              />
              <ResourceMetric
                label="Memory"
                value={agent.memory}
                icon={<HardDrive className="h-3 w-3" />}
              />
              <ResourceMetric
                label="GPU"
                value={agent.gpu}
                icon={<Zap className="h-3 w-3" />}
              />
            </div>

            {/* Stats */}
            <div className="flex items-center justify-between text-xs opacity-70">
              <span>Tasks: {agent.tasksCompleted}</span>
              <span>Uptime: {agent.uptime}</span>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-between pt-2">
              <div className="flex items-center gap-2">
                <Button size="sm" variant="ghost" icon={<Play className="h-3 w-3" />} />
                <Button size="sm" variant="ghost" icon={<Pause className="h-3 w-3" />} />
                <Button size="sm" variant="ghost" icon={<Square className="h-3 w-3" />} />
              </div>
              <Button size="sm" variant="ghost" icon={<MoreVertical className="h-3 w-3" />} />
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function ResourceMetric({ label, value, icon }: {
  label: string;
  value: number;
  icon: React.ReactNode;
}) {
  const getColor = (value: number) => {
    if (value > 80) return 'text-red-400';
    if (value > 60) return 'text-yellow-400';
    return 'text-green-400';
  };

  return (
    <div className="text-center">
      <div className="flex items-center justify-center gap-1 mb-1">
        {icon}
        <span className="label text-xs">{label}</span>
      </div>
      <div className={`text-sm font-medium ${getColor(value)}`}>
        {value}%
      </div>
    </div>
  );
}

function AgentDetails({ agent }: { agent: typeof mockAgents[0] }) {
  return (
    <div className="space-y-6">
      {/* Agent Info */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="label text-xs mb-1">Agent ID</p>
          <p className="font-mono text-sm">{agent.id}</p>
        </div>
        <div>
          <p className="label text-xs mb-1">Type</p>
          <p className="text-sm">{agent.type}</p>
        </div>
        <div>
          <p className="label text-xs mb-1">Location</p>
          <p className="text-sm">{agent.location}</p>
        </div>
        <div>
          <p className="label text-xs mb-1">Version</p>
          <p className="text-sm">{agent.version}</p>
        </div>
      </div>

      {/* Resource Usage */}
      <div>
        <h3 className="label text-sm mb-3">Resource Usage</h3>
        <div className="space-y-3">
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-sm">CPU</span>
              <span className="text-sm">{agent.cpu}%</span>
            </div>
            <div className="w-full bg-white/10 dark:bg-black/20 rounded-full h-2">
              <div
                className="bg-blue-400 h-2 rounded-full transition-all duration-300"
                style={{ width: `${agent.cpu}%` }}
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-sm">Memory</span>
              <span className="text-sm">{agent.memory}%</span>
            </div>
            <div className="w-full bg-white/10 dark:bg-black/20 rounded-full h-2">
              <div
                className="bg-green-400 h-2 rounded-full transition-all duration-300"
                style={{ width: `${agent.memory}%` }}
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-sm">GPU</span>
              <span className="text-sm">{agent.gpu}%</span>
            </div>
            <div className="w-full bg-white/10 dark:bg-black/20 rounded-full h-2">
              <div
                className="bg-purple-400 h-2 rounded-full transition-all duration-300"
                style={{ width: `${agent.gpu}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <Button variant="primary" icon={<Settings className="h-4 w-4" />}>
          Configure
        </Button>
        <Button variant="secondary" icon={<Activity className="h-4 w-4" />}>
          View Logs
        </Button>
        <Button variant="danger" icon={<Trash2 className="h-4 w-4" />}>
          Terminate
        </Button>
      </div>
    </div>
  );
}

function DeployAgentForm({ onClose }: { onClose: () => void }) {
  return (
    <div className="space-y-4">
      <Input label="Agent Name" placeholder="Enter agent name..." />
      
      <div>
        <label className="label block mb-2">Agent Type</label>
        <select className="input w-full">
          <option>Neural Mesh</option>
          <option>Quantum Scheduler</option>
          <option>Universal I/O</option>
          <option>Security Monitor</option>
          <option>Analytics Engine</option>
        </select>
      </div>

      <div>
        <label className="label block mb-2">Deployment Region</label>
        <select className="input w-full">
          <option>us-east-1</option>
          <option>us-west-2</option>
          <option>eu-west-1</option>
          <option>ap-southeast-1</option>
        </select>
      </div>

      <Input label="Resource Allocation" placeholder="CPU, Memory, GPU requirements..." />

      <div className="flex gap-3 pt-4">
        <Button variant="primary" className="flex-1">
          Deploy Agent
        </Button>
        <Button variant="ghost" onClick={onClose}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
