'use client';

import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Users,
  Bot,
  Activity,
  Zap,
  Target,
  Settings,
  Play,
  Pause,
  Square,
  RotateCcw,
  Plus,
  Minus,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Cpu,
  HardDrive,
  Network
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Modal } from '@/components/ui/Modal';
import { Layout } from '@/components/layout/Layout';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';

// Mock swarm data
const swarmClusters = [
  {
    id: 'cluster-001',
    name: 'Neural Processing Cluster',
    type: 'neural-mesh',
    status: 'active',
    agents: 24,
    maxAgents: 50,
    cpu: 78,
    memory: 65,
    gpu: 89,
    throughput: 1247,
    location: 'us-east-1a',
    uptime: '2d 14h 32m',
    tasks: ['Document Analysis', 'Pattern Recognition', 'NLP Processing']
  },
  {
    id: 'cluster-002',
    name: 'Quantum Coordination Hub',
    type: 'quantum-scheduler',
    status: 'active',
    agents: 12,
    maxAgents: 25,
    cpu: 45,
    memory: 34,
    gpu: 12,
    throughput: 892,
    location: 'us-west-2a',
    uptime: '1d 8h 15m',
    tasks: ['Task Distribution', 'Load Balancing', 'Optimization']
  },
  {
    id: 'cluster-003',
    name: 'Data Processing Array',
    type: 'universal-io',
    status: 'scaling',
    agents: 18,
    maxAgents: 40,
    cpu: 67,
    memory: 54,
    gpu: 23,
    throughput: 2156,
    location: 'eu-west-1a',
    uptime: '4d 2h 7m',
    tasks: ['Data Ingestion', 'Format Conversion', 'Stream Processing']
  },
  {
    id: 'cluster-004',
    name: 'Security Operations Center',
    type: 'security',
    status: 'warning',
    agents: 8,
    maxAgents: 15,
    cpu: 91,
    memory: 87,
    gpu: 45,
    throughput: 445,
    location: 'us-east-1c',
    uptime: '12h 45m',
    tasks: ['Threat Detection', 'Anomaly Analysis', 'Compliance Monitoring']
  }
];

export default function SwarmPage() {
  const [selectedCluster, setSelectedCluster] = useState<typeof swarmClusters[0] | null>(null);
  const [showScaleModal, setShowScaleModal] = useState(false);
  const snap = useSnapshot(store);

  const swarmStats = useMemo(() => {
    const totalAgents = swarmClusters.reduce((sum, cluster) => sum + cluster.agents, 0);
    const maxCapacity = swarmClusters.reduce((sum, cluster) => sum + cluster.maxAgents, 0);
    const avgCpu = Math.round(swarmClusters.reduce((sum, cluster) => sum + cluster.cpu, 0) / swarmClusters.length);
    const avgMemory = Math.round(swarmClusters.reduce((sum, cluster) => sum + cluster.memory, 0) / swarmClusters.length);
    const totalThroughput = swarmClusters.reduce((sum, cluster) => sum + cluster.throughput, 0);
    
    return {
      totalAgents,
      maxCapacity,
      utilization: Math.round((totalAgents / maxCapacity) * 100),
      avgCpu,
      avgMemory,
      totalThroughput,
      activeClusters: swarmClusters.filter(c => c.status === 'active').length,
      totalClusters: swarmClusters.length
    };
  }, []);

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <Users className="h-8 w-8 text-day-accent dark:text-night-text" />
              Swarm Control Center
            </h1>
            <p className="text-sm opacity-70 mt-1">
              Orchestrate and manage your distributed agent clusters
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="success" className="px-3 py-1">
              {swarmStats.activeClusters}/{swarmStats.totalClusters} Clusters Active
            </Badge>
            <Badge variant="info" className="px-3 py-1">
              {swarmStats.totalAgents}/{swarmStats.maxCapacity} Agents
            </Badge>
            <Button
              variant="primary"
              icon={<Plus className="h-4 w-4" />}
              onClick={() => setShowScaleModal(true)}
            >
              Scale Swarm
            </Button>
          </div>
        </div>

        {/* Swarm Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <SwarmMetricCard
            title="Total Agents"
            value={swarmStats.totalAgents}
            subtitle={`${swarmStats.utilization}% capacity`}
            icon={<Bot className="h-6 w-6" />}
            color="blue"
          />
          <SwarmMetricCard
            title="Throughput"
            value={`${swarmStats.totalThroughput}/min`}
            subtitle="Tasks processed"
            icon={<Activity className="h-6 w-6" />}
            color="green"
          />
          <SwarmMetricCard
            title="Avg CPU"
            value={`${swarmStats.avgCpu}%`}
            subtitle="Across all clusters"
            icon={<Cpu className="h-6 w-6" />}
            color="yellow"
          />
          <SwarmMetricCard
            title="Avg Memory"
            value={`${swarmStats.avgMemory}%`}
            subtitle="Memory utilization"
            icon={<HardDrive className="h-6 w-6" />}
            color="purple"
          />
        </div>

        {/* Swarm Clusters */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Users className="h-5 w-5" />
                Agent Clusters ({swarmClusters.length})
              </span>
              <div className="flex gap-2">
                <Button variant="ghost" size="sm" icon={<RotateCcw className="h-4 w-4" />}>
                  Refresh
                </Button>
                <Button variant="ghost" size="sm" icon={<Settings className="h-4 w-4" />}>
                  Configure
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {swarmClusters.map((cluster, index) => (
                <ClusterCard
                  key={cluster.id}
                  cluster={cluster}
                  index={index}
                  onSelect={() => setSelectedCluster(cluster)}
                />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Swarm Performance */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Performance Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Overall Efficiency</span>
                  <Badge variant="success">94.2%</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Load Distribution</span>
                  <Badge variant="info">Balanced</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Fault Tolerance</span>
                  <Badge variant="success">High</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Auto-scaling</span>
                  <Badge variant="success">Enabled</Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Coordination Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-day-accent dark:text-night-text mb-2">
                    {snap.meta.nodes || swarmStats.totalAgents}
                  </div>
                  <p className="text-sm opacity-70">Coordinated Agents</p>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div>
                    <div className="text-xl font-bold text-green-400">
                      {snap.meta.rps || 156}
                    </div>
                    <p className="text-xs opacity-70">Tasks/sec</p>
                  </div>
                  <div>
                    <div className="text-xl font-bold text-yellow-400">
                      {snap.meta.queueDepth || 23}
                    </div>
                    <p className="text-xs opacity-70">Queue Depth</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Cluster Details Modal */}
        <Modal
          isOpen={!!selectedCluster}
          onClose={() => setSelectedCluster(null)}
          title={selectedCluster?.name}
          size="lg"
        >
          {selectedCluster && <ClusterDetails cluster={selectedCluster} />}
        </Modal>

        {/* Scale Swarm Modal */}
        <Modal
          isOpen={showScaleModal}
          onClose={() => setShowScaleModal(false)}
          title="Scale Swarm Cluster"
          size="md"
        >
          <ScaleSwarmForm onClose={() => setShowScaleModal(false)} />
        </Modal>
      </div>
    </Layout>
  );
}

function SwarmMetricCard({ title, value, subtitle, icon, color }: {
  title: string;
  value: string | number;
  subtitle: string;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'yellow' | 'purple';
}) {
  const colorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    yellow: 'text-yellow-400',
    purple: 'text-purple-400'
  };

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="label text-xs">{title}</p>
            <p className="text-xl font-bold mt-1">{value}</p>
            <p className="text-xs opacity-60 mt-1">{subtitle}</p>
          </div>
          <div className={colorClasses[color]}>
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function ClusterCard({ cluster, index, onSelect }: {
  cluster: typeof swarmClusters[0];
  index: number;
  onSelect: () => void;
}) {
  const statusConfig = {
    active: { badge: 'success', icon: CheckCircle, color: 'text-green-400' },
    scaling: { badge: 'warning', icon: Clock, color: 'text-yellow-400' },
    warning: { badge: 'warning', icon: AlertTriangle, color: 'text-yellow-400' },
    error: { badge: 'danger', icon: AlertTriangle, color: 'text-red-400' }
  };

  const config = statusConfig[cluster.status as keyof typeof statusConfig];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
    >
      <Card hover className="cursor-pointer" onClick={onSelect}>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-day-accent/20 dark:bg-night-text/20 flex items-center justify-center">
                <Users className="w-5 h-5 text-day-accent dark:text-night-text" />
              </div>
              <div>
                <CardTitle className="text-sm">{cluster.name}</CardTitle>
                <p className="text-xs opacity-60">{cluster.id}</p>
              </div>
            </div>
            <Badge variant={config.badge as any} size="sm">
              {cluster.status}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="pt-0">
          <div className="space-y-3">
            {/* Agent Count */}
            <div className="flex items-center justify-between">
              <span className="text-sm">Agents</span>
              <span className="text-sm font-mono">
                {cluster.agents}/{cluster.maxAgents}
              </span>
            </div>

            {/* Resource Usage */}
            <div className="grid grid-cols-3 gap-3">
              <div className="text-center">
                <p className="text-xs opacity-60">CPU</p>
                <p className="text-sm font-medium">{cluster.cpu}%</p>
              </div>
              <div className="text-center">
                <p className="text-xs opacity-60">Memory</p>
                <p className="text-sm font-medium">{cluster.memory}%</p>
              </div>
              <div className="text-center">
                <p className="text-xs opacity-60">GPU</p>
                <p className="text-sm font-medium">{cluster.gpu}%</p>
              </div>
            </div>

            {/* Throughput */}
            <div className="flex items-center justify-between">
              <span className="text-sm">Throughput</span>
              <span className="text-sm font-mono">{cluster.throughput}/min</span>
            </div>

            {/* Quick Actions */}
            <div className="flex items-center justify-between pt-2">
              <div className="flex items-center gap-2">
                <Button size="sm" variant="ghost" icon={<Play className="h-3 w-3" />} />
                <Button size="sm" variant="ghost" icon={<Pause className="h-3 w-3" />} />
                <Button size="sm" variant="ghost" icon={<Settings className="h-3 w-3" />} />
              </div>
              <div className="flex items-center gap-1">
                <Button size="sm" variant="ghost" icon={<Minus className="h-3 w-3" />} />
                <Button size="sm" variant="ghost" icon={<Plus className="h-3 w-3" />} />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function ClusterDetails({ cluster }: { cluster: typeof swarmClusters[0] }) {
  return (
    <div className="space-y-6">
      {/* Cluster Overview */}
      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="label text-sm mb-3">Cluster Information</h3>
          <div className="space-y-3">
            <div>
              <p className="label text-xs">Cluster ID</p>
              <p className="font-mono text-sm">{cluster.id}</p>
            </div>
            <div>
              <p className="label text-xs">Type</p>
              <p className="text-sm">{cluster.type}</p>
            </div>
            <div>
              <p className="label text-xs">Location</p>
              <p className="text-sm">{cluster.location}</p>
            </div>
            <div>
              <p className="label text-xs">Uptime</p>
              <p className="text-sm">{cluster.uptime}</p>
            </div>
          </div>
        </div>

        <div>
          <h3 className="label text-sm mb-3">Performance Metrics</h3>
          <div className="space-y-3">
            <div>
              <p className="label text-xs">Agent Count</p>
              <p className="text-sm">{cluster.agents}/{cluster.maxAgents}</p>
            </div>
            <div>
              <p className="label text-xs">Throughput</p>
              <p className="text-sm">{cluster.throughput} tasks/min</p>
            </div>
            <div>
              <p className="label text-xs">Efficiency</p>
              <p className="text-sm">94.2%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Resource Usage */}
      <div>
        <h3 className="label text-sm mb-3">Resource Usage</h3>
        <div className="space-y-3">
          <ResourceBar label="CPU" usage={cluster.cpu} />
          <ResourceBar label="Memory" usage={cluster.memory} />
          <ResourceBar label="GPU" usage={cluster.gpu} />
        </div>
      </div>

      {/* Active Tasks */}
      <div>
        <h3 className="label text-sm mb-3">Active Tasks</h3>
        <div className="space-y-2">
          {cluster.tasks.map((task, index) => (
            <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-white/5 dark:bg-black/20">
              <span className="text-sm">{task}</span>
              <Badge variant="info" size="sm">Running</Badge>
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <Button variant="primary" icon={<Settings className="h-4 w-4" />}>
          Configure
        </Button>
        <Button variant="secondary" icon={<Plus className="h-4 w-4" />}>
          Scale Up
        </Button>
        <Button variant="secondary" icon={<Minus className="h-4 w-4" />}>
          Scale Down
        </Button>
        <Button variant="danger" icon={<Square className="h-4 w-4" />}>
          Shutdown
        </Button>
      </div>
    </div>
  );
}

function ResourceBar({ label, usage }: { label: string; usage: number }) {
  const getColor = (usage: number) => {
    if (usage > 80) return 'bg-red-400';
    if (usage > 60) return 'bg-yellow-400';
    return 'bg-green-400';
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm">{label}</span>
        <span className="text-xs">{usage}%</span>
      </div>
      <div className="w-full bg-white/10 dark:bg-black/20 rounded-full h-2">
        <motion.div
          className={`h-2 rounded-full ${getColor(usage)}`}
          initial={{ width: 0 }}
          animate={{ width: `${usage}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}

function ScaleSwarmForm({ onClose }: { onClose: () => void }) {
  const [selectedCluster, setSelectedCluster] = useState('');
  const [targetAgents, setTargetAgents] = useState(10);

  return (
    <div className="space-y-4">
      <div>
        <label className="label block mb-2">Select Cluster</label>
        <select 
          className="input w-full"
          value={selectedCluster}
          onChange={(e) => setSelectedCluster(e.target.value)}
        >
          <option value="">Choose a cluster...</option>
          {swarmClusters.map(cluster => (
            <option key={cluster.id} value={cluster.id}>
              {cluster.name} ({cluster.agents}/{cluster.maxAgents} agents)
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="label block mb-2">Target Agent Count</label>
        <input
          type="range"
          min="1"
          max="100"
          value={targetAgents}
          onChange={(e) => setTargetAgents(parseInt(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs opacity-60 mt-1">
          <span>1</span>
          <span className="font-medium">{targetAgents} agents</span>
          <span>100</span>
        </div>
      </div>

      <div className="p-4 rounded-lg bg-white/5 dark:bg-black/20">
        <div className="flex items-center gap-2 mb-2">
          <AlertTriangle className="h-4 w-4 text-yellow-400" />
          <span className="text-sm font-medium">Scaling Impact</span>
        </div>
        <p className="text-xs opacity-70">
          Scaling will temporarily affect cluster performance during agent initialization.
          Estimated completion: 2-5 minutes.
        </p>
      </div>

      <div className="flex gap-3 pt-4">
        <Button variant="primary" className="flex-1">
          Scale Cluster
        </Button>
        <Button variant="ghost" onClick={onClose}>
          Cancel
        </Button>
      </div>
    </div>
  );
}