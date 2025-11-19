'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Zap,
  Atom,
  Activity,
  Clock,
  Target,
  Layers,
  BarChart3,
  Play,
  Pause,
  Square,
  Settings,
  AlertTriangle,
  CheckCircle,
  Plus
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Modal } from '@/components/ui/Modal';
import { Layout } from '@/components/layout/Layout';
import { getQuantumStatus, getQuantumTasks, createQuantumTask } from '@/lib/api';

// Mock quantum tasks data
const mockQuantumTasks = [
  {
    id: 'qt-001',
    description: 'Neural network hyperparameter optimization',
    status: 'running',
    progress: 67,
    targetAgentCount: 1000,
    currentAgentCount: 847,
    coherenceLevel: 'HIGH',
    priority: 'critical',
    estimatedCompletion: new Date(Date.now() + 45 * 60000),
    quantumEfficiency: 0.89,
    entanglementStrength: 0.94,
    decoherenceRate: 0.03
  },
  {
    id: 'qt-002',
    description: 'Large-scale data processing coordination',
    status: 'queued',
    progress: 0,
    targetAgentCount: 500,
    currentAgentCount: 0,
    coherenceLevel: 'MEDIUM',
    priority: 'high',
    estimatedCompletion: new Date(Date.now() + 120 * 60000),
    quantumEfficiency: 0.0,
    entanglementStrength: 0.0,
    decoherenceRate: 0.0
  },
  {
    id: 'qt-003',
    description: 'Multi-agent reinforcement learning',
    status: 'completed',
    progress: 100,
    targetAgentCount: 750,
    currentAgentCount: 750,
    coherenceLevel: 'HIGH',
    priority: 'medium',
    completedAt: new Date(Date.now() - 30 * 60000),
    quantumEfficiency: 0.96,
    entanglementStrength: 0.92,
    decoherenceRate: 0.01
  }
];

export default function QuantumPage() {
  const [tasks, setTasks] = useState(mockQuantumTasks);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedTask, setSelectedTask] = useState<typeof mockQuantumTasks[0] | null>(null);
  const [quantumStatus, setQuantumStatus] = useState({
    totalClusters: 12,
    activeAgents: 2847,
    coherenceScore: 0.91,
    quantumStates: 'SUPERPOSITION',
    entangledPairs: 1423,
    decoherenceRate: 0.02,
    quantumVolume: 64,
    fidelity: 0.95
  });

  useEffect(() => {
    const loadQuantumStatus = async () => {
      try {
        const status = await getQuantumStatus();
        setQuantumStatus(prev => ({ ...prev, ...status }));
        
        const quantumTasks = await getQuantumTasks();
        setTasks(quantumTasks);
      } catch (error) {
        console.error('Failed to load quantum status:', error);
      }
    };

    loadQuantumStatus();
    const interval = setInterval(loadQuantumStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const taskStats = {
    total: tasks.length,
    running: tasks.filter(t => t.status === 'running').length,
    queued: tasks.filter(t => t.status === 'queued').length,
    completed: tasks.filter(t => t.status === 'completed').length,
    failed: tasks.filter(t => t.status === 'failed').length
  };

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <Atom className="h-8 w-8 text-purple-400" />
              Quantum Scheduler
            </h1>
            <p className="text-sm opacity-70 mt-1">
              Million-scale agent coordination using quantum computing principles
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="info" className="px-3 py-1">
              Quantum Volume: {quantumStatus.quantumVolume}
            </Badge>
            <Badge 
              variant={quantumStatus.coherenceScore > 0.9 ? 'success' : 'warning'} 
              className="px-3 py-1"
            >
              Coherence: {Math.round(quantumStatus.coherenceScore * 100)}%
            </Badge>
            <Button
              variant="primary"
              icon={<Plus className="h-4 w-4" />}
              onClick={() => setShowCreateModal(true)}
            >
              Create Quantum Task
            </Button>
          </div>
        </div>

        {/* Quantum Status Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <QuantumMetricCard
            title="Quantum Clusters"
            value={quantumStatus.totalClusters}
            icon={<Layers className="h-6 w-6" />}
            color="purple"
            subtitle="Active quantum computing clusters"
          />
          <QuantumMetricCard
            title="Entangled Agents"
            value={quantumStatus.activeAgents}
            icon={<Zap className="h-6 w-6" />}
            color="blue"
            subtitle="Agents in quantum superposition"
          />
          <QuantumMetricCard
            title="Entangled Pairs"
            value={quantumStatus.entangledPairs}
            icon={<Target className="h-6 w-6" />}
            color="green"
            subtitle="Quantum entangled agent pairs"
          />
          <QuantumMetricCard
            title="Fidelity"
            value={`${Math.round(quantumStatus.fidelity * 100)}%`}
            icon={<Activity className="h-6 w-6" />}
            color="yellow"
            subtitle="Quantum state fidelity"
          />
        </div>

        {/* Quantum State Visualization */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Atom className="h-5 w-5" />
              Quantum State Visualization
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64 bg-gradient-to-br from-purple-900/20 to-blue-900/20 rounded-lg relative overflow-hidden flex items-center justify-center">
              {/* Quantum visualization */}
              <div className="absolute inset-0">
                {Array.from({ length: 30 }).map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute w-2 h-2 rounded-full"
                    style={{
                      background: `hsl(${(i * 30) % 360}, 70%, 60%)`,
                      left: `${Math.random() * 100}%`,
                      top: `${Math.random() * 100}%`,
                    }}
                    animate={{
                      scale: [0.5, 1.5, 0.5],
                      opacity: [0.3, 1, 0.3],
                      rotate: [0, 360],
                    }}
                    transition={{
                      duration: 3 + Math.random() * 2,
                      repeat: Infinity,
                      delay: Math.random() * 2,
                      ease: 'easeInOut',
                    }}
                  />
                ))}
              </div>
              
              <div className="text-center z-10">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 10, repeat: Infinity, ease: 'linear' }}
                >
                  <Atom className="h-20 w-20 text-purple-400 mx-auto mb-4" />
                </motion.div>
                <p className="text-lg font-semibold">Quantum State: {quantumStatus.quantumStates}</p>
                <p className="text-sm opacity-70">
                  Coherence: {Math.round(quantumStatus.coherenceScore * 100)}% | 
                  Decoherence Rate: {(quantumStatus.decoherenceRate * 100).toFixed(2)}%
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Task Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <TaskStatCard title="Total" value={taskStats.total} color="blue" />
          <TaskStatCard title="Running" value={taskStats.running} color="green" />
          <TaskStatCard title="Queued" value={taskStats.queued} color="yellow" />
          <TaskStatCard title="Completed" value={taskStats.completed} color="purple" />
          <TaskStatCard title="Failed" value={taskStats.failed} color="red" />
        </div>

        {/* Quantum Tasks */}
        <Card>
          <CardHeader>
            <CardTitle>Quantum Tasks</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {tasks.map((task, index) => (
                <QuantumTaskCard
                  key={task.id}
                  task={task}
                  index={index}
                  onSelect={() => setSelectedTask(task)}
                />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Performance Metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Quantum Efficiency Over Time
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-48 flex items-end gap-2">
                {Array.from({ length: 24 }).map((_, i) => {
                  const efficiency = 0.7 + Math.random() * 0.25;
                  return (
                    <motion.div
                      key={i}
                      className="flex-1 bg-purple-400/30 rounded-t"
                      initial={{ height: 0 }}
                      animate={{ height: `${efficiency * 100}%` }}
                      transition={{ duration: 0.5, delay: i * 0.05 }}
                    />
                  );
                })}
              </div>
              <div className="mt-4 text-center">
                <p className="text-sm opacity-70">Average Efficiency: 89.4%</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Quantum Coherence Levels</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <CoherenceLevel level="HIGH" count={3} percentage={45} color="bg-green-400" />
                <CoherenceLevel level="MEDIUM" count={7} percentage={35} color="bg-yellow-400" />
                <CoherenceLevel level="LOW" count={2} percentage={20} color="bg-red-400" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Task Details Modal */}
        <Modal
          isOpen={!!selectedTask}
          onClose={() => setSelectedTask(null)}
          title={selectedTask?.description}
          size="lg"
        >
          {selectedTask && <QuantumTaskDetails task={selectedTask} />}
        </Modal>

        {/* Create Task Modal */}
        <Modal
          isOpen={showCreateModal}
          onClose={() => setShowCreateModal(false)}
          title="Create Quantum Task"
          size="md"
        >
          <CreateQuantumTaskForm onClose={() => setShowCreateModal(false)} />
        </Modal>
      </div>
    </Layout>
  );
}

function QuantumMetricCard({ title, value, icon, color, subtitle }: {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
  subtitle: string;
}) {
  const colorClasses = {
    purple: 'text-purple-400',
    blue: 'text-blue-400',
    green: 'text-green-400',
    yellow: 'text-yellow-400'
  };

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <div>
            <p className="label text-xs">{title}</p>
            <p className="text-xl font-bold mt-1">{value}</p>
          </div>
          <div className={colorClasses[color as keyof typeof colorClasses]}>
            {icon}
          </div>
        </div>
        <p className="text-xs opacity-60">{subtitle}</p>
      </CardContent>
    </Card>
  );
}

function TaskStatCard({ title, value, color }: {
  title: string;
  value: number;
  color: string;
}) {
  const colorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    yellow: 'text-yellow-400',
    purple: 'text-purple-400',
    red: 'text-red-400'
  };

  return (
    <Card>
      <CardContent className="p-4 text-center">
        <p className="label text-xs">{title}</p>
        <p className={`text-2xl font-bold mt-1 ${colorClasses[color as keyof typeof colorClasses]}`}>
          {value}
        </p>
      </CardContent>
    </Card>
  );
}

function QuantumTaskCard({ task, index, onSelect }: {
  task: typeof mockQuantumTasks[0];
  index: number;
  onSelect: () => void;
}) {
  const statusConfig = {
    running: { badge: 'info', icon: Play, color: 'text-blue-400' },
    queued: { badge: 'warning', icon: Clock, color: 'text-yellow-400' },
    completed: { badge: 'success', icon: CheckCircle, color: 'text-green-400' },
    failed: { badge: 'danger', icon: AlertTriangle, color: 'text-red-400' }
  };

  const config = statusConfig[task.status as keyof typeof statusConfig];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      className="p-4 rounded-lg bg-white/5 dark:bg-black/20 border border-white/10 dark:border-red-900/40 cursor-pointer hover:bg-white/10 dark:hover:bg-black/30"
      onClick={onSelect}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="font-medium text-sm mb-1">{task.description}</h3>
          <div className="flex items-center gap-3 text-xs opacity-60">
            <span>Task: {task.id}</span>
            <span>Agents: {task.currentAgentCount}/{task.targetAgentCount}</span>
            <span>Coherence: {task.coherenceLevel}</span>
          </div>
        </div>
        <Badge variant={config.badge as any} size="sm">
          {task.status}
        </Badge>
      </div>

      {/* Progress Bar */}
      <div className="mb-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs">Progress</span>
          <span className="text-xs">{task.progress}%</span>
        </div>
        <div className="w-full bg-white/10 dark:bg-black/20 rounded-full h-2">
          <motion.div
            className="bg-purple-400 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${task.progress}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
          />
        </div>
      </div>

      {/* Quantum Metrics */}
      <div className="grid grid-cols-3 gap-3 text-xs">
        <div className="text-center">
          <p className="opacity-60">Efficiency</p>
          <p className="font-medium">{Math.round(task.quantumEfficiency * 100)}%</p>
        </div>
        <div className="text-center">
          <p className="opacity-60">Entanglement</p>
          <p className="font-medium">{Math.round(task.entanglementStrength * 100)}%</p>
        </div>
        <div className="text-center">
          <p className="opacity-60">Decoherence</p>
          <p className="font-medium">{(task.decoherenceRate * 100).toFixed(1)}%</p>
        </div>
      </div>
    </motion.div>
  );
}

function CoherenceLevel({ level, count, percentage, color }: {
  level: string;
  count: number;
  percentage: number;
  color: string;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium">{level}</span>
        <span className="text-sm opacity-60">{count} tasks ({percentage}%)</span>
      </div>
      <div className="w-full bg-white/10 dark:bg-black/20 rounded-full h-2">
        <motion.div
          className={`h-2 rounded-full ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}

function QuantumTaskDetails({ task }: { task: typeof mockQuantumTasks[0] }) {
  return (
    <div className="space-y-6">
      {/* Task Overview */}
      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="label text-sm mb-3">Task Information</h3>
          <div className="space-y-3">
            <div>
              <p className="label text-xs">Task ID</p>
              <p className="font-mono text-sm">{task.id}</p>
            </div>
            <div>
              <p className="label text-xs">Priority</p>
              <Badge variant="info" size="sm">{task.priority}</Badge>
            </div>
            <div>
              <p className="label text-xs">Coherence Level</p>
              <Badge variant="success" size="sm">{task.coherenceLevel}</Badge>
            </div>
          </div>
        </div>

        <div>
          <h3 className="label text-sm mb-3">Agent Allocation</h3>
          <div className="space-y-3">
            <div>
              <p className="label text-xs">Target Agents</p>
              <p className="text-sm">{task.targetAgentCount.toLocaleString()}</p>
            </div>
            <div>
              <p className="label text-xs">Current Agents</p>
              <p className="text-sm">{task.currentAgentCount.toLocaleString()}</p>
            </div>
            <div>
              <p className="label text-xs">Allocation Progress</p>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex-1 bg-white/10 dark:bg-black/20 rounded-full h-2">
                  <div
                    className="bg-purple-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(task.currentAgentCount / task.targetAgentCount) * 100}%` }}
                  />
                </div>
                <span className="text-sm font-mono">
                  {Math.round((task.currentAgentCount / task.targetAgentCount) * 100)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quantum Metrics */}
      <div>
        <h3 className="label text-sm mb-3">Quantum Metrics</h3>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <p className="label text-xs mb-1">Quantum Efficiency</p>
            <p className="text-lg font-bold">{Math.round(task.quantumEfficiency * 100)}%</p>
          </div>
          <div>
            <p className="label text-xs mb-1">Entanglement Strength</p>
            <p className="text-lg font-bold">{Math.round(task.entanglementStrength * 100)}%</p>
          </div>
          <div>
            <p className="label text-xs mb-1">Decoherence Rate</p>
            <p className="text-lg font-bold">{(task.decoherenceRate * 100).toFixed(2)}%</p>
          </div>
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
        <Button variant="secondary" icon={<Settings className="h-4 w-4" />}>
          Configure
        </Button>
        <Button variant="danger" icon={<Square className="h-4 w-4" />}>
          Terminate
        </Button>
      </div>
    </div>
  );
}

function CreateQuantumTaskForm({ onClose }: { onClose: () => void }) {
  const [formData, setFormData] = useState({
    description: '',
    targetAgentCount: 1000,
    coherenceLevel: 'HIGH' as 'LOW' | 'MEDIUM' | 'HIGH',
    priority: 'medium'
  });

  const handleSubmit = async () => {
    try {
      await createQuantumTask(formData);
      onClose();
    } catch (error) {
      console.error('Failed to create quantum task:', error);
    }
  };

  return (
    <div className="space-y-4">
      <Input
        label="Task Description"
        placeholder="Describe the quantum task..."
        value={formData.description}
        onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
      />
      
      <Input
        label="Target Agent Count"
        type="number"
        placeholder="1000"
        value={formData.targetAgentCount}
        onChange={(e) => setFormData(prev => ({ ...prev, targetAgentCount: parseInt(e.target.value) }))}
      />

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="label block mb-2">Coherence Level</label>
          <select 
            className="input w-full"
            value={formData.coherenceLevel}
            onChange={(e) => setFormData(prev => ({ ...prev, coherenceLevel: e.target.value as any }))}
          >
            <option value="LOW">Low</option>
            <option value="MEDIUM">Medium</option>
            <option value="HIGH">High</option>
          </select>
        </div>

        <div>
          <label className="label block mb-2">Priority</label>
          <select 
            className="input w-full"
            value={formData.priority}
            onChange={(e) => setFormData(prev => ({ ...prev, priority: e.target.value }))}
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>
      </div>

      <div className="flex gap-3 pt-4">
        <Button variant="primary" className="flex-1" onClick={handleSubmit}>
          Create Quantum Task
        </Button>
        <Button variant="ghost" onClick={onClose}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

