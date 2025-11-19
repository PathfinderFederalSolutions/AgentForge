'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { 
  Network, 
  Database, 
  Brain, 
  Users, 
  Activity, 
  Zap, 
  Globe,
  Clock,
  TrendingUp,
  MessageSquare,
  Settings,
  RefreshCw,
  CheckCircle,
  AlertTriangle,
  Info
} from 'lucide-react';

interface NeuralMeshStatus {
  neural_mesh_status: {
    mode: string;
    intelligence_level: string;
    active_agents: number;
    total_memories: number;
    sync_operations_per_second: number;
    memory_consistency_rate: number;
    system_health: number;
  };
  collective_intelligence: {
    emergence_score: number;
    collaboration_effectiveness: number;
    knowledge_synthesis_rate: number;
    intelligence_amplification_factor: number;
  };
  active_swarms: number;
  integration_health: {
    neural_mesh_integrated: boolean;
    agent_intelligence_integrated: boolean;
    communication_system_active: boolean;
    distributed_memory_active: boolean;
    replication_active: boolean;
  };
}

interface MemoryTierStats {
  tier: string;
  access_time: string;
  description: string;
  memory_count: number;
  hit_rate: number;
}

interface AgentCollaboration {
  collaboration_id: string;
  initiator: string;
  participants: string[];
  objective: string;
  status: string;
  started_at: number;
}

export default function NeuralMeshPage() {
  const [neuralMeshStatus, setNeuralMeshStatus] = useState<NeuralMeshStatus | null>(null);
  const [memoryStats, setMemoryStats] = useState<MemoryTierStats[]>([]);
  const [activeCollaborations, setActiveCollaborations] = useState<AgentCollaboration[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  // Demo states
  const [collaborationObjective, setCollaborationObjective] = useState('');
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);
  const [knowledgeDomain, setKnowledgeDomain] = useState('');
  const [memoryQuery, setMemoryQuery] = useState('');
  const [queryResults, setQueryResults] = useState<any[]>([]);

  useEffect(() => {
    loadNeuralMeshData();
    
    // Refresh every 15 seconds for real-time updates
    const interval = setInterval(loadNeuralMeshData, 15000);
    return () => clearInterval(interval);
  }, []);

  const loadNeuralMeshData = async () => {
    try {
      // Load neural mesh status
      const statusResponse = await fetch('http://localhost:8001/v1/ai/neural-mesh/status');
      const status = await statusResponse.json();
      setNeuralMeshStatus(status);

      // Load memory tier statistics (mock data based on actual implementation)
      setMemoryStats([
        {
          tier: 'L1',
          access_time: '<1ms',
          description: 'Working Memory (Redis)',
          memory_count: Math.floor(Math.random() * 1000) + 500,
          hit_rate: 0.95 + Math.random() * 0.05
        },
        {
          tier: 'L2', 
          access_time: '<5ms',
          description: 'Short-term Memory (Redis)',
          memory_count: Math.floor(Math.random() * 5000) + 2000,
          hit_rate: 0.90 + Math.random() * 0.08
        },
        {
          tier: 'L3',
          access_time: '<50ms', 
          description: 'Long-term Memory (PostgreSQL)',
          memory_count: Math.floor(Math.random() * 20000) + 10000,
          hit_rate: 0.85 + Math.random() * 0.10
        },
        {
          tier: 'L4',
          access_time: '<200ms',
          description: 'Archive Memory (Compressed)',
          memory_count: Math.floor(Math.random() * 100000) + 50000,
          hit_rate: 0.75 + Math.random() * 0.15
        }
      ]);

      setIsLoading(false);

    } catch (error) {
      console.error('Failed to load neural mesh data:', error);
      setIsLoading(false);
    }
  };

  const facilitateCollaboration = async () => {
    if (!collaborationObjective.trim() || selectedAgents.length < 2) {
      alert('Please provide an objective and select at least 2 agents');
      return;
    }

    try {
      const response = await fetch('http://localhost:8001/v1/ai/collaboration/facilitate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          initiator_agent: selectedAgents[0],
          target_agents: selectedAgents.slice(1),
          collaboration_objective: collaborationObjective,
          shared_context: { initiated_from: 'frontend' }
        })
      });

      const result = await response.json();
      
      // Add to active collaborations
      setActiveCollaborations(prev => [...prev, {
        collaboration_id: result.collaboration_id,
        initiator: result.initiator_agent,
        participants: result.target_agents,
        objective: collaborationObjective,
        status: 'active',
        started_at: Date.now()
      }]);

      setCollaborationObjective('');
      setSelectedAgents([]);
      
      alert(`Collaboration facilitated: ${result.collaboration_id}`);

    } catch (error) {
      console.error('Error facilitating collaboration:', error);
      alert('Failed to facilitate collaboration');
    }
  };

  const synthesizeKnowledge = async () => {
    if (!knowledgeDomain.trim()) {
      alert('Please specify a knowledge domain');
      return;
    }

    try {
      const response = await fetch('http://localhost:8001/v1/ai/knowledge/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          knowledge_domain: knowledgeDomain,
          contributing_agents: null // Use all agents
        })
      });

      const result = await response.json();
      
      alert(`Knowledge synthesized for domain: ${knowledgeDomain}\nContributing agents: ${result.contributing_agents}\nConfidence: ${(result.confidence * 100).toFixed(1)}%`);
      
      setKnowledgeDomain('');

    } catch (error) {
      console.error('Error synthesizing knowledge:', error);
      alert('Failed to synthesize knowledge');
    }
  };

  const queryMemories = async () => {
    if (!memoryQuery.trim()) {
      alert('Please enter a memory query');
      return;
    }

    try {
      const response = await fetch(`http://localhost:8001/v1/ai/neural-mesh/memory/system?query=${encodeURIComponent(memoryQuery)}&strategy=hybrid&limit=10`);
      const result = await response.json();
      
      setQueryResults([result, ...queryResults.slice(0, 4)]); // Keep last 5 results
      setMemoryQuery('');

    } catch (error) {
      console.error('Error querying memories:', error);
      alert('Failed to query memories');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Network className="h-8 w-8 animate-pulse mx-auto mb-4" />
          <p>Loading Neural Mesh Status...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Network className="h-8 w-8" />
          Neural Mesh Control Center
        </h1>
        <Button onClick={loadNeuralMeshData} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* System Status Overview */}
      {neuralMeshStatus && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="p-6 text-center">
            <div className={`inline-flex items-center justify-center w-12 h-12 rounded-full mb-3 ${
              neuralMeshStatus.neural_mesh_status.system_health > 0.8 ? 'bg-green-100 text-green-600' : 
              neuralMeshStatus.neural_mesh_status.system_health > 0.6 ? 'bg-yellow-100 text-yellow-600' : 
              'bg-red-100 text-red-600'
            }`}>
              <Activity className="h-6 w-6" />
            </div>
            <div className="text-2xl font-bold">
              {(neuralMeshStatus.neural_mesh_status.system_health * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">System Health</div>
          </Card>

          <Card className="p-6 text-center">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-blue-100 text-blue-600 mb-3">
              <Users className="h-6 w-6" />
            </div>
            <div className="text-2xl font-bold text-blue-600">
              {neuralMeshStatus.neural_mesh_status.active_agents}
            </div>
            <div className="text-sm text-gray-600">Active Agents</div>
          </Card>

          <Card className="p-6 text-center">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-purple-100 text-purple-600 mb-3">
              <Database className="h-6 w-6" />
            </div>
            <div className="text-2xl font-bold text-purple-600">
              {neuralMeshStatus.neural_mesh_status.total_memories.toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">Total Memories</div>
          </Card>

          <Card className="p-6 text-center">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-orange-100 text-orange-600 mb-3">
              <Zap className="h-6 w-6" />
            </div>
            <div className="text-2xl font-bold text-orange-600">
              {neuralMeshStatus.collective_intelligence.intelligence_amplification_factor.toFixed(1)}x
            </div>
            <div className="text-sm text-gray-600">Intelligence Amplification</div>
          </Card>
        </div>
      )}

      {/* Memory Tier Architecture */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Database className="h-5 w-5" />
          4-Tier Memory Architecture
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {memoryStats.map((tier) => (
            <div key={tier.tier} className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-lg">{tier.tier}</span>
                <Badge variant="outline">{tier.access_time}</Badge>
              </div>
              <div className="text-sm text-gray-600 mb-3">
                {tier.description}
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Memories:</span>
                  <span>{tier.memory_count.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Hit Rate:</span>
                  <span>{(tier.hit_rate * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full" 
                    style={{ width: `${tier.hit_rate * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Collective Intelligence Metrics */}
      {neuralMeshStatus && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Collective Intelligence Metrics
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <TrendingUp className="h-6 w-6 mx-auto mb-2 text-green-600" />
              <div className="text-xl font-bold text-green-600">
                {(neuralMeshStatus.collective_intelligence.emergence_score * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Emergence Score</div>
            </div>

            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <Users className="h-6 w-6 mx-auto mb-2 text-blue-600" />
              <div className="text-xl font-bold text-blue-600">
                {(neuralMeshStatus.collective_intelligence.collaboration_effectiveness * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Collaboration Effectiveness</div>
            </div>

            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <Brain className="h-6 w-6 mx-auto mb-2 text-purple-600" />
              <div className="text-xl font-bold text-purple-600">
                {neuralMeshStatus.collective_intelligence.knowledge_synthesis_rate.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600">Knowledge Synthesis Rate</div>
            </div>

            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <Zap className="h-6 w-6 mx-auto mb-2 text-orange-600" />
              <div className="text-xl font-bold text-orange-600">
                {neuralMeshStatus.collective_intelligence.intelligence_amplification_factor.toFixed(1)}x
              </div>
              <div className="text-sm text-gray-600">Intelligence Amplification</div>
            </div>
          </div>
        </Card>
      )}

      {/* System Integration Status */}
      {neuralMeshStatus && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Integration Health
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {Object.entries(neuralMeshStatus.integration_health).map(([system, status]) => (
              <div key={system} className="flex items-center gap-2 p-3 border rounded-lg">
                {status ? (
                  <CheckCircle className="h-5 w-5 text-green-600" />
                ) : (
                  <AlertTriangle className="h-5 w-5 text-red-600" />
                )}
                <div>
                  <div className="font-medium text-sm">
                    {system.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </div>
                  <div className={`text-xs ${status ? 'text-green-600' : 'text-red-600'}`}>
                    {status ? 'Active' : 'Inactive'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Interactive Neural Mesh Operations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Agent Collaboration */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Facilitate Agent Collaboration
          </h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Collaboration Objective</label>
              <Input
                value={collaborationObjective}
                onChange={(e) => setCollaborationObjective(e.target.value)}
                placeholder="Analyze system performance and identify optimization opportunities"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Select Agents (comma-separated IDs)</label>
              <Input
                value={selectedAgents.join(', ')}
                onChange={(e) => setSelectedAgents(e.target.value.split(',').map(s => s.trim()).filter(s => s))}
                placeholder="agent-001, agent-002, agent-003"
              />
            </div>
            
            <Button 
              onClick={facilitateCollaboration}
              disabled={!collaborationObjective.trim() || selectedAgents.length < 2}
              className="w-full"
            >
              Facilitate Collaboration
            </Button>
          </div>
        </Card>

        {/* Knowledge Synthesis */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Synthesize Collective Knowledge
          </h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Knowledge Domain</label>
              <Input
                value={knowledgeDomain}
                onChange={(e) => setKnowledgeDomain(e.target.value)}
                placeholder="cybersecurity, performance_optimization, system_architecture"
              />
            </div>
            
            <Button 
              onClick={synthesizeKnowledge}
              disabled={!knowledgeDomain.trim()}
              className="w-full"
            >
              Synthesize Knowledge
            </Button>
          </div>
        </Card>
      </div>

      {/* Memory Query Interface */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Database className="h-5 w-5" />
          Neural Mesh Memory Query
        </h2>
        <div className="flex gap-4 mb-4">
          <Input
            value={memoryQuery}
            onChange={(e) => setMemoryQuery(e.target.value)}
            placeholder="Query neural mesh memories..."
            className="flex-1"
          />
          <Button onClick={queryMemories} disabled={!memoryQuery.trim()}>
            Query Memories
          </Button>
        </div>
        
        {queryResults.length > 0 && (
          <div className="space-y-4">
            <h3 className="font-medium">Query Results</h3>
            {queryResults.map((result, index) => (
              <div key={index} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">Query: {result.query}</span>
                  <Badge variant="info">
                    {result.count} memories found
                  </Badge>
                </div>
                <div className="text-sm text-gray-600">
                  Strategy: {result.strategy} | Retrieved in: {result.retrieved_at}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Active Collaborations */}
      {activeCollaborations.length > 0 && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Users className="h-5 w-5" />
            Active Agent Collaborations
          </h2>
          <div className="space-y-4">
            {activeCollaborations.map((collab) => (
              <div key={collab.collaboration_id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{collab.collaboration_id}</span>
                  <Badge variant="success">{collab.status}</Badge>
                </div>
                <div className="text-sm text-gray-600 mb-2">
                  <strong>Objective:</strong> {collab.objective}
                </div>
                <div className="text-sm text-gray-600 mb-2">
                  <strong>Participants:</strong> {collab.participants.join(', ')}
                </div>
                <div className="text-xs text-gray-500">
                  Started: {new Date(collab.started_at).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Real-time Synchronization Metrics */}
      {neuralMeshStatus && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Real-time Synchronization
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <Clock className="h-6 w-6 mx-auto mb-2 text-blue-600" />
              <div className="text-xl font-bold text-blue-600">
                {neuralMeshStatus.neural_mesh_status.sync_operations_per_second.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600">Sync Operations/sec</div>
            </div>

            <div className="text-center p-4 bg-green-50 rounded-lg">
              <CheckCircle className="h-6 w-6 mx-auto mb-2 text-green-600" />
              <div className="text-xl font-bold text-green-600">
                {(neuralMeshStatus.neural_mesh_status.memory_consistency_rate * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Memory Consistency</div>
            </div>

            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <Globe className="h-6 w-6 mx-auto mb-2 text-purple-600" />
              <div className="text-xl font-bold text-purple-600">
                {neuralMeshStatus.neural_mesh_status.mode.toUpperCase()}
              </div>
              <div className="text-sm text-gray-600">Operation Mode</div>
            </div>
          </div>
        </Card>
      )}

      {/* Neural Mesh Features Demo */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Neural Mesh Capabilities Demo</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Button 
            onClick={() => alert('Memory synchronization demo - agents sharing knowledge in real-time')}
            className="h-20 flex flex-col items-center justify-center gap-2"
          >
            <RefreshCw className="h-5 w-5" />
            Memory Sync Demo
          </Button>
          
          <Button 
            onClick={() => alert('Conflict resolution demo - automatic resolution of memory conflicts')}
            className="h-20 flex flex-col items-center justify-center gap-2"
          >
            <Settings className="h-5 w-5" />
            Conflict Resolution
          </Button>
          
          <Button 
            onClick={() => alert('Emergent intelligence demo - detecting spontaneous intelligence patterns')}
            className="h-20 flex flex-col items-center justify-center gap-2"
          >
            <Brain className="h-5 w-5" />
            Emergent Intelligence
          </Button>
          
          <Button 
            onClick={() => alert('Cross-datacenter replication demo - multi-region memory sync')}
            className="h-20 flex flex-col items-center justify-center gap-2"
          >
            <Globe className="h-5 w-5" />
            Cross-DC Replication
          </Button>
        </div>
      </Card>

      {/* Technical Details */}
      {neuralMeshStatus && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Technical Implementation Details</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium mb-3">Neural Mesh Configuration</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Mode:</span>
                  <Badge variant="info">{neuralMeshStatus.neural_mesh_status.mode}</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Intelligence Level:</span>
                  <Badge variant="success">{neuralMeshStatus.neural_mesh_status.intelligence_level}</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Active Swarms:</span>
                  <span>{neuralMeshStatus.active_swarms}</span>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="font-medium mb-3">System Integration</h3>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${neuralMeshStatus.integration_health.distributed_memory_active ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span>Distributed Memory Store</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${neuralMeshStatus.integration_health.communication_system_active ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span>Inter-Agent Communication</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${neuralMeshStatus.integration_health.replication_active ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span>Cross-Datacenter Replication</span>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}