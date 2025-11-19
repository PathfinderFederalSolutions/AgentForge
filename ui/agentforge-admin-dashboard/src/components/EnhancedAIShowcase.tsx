'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Modal } from '@/components/ui/Modal';
import { 
  Brain, 
  Network, 
  Zap, 
  Users, 
  BookOpen, 
  TrendingUp, 
  Settings, 
  Play,
  Pause,
  RotateCcw,
  CheckCircle,
  AlertCircle,
  Clock,
  Activity
} from 'lucide-react';

interface AISystemStatus {
  enhanced_ai_available: boolean;
  neural_mesh_available: boolean;
  systems: {
    master_coordinator?: any;
    llm_integration?: any;
    capabilities?: any;
    learning?: any;
    neural_mesh?: any;
  };
}

interface AgentInfo {
  agent_id: string;
  role: string;
  specializations: string[];
  status: string;
  performance_metrics: any;
  capabilities_available: number;
  neural_mesh_connected: boolean;
  created_at: number;
}

interface SwarmDeployment {
  swarm_id: string;
  agents_deployed: number;
  intelligence_mode: string;
  objective: string;
  estimated_capability_amplification: number;
}

interface CollectiveReasoning {
  reasoning_session_id: string;
  collective_confidence: number;
  intelligence_amplification: number;
  collective_reasoning: string;
  individual_contributions: number;
}

export default function EnhancedAIShowcase() {
  const [aiStatus, setAiStatus] = useState<AISystemStatus | null>(null);
  const [activeAgents, setActiveAgents] = useState<AgentInfo[]>([]);
  const [deployedSwarms, setDeployedSwarms] = useState<SwarmDeployment[]>([]);
  const [reasoningResults, setReasoningResults] = useState<CollectiveReasoning[]>([]);
  
  // Demo states
  const [isCreatingAgent, setIsCreatingAgent] = useState(false);
  const [isDeployingSwarm, setIsDeployingSwarm] = useState(false);
  const [isRunningReasoning, setIsRunningReasoning] = useState(false);
  
  // Form states
  const [newAgentRole, setNewAgentRole] = useState('generalist');
  const [newAgentSpecs, setNewAgentSpecs] = useState('');
  const [swarmObjective, setSwarmObjective] = useState('');
  const [swarmCapabilities, setSwarmCapabilities] = useState('');
  const [reasoningProblem, setReasoningProblem] = useState('');
  const [selectedSwarmId, setSelectedSwarmId] = useState('');
  
  // Modal states
  const [showAgentModal, setShowAgentModal] = useState(false);
  const [showSwarmModal, setShowSwarmModal] = useState(false);
  const [showReasoningModal, setShowReasoningModal] = useState(false);
  const [showKnowledgeModal, setShowKnowledgeModal] = useState(false);

  useEffect(() => {
    loadAIStatus();
    loadActiveAgents();
    
    // Refresh every 30 seconds
    const interval = setInterval(() => {
      loadAIStatus();
      loadActiveAgents();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const loadAIStatus = async () => {
    try {
      const response = await fetch('http://localhost:8001/v1/ai/status');
      const status = await response.json();
      setAiStatus(status);
    } catch (error) {
      console.error('Failed to load AI status:', error);
    }
  };

  const loadActiveAgents = async () => {
    try {
      const response = await fetch('http://localhost:8001/v1/ai/agents/list');
      const data = await response.json();
      setActiveAgents(data.agents || []);
    } catch (error) {
      console.error('Failed to load active agents:', error);
    }
  };

  const createEnhancedAgent = async () => {
    setIsCreatingAgent(true);
    try {
      const response = await fetch('http://localhost:8001/v1/ai/agents/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          role: newAgentRole,
          specializations: newAgentSpecs.split(',').map(s => s.trim()).filter(s => s),
          capabilities: ['reasoning', 'learning', 'collaboration']
        })
      });
      
      const result = await response.json();
      
      // Refresh agents list
      await loadActiveAgents();
      
      // Close modal
      setShowAgentModal(false);
      setNewAgentRole('generalist');
      setNewAgentSpecs('');
      
      alert(`Enhanced agent created: ${result.agent_id}`);
      
    } catch (error) {
      console.error('Error creating agent:', error);
      alert('Failed to create agent');
    } finally {
      setIsCreatingAgent(false);
    }
  };

  const deployIntelligentSwarm = async () => {
    setIsDeployingSwarm(true);
    try {
      const response = await fetch('http://localhost:8001/v1/ai/swarms/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          objective: swarmObjective,
          capabilities: swarmCapabilities.split(',').map(s => s.trim()).filter(s => s),
          specializations: ['analysis', 'research', 'problem_solving'],
          max_agents: 5,
          intelligence_mode: 'collective'
        })
      });
      
      const result = await response.json();
      
      // Add to deployed swarms
      setDeployedSwarms(prev => [...prev, result]);
      
      // Close modal
      setShowSwarmModal(false);
      setSwarmObjective('');
      setSwarmCapabilities('');
      
      alert(`Intelligent swarm deployed: ${result.swarm_id} with ${result.agents_deployed} agents`);
      
    } catch (error) {
      console.error('Error deploying swarm:', error);
      alert('Failed to deploy swarm');
    } finally {
      setIsDeployingSwarm(false);
    }
  };

  const runCollectiveReasoning = async () => {
    if (!selectedSwarmId) {
      alert('Please select a swarm first');
      return;
    }
    
    setIsRunningReasoning(true);
    try {
      const response = await fetch('http://localhost:8001/v1/ai/reasoning/collective', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          swarm_id: selectedSwarmId,
          reasoning_objective: reasoningProblem,
          reasoning_pattern: 'collective_chain_of_thought'
        })
      });
      
      const result = await response.json();
      
      // Add to reasoning results
      setReasoningResults(prev => [...prev, result]);
      
      // Close modal
      setShowReasoningModal(false);
      setReasoningProblem('');
      
      alert(`Collective reasoning completed with ${result.collective_confidence} confidence`);
      
    } catch (error) {
      console.error('Error running collective reasoning:', error);
      alert('Failed to run collective reasoning');
    } finally {
      setIsRunningReasoning(false);
    }
  };

  const runCapabilityDemo = async (capabilityType: string) => {
    try {
      const response = await fetch('http://localhost:8001/v1/ai/demo/intelligent-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysis_request: `Demonstrate ${capabilityType} capabilities`,
          use_swarm: capabilityType === 'swarm',
          agent_count: 3
        })
      });
      
      const result = await response.json();
      alert(`${capabilityType} demo completed successfully!`);
      console.log('Demo result:', result);
      
    } catch (error) {
      console.error('Error running capability demo:', error);
      alert('Demo failed');
    }
  };

  if (!aiStatus) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Activity className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p>Loading AI Systems...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* AI System Status Overview */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="h-6 w-6" />
            Enhanced AI Capabilities
          </h2>
          <div className="flex gap-2">
            <Badge variant={aiStatus.enhanced_ai_available ? "success" : "destructive"}>
              AI Intelligence: {aiStatus.enhanced_ai_available ? "Active" : "Offline"}
            </Badge>
            <Badge variant={aiStatus.neural_mesh_available ? "success" : "destructive"}>
              Neural Mesh: {aiStatus.neural_mesh_available ? "Active" : "Offline"}
            </Badge>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <Users className="h-8 w-8 mx-auto mb-2 text-blue-600" />
            <div className="text-2xl font-bold text-blue-600">
              {activeAgents.length}
            </div>
            <div className="text-sm text-gray-600">Active Intelligent Agents</div>
          </div>
          
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <Network className="h-8 w-8 mx-auto mb-2 text-green-600" />
            <div className="text-2xl font-bold text-green-600">
              {deployedSwarms.length}
            </div>
            <div className="text-sm text-gray-600">Deployed Swarms</div>
          </div>
          
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <Brain className="h-8 w-8 mx-auto mb-2 text-purple-600" />
            <div className="text-2xl font-bold text-purple-600">
              {reasoningResults.length}
            </div>
            <div className="text-sm text-gray-600">Collective Reasoning Sessions</div>
          </div>
        </div>
      </Card>

      {/* Quick Action Buttons */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Button 
          onClick={() => setShowAgentModal(true)}
          className="h-20 flex flex-col items-center justify-center gap-2"
        >
          <Users className="h-6 w-6" />
          Create Enhanced Agent
        </Button>
        
        <Button 
          onClick={() => setShowSwarmModal(true)}
          className="h-20 flex flex-col items-center justify-center gap-2"
        >
          <Network className="h-6 w-6" />
          Deploy Intelligent Swarm
        </Button>
        
        <Button 
          onClick={() => setShowReasoningModal(true)}
          className="h-20 flex flex-col items-center justify-center gap-2"
          disabled={deployedSwarms.length === 0}
        >
          <Brain className="h-6 w-6" />
          Collective Reasoning
        </Button>
        
        <Button 
          onClick={() => runCapabilityDemo('knowledge')}
          className="h-20 flex flex-col items-center justify-center gap-2"
        >
          <BookOpen className="h-6 w-6" />
          Knowledge Demo
        </Button>
      </div>

      {/* Capability Showcase Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Agent Intelligence */}
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="h-5 w-5 text-blue-600" />
            <h3 className="text-lg font-semibold">Agent Intelligence</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span>Multi-Provider LLM</span>
              <Badge variant="success">Active</Badge>
            </div>
            <div className="flex justify-between">
              <span>Advanced Reasoning</span>
              <Badge variant="success">3 Patterns</Badge>
            </div>
            <div className="flex justify-between">
              <span>Learning System</span>
              <Badge variant="success">Continuous</Badge>
            </div>
            <Button 
              size="sm" 
              onClick={() => runCapabilityDemo('reasoning')}
              className="w-full"
            >
              Test Reasoning
            </Button>
          </div>
        </Card>

        {/* Neural Mesh */}
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <Network className="h-5 w-5 text-green-600" />
            <h3 className="text-lg font-semibold">Neural Mesh</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span>Distributed Memory</span>
              <Badge variant="success">4-Tier</Badge>
            </div>
            <div className="flex justify-between">
              <span>Synchronization</span>
              <Badge variant="success">CRDT</Badge>
            </div>
            <div className="flex justify-between">
              <span>Replication</span>
              <Badge variant="success">Multi-Region</Badge>
            </div>
            <Button 
              size="sm" 
              onClick={() => runCapabilityDemo('neural_mesh')}
              className="w-full"
            >
              Test Neural Mesh
            </Button>
          </div>
        </Card>

        {/* Collective Intelligence */}
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="h-5 w-5 text-purple-600" />
            <h3 className="text-lg font-semibold">Collective Intelligence</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span>Swarm Coordination</span>
              <Badge variant="success">Advanced</Badge>
            </div>
            <div className="flex justify-between">
              <span>Emergent Patterns</span>
              <Badge variant="success">Detection</Badge>
            </div>
            <div className="flex justify-between">
              <span>Knowledge Synthesis</span>
              <Badge variant="success">Automatic</Badge>
            </div>
            <Button 
              size="sm" 
              onClick={() => runCapabilityDemo('collective')}
              className="w-full"
            >
              Test Collective Intelligence
            </Button>
          </div>
        </Card>
      </div>

      {/* Active Agents Display */}
      {activeAgents.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Users className="h-5 w-5" />
            Active Intelligent Agents ({activeAgents.length})
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {activeAgents.map((agent) => (
              <div key={agent.agent_id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{agent.agent_id}</span>
                  <Badge variant={agent.status === 'ready' ? 'success' : 'warning'}>
                    {agent.status}
                  </Badge>
                </div>
                <div className="text-sm text-gray-600 space-y-1">
                  <div>Role: {agent.role}</div>
                  <div>Specializations: {agent.specializations.join(', ') || 'None'}</div>
                  <div>Capabilities: {agent.capabilities_available}</div>
                  <div className="flex items-center gap-1">
                    <div className={`w-2 h-2 rounded-full ${agent.neural_mesh_connected ? 'bg-green-500' : 'bg-red-500'}`} />
                    Neural Mesh: {agent.neural_mesh_connected ? 'Connected' : 'Disconnected'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Deployed Swarms */}
      {deployedSwarms.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Network className="h-5 w-5" />
            Deployed Intelligent Swarms ({deployedSwarms.length})
          </h3>
          <div className="space-y-4">
            {deployedSwarms.map((swarm) => (
              <div key={swarm.swarm_id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{swarm.swarm_id}</span>
                  <div className="flex gap-2">
                    <Badge variant="info">{swarm.agents_deployed} Agents</Badge>
                    <Badge variant="success">{swarm.intelligence_mode}</Badge>
                  </div>
                </div>
                <div className="text-sm text-gray-600 mb-2">
                  <strong>Objective:</strong> {swarm.objective}
                </div>
                <div className="text-sm text-gray-600">
                  <strong>Intelligence Amplification:</strong> {swarm.estimated_capability_amplification}x
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Collective Reasoning Results */}
      {reasoningResults.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Collective Reasoning Results ({reasoningResults.length})
          </h3>
          <div className="space-y-4">
            {reasoningResults.map((result) => (
              <div key={result.reasoning_session_id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{result.reasoning_session_id}</span>
                  <div className="flex gap-2">
                    <Badge variant="info">
                      Confidence: {(result.collective_confidence * 100).toFixed(1)}%
                    </Badge>
                    <Badge variant="success">
                      {result.intelligence_amplification}x Amplification
                    </Badge>
                  </div>
                </div>
                <div className="text-sm text-gray-600 mb-2">
                  <strong>Contributors:</strong> {result.individual_contributions} agents
                </div>
                <div className="text-sm bg-gray-50 p-3 rounded">
                  {result.collective_reasoning.substring(0, 200)}...
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Demo Action Buttons */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">AI Capability Demonstrations</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Button 
            onClick={() => runCapabilityDemo('chain_of_thought')}
            variant="outline"
            className="flex flex-col items-center gap-2 h-20"
          >
            <Brain className="h-5 w-5" />
            Chain-of-Thought
          </Button>
          
          <Button 
            onClick={() => runCapabilityDemo('react')}
            variant="outline"
            className="flex flex-col items-center gap-2 h-20"
          >
            <Zap className="h-5 w-5" />
            ReAct Pattern
          </Button>
          
          <Button 
            onClick={() => runCapabilityDemo('tree_of_thoughts')}
            variant="outline"
            className="flex flex-col items-center gap-2 h-20"
          >
            <Network className="h-5 w-5" />
            Tree-of-Thoughts
          </Button>
          
          <Button 
            onClick={() => runCapabilityDemo('knowledge_rag')}
            variant="outline"
            className="flex flex-col items-center gap-2 h-20"
          >
            <BookOpen className="h-5 w-5" />
            Knowledge RAG
          </Button>
        </div>
      </Card>

      {/* Create Agent Modal */}
      <Modal isOpen={showAgentModal} onClose={() => setShowAgentModal(false)}>
        <div className="p-6">
          <h3 className="text-lg font-semibold mb-4">Create Enhanced Agent</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Agent Role</label>
              <select 
                value={newAgentRole} 
                onChange={(e) => setNewAgentRole(e.target.value)}
                className="w-full p-2 border rounded"
              >
                <option value="generalist">Generalist</option>
                <option value="specialist">Specialist</option>
                <option value="coordinator">Coordinator</option>
                <option value="analyzer">Analyzer</option>
                <option value="executor">Executor</option>
                <option value="validator">Validator</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Specializations (comma-separated)</label>
              <Input
                value={newAgentSpecs}
                onChange={(e) => setNewAgentSpecs(e.target.value)}
                placeholder="security, analysis, research"
              />
            </div>
            
            <div className="flex gap-2">
              <Button 
                onClick={createEnhancedAgent} 
                disabled={isCreatingAgent}
                className="flex-1"
              >
                {isCreatingAgent ? 'Creating...' : 'Create Agent'}
              </Button>
              <Button 
                variant="outline" 
                onClick={() => setShowAgentModal(false)}
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      </Modal>

      {/* Deploy Swarm Modal */}
      <Modal isOpen={showSwarmModal} onClose={() => setShowSwarmModal(false)}>
        <div className="p-6">
          <h3 className="text-lg font-semibold mb-4">Deploy Intelligent Swarm</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Swarm Objective</label>
              <Input
                value={swarmObjective}
                onChange={(e) => setSwarmObjective(e.target.value)}
                placeholder="Comprehensive analysis of system performance"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Required Capabilities (comma-separated)</label>
              <Input
                value={swarmCapabilities}
                onChange={(e) => setSwarmCapabilities(e.target.value)}
                placeholder="analysis, optimization, security, performance"
              />
            </div>
            
            <div className="flex gap-2">
              <Button 
                onClick={deployIntelligentSwarm} 
                disabled={isDeployingSwarm}
                className="flex-1"
              >
                {isDeployingSwarm ? 'Deploying...' : 'Deploy Swarm'}
              </Button>
              <Button 
                variant="outline" 
                onClick={() => setShowSwarmModal(false)}
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      </Modal>

      {/* Collective Reasoning Modal */}
      <Modal isOpen={showReasoningModal} onClose={() => setShowReasoningModal(false)}>
        <div className="p-6">
          <h3 className="text-lg font-semibold mb-4">Collective Reasoning</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Select Swarm</label>
              <select 
                value={selectedSwarmId} 
                onChange={(e) => setSelectedSwarmId(e.target.value)}
                className="w-full p-2 border rounded"
              >
                <option value="">Select a swarm...</option>
                {deployedSwarms.map((swarm) => (
                  <option key={swarm.swarm_id} value={swarm.swarm_id}>
                    {swarm.swarm_id} ({swarm.agents_deployed} agents)
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Reasoning Problem</label>
              <Input
                value={reasoningProblem}
                onChange={(e) => setReasoningProblem(e.target.value)}
                placeholder="What is the optimal approach to improve system security?"
              />
            </div>
            
            <div className="flex gap-2">
              <Button 
                onClick={runCollectiveReasoning} 
                disabled={isRunningReasoning || !selectedSwarmId}
                className="flex-1"
              >
                {isRunningReasoning ? 'Reasoning...' : 'Start Collective Reasoning'}
              </Button>
              <Button 
                variant="outline" 
                onClick={() => setShowReasoningModal(false)}
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      </Modal>

      {/* System Metrics */}
      {aiStatus.systems && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            AI System Metrics
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {aiStatus.systems.llm_integration && (
              <div className="text-center p-3 bg-blue-50 rounded">
                <div className="text-lg font-bold text-blue-600">
                  {aiStatus.systems.llm_integration.total_requests || 0}
                </div>
                <div className="text-xs text-gray-600">LLM Requests</div>
              </div>
            )}
            
            {aiStatus.systems.capabilities && (
              <div className="text-center p-3 bg-green-50 rounded">
                <div className="text-lg font-bold text-green-600">
                  {aiStatus.systems.capabilities.total_executions || 0}
                </div>
                <div className="text-xs text-gray-600">Capability Executions</div>
              </div>
            )}
            
            {aiStatus.systems.neural_mesh && (
              <div className="text-center p-3 bg-purple-50 rounded">
                <div className="text-lg font-bold text-purple-600">
                  {aiStatus.systems.neural_mesh.neural_mesh_status?.total_memories || 0}
                </div>
                <div className="text-xs text-gray-600">Neural Mesh Memories</div>
              </div>
            )}
            
            {aiStatus.systems.learning && (
              <div className="text-center p-3 bg-orange-50 rounded">
                <div className="text-lg font-bold text-orange-600">
                  {aiStatus.systems.learning.learning_summary?.feedback_received || 0}
                </div>
                <div className="text-xs text-gray-600">Learning Events</div>
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}
