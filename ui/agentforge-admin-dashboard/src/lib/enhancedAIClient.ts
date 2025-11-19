/**
 * Enhanced AI Client for AgentForge Frontend
 * Provides complete integration with all advanced AI capabilities
 */

const ENHANCED_AI_BASE = 'http://localhost:8001';

export interface EnhancedAIStatus {
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

export interface IntelligentAgent {
  agent_id: string;
  role: string;
  specializations: string[];
  status: string;
  performance_metrics: any;
  capabilities_available: number;
  neural_mesh_connected: boolean;
  created_at: number;
}

export interface IntelligentSwarm {
  swarm_id: string;
  agents_deployed: number;
  intelligence_mode: string;
  objective: string;
  estimated_capability_amplification: number;
  agent_ids: string[];
  coordination_session?: string;
}

export interface CollectiveReasoningResult {
  reasoning_session_id: string;
  swarm_id: string;
  participating_agents: number;
  collective_reasoning: string;
  individual_contributions: number;
  collective_confidence: number;
  intelligence_amplification: number;
  emergent_insights?: string[];
}

export interface KnowledgeQueryResult {
  query: string;
  response: string;
  source_documents: Array<{
    document_id: string;
    content_preview: string;
    relevance_score: number;
    document_type: string;
    metadata: any;
  }>;
  confidence: number;
  reasoning: string;
  token_usage: number;
  processing_time: number;
}

export interface CapabilityExecution {
  execution_id: string;
  capability_name: string;
  success: boolean;
  result: any;
  execution_time: number;
  error_message?: string;
}

class EnhancedAIClient {
  private baseUrl: string;
  private wsConnection: WebSocket | null = null;
  private eventListeners: Map<string, Function[]> = new Map();

  constructor(baseUrl: string = ENHANCED_AI_BASE) {
    this.baseUrl = baseUrl;
  }

  // System Status
  async getAISystemStatus(): Promise<EnhancedAIStatus> {
    const response = await fetch(`${this.baseUrl}/v1/ai/status`);
    return await response.json();
  }

  async getSystemAnalytics(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/analytics/system`);
    return await response.json();
  }

  // Agent Management
  async createEnhancedAgent(
    role: string = 'generalist',
    specializations: string[] = [],
    capabilities: string[] = []
  ): Promise<IntelligentAgent> {
    const response = await fetch(`${this.baseUrl}/v1/ai/agents/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role, specializations, capabilities })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to create agent: ${response.statusText}`);
    }
    
    return await response.json();
  }

  async listActiveAgents(): Promise<{ agents: IntelligentAgent[]; total_agents: number }> {
    const response = await fetch(`${this.baseUrl}/v1/ai/agents/list`);
    return await response.json();
  }

  async getAgentStatus(agentId: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/agents/${agentId}/status`);
    return await response.json();
  }

  // Swarm Management
  async deployIntelligentSwarm(
    objective: string,
    capabilities: string[],
    specializations: string[] = [],
    maxAgents: number = 10,
    intelligenceMode: string = 'collective'
  ): Promise<IntelligentSwarm> {
    const response = await fetch(`${this.baseUrl}/v1/ai/swarms/deploy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        objective,
        capabilities,
        specializations,
        max_agents: maxAgents,
        intelligence_mode: intelligenceMode
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to deploy swarm: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Collective Reasoning
  async coordinateCollectiveReasoning(
    swarmId: string,
    reasoningObjective: string,
    reasoningPattern: string = 'collective_chain_of_thought'
  ): Promise<CollectiveReasoningResult> {
    const response = await fetch(`${this.baseUrl}/v1/ai/reasoning/collective`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        swarm_id: swarmId,
        reasoning_objective: reasoningObjective,
        reasoning_pattern: reasoningPattern
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to coordinate reasoning: ${response.statusText}`);
    }
    
    return await response.json();
  }

  async getReasoningPatterns(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/reasoning/patterns`);
    return await response.json();
  }

  // Knowledge Management
  async queryKnowledgeBase(
    query: string,
    agentId: string = 'frontend_user',
    maxContextDocs: number = 5
  ): Promise<KnowledgeQueryResult> {
    const response = await fetch(`${this.baseUrl}/v1/ai/knowledge/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        agent_id: agentId,
        max_context_docs: maxContextDocs
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to query knowledge base: ${response.statusText}`);
    }
    
    return await response.json();
  }

  async uploadDocument(file: File, agentId: string = 'system'): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('agent_id', agentId);
    formData.append('document_type', 'auto');

    const response = await fetch(`${this.baseUrl}/v1/ai/knowledge/upload`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Failed to upload document: ${response.statusText}`);
    }
    
    return await response.json();
  }

  async synthesizeCollectiveKnowledge(
    knowledgeDomain: string,
    contributingAgents?: string[]
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/knowledge/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        knowledge_domain: knowledgeDomain,
        contributing_agents: contributingAgents
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to synthesize knowledge: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Capabilities System
  async getAvailableCapabilities(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/capabilities/available`);
    return await response.json();
  }

  async executeCapability(
    agentId: string,
    capabilityName: string,
    parameters: any
  ): Promise<CapabilityExecution> {
    const response = await fetch(`${this.baseUrl}/v1/ai/capabilities/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        agent_id: agentId,
        capability_name: capabilityName,
        parameters
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to execute capability: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Learning System
  async submitFeedback(
    agentId: string,
    taskId: string,
    rating: number,
    comments: string = '',
    improvementSuggestions: string[] = []
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/feedback/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        agent_id: agentId,
        task_id: taskId,
        rating,
        comments,
        improvement_suggestions: improvementSuggestions
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to submit feedback: ${response.statusText}`);
    }
    
    return await response.json();
  }

  async getLearningAnalytics(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/learning/analytics`);
    return await response.json();
  }

  // Neural Mesh
  async getNeuralMeshStatus(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/neural-mesh/status`);
    return await response.json();
  }

  async storeMemoryInNeuralMesh(
    agentId: string,
    memoryType: string,
    content: any,
    memoryTier: string = 'L2',
    metadata: any = {}
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/neural-mesh/memory/store`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        agent_id: agentId,
        memory_type: memoryType,
        content,
        memory_tier: memoryTier,
        metadata
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to store memory: ${response.statusText}`);
    }
    
    return await response.json();
  }

  async retrieveAgentMemories(
    agentId: string,
    query: string = '',
    strategy: string = 'hybrid',
    limit: number = 10
  ): Promise<any> {
    const params = new URLSearchParams({
      query,
      strategy,
      limit: limit.toString()
    });
    
    const response = await fetch(`${this.baseUrl}/v1/ai/neural-mesh/memory/${agentId}?${params}`);
    return await response.json();
  }

  async facilitateCollaboration(
    initiatorAgent: string,
    targetAgents: string[],
    collaborationObjective: string,
    sharedContext: any = {}
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/collaboration/facilitate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        initiator_agent: initiatorAgent,
        target_agents: targetAgents,
        collaboration_objective: collaborationObjective,
        shared_context: sharedContext
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to facilitate collaboration: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Task Execution
  async executeIntelligentTask(
    description: string,
    taskType: string = 'general',
    priority: string = 'normal',
    requiredCapabilities: string[] = [],
    context: any = {},
    reasoningPattern?: string
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/tasks/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        description,
        task_type: taskType,
        priority,
        required_capabilities: requiredCapabilities,
        context,
        reasoning_pattern: reasoningPattern
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to execute task: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // Demo Functions
  async runIntelligentAnalysisDemo(
    analysisRequest: string,
    useSwarm: boolean = false,
    agentCount: number = 5
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/demo/intelligent-analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        analysis_request: analysisRequest,
        use_swarm: useSwarm,
        agent_count: agentCount
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to run analysis demo: ${response.statusText}`);
    }
    
    return await response.json();
  }

  async runCollectiveReasoningDemo(
    reasoningProblem: string,
    agentCount: number = 3
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/demo/collective-reasoning`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        reasoning_problem: reasoningProblem,
        agent_count: agentCount
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to run reasoning demo: ${response.statusText}`);
    }
    
    return await response.json();
  }

  async getCapabilitiesShowcase(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/v1/ai/demo/capabilities-showcase`);
    return await response.json();
  }

  // WebSocket Connection
  connectToEnhancedAI(): WebSocket {
    if (this.wsConnection) {
      this.wsConnection.close();
    }

    const wsUrl = `${this.baseUrl.replace('http', 'ws')}/v1/ai/realtime`;
    this.wsConnection = new WebSocket(wsUrl);

    this.wsConnection.onopen = () => {
      console.log('Connected to Enhanced AI WebSocket');
      this.emit('connected', {});
    };

    this.wsConnection.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.emit('message', data);
        
        // Handle specific message types
        if (data.type) {
          this.emit(data.type, data);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.wsConnection.onclose = () => {
      console.log('Disconnected from Enhanced AI WebSocket');
      this.emit('disconnected', {});
    };

    this.wsConnection.onerror = (error) => {
      console.error('Enhanced AI WebSocket error:', error);
      this.emit('error', error);
    };

    return this.wsConnection;
  }

  // Event handling
  on(event: string, callback: Function) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event)!.push(callback);
  }

  off(event: string, callback: Function) {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      const index = listeners.indexOf(callback);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any) {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Error in event listener:', error);
        }
      });
    }
  }

  // Utility methods
  async testSystemHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/ai/health`);
      const health = await response.json();
      return health.status === 'healthy';
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  async runComprehensiveDemo(): Promise<any> {
    try {
      // Create an agent
      const agent = await this.createEnhancedAgent('specialist', ['analysis', 'research']);
      
      // Deploy a small swarm
      const swarm = await this.deployIntelligentSwarm(
        'Demonstrate comprehensive AI capabilities',
        ['analysis', 'reasoning', 'collaboration'],
        ['research', 'problem_solving'],
        3,
        'collective'
      );
      
      // Wait for swarm initialization
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Run collective reasoning
      const reasoning = await this.coordinateCollectiveReasoning(
        swarm.swarm_id,
        'What are the most effective strategies for AI system optimization?',
        'collective_chain_of_thought'
      );
      
      return {
        demo_type: 'comprehensive',
        agent_created: agent,
        swarm_deployed: swarm,
        collective_reasoning: reasoning,
        success: true,
        timestamp: Date.now()
      };
      
    } catch (error) {
      console.error('Comprehensive demo failed:', error);
      return {
        demo_type: 'comprehensive',
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
}

// Export singleton instance
export const enhancedAIClient = new EnhancedAIClient();

// Export class for custom instances
export { EnhancedAIClient };
