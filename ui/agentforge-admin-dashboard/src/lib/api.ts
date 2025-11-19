const API_BASE = process.env.NEXT_PUBLIC_AGENT_API_BASE || 'http://localhost:8000';
const ORCHESTRATOR_API = `${API_BASE}/v1`;
const SWARM_API = `${API_BASE}/v1/jobs`;
const NEURAL_MESH_API = `${API_BASE}/v1/intelligence`;
const QUANTUM_API = `${API_BASE}/v1/predictive`;
const ENTERPRISE_API = `${API_BASE}/v1/enterprise`;
const REALTIME_API = `${API_BASE}/v1/realtime`;

// Types
export type TaskPayload = { 
  input: string; 
  tags?: string[]; 
  priority?: "low"|"normal"|"high";
  agentType?: string;
};

export type Agent = {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'idle' | 'error' | 'starting';
  cpu: number;
  memory: number;
  gpu?: number;
  uptime: string;
  tasksCompleted: number;
  currentTask?: string;
  location: string;
  version: string;
};

export type Job = {
  id: string;
  title: string;
  description: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'paused';
  priority: 'low' | 'medium' | 'high' | 'critical';
  progress: number;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  assignedAgent?: string;
  tags: string[];
};

export type SystemMetrics = {
  cpu: number;
  memory: number;
  network: number;
  disk: number;
  activeAgents: number;
  queueDepth: number;
  requestsPerSecond: number;
};

// Helper function for API calls
async function apiCall(url: string, options?: RequestInit) {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });
  
  if (!response.ok) {
    throw new Error(`API call failed: ${response.status} ${response.statusText}`);
  }
  
  return response.json();
}

// Task/Job Management
export async function submitTask(payload: TaskPayload) {
  return apiCall(`${ORCHESTRATOR_API}/phase/run`, {
    method: 'POST',
    body: JSON.stringify({ phase: 'task_execution', ...payload }),
  });
}

export async function getJobs(): Promise<Job[]> {
  try {
    return await apiCall(`${SWARM_API}/active`);
  } catch (error) {
    console.warn('Failed to fetch jobs from swarm API, using fallback');
    return [];
  }
}

export async function getJobById(id: string): Promise<Job> {
  return apiCall(`${SWARM_API}/jobs/${id}`);
}

export async function createJob(job: Partial<Job>): Promise<Job> {
  return apiCall(`${SWARM_API}/jobs`, {
    method: 'POST',
    body: JSON.stringify(job),
  });
}

export async function updateJob(id: string, updates: Partial<Job>): Promise<Job> {
  return apiCall(`${SWARM_API}/jobs/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  });
}

export async function cancelJob(id: string): Promise<void> {
  return apiCall(`${SWARM_API}/jobs/${id}/cancel`, {
    method: 'POST',
  });
}

// Agent Management
export async function getAgents(): Promise<Agent[]> {
  return apiCall(`${SWARM_API}/agents`);
}

export async function getAgentById(id: string): Promise<Agent> {
  return apiCall(`${SWARM_API}/agents/${id}`);
}

export async function deployAgent(config: {
  name: string;
  type: string;
  region: string;
  resources: { cpu: number; memory: number; gpu?: number };
}): Promise<Agent> {
  return apiCall(`${SWARM_API}/agents`, {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function terminateAgent(id: string): Promise<void> {
  return apiCall(`${SWARM_API}/agents/${id}`, {
    method: 'DELETE',
  });
}

export async function getAgentLogs(id: string): Promise<string[]> {
  return apiCall(`${SWARM_API}/agents/${id}/logs`);
}

// System Health & Monitoring
export async function getHealth() {
  try {
    const orchestratorHealth = await fetch(`${ORCHESTRATOR_API}/health`);
    const swarmHealth = await fetch(`${SWARM_API}/health`);
    
    return {
      ok: orchestratorHealth.ok && swarmHealth.ok,
      orchestrator: orchestratorHealth.ok ? await orchestratorHealth.json() : null,
      swarm: swarmHealth.ok ? await swarmHealth.json() : null,
    };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

export async function getSystemMetrics(): Promise<SystemMetrics> {
  return apiCall(`${SWARM_API}/metrics`);
}

export async function getAlerts() {
  return apiCall(`${SWARM_API}/alerts`);
}

// Neural Mesh API
export async function getNeuralMeshStatus() {
  return apiCall(`${NEURAL_MESH_API}/status`);
}

export async function getNeuralMeshMemory() {
  return apiCall(`${NEURAL_MESH_API}/memory`);
}

export async function queryNeuralMesh(query: string) {
  return apiCall(`${NEURAL_MESH_API}/query`, {
    method: 'POST',
    body: JSON.stringify({ query }),
  });
}

// Quantum Scheduler API
export async function getQuantumStatus() {
  return apiCall(`${QUANTUM_API}/status`);
}

export async function getQuantumTasks() {
  return apiCall(`${QUANTUM_API}/tasks`);
}

export async function createQuantumTask(task: {
  description: string;
  targetAgentCount: number;
  coherenceLevel: 'LOW' | 'MEDIUM' | 'HIGH';
}) {
  return apiCall(`${QUANTUM_API}/tasks`, {
    method: 'POST',
    body: JSON.stringify(task),
  });
}

// Configuration Management
export async function getConfiguration() {
  return apiCall(`${ORCHESTRATOR_API}/config`);
}

export async function updateConfiguration(config: Record<string, any>) {
  return apiCall(`${ORCHESTRATOR_API}/config`, {
    method: 'PUT',
    body: JSON.stringify(config),
  });
}

// WebSocket Management
export function createWebSocketConnection(url: string, callbacks: {
  onOpen?: () => void;
  onMessage?: (data: any) => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
}) {
  const ws = new WebSocket(url);
  
  ws.onopen = callbacks.onOpen || (() => {});
  ws.onclose = callbacks.onClose || (() => {});
  ws.onerror = callbacks.onError || (() => {});
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      callbacks.onMessage?.(data);
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  };
  
  return ws;
}

// Enterprise Management
export async function getOrganizations() {
  return apiCall(`${ENTERPRISE_API}/organizations`);
}

export async function getConnections() {
  return apiCall(`${ENTERPRISE_API}/connections`);
}

export async function getDataFlows(organizationId?: string) {
  const url = organizationId 
    ? `${ENTERPRISE_API}/data-flows?organization_id=${organizationId}`
    : `${ENTERPRISE_API}/data-flows`;
  return apiCall(url);
}

// Real-time updates
export function connectToRealTimeUpdates() {
  return createWebSocketConnection(`${REALTIME_API.replace('http', 'ws')}/ws`, {
    onOpen: () => console.log('Connected to real-time updates'),
    onMessage: (data) => console.log('Real-time update:', data),
    onClose: () => console.log('Disconnected from real-time updates'),
    onError: (error) => console.error('WebSocket error:', error)
  });
}

// System health with all components
export async function getSystemHealth() {
  try {
    const [mainHealth, chatHealth, jobHealth, realtimeHealth] = await Promise.all([
      fetch(`${API_BASE}/health`),
      fetch(`${API_BASE}/v1/chat/health`),
      fetch(`${SWARM_API}/health`),
      fetch(`${REALTIME_API}/health`)
    ]);
    
    return {
      main: mainHealth.ok ? await mainHealth.json() : null,
      chat: chatHealth.ok ? await chatHealth.json() : null,
      jobs: jobHealth.ok ? await jobHealth.json() : null,
      realtime: realtimeHealth.ok ? await realtimeHealth.json() : null,
      overall: mainHealth.ok && chatHealth.ok && jobHealth.ok && realtimeHealth.ok
    };
  } catch (error) {
    return { overall: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

// Batch operations
export async function batchOperation(operations: Array<{
  method: string;
  url: string;
  data?: any;
}>) {
  return apiCall(`${ORCHESTRATOR_API}/batch`, {
    method: 'POST',
    body: JSON.stringify({ operations }),
  });
}
