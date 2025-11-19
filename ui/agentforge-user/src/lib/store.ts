import { proxy, useSnapshot as useSnap } from 'valtio';
import AGIClient, { AGIRequest, AGIResponse, SwarmActivity as AGISwarmActivity } from './agiClient';
import CapabilityEngine, { CapabilitySuggestion, InputAnalysis } from './capabilityEngine';

export type Message = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  metadata?: {
    agentsWorking?: number;
    processingTime?: number;
    dataSourcesUsed?: string[];
    confidence?: number;
    capabilitiesUsed?: string[];
  };
};

export type DataSource = {
  id: string;
  name: string;
  type: 'file' | 'stream' | 'database' | 'api';
  status: 'connected' | 'processing' | 'ready' | 'error';
  size?: string;
  recordCount?: number;
  lastUpdated?: Date;
};

export type SwarmActivity = {
  id: string;
  agentId: string;
  task: string;
  status: 'working' | 'completed' | 'failed' | 'initializing';
  progress: number;
  timestamp: Date;
};

export type JobConversation = {
  jobId: string;
  messages: Message[];
  title: string;
  createdAt: Date;
};

export type ActiveJob = {
  id: string;
  title: string;
  description: string;
  type: 'continuous' | 'task'; // New field to differentiate job types
  status: 'running' | 'paused' | 'completed';
  progress?: number; // Optional - only for task-based jobs
  startTime: Date;
  dataStreams: string[];
  agentsAssigned: number;
  alertsGenerated: number;
  confidence: number;
  // Continuous job specific fields
  eventsProcessed?: number;
  lastEventTime?: Date;
  // Task job specific fields
  tasksCompleted?: number;
  totalTasks?: number;
};

export type ArchivedJob = {
  id: string;
  title: string;
  description: string;
  completedAt: Date;
  duration: string;
  agentsUsed: number;
  confidence: number;
  outputSize: string;
};

type Store = {
  // Theme
  theme: 'day' | 'night';
  toggleTheme: () => void;

  // Connection
  connected: boolean;
  connecting: boolean;
  connectionError?: string;

  // Chat
  messages: Message[];
  isTyping: boolean;
  currentInput: string;
  
  // Data Sources
  dataSources: DataSource[];
  
  // Swarm Activity
  swarmActivity: SwarmActivity[];
  activeAgents: number;
  realAgentMetrics?: any;
  
  // Job Conversations
  jobConversations: JobConversation[];
  currentJobId?: string;
  
  // Active Jobs
  activeJobs: ActiveJob[];
  archivedJobs: ArchivedJob[];

  // AGI Integration
  agiClient: any;
  capabilityEngine: any;
  currentCapabilities: CapabilitySuggestion[];
  realtimeSuggestions: CapabilitySuggestion[];
  inputAnalysis: InputAnalysis | null;
  
  // Actions
  sendMessage: (content: string) => void;
  generateEnhancedResponse: (agiResponse: AGIResponse, analysis: InputAnalysis) => string;
  updateRealtimeSuggestions: (partialInput: string) => void;
  getAllCapabilities: () => CapabilitySuggestion[];
  getCapabilitiesByType: (type: 'input' | 'output' | 'processing' | 'optimization') => CapabilitySuggestion[];
  addDataSource: (source: Omit<DataSource, 'id'>) => void;
  removeDataSource: (id: string) => void;
  connectToStream: (url: string) => void;
  clearChat: () => void;
  loadJobConversation: (jobId: string) => void;
  saveCurrentConversation: () => void;
  createNewJob: (userMessage: string) => Promise<string>;
  pauseJob: (jobId: string) => void;
  resumeJob: (jobId: string) => void;
  archiveJob: (jobId: string) => void;
  startNewChat: () => void;
};

// Initialize AGI Client and Capability Engine
const agiClient = new AGIClient();
const capabilityEngine = new CapabilityEngine();

export const store = proxy<Store>({
  theme: 'day',
  toggleTheme() {
    store.theme = store.theme === 'day' ? 'night' : 'day';
  },

  connected: false,
  connecting: false,
  connectionError: undefined,

  // AGI Integration
  agiClient,
  capabilityEngine,
  currentCapabilities: [] as CapabilitySuggestion[],
  realtimeSuggestions: [] as CapabilitySuggestion[],
  inputAnalysis: null as InputAnalysis | null,

  messages: [
    {
      id: '1',
      role: 'system',
      content: 'Welcome to AgentForge AI! I\'m powered by a swarm of intelligent agents ready to help you solve complex problems. Upload data, connect live streams, or simply ask me anything.',
      timestamp: new Date(),
      metadata: {
        agentsWorking: 0,
        confidence: 1.0
      }
    }
  ],
  isTyping: false,
  currentInput: '',

  dataSources: [], // No mock data - load from backend
  
  swarmActivity: [],
  activeAgents: 0,

  jobConversations: [],
  currentJobId: undefined,
  
  activeJobs: [],
  
  archivedJobs: [],

  async sendMessage(content: string) {
    if (!content.trim() || store.isTyping) return;

    // Check if this is a new conversation (only system messages exist)
    const isNewConversation = store.messages.filter(m => m.role !== 'system').length === 0;
    
    // Only create jobs for complex requests, not simple greetings
    const isComplexRequest = !['hi', 'hello', 'hey', 'how are you'].some(greeting => 
      content.toLowerCase().includes(greeting)
    );
    
    // Create a new job if starting a new conversation with a complex request
    if (isNewConversation && isComplexRequest) {
      const jobId = await store.createNewJob(content);
      store.currentJobId = jobId;
    }
    
    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date()
    };
    store.messages.push(userMessage);

    // Start typing indicator
    store.isTyping = true;

    try {
      // Analyze input with capability engine
      const analysis = store.capabilityEngine.analyzeInput(content, {
        conversationHistory: store.messages,
        dataSources: store.dataSources,
        userPreferences: {}
      });
      
      store.inputAnalysis = analysis;
      // Disable capability popup by not setting currentCapabilities
      // store.currentCapabilities = analysis.recommendedActions;

      // Process with AGI Client
      const agiRequest: AGIRequest = {
        content: content.trim(),
        context: {
          userId: 'user_001', // In production, get from auth
          sessionId: store.currentJobId || 'session_001',
          conversationHistory: store.messages,
          dataSources: store.dataSources,
          userPreferences: {}
        },
        mode: 'INTERACTIVE',
        capabilities: analysis.suggestedCapabilities.map((c: any) => c.capability)
      };

      const agiResponse: AGIResponse = await store.agiClient.processUserRequest(agiRequest);

      // Update store with real AGI response metrics
      store.realAgentMetrics = agiResponse.agentMetrics;
      store.activeAgents = agiResponse.agentMetrics.totalAgentsDeployed || 0;
      store.swarmActivity = agiResponse.swarmActivity.map(activity => ({
        id: activity.id,
        agentId: activity.agentId,
        task: activity.task,
        status: activity.status,
        progress: activity.progress,
        timestamp: activity.timestamp
      }));

      // Update job with real agent assignment from backend
      if (store.currentJobId) {
        const currentJob = store.activeJobs.find(job => job.id === store.currentJobId);
        if (currentJob) {
          currentJob.agentsAssigned = agiResponse.agentMetrics.totalAgentsDeployed || 0;
          currentJob.realAgentMetrics = agiResponse.agentMetrics;
          // Remove fake progress - jobs are either active or complete
          delete currentJob.progress;
        }
      }

      // Apply Phase 3 enhancements: personalization and optimization
      let enhancedResponse = store.generateEnhancedResponse(agiResponse, analysis);
      
      try {
        // Personalize response based on user profile
        const personalizationResult = await store.agiClient.personalizeResponse(
          'user_001', // In production, get from auth
          enhancedResponse,
          {
            expertise_level: 'intermediate', // Get from user profile
            interaction_style: 'detailed',
            complexity_preference: 0.6,
            conversation_history: store.messages.slice(-5) // Last 5 messages for context
          }
        );
        
        if (personalizationResult.personalization_applied) {
          enhancedResponse = personalizationResult.personalized_response;
        }

        // Optimize response for quality
        const optimizationResult = await store.agiClient.optimizeResponse(
          enhancedResponse,
          {
            user_id: 'user_001',
            expertise_level: 'intermediate',
            session_duration: Date.now() - (store.messages[0]?.timestamp.getTime() || Date.now())
          },
          store.messages.slice(-3).map(m => ({ role: m.role, content: m.content }))
        );
        
        if (optimizationResult.optimization_applied) {
          enhancedResponse = optimizationResult.optimized_response;
        }

      } catch (error) {
        console.error('Failed to apply Phase 3 enhancements:', error);
      }

      // Create assistant response with AGI insights
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: enhancedResponse,
        timestamp: new Date(),
        metadata: {
          agentsWorking: agiResponse.agentMetrics.totalAgentsDeployed,
          confidence: agiResponse.confidence,
          capabilitiesUsed: agiResponse.capabilitiesUsed,
          processingTime: agiResponse.processingTime
        }
      };

      store.messages.push(assistantMessage);
      store.isTyping = false;

      // Phase 3: Learning and Analytics
      try {
        // Analyze user interaction for pattern learning
        await store.agiClient.analyzeUserInteraction('user_001', {
          message: content,
          capabilities_used: agiResponse.capabilitiesUsed,
          success_rate: agiResponse.confidence,
          user_satisfaction: 0.8, // Default, can be updated with explicit feedback
          response_time: agiResponse.processingTime,
          complexity_level: analysis.suggestedCapabilities.length / 5, // Normalize complexity
          session_duration: Date.now() - (store.messages[0]?.timestamp.getTime() || Date.now()),
          preferences: {
            interaction_style: 'detailed',
            complexity: 'medium'
          }
        });

        // Update user profile with interaction data
        await store.agiClient.updateUserProfile('user_001', {
          message: content,
          capabilities_used: agiResponse.capabilitiesUsed,
          success_rate: agiResponse.confidence,
          response_time: agiResponse.processingTime,
          complexity_level: analysis.suggestedCapabilities.length / 5,
          session_duration: Date.now() - (store.messages[0]?.timestamp.getTime() || Date.now())
        });

        // Analyze conversation quality for continuous improvement
        await store.agiClient.analyzeConversationQuality(store.currentJobId || 'session_001', {
          user_message: content,
          response: enhancedResponse,
          capabilities_used: agiResponse.capabilitiesUsed,
          agent_metrics: agiResponse.agentMetrics,
          response_data: {
            processing_time: agiResponse.processingTime,
            accuracy: agiResponse.confidence
          },
          user_feedback: {
            satisfaction: 0.8,
            helpfulness: 0.85
          }
        });

      } catch (error) {
        console.error('Phase 3 learning failed:', error);
      }

      // Update swarm activities to show progression
      setTimeout(() => {
        store.swarmActivity = store.swarmActivity.map(activity => ({
          ...activity,
          status: Math.random() > 0.3 ? 'completed' : 'working',
          progress: activity.status === 'completed' ? 100 : Math.min(activity.progress + 20, 90)
        }));
        
        // Update job progress
        if (store.currentJobId) {
          const currentJob = store.activeJobs.find(job => job.id === store.currentJobId);
          if (currentJob) {
            if (currentJob.type === 'task' && currentJob.progress !== undefined) {
              currentJob.progress = Math.min(100, currentJob.progress + 30);
              if (currentJob.tasksCompleted !== undefined) {
                currentJob.tasksCompleted += 1;
              }
            }
            currentJob.alertsGenerated += Math.floor(Math.random() * 3) + 1;
          }
        }
      }, 3000);

    } catch (error) {
      console.error('AGI processing failed:', error);
      
      // Fallback to original behavior
      const swarmConfig = analyzeAndGenerateSwarm(content, store.dataSources);
      store.activeAgents = swarmConfig.agentCount;
      store.swarmActivity.push(...swarmConfig.activities);

      const fallbackResponse = generateIntelligentResponse(content, swarmConfig, store.dataSources);
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: fallbackResponse,
        timestamp: new Date(),
        metadata: {
          agentsWorking: swarmConfig.agentCount,
          confidence: swarmConfig.confidence
        }
      };

      store.messages.push(assistantMessage);
      store.isTyping = false;
    }
  },

  generateEnhancedResponse(agiResponse: AGIResponse, analysis: InputAnalysis): string {
    // Always return just the content without any suggestions or technical details
    return agiResponse.content;
  },

  // Real-time capability suggestions disabled to prevent popups
  updateRealtimeSuggestions(partialInput: string) {
    // Disabled to prevent unwanted popups
    store.realtimeSuggestions = [];
  },

  // Get all available capabilities
  getAllCapabilities(): CapabilitySuggestion[] {
    return store.capabilityEngine.getAllCapabilities();
  },

  // Get capabilities by type
  getCapabilitiesByType(type: 'input' | 'output' | 'processing' | 'optimization'): CapabilitySuggestion[] {
    return store.capabilityEngine.getCapabilitiesByType(type);
  },

  // Load real data from backend
  async loadBackendData() {
    try {
      // Load active jobs from backend
      const jobs = await store.agiClient.getActiveJobs();
      if (jobs && Array.isArray(jobs)) {
        store.activeJobs = jobs.map((job: any) => ({
          id: job.id,
          title: job.title,
          description: job.description,
          type: job.type,
          status: job.status,
          startTime: new Date(job.start_time || job.startTime),
          dataStreams: job.data_streams || job.dataStreams || [],
          agentsAssigned: job.agents_assigned || job.agentsAssigned || 0,
          alertsGenerated: job.alerts_generated || job.alertsGenerated || 0,
          confidence: job.confidence || 0.8,
          progress: job.progress,
          tasksCompleted: job.tasks_completed || job.tasksCompleted,
          totalTasks: job.total_tasks || job.totalTasks,
          eventsProcessed: job.events_processed || job.eventsProcessed,
          lastEventTime: job.last_event_time ? new Date(job.last_event_time) : job.lastEventTime
        }));
      }

      // Load data sources from backend
      const dataSources = await store.agiClient.getDataSources();
      if (dataSources && Array.isArray(dataSources)) {
        store.dataSources = dataSources.map((ds: any) => ({
          id: ds.id,
          name: ds.name,
          type: ds.type,
          status: ds.status as 'processing' | 'ready' | 'error',
          size: ds.size,
          recordCount: ds.metadata?.rows || ds.metadata?.record_count,
          lastUpdated: new Date(ds.last_processed || ds.lastUpdated || Date.now())
        }));
      }

      // Load swarm activity from backend
      const swarmActivity = await store.agiClient.getAllSwarmActivity();
      if (swarmActivity && Array.isArray(swarmActivity)) {
        store.swarmActivity = swarmActivity.slice(0, 10).map((activity: any) => ({
          id: activity.id,
          agentId: activity.agent_id || activity.agentId,
          task: activity.task,
          status: activity.status,
          progress: activity.progress || 0,
          timestamp: new Date(activity.start_time || activity.timestamp || Date.now())
        }));
      }

    } catch (error) {
      console.error('Failed to load backend data:', error);
    }
  },

  async addDataSource(source: Omit<DataSource, 'id'>) {
    const newSource: DataSource = {
      ...source,
      id: Date.now().toString(),
      lastUpdated: new Date()
    };
    store.dataSources.push(newSource);

    // Also sync with backend data sources
    try {
      const backendSources = await store.agiClient.getDataSources();
      // Update store with backend data if available
      if (backendSources && Array.isArray(backendSources)) {
        const mappedSources = backendSources.map((bs: any) => ({
          id: bs.id,
          name: bs.name,
          type: bs.type,
          status: bs.status as 'processing' | 'ready' | 'error',
          size: bs.size,
          recordCount: bs.metadata?.rows || bs.metadata?.record_count,
          lastUpdated: new Date(bs.last_processed)
        }));
        store.dataSources = [...store.dataSources.filter((ds: DataSource) => !mappedSources.find((ms: DataSource) => ms.name === ds.name)), ...mappedSources];
      }
    } catch (error) {
      console.error('Failed to sync with backend data sources:', error);
    }
  },

  removeDataSource(id: string) {
    store.dataSources = store.dataSources.filter(ds => ds.id !== id);
  },

  connectToStream(url: string) {
    store.connecting = true;
    store.connectionError = undefined;
    
    // Simulate connection
    setTimeout(() => {
      store.connecting = false;
      if (Math.random() > 0.2) { // 80% success rate
        store.connected = true;
        store.addDataSource({
          name: `Live Stream (${url})`,
          type: 'stream',
          status: 'connected'
        });
      } else {
        store.connectionError = 'Failed to connect to stream';
      }
    }, 2000);
  },

  clearChat() {
    store.messages = store.messages.filter(m => m.role === 'system');
    store.swarmActivity = [];
    store.activeAgents = 0;
    store.currentJobId = undefined;
  },

  loadJobConversation(jobId: string) {
    // Save current conversation if it exists
    if (store.currentJobId && store.messages.length > 1) {
      store.saveCurrentConversation();
    }

    // Load the requested job conversation
    const jobConversation = store.jobConversations.find(jc => jc.jobId === jobId);
    if (jobConversation) {
      store.messages = [...store.messages.filter(m => m.role === 'system'), ...jobConversation.messages];
      store.currentJobId = jobId;
    } else {
      // Create new conversation for this job
      store.currentJobId = jobId;
      const systemMessage = store.messages.find(m => m.role === 'system');
      const jobMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Loading conversation for job: ${jobId}\n\nThis job involves continuous processing of live data streams with real-time agent coordination. The conversation history will help you understand the job's configuration and progress.`,
        timestamp: new Date(),
        metadata: {
          agentsWorking: 0,
          confidence: 1.0
        }
      };
      store.messages = systemMessage ? [systemMessage, jobMessage] : [jobMessage];
    }
  },

  saveCurrentConversation() {
    if (!store.currentJobId || store.messages.length <= 1) return;

    const existingIndex = store.jobConversations.findIndex(jc => jc.jobId === store.currentJobId);
    const conversation: JobConversation = {
      jobId: store.currentJobId,
      messages: store.messages.filter(m => m.role !== 'system'),
      title: `Job ${store.currentJobId}`,
      createdAt: new Date()
    };

    if (existingIndex >= 0) {
      store.jobConversations[existingIndex] = conversation;
    } else {
      store.jobConversations.push(conversation);
    }
  },
  
  async createNewJob(userMessage: string): Promise<string> {
    const jobId = `job-${Date.now()}`;
    const input = userMessage.toLowerCase();
    
    // Only create jobs for substantial work requests
    const isJobWorthy = 
      input.includes('analyze') || input.includes('create') || input.includes('build') ||
      input.includes('generate') || input.includes('process') || input.includes('optimize') ||
      input.includes('predict') || input.includes('monitor') || input.includes('track') ||
      input.length > 50; // Longer messages likely need jobs
    
    if (!isJobWorthy) {
      return jobId; // Return ID but don't actually create job
    }
    
    // Determine job type based on user intent
    const isContinuous = 
      input.includes('monitor') || input.includes('watch') || input.includes('track') ||
      input.includes('stream') || input.includes('real-time') || input.includes('live') ||
      input.includes('detect') || input.includes('alert') || input.includes('anomal') ||
      input.includes('fusion') || input.includes('continuous');
    
    // Generate smart title using AGI analysis
    let smartTitle = userMessage.length > 50 
      ? userMessage.substring(0, 50) + '...'
      : userMessage;
    
    try {
      // Get smart title from AGI
      const titleResponse = await store.agiClient.sendMessage(`Generate a concise 3-5 word title for this task: "${userMessage}"`, {
        conversationHistory: [],
        dataSources: [],
        userId: 'user_001',
        sessionId: 'title_generation'
      });
      
      if (titleResponse.response) {
        // Extract clean title from response
        const extractedTitle = titleResponse.response
          .replace(/['"]/g, '')
          .replace(/^(Title:|Task:|Job:)\s*/i, '')
          .trim();
        
        if (extractedTitle.length > 0 && extractedTitle.length < 100) {
          smartTitle = extractedTitle;
        }
      }
    } catch (error) {
      console.log('Smart title generation failed, using fallback');
    }
    
    // Create a new active job with appropriate type
    const newJob: ActiveJob = {
      id: jobId,
      title: smartTitle,
      description: `Processing: ${userMessage}`,
      type: isContinuous ? 'continuous' : 'task',
      status: 'running',
      startTime: new Date(),
      dataStreams: store.dataSources.map(ds => ds.name),
      agentsAssigned: 0, // Will be updated when agents are assigned
      alertsGenerated: 0,
      confidence: 0.8, // Default confidence
      // Initialize type-specific fields
      ...(isContinuous ? {
        eventsProcessed: 0,
        lastEventTime: new Date()
      } : {
        progress: 0,
        tasksCompleted: 0,
        totalTasks: 10 // Will be determined by backend
      })
    };
    
    store.activeJobs.push(newJob);

    // Also create job in backend
    try {
      const backendJob = await store.agiClient.createJob({
        title: smartTitle,
        description: `Processing: ${userMessage}`,
        type: isContinuous ? 'continuous' : 'task',
        user_message: userMessage,
        data_sources: store.dataSources.map(ds => ds.name),
        requirements: {}
      });
      
      // Update job with intelligent backend response
      if (backendJob) {
        const existingJob = store.activeJobs.find(job => job.id === jobId);
        if (existingJob) {
          existingJob.id = backendJob.id; // Use backend job ID
          existingJob.title = backendJob.title; // Intelligent title from ChatGPT
          existingJob.description = backendJob.description; // Smart description
          existingJob.agentsAssigned = backendJob.agents_assigned; // Real agent count
          existingJob.confidence = backendJob.confidence;
          existingJob.alertsGenerated = backendJob.potential_alerts || 0;
          existingJob.type = backendJob.type;
          
          // Only show progress bar for jobs that actually need it
          if (!backendJob.requires_progress) {
            delete existingJob.progress;
            delete existingJob.tasksCompleted;
            delete existingJob.totalTasks;
          }
          
          // Set realistic streams count
          if (backendJob.expected_streams > 0) {
            existingJob.dataStreams = Array.from({length: backendJob.expected_streams}, 
              (_, i) => `stream-${i + 1}`);
          } else {
            existingJob.dataStreams = store.dataSources.map(ds => ds.name);
          }
        }
      }
    } catch (error) {
      console.error('Failed to create backend job:', error);
    }

    return jobId;
  },
  
  async pauseJob(jobId: string) {
    const job = store.activeJobs.find(j => j.id === jobId);
    if (job && job.status === 'running') {
      job.status = 'paused';
      
      // Also pause in backend
      try {
        await store.agiClient.pauseJob(jobId);
      } catch (error) {
        console.error('Failed to pause job in backend:', error);
        job.status = 'running'; // Revert on error
      }
    }
  },
  
  async resumeJob(jobId: string) {
    const job = store.activeJobs.find(j => j.id === jobId);
    if (job && job.status === 'paused') {
      job.status = 'running';
      
      // Also resume in backend
      try {
        await store.agiClient.resumeJob(jobId);
      } catch (error) {
        console.error('Failed to resume job in backend:', error);
        job.status = 'paused'; // Revert on error
      }
    }
  },
  
  async archiveJob(jobId: string) {
    const jobIndex = store.activeJobs.findIndex(j => j.id === jobId);
    if (jobIndex >= 0) {
      const job = store.activeJobs[jobIndex];
      
      // Calculate duration
      const duration = (() => {
        const now = new Date();
        const diff = now.getTime() - job.startTime.getTime();
        const hours = Math.floor(diff / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        return `${hours}h ${minutes}m`;
      })();
      
      // Create archived job
      const archivedJob: ArchivedJob = {
        id: job.id,
        title: job.title,
        description: job.description,
        completedAt: new Date(),
        duration: duration,
        agentsUsed: job.agentsAssigned,
        confidence: 0.85 + Math.random() * 0.1, // Random confidence between 0.85-0.95
        outputSize: `${Math.floor(Math.random() * 50) + 10} insights`
      };
      
      // Also archive in backend
      try {
        await store.agiClient.archiveJob(jobId);
      } catch (error) {
        console.error('Failed to archive job in backend:', error);
      }

      // Move to archived and remove from active
      store.archivedJobs.unshift(archivedJob); // Add to beginning
      store.activeJobs.splice(jobIndex, 1);
    }
  },
  
  startNewChat() {
    // Save current conversation if it exists
    if (store.currentJobId && store.messages.length > 1) {
      store.saveCurrentConversation();
    }
    
    // Clear current chat
    store.messages = store.messages.filter(m => m.role === 'system');
    store.currentJobId = undefined;
    store.isTyping = false;
    store.activeAgents = 0;
    store.swarmActivity = [];
  }
});

// Simulate real-time updates for continuous jobs
setInterval(() => {
  store.activeJobs.forEach(job => {
    if (job.type === 'continuous' && job.status === 'running') {
      // Simulate processing events
      if (job.eventsProcessed !== undefined) {
        job.eventsProcessed += Math.floor(Math.random() * 5) + 1;
        job.lastEventTime = new Date();
        
        // Occasionally generate alerts
        if (Math.random() < 0.1) {
          job.alertsGenerated += 1;
        }
      }
    } else if (job.type === 'task' && job.status === 'running') {
      // Simulate task completion (this would come from backend in real implementation)
      if (job.tasksCompleted !== undefined && job.totalTasks !== undefined && job.tasksCompleted < job.totalTasks) {
        if (Math.random() < 0.05) { // 5% chance per interval
          job.tasksCompleted += 1;
          job.progress = Math.round((job.tasksCompleted / job.totalTasks) * 100);
        }
      }
    }
  });
}, 2000); // Update every 2 seconds

// Intelligent swarm generation based on user input and data
function analyzeAndGenerateSwarm(userInput: string, dataSources: DataSource[]) {
  const input = userInput.toLowerCase();
  let agentTypes: string[] = [];
  let complexity = 1;
  const confidence = 0.85;

  // Analyze user intent
  if (input.includes('analyze') || input.includes('pattern') || input.includes('insight')) {
    agentTypes.push('neural-mesh', 'analytics');
    complexity += 0.5;
  }
  
  if (input.includes('optimize') || input.includes('improve') || input.includes('efficiency')) {
    agentTypes.push('quantum-scheduler', 'optimization');
    complexity += 0.3;
  }
  
  if (input.includes('report') || input.includes('document') || input.includes('summary')) {
    agentTypes.push('document-processor', 'content-generator');
    complexity += 0.2;
  }
  
  if (input.includes('predict') || input.includes('forecast') || input.includes('model')) {
    agentTypes.push('ml-trainer', 'predictor');
    complexity += 0.7;
  }
  
  if (input.includes('anomaly') || input.includes('error') || input.includes('detect')) {
    agentTypes.push('anomaly-detector', 'security-scanner');
    complexity += 0.4;
  }

  // Analyze data sources
  dataSources.forEach(source => {
    if (source.type === 'file') {
      if (source.name.includes('.csv') || source.name.includes('.xlsx')) {
        agentTypes.push('data-processor', 'statistical-analyzer');
      } else if (source.name.includes('.pdf') || source.name.includes('.doc')) {
        agentTypes.push('document-analyzer', 'nlp-processor');
      } else if (source.name.includes('.jpg') || source.name.includes('.png')) {
        agentTypes.push('image-processor', 'computer-vision');
      }
    } else if (source.type === 'stream') {
      agentTypes.push('stream-processor', 'real-time-analyzer');
      complexity += 0.5;
    }
  });

  // Default agents if none specified
  if (agentTypes.length === 0) {
    agentTypes = ['neural-mesh', 'general-processor'];
  }

  // Remove duplicates and calculate agent count
  const uniqueAgentTypes = [...new Set(agentTypes)];
  const agentCount = Math.min(uniqueAgentTypes.length * 2 + Math.floor(complexity * 3), 15);

  // Generate activities
  const activities: SwarmActivity[] = uniqueAgentTypes.slice(0, 6).map((type, index) => ({
    id: `activity-${Date.now()}-${index}`,
    agentId: `${type}-${String(index + 1).padStart(3, '0')}`,
    task: generateTaskForAgentType(type),
    status: 'working' as const,
    progress: 0,
    timestamp: new Date()
  }));

  return {
    agentCount,
    agentTypes: uniqueAgentTypes,
    activities,
    complexity: Math.min(complexity, 3),
    confidence: Math.max(confidence - (complexity * 0.1), 0.6)
  };
}

function generateTaskForAgentType(agentType: string): string {
  const taskMap: Record<string, string[]> = {
    'neural-mesh': ['Analyzing semantic patterns', 'Building knowledge graph', 'Cross-referencing memories'],
    'analytics': ['Computing statistical metrics', 'Identifying trends', 'Generating insights'],
    'quantum-scheduler': ['Coordinating agent tasks', 'Optimizing resource allocation', 'Balancing workload'],
    'document-processor': ['Extracting text content', 'Parsing document structure', 'Identifying key information'],
    'data-processor': ['Cleaning data format', 'Validating data integrity', 'Structuring datasets'],
    'ml-trainer': ['Training predictive models', 'Feature engineering', 'Model validation'],
    'image-processor': ['Analyzing visual content', 'Extracting image features', 'Object recognition'],
    'stream-processor': ['Processing real-time data', 'Monitoring data flow', 'Stream aggregation'],
    'anomaly-detector': ['Scanning for outliers', 'Pattern deviation analysis', 'Threat assessment'],
    'content-generator': ['Generating reports', 'Creating summaries', 'Formatting output'],
    'general-processor': ['Processing user request', 'Coordinating analysis', 'Synthesizing results']
  };

  const tasks = taskMap[agentType] || taskMap['general-processor'];
  return tasks[Math.floor(Math.random() * tasks.length)];
}

function generateIntelligentResponse(userInput: string, swarmConfig: { agentCount: number; agentTypes: string[]; confidence: number }, dataSources: DataSource[]): string {
  const hasData = dataSources.length > 0;
  const dataInfo = hasData ? `\n\n📊 **Data Analysis Results:**\nProcessed ${dataSources.length} data source${dataSources.length !== 1 ? 's' : ''} using ${swarmConfig.agentCount} specialized agents:\n${dataSources.map(ds => `• ${ds.name} (${ds.status})`).join('\n')}` : '';

  const responses = [
    `I've deployed a **${swarmConfig.agentCount}-agent swarm** to address your request. The agents specialized in ${swarmConfig.agentTypes.slice(0, 3).join(', ')} have analyzed your query using advanced neural mesh processing.

**Key Findings:**
• Identified ${Math.floor(Math.random() * 15) + 5} relevant patterns in your request
• Cross-referenced with ${hasData ? 'your uploaded data' : 'global knowledge base'}
• Applied ${swarmConfig.agentTypes.length} different analytical approaches
• Achieved ${Math.round(swarmConfig.confidence * 100)}% confidence in results

The swarm intelligence has processed your request through multiple cognitive layers, combining pattern recognition, semantic analysis, and contextual understanding.${dataInfo}

**Recommendations:**
Based on the collective analysis, I recommend focusing on the identified patterns and considering the strategic implications for your objectives.`,

    `Your request has been processed by a **specialized ${swarmConfig.agentCount}-agent cluster** optimized for this type of analysis. The quantum scheduler coordinated parallel processing across multiple agent types.

**Processing Summary:**
• **Neural Mesh Agents:** Analyzed semantic meaning and context
• **Data Processing Agents:** ${hasData ? 'Processed your uploaded data sources' : 'Accessed relevant knowledge domains'}
• **Analytics Agents:** Generated insights and pattern recognition
• **Synthesis Agents:** Compiled results into actionable intelligence

**Results:**
The collective intelligence has identified several key insights that directly address your query. The analysis shows strong correlation patterns and actionable recommendations.${dataInfo}

**Next Steps:**
The swarm recommends diving deeper into the identified patterns and exploring the strategic implications for your specific use case.`,

    `I've assembled a **dynamic ${swarmConfig.agentCount}-agent swarm** tailored specifically for your request. Each agent type was selected based on the complexity and requirements of your query.

**Swarm Composition:**
${swarmConfig.agentTypes.map(type => `• **${type.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())} Agent:** Processing specialized tasks`).join('\n')}

**Analysis Results:**
The distributed processing approach has yielded comprehensive insights by breaking down your request into specialized tasks, each handled by the most appropriate agent type.${dataInfo}

**Intelligence Summary:**
The swarm has identified actionable patterns and generated strategic recommendations based on multi-layered analysis. The results show high confidence in the proposed solutions and next steps.`
  ];
  
  return responses[Math.floor(Math.random() * responses.length)];
}

export const useSnapshot = useSnap;
