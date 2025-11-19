import { store } from './store';

export interface SyncPayload {
  type: string;
  dataType: string;
  data: any;
  timestamp: number;
  source: 'user_interface';
  userId?: string;
  sessionId?: string;
}

export class AdminDataSync {
  private adminWs: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private adminApiBase = 'http://localhost:8000/api';
  private adminWsUrl = 'ws://localhost:8000/v1/realtime/ws';
  private userId: string;
  private sessionId: string;
  private isConnected = false;
  private initialized = false;

  constructor() {
    // Generate unique user and session IDs
    this.userId = this.generateUserId();
    this.sessionId = this.generateSessionId();
  }

  // Initialize only when called from client-side
  public initialize() {
    if (this.initialized || typeof window === 'undefined') {
      return;
    }
    
    this.initialized = true;
    this.connectToAdmin();
    this.setupDataWatchers();
    this.startHeartbeat();
  }

  private generateUserId(): string {
    // In production, this would come from authentication
    return `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private connectToAdmin() {
    // Only connect in browser environment
    if (typeof WebSocket === 'undefined') {
      console.log('WebSocket not available (SSR environment)');
      return;
    }
    
    try {
      this.adminWs = new WebSocket(this.adminWsUrl);
      
      this.adminWs.onopen = () => {
        console.log('Connected to admin interface');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.syncInitialData();
        this.sendHeartbeat();
      };

      this.adminWs.onclose = () => {
        console.log('Admin connection closed');
        this.isConnected = false;
        this.reconnectToAdmin();
      };

      this.adminWs.onerror = (error) => {
        console.error('Admin WebSocket error:', error);
        this.isConnected = false;
      };

    } catch (error) {
      console.error('Failed to connect to admin interface:', error);
      this.reconnectToAdmin();
    }
  }

  private reconnectToAdmin() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Reconnecting to admin interface (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connectToAdmin();
      }, Math.pow(2, this.reconnectAttempts) * 1000);
    }
  }

  private startHeartbeat() {
    setInterval(() => {
      if (this.isConnected) {
        this.sendHeartbeat();
      }
    }, 30000); // Send heartbeat every 30 seconds
  }

  private sendHeartbeat() {
    // Check if we're in browser environment to avoid SSR errors
    const userAgent = typeof navigator !== 'undefined' ? navigator.userAgent : 'Server';
    const url = typeof window !== 'undefined' ? window.location.href : 'http://localhost:3002';
    
    this.syncToAdmin('heartbeat', {
      userId: this.userId,
      sessionId: this.sessionId,
      timestamp: Date.now(),
      userAgent,
      url
    });
  }

  // Main sync method
  public syncToAdmin(dataType: string, data: any) {
    const payload: SyncPayload = {
      type: 'user_data_sync',
      dataType,
      data,
      timestamp: Date.now(),
      source: 'user_interface',
      userId: this.userId,
      sessionId: this.sessionId
    };

    // Send via WebSocket (primary) - only in browser
    if (typeof WebSocket !== 'undefined' && this.adminWs) {
      try {
        if (this.adminWs.readyState === WebSocket.OPEN) {
          this.adminWs.send(JSON.stringify(payload));
        }
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
      }
    }

    // Send via HTTP API (fallback)
    this.syncViaAPI(dataType, data);
  }

  private async syncViaAPI(dataType: string, data: any) {
    try {
      await fetch(`${this.adminApiBase}/sync/${dataType}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data,
          timestamp: Date.now(),
          source: 'user_interface',
          userId: this.userId,
          sessionId: this.sessionId
        })
      });
    } catch (error) {
      console.error('Failed to sync via API:', error);
    }
  }

  private async syncInitialData() {
    // Sync all existing data when connection is established
    this.syncToAdmin('user_session_start', {
      userId: this.userId,
      sessionId: this.sessionId,
      startTime: Date.now(),
      userAgent: navigator.userAgent,
      url: window.location.href
    });
    
    this.syncToAdmin('jobs', {
      active: store.activeJobs,
      archived: store.archivedJobs
    });
    
    this.syncToAdmin('conversations', store.jobConversations);
    this.syncToAdmin('dataSources', store.dataSources);
    this.syncToAdmin('swarmActivity', store.swarmActivity);
    this.syncToAdmin('messages', store.messages);
  }

  private setupDataWatchers() {
    // Watch for changes in user interface data and sync them
    
    // Watch job changes
    const originalCreateNewJob = store.createNewJob;
    store.createNewJob = (userMessage: string) => {
      const jobId = originalCreateNewJob.call(store, userMessage);
      const newJob = store.activeJobs.find(j => j.id === jobId);
      
      this.syncToAdmin('job_created', {
        jobId,
        userMessage,
        job: newJob,
        timestamp: Date.now()
      });
      return jobId;
    };

    // Watch job status changes
    const originalPauseJob = store.pauseJob;
    store.pauseJob = (jobId: string) => {
      originalPauseJob.call(store, jobId);
      const job = store.activeJobs.find(j => j.id === jobId);
      this.syncToAdmin('job_paused', { 
        jobId, 
        job,
        timestamp: Date.now() 
      });
    };

    const originalResumeJob = store.resumeJob;
    store.resumeJob = (jobId: string) => {
      originalResumeJob.call(store, jobId);
      const job = store.activeJobs.find(j => j.id === jobId);
      this.syncToAdmin('job_resumed', { 
        jobId, 
        job,
        timestamp: Date.now() 
      });
    };

    const originalArchiveJob = store.archiveJob;
    store.archiveJob = (jobId: string) => {
      originalArchiveJob.call(store, jobId);
      const job = store.archivedJobs.find(j => j.id === jobId);
      this.syncToAdmin('job_archived', { 
        jobId, 
        job,
        timestamp: Date.now() 
      });
    };

    // Watch message changes
    const originalSendMessage = store.sendMessage;
    store.sendMessage = (content: string) => {
      originalSendMessage.call(store, content);
      this.syncToAdmin('message_sent', {
        content,
        currentJobId: store.currentJobId,
        timestamp: Date.now()
      });
    };

    // Watch data source changes
    const originalAddDataSource = store.addDataSource;
    store.addDataSource = (source: any) => {
      originalAddDataSource.call(store, source);
      this.syncToAdmin('data_source_added', {
        source,
        timestamp: Date.now()
      });
    };

    const originalRemoveDataSource = store.removeDataSource;
    store.removeDataSource = (id: string) => {
      originalRemoveDataSource.call(store, id);
      this.syncToAdmin('data_source_removed', {
        sourceId: id,
        timestamp: Date.now()
      });
    };

    // Watch theme changes
    const originalToggleTheme = store.toggleTheme;
    store.toggleTheme = () => {
      originalToggleTheme.call(store);
      this.syncToAdmin('theme_changed', {
        theme: store.theme,
        timestamp: Date.now()
      });
    };
  }

  // Public methods for manual sync
  public syncCurrentState() {
    this.syncToAdmin('current_state', {
      activeJobs: store.activeJobs,
      archivedJobs: store.archivedJobs,
      dataSources: store.dataSources,
      messages: store.messages,
      swarmActivity: store.swarmActivity,
      currentJobId: store.currentJobId,
      theme: store.theme
    });
  }

  public syncJobProgress(jobId: string, progress: number) {
    this.syncToAdmin('job_progress', {
      jobId,
      progress,
      timestamp: Date.now()
    });
  }

  public syncSwarmActivity(activity: any) {
    this.syncToAdmin('swarm_activity', {
      activity,
      timestamp: Date.now()
    });
  }

  public disconnect() {
    if (this.adminWs) {
      this.adminWs.close();
    }
    this.syncToAdmin('user_session_end', {
      userId: this.userId,
      sessionId: this.sessionId,
      endTime: Date.now()
    });
  }
}

// Initialize the sync service
export const adminSync = new AdminDataSync();

// Cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    adminSync.disconnect();
  });
}
