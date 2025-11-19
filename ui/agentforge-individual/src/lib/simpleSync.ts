/**
 * Simple Sync - SSR-Safe version without WebSocket
 * Handles data synchronization without browser-only APIs
 */

export class SimpleSync {
  private userId: string;
  private sessionId: string;
  private apiBase = 'http://localhost:8000/api';

  constructor() {
    this.userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Initialize only when called from client-side
  public initialize() {
    if (typeof window === 'undefined') {
      console.log('SimpleSync: SSR environment, skipping initialization');
      return;
    }
    
    console.log('SimpleSync initialized for user:', this.userId);
    this.syncInitialData();
    this.startPeriodicSync();
  }

  private async syncInitialData() {
    if (typeof window === 'undefined') return;
    
    try {
      await this.syncToAPI('user_session_start', {
        userId: this.userId,
        sessionId: this.sessionId,
        startTime: Date.now(),
        userAgent: navigator?.userAgent || 'Unknown',
        url: window?.location?.href || 'http://localhost:3002'
      });
    } catch (error) {
      console.log('Initial sync failed:', error);
    }
  }

  private startPeriodicSync() {
    if (typeof window === 'undefined') return;
    
    // Send heartbeat every 30 seconds
    setInterval(() => {
      this.syncToAPI('heartbeat', {
        userId: this.userId,
        sessionId: this.sessionId,
        timestamp: Date.now(),
        userAgent: navigator?.userAgent || 'Unknown',
        url: window?.location?.href || 'http://localhost:3002'
      }).catch(error => {
        console.log('Heartbeat sync failed:', error);
      });
    }, 30000);
  }

  public async syncToAPI(dataType: string, data: any) {
    if (typeof window === 'undefined') return;
    
    try {
      const payload = {
        type: 'user_data_sync',
        dataType,
        data,
        timestamp: Date.now(),
        source: 'user_interface',
        userId: this.userId,
        sessionId: this.sessionId
      };

      const response = await fetch(`${this.apiBase}/sync/${dataType}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        console.log(`Sync failed for ${dataType}:`, response.status);
      }
    } catch (error) {
      console.log(`Sync error for ${dataType}:`, error);
    }
  }

  public syncCurrentState() {
    if (typeof window === 'undefined') return;
    
    // This would sync current application state
    // For now, just send a heartbeat
    this.syncToAPI('heartbeat', {
      userId: this.userId,
      sessionId: this.sessionId,
      timestamp: Date.now()
    }).catch(error => {
      console.log('State sync failed:', error);
    });
  }

  public disconnect() {
    if (typeof window === 'undefined') return;
    
    this.syncToAPI('user_session_end', {
      userId: this.userId,
      sessionId: this.sessionId,
      endTime: Date.now()
    }).catch(error => {
      console.log('Disconnect sync failed:', error);
    });
  }
}

// Create singleton instance
export const simpleSync = new SimpleSync();
