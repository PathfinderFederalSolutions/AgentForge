import { NextApiRequest } from 'next';
import { WebSocketServer } from 'ws';
import { store } from '@/lib/state';

let wss: WebSocketServer | null = null;

export default function handler(req: NextApiRequest, res: any) {
  if (!res.socket.server.wss) {
    console.log('Setting up WebSocket server...');
    
    wss = new WebSocketServer({ 
      port: 3001,
      perMessageDeflate: false 
    });

    wss.on('connection', (ws, request) => {
      console.log('New WebSocket connection established');

      ws.on('message', (message) => {
        try {
          const data = JSON.parse(message.toString());
          console.log('Received WebSocket message:', data);

          // Handle different message types
          switch (data.type) {
            case 'user_data_sync':
              handleUserDataSync(data);
              break;
            
            case 'ping':
              ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
              break;
            
            default:
              console.log('Unknown message type:', data.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      });

      ws.on('close', () => {
        console.log('WebSocket connection closed');
      });

      ws.on('error', (error) => {
        console.error('WebSocket error:', error);
      });

      // Send welcome message
      ws.send(JSON.stringify({
        type: 'connection_established',
        timestamp: Date.now(),
        message: 'Connected to AgentForge Admin Interface'
      }));
    });

    res.socket.server.wss = wss;
    console.log('WebSocket server started on port 3001');
  }

  res.end();
}

function handleUserDataSync(data: any) {
  const { dataType, data: syncData, userId, sessionId, timestamp } = data;
  
  console.log(`Processing sync data: ${dataType} from user ${userId}`);

  try {
    switch (dataType) {
      case 'user_session_start':
        console.log(`User session started: ${userId} (${sessionId})`);
        break;

      case 'job_created':
        // Add job to admin interface
        const newJob = {
          id: syncData.jobId,
          title: syncData.userMessage.substring(0, 50) + '...',
          description: syncData.userMessage,
          status: 'queued' as const,
          priority: 'medium' as const,
          progress: 0,
          createdAt: new Date(timestamp),
          startedAt: undefined,
          completedAt: undefined,
          assignedAgent: undefined,
          tags: ['user-generated'],
          userId,
          sessionId,
          source: 'user_interface'
        };
        
        store.jobs.unshift(newJob);
        
        // Update user session metrics
        if (!store.meta.userSessions) {
          store.meta.userSessions = {};
        }
        if (!store.meta.userSessions[userId]) {
          store.meta.userSessions[userId] = {
            sessionId,
            jobsCreated: 0,
            messagesent: 0,
            dataSourcesAdded: 0,
            lastActivity: timestamp
          };
        }
        store.meta.userSessions[userId].jobsCreated += 1;
        store.meta.userSessions[userId].lastActivity = timestamp;
        break;

      case 'job_paused':
      case 'job_resumed':
      case 'job_archived':
        const job = store.jobs.find(j => j.id === syncData.jobId);
        if (job) {
          job.status = dataType === 'job_paused' ? 'paused' : 
                      dataType === 'job_resumed' ? 'running' : 'completed';
          if (dataType === 'job_archived') {
            job.completedAt = new Date(timestamp);
          }
          (job as any).updatedAt = timestamp;
        }
        break;

      case 'message_sent':
        if (store.meta.userSessions && store.meta.userSessions[userId]) {
          store.meta.userSessions[userId].messagesent += 1;
          store.meta.userSessions[userId].lastActivity = timestamp;
        }
        store.meta.userInteractions = (store.meta.userInteractions || 0) + 1;
        store.meta.lastUserActivity = timestamp;
        break;

      case 'data_source_added':
        if (store.meta.userSessions && store.meta.userSessions[userId]) {
          store.meta.userSessions[userId].dataSourcesAdded += 1;
          store.meta.userSessions[userId].lastActivity = timestamp;
        }
        store.meta.totalDataSources = (store.meta.totalDataSources || 0) + 1;
        break;

      case 'heartbeat':
        if (store.meta.userSessions && store.meta.userSessions[userId]) {
          store.meta.userSessions[userId].lastActivity = timestamp;
        }
        break;

      case 'current_state':
        console.log(`Full state sync from ${userId}:`, {
          activeJobs: syncData.activeJobs?.length || 0,
          archivedJobs: syncData.archivedJobs?.length || 0,
          dataSources: syncData.dataSources?.length || 0,
          messages: syncData.messages?.length || 0
        });
        break;

      default:
        console.log(`Unknown sync data type: ${dataType}`);
    }

    // Broadcast updates to all connected admin clients if needed
    if (wss) {
      const updateMessage = {
        type: 'admin_update',
        dataType,
        userId,
        timestamp,
        metrics: {
          totalJobs: store.jobs.length,
          activeSessions: store.meta.userSessions ? Object.keys(store.meta.userSessions).length : 0,
          userInteractions: store.meta.userInteractions || 0
        }
      };

      wss.clients.forEach((client) => {
        if (client.readyState === 1) { // WebSocket.OPEN
          client.send(JSON.stringify(updateMessage));
        }
      });
    }

  } catch (error) {
    console.error('Error handling user data sync:', error);
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
}
