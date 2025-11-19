import { NextApiRequest, NextApiResponse } from 'next';
import { store } from '@/lib/state';

interface SyncRequest {
  data: any;
  timestamp: number;
  source: 'user_interface';
  userId: string;
  sessionId: string;
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { dataType } = req.query;
  const { data, timestamp, source, userId, sessionId }: SyncRequest = req.body;

  try {
    console.log(`Received sync data for ${dataType} from ${userId}:`, data);

    switch (dataType) {
      case 'user_session_start':
        // Track new user session
        console.log(`User session started: ${userId} (${sessionId})`);
        // You could store this in a database or update metrics
        break;

      case 'job_created':
        // Add job to admin interface
        const newJob = {
          id: data.jobId,
          title: data.userMessage.substring(0, 50) + '...',
          description: data.userMessage,
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
        
        // Add to jobs array (in production, this would go to a database)
        store.jobs.unshift(newJob);
        
        // Update metrics
        if (!store.meta.userSessions) {
          store.meta.userSessions = {};
        }
        if (!store.meta.userSessions[userId]) {
          store.meta.userSessions[userId] = {
            sessionId,
            jobsCreated: 0,
            messagessent: 0,
            dataSourcesAdded: 0,
            lastActivity: timestamp
          };
        }
        store.meta.userSessions[userId].jobsCreated += 1;
        store.meta.userSessions[userId].lastActivity = timestamp;
        break;

      case 'job_paused':
        // Update job status
        const pausedJob = store.jobs.find(j => j.id === data.jobId);
        if (pausedJob) {
          pausedJob.status = 'paused';
          (pausedJob as any).updatedAt = timestamp;
        }
        break;

      case 'job_resumed':
        // Update job status
        const resumedJob = store.jobs.find(j => j.id === data.jobId);
        if (resumedJob) {
          resumedJob.status = 'running';
          (resumedJob as any).updatedAt = timestamp;
        }
        break;

      case 'job_archived':
        // Move job to completed
        const archivedJob = store.jobs.find(j => j.id === data.jobId);
        if (archivedJob) {
          archivedJob.status = 'completed';
          archivedJob.completedAt = new Date(timestamp);
          (archivedJob as any).updatedAt = timestamp;
        }
        break;

      case 'job_progress':
        // Update job progress
        const progressJob = store.jobs.find(j => j.id === data.jobId);
        if (progressJob) {
          progressJob.progress = data.progress;
          (progressJob as any).updatedAt = timestamp;
        }
        break;

      case 'message_sent':
        // Track user message
        if (store.meta.userSessions && store.meta.userSessions[userId]) {
          store.meta.userSessions[userId].messagesent += 1;
          store.meta.userSessions[userId].lastActivity = timestamp;
        }
        
        // Update general metrics
        store.meta.userInteractions = (store.meta.userInteractions || 0) + 1;
        store.meta.lastUserActivity = timestamp;
        break;

      case 'data_source_added':
        // Track data source addition
        if (store.meta.userSessions && store.meta.userSessions[userId]) {
          store.meta.userSessions[userId].dataSourcesAdded += 1;
          store.meta.userSessions[userId].lastActivity = timestamp;
        }
        
        // Update general metrics
        store.meta.totalDataSources = (store.meta.totalDataSources || 0) + 1;
        break;

      case 'data_source_removed':
        // Track data source removal
        store.meta.totalDataSources = Math.max(0, (store.meta.totalDataSources || 1) - 1);
        break;

      case 'swarm_activity':
        // Track swarm activity from user interface
        // This could be used to show real-time agent activity in admin
        console.log('Swarm activity from user:', data.activity);
        break;

      case 'theme_changed':
        // Track user preferences
        console.log(`User ${userId} changed theme to:`, data.theme);
        break;

      case 'heartbeat':
        // Update user session heartbeat
        if (store.meta.userSessions && store.meta.userSessions[userId]) {
          store.meta.userSessions[userId].lastActivity = timestamp;
        }
        break;

      case 'current_state':
        // Full state sync from user interface
        console.log(`Full state sync from ${userId}:`, {
          activeJobs: data.activeJobs?.length || 0,
          archivedJobs: data.archivedJobs?.length || 0,
          dataSources: data.dataSources?.length || 0,
          messages: data.messages?.length || 0
        });
        break;

      case 'user_session_end':
        // Track user session end
        console.log(`User session ended: ${userId} (${sessionId})`);
        break;

      default:
        console.log(`Unknown sync data type: ${dataType}`, data);
    }

    res.status(200).json({ success: true, timestamp: Date.now() });
  } catch (error) {
    console.error('Sync error:', error);
    res.status(500).json({ error: 'Sync failed' });
  }
}
