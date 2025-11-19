'use client';

import { motion } from 'framer-motion';
import { 
  Activity, 
  Bot, 
  Briefcase, 
  Cpu, 
  Database, 
  Zap, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  Clock,
  Users,
  Brain,
  Network,
  Server
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';
import { HUD } from '@/components/HUD';
import { useMemo } from 'react';

export default function Dashboard() {
  const snap = useSnapshot(store);

  // Calculate metrics
  const metrics = useMemo(() => {
    const jobCounts = snap.jobs.reduce((acc, job) => {
      acc[job.status] = (acc[job.status] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const userJobCounts = snap.jobs.filter(j => (j as any).source === 'user_interface').reduce((acc, job) => {
      acc[job.status] = (acc[job.status] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const activeSessions = snap.meta.userSessions ? Object.keys(snap.meta.userSessions).length : 0;
    const totalUserJobs = snap.jobs.filter(j => (j as any).source === 'user_interface').length;

    return {
      totalJobs: snap.jobs.length,
      activeJobs: jobCounts.running || 0,
      completedJobs: jobCounts.completed || 0,
      failedJobs: jobCounts.failed || 0,
      queuedJobs: jobCounts.queued || 0,
      nodes: snap.meta.nodes || 0,
      queueDepth: snap.meta.queueDepth || 0,
      rps: snap.meta.rps || 0,
      // User interface metrics
      activeSessions,
      totalUserJobs,
      userActiveJobs: userJobCounts.running || 0,
      userCompletedJobs: userJobCounts.completed || 0,
      userInteractions: snap.meta.userInteractions || 0,
      totalDataSources: snap.meta.totalDataSources || 0,
      lastUserActivity: snap.meta.lastUserActivity
    };
  }, [snap.jobs, snap.meta]);

  const systemHealth = useMemo(() => {
    const healthScore = snap.connected ? 95 : 0;
    const status = healthScore > 90 ? 'excellent' : healthScore > 70 ? 'good' : healthScore > 50 ? 'warning' : 'critical';
    return { score: healthScore, status };
  }, [snap.connected]);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-day-text dark:text-night-text">
            Mission Control Dashboard
          </h1>
          <p className="text-sm text-day-text/70 dark:text-night-text/70 mt-1">
            Real-time monitoring and control of your agent swarm operations
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge 
            variant={snap.connected ? 'success' : 'danger'}
            className="px-3 py-1"
          >
            {snap.connected ? 'SYSTEMS ONLINE' : 'SYSTEMS OFFLINE'}
          </Badge>
          <Badge variant="info" className="px-3 py-1">
            Health: {systemHealth.score}%
          </Badge>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Active Agents"
          value={metrics.nodes.toString()}
          icon={<Bot className="h-6 w-6" />}
          trend="+12%"
          trendUp={true}
          color="blue"
        />
        <MetricCard
          title="Jobs Processed"
          value={metrics.totalJobs.toString()}
          icon={<Briefcase className="h-6 w-6" />}
          trend="+8%"
          trendUp={true}
          color="green"
        />
        <MetricCard
          title="Queue Depth"
          value={metrics.queueDepth.toString()}
          icon={<Clock className="h-6 w-6" />}
          trend="-5%"
          trendUp={false}
          color="yellow"
        />
        <MetricCard
          title="Requests/sec"
          value={metrics.rps.toString()}
          icon={<Zap className="h-6 w-6" />}
          trend="+23%"
          trendUp={true}
          color="purple"
        />
      </div>

      {/* User Interface Metrics */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Users className="h-5 w-5 text-day-accent dark:text-night-accent" />
          <h2 className="text-lg font-semibold text-day-text dark:text-night-text">
            User Interface Activity
          </h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            title="Active User Sessions"
            value={metrics.activeSessions.toString()}
            icon={<Users className="h-5 w-5" />}
            trend={metrics.activeSessions > 0 ? "+100%" : "0%"}
            trendUp={metrics.activeSessions > 0}
            color="blue"
          />
          
          <MetricCard
            title="User Jobs"
            value={metrics.totalUserJobs.toString()}
            icon={<Briefcase className="h-5 w-5" />}
            trend={metrics.userActiveJobs > 0 ? `+${metrics.userActiveJobs}` : "0"}
            trendUp={metrics.userActiveJobs > 0}
            color="green"
          />
          
          <MetricCard
            title="User Interactions"
            value={metrics.userInteractions.toString()}
            icon={<Activity className="h-5 w-5" />}
            trend="+15%"
            trendUp={true}
            color="orange"
          />
          
          <MetricCard
            title="Data Sources"
            value={metrics.totalDataSources.toString()}
            icon={<Database className="h-5 w-5" />}
            trend={metrics.totalDataSources > 0 ? `+${metrics.totalDataSources}` : "0"}
            trendUp={metrics.totalDataSources > 0}
            color="purple"
          />
        </div>

        {/* User Sessions Detail */}
        {snap.meta.userSessions && Object.keys(snap.meta.userSessions).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-5 w-5" />
                Active User Sessions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(snap.meta.userSessions).map(([userId, session]) => (
                  <div key={userId} className="flex items-center justify-between p-4 bg-day-cardBg dark:bg-night-cardBg rounded-lg border border-day-border dark:border-night-border">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-day-accent dark:bg-night-accent rounded-full flex items-center justify-center">
                        <span className="text-white text-sm font-bold">
                          {userId.slice(-2).toUpperCase()}
                        </span>
                      </div>
                      <div>
                        <p className="font-medium text-day-text dark:text-night-text">
                          User {userId.slice(-8)}
                        </p>
                        <p className="text-sm text-day-text/70 dark:text-night-text/70">
                          Last active: {typeof window !== 'undefined' ? new Date(session.lastActivity).toLocaleTimeString() : 'Loading...'}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-6 text-sm">
                      <div className="text-center">
                        <div className="font-semibold text-day-text dark:text-night-text">
                          {session.jobsCreated}
                        </div>
                        <div className="text-day-text/70 dark:text-night-text/70">Jobs</div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold text-day-text dark:text-night-text">
                          {session.messagesent}
                        </div>
                        <div className="text-day-text/70 dark:text-night-text/70">Messages</div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold text-day-text dark:text-night-text">
                          {session.dataSourcesAdded}
                        </div>
                        <div className="text-day-text/70 dark:text-night-text/70">Data Sources</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* System Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* System Health */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Health
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">Overall Health</span>
                <Badge variant={systemHealth.status === 'excellent' ? 'success' : 'warning'}>
                  {systemHealth.score}%
                </Badge>
              </div>
              
              <div className="space-y-3">
                <HealthIndicator label="Neural Mesh" status="operational" />
                <HealthIndicator label="Quantum Scheduler" status="operational" />
                <HealthIndicator label="Agent Swarm" status="operational" />
                <HealthIndicator label="Data Pipeline" status={snap.connected ? "operational" : "degraded"} />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Resource Usage */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Resource Usage
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <ResourceBar label="CPU" usage={75} />
              <ResourceBar label="Memory" usage={62} />
              <ResourceBar label="GPU" usage={89} />
              <ResourceBar label="Network" usage={34} />
              <ResourceBar label="Storage" usage={45} />
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Recent Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <ActivityItem
                title="Agent deployment completed"
                subtitle="GPU cluster scaled to 12 nodes"
                time="2 min ago"
                type="success"
              />
              <ActivityItem
                title="Neural mesh optimization"
                subtitle="Memory usage reduced by 15%"
                time="5 min ago"
                type="info"
              />
              <ActivityItem
                title="High queue depth detected"
                subtitle="50+ jobs pending processing"
                time="8 min ago"
                type="warning"
              />
              <ActivityItem
                title="Quantum scheduler update"
                subtitle="Performance improvements deployed"
                time="12 min ago"
                type="success"
              />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Enhanced HUD */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5" />
            Swarm Operations Console
          </CardTitle>
        </CardHeader>
        <CardContent>
          <HUD />
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Server className="h-5 w-5" />
            Quick Actions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <QuickAction
              icon={<Bot className="h-6 w-6" />}
              label="Deploy Agent"
              href="/agents/deploy"
            />
            <QuickAction
              icon={<Briefcase className="h-6 w-6" />}
              label="Create Job"
              href="/jobs/create"
            />
            <QuickAction
              icon={<Brain className="h-6 w-6" />}
              label="Neural Mesh"
              href="/neural-mesh"
            />
            <QuickAction
              icon={<Zap className="h-6 w-6" />}
              label="Quantum Queue"
              href="/quantum"
            />
            <QuickAction
              icon={<Activity className="h-6 w-6" />}
              label="Monitoring"
              href="/monitoring"
            />
            <QuickAction
              icon={<Database className="h-6 w-6" />}
              label="Data Pipeline"
              href="/data"
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Component helpers
function MetricCard({ title, value, icon, trend, trendUp, color }: {
  title: string;
  value: string;
  icon: React.ReactNode;
  trend: string;
  trendUp: boolean;
  color: string;
}) {
  const colorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    yellow: 'text-yellow-400',
    purple: 'text-purple-400'
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card hover>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="label text-xs">{title}</p>
              <p className="text-2xl font-bold mt-2">{value}</p>
              <div className={`flex items-center gap-1 mt-2 text-sm ${trendUp ? 'text-green-400' : 'text-red-400'}`}>
                <TrendingUp className={`h-4 w-4 ${!trendUp && 'rotate-180'}`} />
                {trend}
              </div>
            </div>
            <div className={colorClasses[color as keyof typeof colorClasses]}>
              {icon}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function HealthIndicator({ label, status }: { label: string; status: 'operational' | 'degraded' | 'critical' }) {
  const statusConfig = {
    operational: { icon: CheckCircle, color: 'text-green-400', bg: 'bg-green-400/20' },
    degraded: { icon: AlertTriangle, color: 'text-yellow-400', bg: 'bg-yellow-400/20' },
    critical: { icon: AlertTriangle, color: 'text-red-400', bg: 'bg-red-400/20' }
  };

  const { icon: Icon, color, bg } = statusConfig[status];

  return (
    <div className="flex items-center justify-between">
      <span className="text-sm">{label}</span>
      <div className={`flex items-center gap-2 px-2 py-1 rounded-lg ${bg}`}>
        <Icon className={`h-3 w-3 ${color}`} />
        <span className={`text-xs capitalize ${color}`}>{status}</span>
      </div>
    </div>
  );
}

function ResourceBar({ label, usage }: { label: string; usage: number }) {
  const getColor = (usage: number) => {
    if (usage > 80) return 'bg-red-400';
    if (usage > 60) return 'bg-yellow-400';
    return 'bg-green-400';
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm">{label}</span>
        <span className="text-xs">{usage}%</span>
      </div>
      <div className="w-full bg-white/10 dark:bg-black/20 rounded-full h-2">
        <motion.div
          className={`h-2 rounded-full ${getColor(usage)}`}
          initial={{ width: 0 }}
          animate={{ width: `${usage}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}

function ActivityItem({ title, subtitle, time, type }: {
  title: string;
  subtitle: string;
  time: string;
  type: 'success' | 'info' | 'warning' | 'error';
}) {
  const typeConfig = {
    success: { icon: CheckCircle, color: 'text-green-400' },
    info: { icon: Activity, color: 'text-blue-400' },
    warning: { icon: AlertTriangle, color: 'text-yellow-400' },
    error: { icon: AlertTriangle, color: 'text-red-400' }
  };

  const { icon: Icon, color } = typeConfig[type];

  return (
    <div className="flex items-start gap-3 p-3 rounded-lg bg-white/5 dark:bg-black/20">
      <Icon className={`h-4 w-4 mt-0.5 ${color}`} />
      <div className="flex-1">
        <p className="text-sm font-medium">{title}</p>
        <p className="text-xs opacity-70">{subtitle}</p>
        <p className="text-xs opacity-50 mt-1">{time}</p>
      </div>
    </div>
  );
}

function QuickAction({ icon, label, href }: { icon: React.ReactNode; label: string; href: string }) {
  return (
    <motion.a
      href={href}
      className="flex flex-col items-center gap-2 p-4 rounded-xl bg-white/5 dark:bg-black/20 hover:bg-white/10 dark:hover:bg-black/30 transition-colors border border-white/10 dark:border-red-900/40"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <div className="text-day-accent dark:text-night-text">
        {icon}
      </div>
      <span className="text-xs font-medium text-center">{label}</span>
    </motion.a>
  );
}

