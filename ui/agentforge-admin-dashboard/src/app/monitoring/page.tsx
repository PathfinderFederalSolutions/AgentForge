'use client';

import { useState, useMemo, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Zap,
  Cpu,
  HardDrive,
  Network,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Server,
  Database,
  Wifi
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Layout } from '@/components/layout/Layout';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';

// Mock monitoring data
const generateMetricData = (points = 24) => {
  return Array.from({ length: points }, (_, i) => ({
    timestamp: new Date(Date.now() - (points - i) * 60000),
    value: Math.random() * 100
  }));
};

const systemMetrics = {
  cpu: generateMetricData(),
  memory: generateMetricData(),
  network: generateMetricData(),
  disk: generateMetricData(),
  requests: generateMetricData().map(p => ({ ...p, value: p.value * 10 }))
};

const alerts = [
  {
    id: 'alert-001',
    severity: 'critical',
    title: 'High Memory Usage',
    message: 'Agent cluster memory usage exceeded 90% threshold',
    timestamp: new Date(Date.now() - 5 * 60000),
    source: 'Neural Processor Alpha',
    acknowledged: false
  },
  {
    id: 'alert-002',
    severity: 'warning',
    title: 'Queue Depth High',
    message: 'Job queue depth approaching capacity (85%)',
    timestamp: new Date(Date.now() - 12 * 60000),
    source: 'Quantum Scheduler',
    acknowledged: false
  },
  {
    id: 'alert-003',
    severity: 'info',
    title: 'Agent Deployment Complete',
    message: 'Successfully deployed 3 new GPU agents',
    timestamp: new Date(Date.now() - 25 * 60000),
    source: 'Deployment Manager',
    acknowledged: true
  },
  {
    id: 'alert-004',
    severity: 'critical',
    title: 'Connection Lost',
    message: 'Lost connection to agent cluster in us-west-2',
    timestamp: new Date(Date.now() - 45 * 60000),
    source: 'Network Monitor',
    acknowledged: false
  }
];

export default function MonitoringPage() {
  const [timeRange, setTimeRange] = useState('1h');
  const [refreshInterval, setRefreshInterval] = useState(30);
  const snap = useSnapshot(store);

  // Auto-refresh logic
  useEffect(() => {
    const interval = setInterval(() => {
      // In a real app, this would refresh the metrics
      console.log('Refreshing metrics...');
    }, refreshInterval * 1000);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  const currentMetrics = useMemo(() => {
    const latest = (data: typeof systemMetrics.cpu) => data[data.length - 1]?.value || 0;
    
    return {
      cpu: Math.round(latest(systemMetrics.cpu)),
      memory: Math.round(latest(systemMetrics.memory)),
      network: Math.round(latest(systemMetrics.network)),
      disk: Math.round(latest(systemMetrics.disk)),
      requests: Math.round(latest(systemMetrics.requests))
    };
  }, []);

  const systemHealth = useMemo(() => {
    const criticalAlerts = alerts.filter(a => a.severity === 'critical' && !a.acknowledged).length;
    const warningAlerts = alerts.filter(a => a.severity === 'warning' && !a.acknowledged).length;
    
    if (criticalAlerts > 0) return { status: 'critical', score: 45 };
    if (warningAlerts > 0) return { status: 'warning', score: 75 };
    return { status: 'healthy', score: 95 };
  }, []);

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">System Monitoring</h1>
            <p className="text-sm opacity-70 mt-1">
              Real-time monitoring of your AgentForge infrastructure
            </p>
          </div>
          <div className="flex items-center gap-3">
            <select
              className="input w-auto"
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
            >
              <option value="15m">Last 15 minutes</option>
              <option value="1h">Last hour</option>
              <option value="6h">Last 6 hours</option>
              <option value="24h">Last 24 hours</option>
            </select>
            <Badge 
              variant={snap.connected ? 'success' : 'danger'}
              className="px-3 py-1"
            >
              {snap.connected ? 'LIVE' : 'DISCONNECTED'}
            </Badge>
          </div>
        </div>

        {/* System Health Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                System Health
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center">
                <div className="relative w-24 h-24 mx-auto mb-4">
                  <svg className="w-24 h-24 transform -rotate-90">
                    <circle
                      cx="48"
                      cy="48"
                      r="40"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      className="opacity-20"
                    />
                    <motion.circle
                      cx="48"
                      cy="48"
                      r="40"
                      stroke={systemHealth.status === 'healthy' ? '#10B981' : systemHealth.status === 'warning' ? '#F59E0B' : '#EF4444'}
                      strokeWidth="8"
                      fill="none"
                      strokeLinecap="round"
                      initial={{ strokeDasharray: 0, strokeDashoffset: 0 }}
                      animate={{
                        strokeDasharray: 2 * Math.PI * 40,
                        strokeDashoffset: 2 * Math.PI * 40 * (1 - systemHealth.score / 100)
                      }}
                      transition={{ duration: 1, ease: 'easeOut' }}
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-2xl font-bold">{systemHealth.score}%</span>
                  </div>
                </div>
                <Badge 
                  variant={systemHealth.status === 'healthy' ? 'success' : systemHealth.status === 'warning' ? 'warning' : 'danger'}
                  className="capitalize"
                >
                  {systemHealth.status}
                </Badge>
              </div>
            </CardContent>
          </Card>

          <div className="lg:col-span-3 grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              title="CPU Usage"
              value={currentMetrics.cpu}
              unit="%"
              icon={<Cpu className="h-5 w-5" />}
              trend={5}
              color="blue"
            />
            <MetricCard
              title="Memory"
              value={currentMetrics.memory}
              unit="%"
              icon={<HardDrive className="h-5 w-5" />}
              trend={-2}
              color="green"
            />
            <MetricCard
              title="Network"
              value={currentMetrics.network}
              unit="MB/s"
              icon={<Network className="h-5 w-5" />}
              trend={12}
              color="purple"
            />
            <MetricCard
              title="Disk I/O"
              value={currentMetrics.disk}
              unit="MB/s"
              icon={<HardDrive className="h-5 w-5" />}
              trend={-8}
              color="yellow"
            />
          </div>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>CPU & Memory Usage</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-end gap-1">
                {systemMetrics.cpu.slice(-20).map((point, i) => (
                  <motion.div
                    key={i}
                    className="flex-1 bg-blue-400/20 rounded-t"
                    initial={{ height: 0 }}
                    animate={{ height: `${(point.value / 100) * 100}%` }}
                    transition={{ duration: 0.5, delay: i * 0.05 }}
                  />
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Request Rate</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-end gap-1">
                {systemMetrics.requests.slice(-20).map((point, i) => (
                  <motion.div
                    key={i}
                    className="flex-1 bg-green-400/20 rounded-t"
                    initial={{ height: 0 }}
                    animate={{ height: `${(point.value / 1000) * 100}%` }}
                    transition={{ duration: 0.5, delay: i * 0.05 }}
                  />
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Alerts and Status */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Active Alerts */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Active Alerts ({alerts.filter(a => !a.acknowledged).length})
                </span>
                <Badge variant="danger" size="sm">
                  {alerts.filter(a => a.severity === 'critical' && !a.acknowledged).length} Critical
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="max-h-80 overflow-y-auto">
                {alerts.map((alert) => (
                  <AlertItem key={alert.id} alert={alert} />
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Service Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                Service Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <ServiceStatus
                  name="Neural Mesh"
                  status="operational"
                  uptime="99.9%"
                  icon={<Database className="h-4 w-4" />}
                />
                <ServiceStatus
                  name="Quantum Scheduler"
                  status="operational"
                  uptime="99.8%"
                  icon={<Zap className="h-4 w-4" />}
                />
                <ServiceStatus
                  name="Agent Swarm"
                  status="degraded"
                  uptime="98.2%"
                  icon={<Server className="h-4 w-4" />}
                />
                <ServiceStatus
                  name="API Gateway"
                  status="operational"
                  uptime="99.9%"
                  icon={<Wifi className="h-4 w-4" />}
                />
                <ServiceStatus
                  name="WebSocket"
                  status={snap.connected ? "operational" : "down"}
                  uptime={snap.connected ? "100%" : "0%"}
                  icon={<Network className="h-4 w-4" />}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Agent Performance */}
        <Card>
          <CardHeader>
            <CardTitle>Agent Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-day-accent dark:text-night-text">
                  {snap.meta.nodes || 0}
                </div>
                <div className="label text-xs mt-1">Active Agents</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400">
                  {snap.meta.rps || 0}
                </div>
                <div className="label text-xs mt-1">Requests/sec</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-yellow-400">
                  {snap.meta.queueDepth || 0}
                </div>
                <div className="label text-xs mt-1">Queue Depth</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">
                  {snap.jobs.length}
                </div>
                <div className="label text-xs mt-1">Total Jobs</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}

function MetricCard({ title, value, unit, icon, trend, color }: {
  title: string;
  value: number;
  unit: string;
  icon: React.ReactNode;
  trend: number;
  color: 'blue' | 'green' | 'purple' | 'yellow';
}) {
  const colorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    purple: 'text-purple-400',
    yellow: 'text-yellow-400'
  };

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="label text-xs">{title}</p>
            <p className="text-xl font-bold mt-1">
              {value}{unit}
            </p>
            <div className={`flex items-center gap-1 mt-2 text-xs ${trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
              {trend > 0 ? (
                <TrendingUp className="h-3 w-3" />
              ) : (
                <TrendingDown className="h-3 w-3" />
              )}
              {Math.abs(trend)}%
            </div>
          </div>
          <div className={colorClasses[color]}>
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function AlertItem({ alert }: { alert: typeof alerts[0] }) {
  const severityConfig = {
    critical: { icon: XCircle, color: 'text-red-400', bg: 'bg-red-400/10' },
    warning: { icon: AlertTriangle, color: 'text-yellow-400', bg: 'bg-yellow-400/10' },
    info: { icon: CheckCircle, color: 'text-blue-400', bg: 'bg-blue-400/10' }
  };

  const config = severityConfig[alert.severity as keyof typeof severityConfig];

  return (
    <div className={`p-4 border-b border-white/10 dark:border-red-900/40 last:border-b-0 ${!alert.acknowledged ? config.bg : 'opacity-50'}`}>
      <div className="flex items-start gap-3">
        <config.icon className={`h-5 w-5 mt-0.5 ${config.color}`} />
        <div className="flex-1">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-sm">{alert.title}</h4>
            <div className="flex items-center gap-2">
              <Badge variant={alert.severity === 'critical' ? 'danger' : alert.severity === 'warning' ? 'warning' : 'info'} size="sm">
                {alert.severity}
              </Badge>
              {!alert.acknowledged && (
                <button className="text-xs text-day-accent dark:text-night-text hover:underline">
                  Acknowledge
                </button>
              )}
            </div>
          </div>
          <p className="text-sm opacity-80 mt-1">{alert.message}</p>
          <div className="flex items-center gap-4 mt-2 text-xs opacity-60">
            <span>Source: {alert.source}</span>
            <span>{alert.timestamp.toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function ServiceStatus({ name, status, uptime, icon }: {
  name: string;
  status: 'operational' | 'degraded' | 'down';
  uptime: string;
  icon: React.ReactNode;
}) {
  const statusConfig = {
    operational: { color: 'text-green-400', bg: 'bg-green-400/20' },
    degraded: { color: 'text-yellow-400', bg: 'bg-yellow-400/20' },
    down: { color: 'text-red-400', bg: 'bg-red-400/20' }
  };

  const config = statusConfig[status];

  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg ${config.bg}`}>
          <div className={config.color}>
            {icon}
          </div>
        </div>
        <div>
          <p className="text-sm font-medium">{name}</p>
          <p className="text-xs opacity-60">Uptime: {uptime}</p>
        </div>
      </div>
      <Badge 
        variant={status === 'operational' ? 'success' : status === 'degraded' ? 'warning' : 'danger'} 
        size="sm"
      >
        {status}
      </Badge>
    </div>
  );
}
