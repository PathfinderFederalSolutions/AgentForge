'use client';

import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  PieChart,
  Activity,
  Clock,
  Target,
  Zap,
  Users,
  Briefcase,
  Calendar,
  Download,
  Filter,
  RefreshCw
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Layout } from '@/components/layout/Layout';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';

// Mock analytics data
const analyticsData = {
  performance: {
    taskCompletionRate: 94.2,
    averageTaskTime: 127,
    successRate: 96.8,
    errorRate: 3.2,
    throughputTrend: 12.5,
    efficiencyScore: 89.4
  },
  usage: {
    totalTasks: 15847,
    completedTasks: 14923,
    failedTasks: 507,
    activeTasks: 417,
    agentUtilization: 78.3,
    resourceEfficiency: 85.7
  },
  trends: {
    daily: Array.from({ length: 30 }, (_, i) => ({
      date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000),
      tasks: Math.floor(Math.random() * 500) + 200,
      agents: Math.floor(Math.random() * 20) + 10,
      efficiency: Math.random() * 20 + 80
    })),
    hourly: Array.from({ length: 24 }, (_, i) => ({
      hour: i,
      requests: Math.floor(Math.random() * 100) + 50,
      latency: Math.random() * 200 + 100,
      errors: Math.floor(Math.random() * 10)
    }))
  }
};

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('7d');
  const [chartType, setChartType] = useState('line');
  const snap = useSnapshot(store);

  const currentMetrics = useMemo(() => {
    return {
      totalJobs: snap.jobs.length,
      completedJobs: snap.jobs.filter(j => j.status === 'completed').length,
      failedJobs: snap.jobs.filter(j => j.status === 'failed').length,
      activeJobs: snap.jobs.filter(j => j.status === 'running').length,
      successRate: snap.jobs.length > 0 ? 
        (snap.jobs.filter(j => j.status === 'completed').length / snap.jobs.length * 100) : 0
    };
  }, [snap.jobs]);

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <BarChart3 className="h-8 w-8 text-purple-400" />
              Analytics & Insights
            </h1>
            <p className="text-sm opacity-70 mt-1">
              Performance analytics and operational insights for your agent swarm
            </p>
          </div>
          <div className="flex items-center gap-3">
            <select
              className="input w-auto"
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
            <Button variant="ghost" icon={<RefreshCw className="h-4 w-4" />}>
              Refresh
            </Button>
            <Button variant="ghost" icon={<Download className="h-4 w-4" />}>
              Export
            </Button>
          </div>
        </div>

        {/* Key Performance Indicators */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <KPICard
            title="Success Rate"
            value={`${Math.round(currentMetrics.successRate)}%`}
            change={+2.3}
            icon={<Target className="h-5 w-5" />}
            color="green"
          />
          <KPICard
            title="Avg Task Time"
            value="127s"
            change={-8.1}
            icon={<Clock className="h-5 w-5" />}
            color="blue"
          />
          <KPICard
            title="Throughput"
            value="1.2k/min"
            change={+12.5}
            icon={<Activity className="h-5 w-5" />}
            color="purple"
          />
          <KPICard
            title="Agent Utilization"
            value="78.3%"
            change={+5.7}
            icon={<Users className="h-5 w-5" />}
            color="yellow"
          />
          <KPICard
            title="Error Rate"
            value="3.2%"
            change={-1.4}
            icon={<TrendingDown className="h-5 w-5" />}
            color="red"
          />
          <KPICard
            title="Efficiency"
            value="89.4%"
            change={+3.2}
            icon={<Zap className="h-5 w-5" />}
            color="green"
          />
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Task Performance Over Time */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Task Performance Trends
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-end gap-1">
                {analyticsData.trends.daily.slice(-14).map((point, i) => (
                  <motion.div
                    key={i}
                    className="flex-1 bg-gradient-to-t from-blue-400/20 to-blue-400/60 rounded-t"
                    initial={{ height: 0 }}
                    animate={{ height: `${(point.tasks / 700) * 100}%` }}
                    transition={{ duration: 0.5, delay: i * 0.05 }}
                  />
                ))}
              </div>
              <div className="flex justify-between text-xs opacity-60 mt-2">
                <span>14 days ago</span>
                <span>Today</span>
              </div>
            </CardContent>
          </Card>

          {/* Agent Utilization */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-5 w-5" />
                Agent Utilization
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-end gap-1">
                {analyticsData.trends.daily.slice(-14).map((point, i) => (
                  <motion.div
                    key={i}
                    className="flex-1 bg-gradient-to-t from-green-400/20 to-green-400/60 rounded-t"
                    initial={{ height: 0 }}
                    animate={{ height: `${point.efficiency}%` }}
                    transition={{ duration: 0.5, delay: i * 0.05 }}
                  />
                ))}
              </div>
              <div className="flex justify-between text-xs opacity-60 mt-2">
                <span>14 days ago</span>
                <span>Today</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Detailed Analytics */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Task Distribution */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PieChart className="h-5 w-5" />
                Task Distribution
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <TaskTypeBar type="Neural Processing" count={5847} percentage={37} color="bg-blue-400" />
                <TaskTypeBar type="Data Analysis" count={4932} percentage={31} color="bg-green-400" />
                <TaskTypeBar type="ML Training" count={3421} percentage={22} color="bg-purple-400" />
                <TaskTypeBar type="Security Scans" count={1647} percentage={10} color="bg-red-400" />
              </div>
            </CardContent>
          </Card>

          {/* Performance Breakdown */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <PerformanceMetric
                  label="Task Completion Rate"
                  value={94.2}
                  target={95}
                  unit="%"
                />
                <PerformanceMetric
                  label="Average Response Time"
                  value={127}
                  target={100}
                  unit="ms"
                />
                <PerformanceMetric
                  label="Resource Efficiency"
                  value={85.7}
                  target={90}
                  unit="%"
                />
                <PerformanceMetric
                  label="Error Rate"
                  value={3.2}
                  target={5}
                  unit="%"
                  inverse
                />
              </div>
            </CardContent>
          </Card>

          {/* Real-time Stats */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Real-time Statistics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-day-accent dark:text-night-text">
                    {snap.jobs.length}
                  </div>
                  <p className="text-xs opacity-70">Active Jobs</p>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div>
                    <div className="text-lg font-bold text-green-400">
                      {currentMetrics.completedJobs}
                    </div>
                    <p className="text-xs opacity-70">Completed</p>
                  </div>
                  <div>
                    <div className="text-lg font-bold text-red-400">
                      {currentMetrics.failedJobs}
                    </div>
                    <p className="text-xs opacity-70">Failed</p>
                  </div>
                </div>

                <div className="text-center">
                  <div className="text-lg font-bold text-purple-400">
                    {Math.round(currentMetrics.successRate)}%
                  </div>
                  <p className="text-xs opacity-70">Success Rate</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Hourly Performance */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-5 w-5" />
              24-Hour Performance Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-48 flex items-end gap-1">
              {analyticsData.trends.hourly.map((point, i) => (
                <motion.div
                  key={i}
                  className="flex-1 bg-gradient-to-t from-purple-400/20 to-purple-400/60 rounded-t"
                  initial={{ height: 0 }}
                  animate={{ height: `${(point.requests / 150) * 100}%` }}
                  transition={{ duration: 0.5, delay: i * 0.02 }}
                />
              ))}
            </div>
            <div className="flex justify-between text-xs opacity-60 mt-2">
              <span>00:00</span>
              <span>12:00</span>
              <span>23:59</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}

function KPICard({ title, value, change, icon, color }: {
  title: string;
  value: string;
  change: number;
  icon: React.ReactNode;
  color: 'green' | 'blue' | 'purple' | 'yellow' | 'red';
}) {
  const colorClasses = {
    green: 'text-green-400',
    blue: 'text-blue-400',
    purple: 'text-purple-400',
    yellow: 'text-yellow-400',
    red: 'text-red-400'
  };

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <div className={colorClasses[color]}>
            {icon}
          </div>
          <div className={`flex items-center gap-1 text-xs ${change > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {change > 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
            {Math.abs(change)}%
          </div>
        </div>
        <div>
          <p className="label text-xs">{title}</p>
          <p className="text-lg font-bold mt-1">{value}</p>
        </div>
      </CardContent>
    </Card>
  );
}

function TaskTypeBar({ type, count, percentage, color }: {
  type: string;
  count: number;
  percentage: number;
  color: string;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium">{type}</span>
        <span className="text-sm opacity-60">{count.toLocaleString()} ({percentage}%)</span>
      </div>
      <div className="w-full bg-white/10 dark:bg-black/20 rounded-full h-2">
        <motion.div
          className={`h-2 rounded-full ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}

function PerformanceMetric({ label, value, target, unit, inverse = false }: {
  label: string;
  value: number;
  target: number;
  unit: string;
  inverse?: boolean;
}) {
  const isGood = inverse ? value <= target : value >= target;
  const percentage = inverse ? 
    Math.min((target / value) * 100, 100) : 
    Math.min((value / target) * 100, 100);

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm">{label}</span>
        <span className={`text-sm font-medium ${isGood ? 'text-green-400' : 'text-yellow-400'}`}>
          {value}{unit}
        </span>
      </div>
      <div className="w-full bg-white/10 dark:bg-black/20 rounded-full h-2">
        <motion.div
          className={`h-2 rounded-full ${isGood ? 'bg-green-400' : 'bg-yellow-400'}`}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
      </div>
      <div className="flex justify-between text-xs opacity-60 mt-1">
        <span>Target: {target}{unit}</span>
        <span>{Math.round(percentage)}%</span>
      </div>
    </div>
  );
}