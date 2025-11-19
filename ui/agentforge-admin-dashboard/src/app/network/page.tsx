'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Network as NetworkIcon,
  Globe,
  Wifi,
  Server,
  Activity,
  Shield,
  Zap,
  AlertTriangle,
  CheckCircle,
  Clock,
  Database,
  Router,
  Satellite,
  MapPin
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Layout } from '@/components/layout/Layout';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';

// Mock network topology data
const networkNodes = [
  {
    id: 'node-001',
    name: 'Primary Gateway',
    type: 'gateway',
    status: 'active',
    location: 'us-east-1a',
    connections: 45,
    latency: 12,
    bandwidth: 89.5,
    uptime: 99.9
  },
  {
    id: 'node-002',
    name: 'Neural Mesh Hub',
    type: 'neural-mesh',
    status: 'active',
    location: 'us-west-2a',
    connections: 127,
    latency: 8,
    bandwidth: 76.3,
    uptime: 99.8
  },
  {
    id: 'node-003',
    name: 'Quantum Router',
    type: 'quantum',
    status: 'active',
    location: 'eu-west-1a',
    connections: 89,
    latency: 15,
    bandwidth: 92.1,
    uptime: 99.7
  },
  {
    id: 'node-004',
    name: 'Security Perimeter',
    type: 'security',
    status: 'warning',
    location: 'ap-southeast-1a',
    connections: 34,
    latency: 23,
    bandwidth: 67.8,
    uptime: 98.2
  }
];

export default function NetworkPage() {
  const [selectedNode, setSelectedNode] = useState<typeof networkNodes[0] | null>(null);
  const [viewMode, setViewMode] = useState<'topology' | 'performance' | 'security'>('topology');
  const snap = useSnapshot(store);

  const networkStats = {
    totalNodes: networkNodes.length,
    activeNodes: networkNodes.filter(n => n.status === 'active').length,
    totalConnections: networkNodes.reduce((sum, node) => sum + node.connections, 0),
    avgLatency: Math.round(networkNodes.reduce((sum, node) => sum + node.latency, 0) / networkNodes.length),
    avgBandwidth: Math.round(networkNodes.reduce((sum, node) => sum + node.bandwidth, 0) / networkNodes.length),
    avgUptime: Math.round(networkNodes.reduce((sum, node) => sum + node.uptime, 0) / networkNodes.length * 10) / 10
  };

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <NetworkIcon className="h-8 w-8 text-blue-400" />
              Network Topology
            </h1>
            <p className="text-sm opacity-70 mt-1">
              Monitor and manage your distributed network infrastructure
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex gap-1 bg-white/5 dark:bg-black/20 rounded-lg p-1">
              <button
                className={`px-3 py-1 text-xs rounded-md transition-colors ${
                  viewMode === 'topology' 
                    ? 'bg-day-accent text-white dark:bg-night-text dark:text-black' 
                    : 'hover:bg-white/10'
                }`}
                onClick={() => setViewMode('topology')}
              >
                Topology
              </button>
              <button
                className={`px-3 py-1 text-xs rounded-md transition-colors ${
                  viewMode === 'performance' 
                    ? 'bg-day-accent text-white dark:bg-night-text dark:text-black' 
                    : 'hover:bg-white/10'
                }`}
                onClick={() => setViewMode('performance')}
              >
                Performance
              </button>
              <button
                className={`px-3 py-1 text-xs rounded-md transition-colors ${
                  viewMode === 'security' 
                    ? 'bg-day-accent text-white dark:bg-night-text dark:text-black' 
                    : 'hover:bg-white/10'
                }`}
                onClick={() => setViewMode('security')}
              >
                Security
              </button>
            </div>
            <Badge 
              variant={snap.connected ? 'success' : 'danger'}
              className="px-3 py-1"
            >
              {snap.connected ? 'CONNECTED' : 'DISCONNECTED'}
            </Badge>
          </div>
        </div>

        {/* Network Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <NetworkMetricCard
            title="Active Nodes"
            value={networkStats.activeNodes}
            subtitle={`${networkStats.totalNodes} total`}
            icon={<Server className="h-6 w-6" />}
            color="blue"
          />
          <NetworkMetricCard
            title="Connections"
            value={networkStats.totalConnections}
            subtitle="Active connections"
            icon={<Wifi className="h-6 w-6" />}
            color="green"
          />
          <NetworkMetricCard
            title="Avg Latency"
            value={`${networkStats.avgLatency}ms`}
            subtitle="Response time"
            icon={<Zap className="h-6 w-6" />}
            color="yellow"
          />
          <NetworkMetricCard
            title="Uptime"
            value={`${networkStats.avgUptime}%`}
            subtitle="Network availability"
            icon={<Activity className="h-6 w-6" />}
            color="purple"
          />
        </div>

        {/* Network Visualization */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Globe className="h-5 w-5" />
              Network Topology Visualization
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-96 bg-gradient-to-br from-day-bg to-day-grid dark:from-night-bg dark:to-night-grid rounded-lg relative overflow-hidden">
              {/* Network visualization */}
              <div className="absolute inset-0 flex items-center justify-center">
                <svg width="100%" height="100%" className="absolute inset-0">
                  {/* Connection lines */}
                  {networkNodes.map((node, i) => (
                    networkNodes.slice(i + 1).map((targetNode, j) => (
                      <motion.line
                        key={`${node.id}-${targetNode.id}`}
                        x1={`${20 + (i % 3) * 30}%`}
                        y1={`${20 + Math.floor(i / 3) * 30}%`}
                        x2={`${20 + ((i + j + 1) % 3) * 30}%`}
                        y2={`${20 + Math.floor((i + j + 1) / 3) * 30}%`}
                        stroke="currentColor"
                        strokeWidth="1"
                        className="text-day-accent/30 dark:text-night-text/30"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: 2, delay: (i + j) * 0.1 }}
                      />
                    ))
                  ))}
                </svg>

                {/* Network nodes */}
                {networkNodes.map((node, i) => (
                  <motion.div
                    key={node.id}
                    className="absolute cursor-pointer"
                    style={{
                      left: `${20 + (i % 3) * 30}%`,
                      top: `${20 + Math.floor(i / 3) * 30}%`,
                      transform: 'translate(-50%, -50%)'
                    }}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.5, delay: i * 0.2 }}
                    whileHover={{ scale: 1.2 }}
                    onClick={() => setSelectedNode(node)}
                  >
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center border-2 ${
                      node.status === 'active' 
                        ? 'bg-green-400/20 border-green-400' 
                        : 'bg-yellow-400/20 border-yellow-400'
                    }`}>
                      {node.type === 'gateway' && <Router className="h-6 w-6" />}
                      {node.type === 'neural-mesh' && <NetworkIcon className="h-6 w-6" />}
                      {node.type === 'quantum' && <Zap className="h-6 w-6" />}
                      {node.type === 'security' && <Shield className="h-6 w-6" />}
                    </div>
                    <div className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 text-center">
                      <p className="text-xs font-medium whitespace-nowrap">{node.name}</p>
                      <p className="text-xs opacity-60">{node.location}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Network Nodes */}
        <Card>
          <CardHeader>
            <CardTitle>Network Nodes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {networkNodes.map((node, index) => (
                <NetworkNodeCard
                  key={node.id}
                  node={node}
                  index={index}
                  onSelect={() => setSelectedNode(node)}
                />
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}

function NetworkMetricCard({ title, value, subtitle, icon, color }: {
  title: string;
  value: string | number;
  subtitle: string;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'yellow' | 'purple';
}) {
  const colorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    yellow: 'text-yellow-400',
    purple: 'text-purple-400'
  };

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="label text-xs">{title}</p>
            <p className="text-xl font-bold mt-1">{value}</p>
            <p className="text-xs opacity-60 mt-1">{subtitle}</p>
          </div>
          <div className={colorClasses[color]}>
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function NetworkNodeCard({ node, index, onSelect }: {
  node: typeof networkNodes[0];
  index: number;
  onSelect: () => void;
}) {
  const statusConfig = {
    active: { badge: 'success', icon: CheckCircle, color: 'text-green-400' },
    warning: { badge: 'warning', icon: AlertTriangle, color: 'text-yellow-400' },
    error: { badge: 'danger', icon: AlertTriangle, color: 'text-red-400' }
  };

  const config = statusConfig[node.status as keyof typeof statusConfig];

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      className="p-4 rounded-lg bg-white/5 dark:bg-black/20 border border-white/10 dark:border-red-900/40 cursor-pointer hover:bg-white/10 dark:hover:bg-black/30"
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
            node.status === 'active' 
              ? 'bg-green-400/20 text-green-400' 
              : 'bg-yellow-400/20 text-yellow-400'
          }`}>
            {node.type === 'gateway' && <Router className="h-5 w-5" />}
            {node.type === 'neural-mesh' && <NetworkIcon className="h-5 w-5" />}
            {node.type === 'quantum' && <Zap className="h-5 w-5" />}
            {node.type === 'security' && <Shield className="h-5 w-5" />}
          </div>
          
          <div>
            <h3 className="font-medium text-sm">{node.name}</h3>
            <p className="text-xs opacity-60">{node.id} • {node.location}</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <p className="opacity-60">Latency</p>
                <p className="font-medium">{node.latency}ms</p>
              </div>
              <div>
                <p className="opacity-60">Bandwidth</p>
                <p className="font-medium">{node.bandwidth}%</p>
              </div>
              <div>
                <p className="opacity-60">Connections</p>
                <p className="font-medium">{node.connections}</p>
              </div>
              <div>
                <p className="opacity-60">Uptime</p>
                <p className="font-medium">{node.uptime}%</p>
              </div>
            </div>
          </div>
          
          <Badge variant={config.badge as any} size="sm">
            {node.status}
          </Badge>
        </div>
      </div>
    </motion.div>
  );
}