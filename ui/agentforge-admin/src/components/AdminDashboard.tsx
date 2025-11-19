'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Users, 
  Activity, 
  BarChart3, 
  Settings, 
  Shield, 
  Zap,
  Brain,
  Database,
  Globe,
  TrendingUp
} from 'lucide-react';

interface OrganizationData {
  organization_id: string;
  name: string;
  total_users: number;
  active_users: number;
  admin_connections: number;
  user_connections: number;
  subscription_tier: string;
}

interface ConnectionData {
  connection_id: string;
  user_id: string;
  user_tier: string;
  connection_type: string;
  organization_id?: string;
  connected_duration: number;
  last_activity: string;
}

interface SystemMetrics {
  total_connections: number;
  total_organizations: number;
  total_data_flows: number;
  active_agents: number;
  system_load: number;
  memory_usage: Record<string, number>;
  quantum_coherence: number;
}

export default function AdminDashboard() {
  const [organizations, setOrganizations] = useState<OrganizationData[]>([]);
  const [connections, setConnections] = useState<ConnectionData[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [selectedOrg, setSelectedOrg] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'organizations' | 'connections' | 'analytics'>('overview');

  useEffect(() => {
    loadDashboardData();
    
    // Refresh data every 30 seconds
    const interval = setInterval(loadDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async () => {
    try {
      // Load organizations
      const orgResponse = await fetch('http://localhost:8000/v1/enterprise/organizations');
      const orgData = await orgResponse.json();
      setOrganizations(orgData.organizations || []);

      // Load connections
      const connResponse = await fetch('http://localhost:8000/v1/enterprise/connections');
      const connData = await connResponse.json();
      setConnections(connData.connections || []);

      // Mock system metrics (in production, get from monitoring)
      setSystemMetrics({
        total_connections: connData.summary?.total_connections || 0,
        total_organizations: orgData.total_organizations || 0,
        total_data_flows: Math.floor(Math.random() * 1000) + 500,
        active_agents: Math.floor(Math.random() * 50) + 20,
        system_load: Math.random() * 0.8 + 0.1,
        memory_usage: {
          L1: Math.random() * 0.8 + 0.1,
          L2: Math.random() * 0.8 + 0.2,
          L3: Math.random() * 0.7 + 0.2,
          L4: Math.random() * 0.6 + 0.1
        },
        quantum_coherence: Math.random() * 0.3 + 0.7
      });

    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    }
  };

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[#8FA8C4] text-sm">Total Organizations</p>
              <p className="text-2xl font-bold text-[#D6E2F0]">
                {systemMetrics?.total_organizations || 0}
              </p>
            </div>
            <Globe className="w-8 h-8 text-[#00A39B]" />
          </div>
        </div>

        <div className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[#8FA8C4] text-sm">Active Connections</p>
              <p className="text-2xl font-bold text-[#D6E2F0]">
                {systemMetrics?.total_connections || 0}
              </p>
            </div>
            <Users className="w-8 h-8 text-[#00A39B]" />
          </div>
        </div>

        <div className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[#8FA8C4] text-sm">Active Agents</p>
              <p className="text-2xl font-bold text-[#D6E2F0]">
                {systemMetrics?.active_agents || 0}
              </p>
            </div>
            <Brain className="w-8 h-8 text-[#00A39B]" />
          </div>
        </div>

        <div className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[#8FA8C4] text-sm">Data Flows</p>
              <p className="text-2xl font-bold text-[#D6E2F0]">
                {systemMetrics?.total_data_flows || 0}
              </p>
            </div>
            <Activity className="w-8 h-8 text-[#00A39B]" />
          </div>
        </div>
      </div>

      {/* System Health */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
          <h3 className="text-lg font-semibold text-[#D6E2F0] mb-4 flex items-center">
            <Zap className="w-5 h-5 mr-2 text-[#00A39B]" />
            System Performance
          </h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-[#8FA8C4]">System Load</span>
                <span className="text-[#D6E2F0]">{((systemMetrics?.system_load || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-[#1E2A3E] rounded-full h-2">
                <div 
                  className="bg-[#00A39B] h-2 rounded-full transition-all duration-300"
                  style={{ width: `${(systemMetrics?.system_load || 0) * 100}%` }}
                />
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-[#8FA8C4]">Quantum Coherence</span>
                <span className="text-[#D6E2F0]">{((systemMetrics?.quantum_coherence || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-[#1E2A3E] rounded-full h-2">
                <div 
                  className="bg-[#00A39B] h-2 rounded-full transition-all duration-300"
                  style={{ width: `${(systemMetrics?.quantum_coherence || 0) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
          <h3 className="text-lg font-semibold text-[#D6E2F0] mb-4 flex items-center">
            <Database className="w-5 h-5 mr-2 text-[#00A39B]" />
            Memory Utilization
          </h3>
          <div className="space-y-3">
            {systemMetrics?.memory_usage && Object.entries(systemMetrics.memory_usage).map(([tier, usage]) => (
              <div key={tier}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-[#8FA8C4]">{tier} Memory</span>
                  <span className="text-[#D6E2F0]">{(usage * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-[#1E2A3E] rounded-full h-2">
                  <div 
                    className="bg-[#00A39B] h-2 rounded-full transition-all duration-300"
                    style={{ width: `${usage * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderOrganizations = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-[#D6E2F0]">Organizations</h3>
        <button className="px-4 py-2 bg-[#00A39B] text-white rounded-lg hover:bg-[#008A84] transition-colors">
          Create Organization
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {organizations.map((org) => (
          <motion.div
            key={org.organization_id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg hover:border-[#00A39B] transition-colors cursor-pointer"
            onClick={() => setSelectedOrg(org.organization_id)}
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <h4 className="text-lg font-semibold text-[#D6E2F0]">{org.name}</h4>
                <p className="text-sm text-[#8FA8C4] capitalize">{org.subscription_tier}</p>
              </div>
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                org.active_users > 0 ? 'bg-[#00A39B] text-white' : 'bg-[#1E2A3E] text-[#8FA8C4]'
              }`}>
                {org.active_users > 0 ? 'Active' : 'Inactive'}
              </span>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-[#8FA8C4]">Total Users:</span>
                <span className="text-[#D6E2F0]">{org.total_users}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-[#8FA8C4]">Active Now:</span>
                <span className="text-[#00A39B]">{org.active_users}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-[#8FA8C4]">Admin Connections:</span>
                <span className="text-[#D6E2F0]">{org.admin_connections}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );

  const renderConnections = () => (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold text-[#D6E2F0]">Active Connections</h3>

      <div className="overflow-x-auto">
        <table className="w-full bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
          <thead className="bg-[#1E2A3E]">
            <tr>
              <th className="px-4 py-3 text-left text-sm font-medium text-[#D6E2F0]">User ID</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-[#D6E2F0]">Tier</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-[#D6E2F0]">Type</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-[#D6E2F0]">Organization</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-[#D6E2F0]">Duration</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-[#D6E2F0]">Status</th>
            </tr>
          </thead>
          <tbody>
            {connections.map((conn, index) => (
              <tr key={conn.connection_id} className={index % 2 === 0 ? 'bg-[#0D1421]' : 'bg-[#1E2A3E]/30'}>
                <td className="px-4 py-3 text-sm text-[#D6E2F0]">{conn.user_id}</td>
                <td className="px-4 py-3 text-sm">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    conn.user_tier === 'admin' ? 'bg-[#FF4444] text-white' :
                    conn.user_tier === 'enterprise_user' ? 'bg-[#00A39B] text-white' :
                    'bg-[#FFB800] text-black'
                  }`}>
                    {conn.user_tier.replace('_', ' ').toUpperCase()}
                  </span>
                </td>
                <td className="px-4 py-3 text-sm text-[#8FA8C4] capitalize">
                  {conn.connection_type.replace('_', ' ')}
                </td>
                <td className="px-4 py-3 text-sm text-[#8FA8C4]">
                  {conn.organization_id || 'Personal'}
                </td>
                <td className="px-4 py-3 text-sm text-[#8FA8C4]">
                  {Math.floor(conn.connected_duration / 60)}m
                </td>
                <td className="px-4 py-3 text-sm">
                  <span className="w-2 h-2 bg-[#00A39B] rounded-full inline-block"></span>
                  <span className="ml-2 text-[#8FA8C4]">Active</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  const renderAnalytics = () => (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold text-[#D6E2F0]">System Analytics</h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Connection Distribution */}
        <div className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
          <h4 className="text-lg font-semibold text-[#D6E2F0] mb-4">Connection Distribution</h4>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-[#8FA8C4]">Individual Users (3002)</span>
              <div className="flex items-center space-x-2">
                <div className="w-20 bg-[#1E2A3E] rounded-full h-2">
                  <div className="bg-[#00A39B] h-2 rounded-full" style={{ width: '60%' }} />
                </div>
                <span className="text-[#D6E2F0] text-sm">
                  {connections.filter(c => c.connection_type === 'individual_ui').length}
                </span>
              </div>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-[#8FA8C4]">Admin Users (3001)</span>
              <div className="flex items-center space-x-2">
                <div className="w-20 bg-[#1E2A3E] rounded-full h-2">
                  <div className="bg-[#FFB800] h-2 rounded-full" style={{ width: '25%' }} />
                </div>
                <span className="text-[#D6E2F0] text-sm">
                  {connections.filter(c => c.connection_type === 'admin_ui').length}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* AGI Performance */}
        <div className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
          <h4 className="text-lg font-semibold text-[#D6E2F0] mb-4">AGI Performance</h4>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-[#8FA8C4]">System Load</span>
              <span className="text-[#00A39B]">{((systemMetrics?.system_load || 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#8FA8C4]">Quantum Coherence</span>
              <span className="text-[#00A39B]">{((systemMetrics?.quantum_coherence || 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#8FA8C4]">Active Agents</span>
              <span className="text-[#00A39B]">{systemMetrics?.active_agents || 0}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Memory Usage Chart */}
      <div className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
        <h4 className="text-lg font-semibold text-[#D6E2F0] mb-4">Neural Mesh Memory Usage</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {systemMetrics?.memory_usage && Object.entries(systemMetrics.memory_usage).map(([tier, usage]) => (
            <div key={tier} className="text-center">
              <div className="text-sm text-[#8FA8C4] mb-2">{tier} Memory</div>
              <div className="relative w-16 h-16 mx-auto">
                <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 36 36">
                  <path
                    d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                    fill="none"
                    stroke="#1E2A3E"
                    strokeWidth="2"
                  />
                  <path
                    d="m18,2.0845 a 15.9155,15.9155 0 0,1 0,31.831 a 15.9155,15.9155 0 0,1 0,-31.831"
                    fill="none"
                    stroke="#00A39B"
                    strokeWidth="2"
                    strokeDasharray={`${usage * 100}, 100`}
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-xs font-medium text-[#D6E2F0]">
                    {(usage * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#05080D] text-[#D6E2F0]">
      {/* Header */}
      <div className="border-b border-[#1E2A3E] bg-[#0A0F1C]">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-[#D6E2F0]">
                🚀 AgentForge Admin Dashboard
              </h1>
              <p className="text-[#8FA8C4]">
                Technical administration and system monitoring
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-[#8FA8C4]">
                Port 3001 - Admin Interface
              </div>
              <div className="w-2 h-2 bg-[#00A39B] rounded-full animate-pulse"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="border-b border-[#1E2A3E] bg-[#0A0F1C]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex space-x-8">
            {[
              { key: 'overview', icon: BarChart3, label: 'Overview' },
              { key: 'organizations', icon: Globe, label: 'Organizations' },
              { key: 'connections', icon: Users, label: 'Connections' },
              { key: 'analytics', icon: TrendingUp, label: 'Analytics' }
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as any)}
                className={`flex items-center space-x-2 px-4 py-3 border-b-2 transition-colors ${
                  activeTab === tab.key
                    ? 'border-[#00A39B] text-[#00A39B]'
                    : 'border-transparent text-[#8FA8C4] hover:text-[#D6E2F0]'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'organizations' && renderOrganizations()}
        {activeTab === 'connections' && renderConnections()}
        {activeTab === 'analytics' && renderAnalytics()}
      </div>
    </div>
  );
}
