'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Shield,
  Lock,
  Unlock,
  Key,
  Eye,
  EyeOff,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  Users,
  Globe,
  Server,
  FileText,
  Clock,
  Zap,
  Ban,
  UserX,
  Settings,
  Trash2
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Modal } from '@/components/ui/Modal';
import { Layout } from '@/components/layout/Layout';

// Mock security data
const securityAlerts = [
  {
    id: 'sec-001',
    severity: 'critical',
    title: 'Unauthorized Access Attempt',
    description: 'Multiple failed login attempts from IP 192.168.1.100',
    timestamp: new Date(Date.now() - 5 * 60000),
    source: 'Authentication System',
    status: 'active',
    actions: ['Block IP', 'Notify Admin', 'Increase Monitoring']
  },
  {
    id: 'sec-002',
    severity: 'warning',
    title: 'Unusual API Usage Pattern',
    description: 'API key sk-abc123 exceeded normal usage by 300%',
    timestamp: new Date(Date.now() - 15 * 60000),
    source: 'API Gateway',
    status: 'investigating',
    actions: ['Rate Limit', 'Contact User', 'Monitor Activity']
  },
  {
    id: 'sec-003',
    severity: 'info',
    title: 'Security Scan Completed',
    description: 'Automated vulnerability scan completed successfully',
    timestamp: new Date(Date.now() - 30 * 60000),
    source: 'Security Scanner',
    status: 'resolved',
    actions: ['Generate Report', 'Update Baselines']
  }
];

const accessLogs = [
  {
    id: 'log-001',
    timestamp: new Date(Date.now() - 2 * 60000),
    user: 'admin@agentforge.ai',
    action: 'Dashboard Access',
    resource: '/dashboard',
    ip: '10.0.1.45',
    status: 'success'
  },
  {
    id: 'log-002',
    timestamp: new Date(Date.now() - 5 * 60000),
    user: 'operator@agentforge.ai',
    action: 'Agent Deployment',
    resource: '/agents/deploy',
    ip: '10.0.1.67',
    status: 'success'
  },
  {
    id: 'log-003',
    timestamp: new Date(Date.now() - 8 * 60000),
    user: 'unknown',
    action: 'Failed Login',
    resource: '/auth/login',
    ip: '192.168.1.100',
    status: 'blocked'
  }
];

export default function SecurityPage() {
  const [selectedAlert, setSelectedAlert] = useState<typeof securityAlerts[0] | null>(null);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [alertFilter, setAlertFilter] = useState('all');

  const securityStats = {
    totalAlerts: securityAlerts.length,
    criticalAlerts: securityAlerts.filter(a => a.severity === 'critical').length,
    activeThreats: securityAlerts.filter(a => a.status === 'active').length,
    blockedIPs: 23,
    activeUsers: 12,
    failedLogins: 5,
    securityScore: 87
  };

  const filteredAlerts = securityAlerts.filter(alert => {
    if (alertFilter === 'all') return true;
    return alert.severity === alertFilter;
  });

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <Shield className="h-8 w-8 text-red-400" />
              Security Center
            </h1>
            <p className="text-sm opacity-70 mt-1">
              Monitor security threats, manage access, and protect your infrastructure
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge 
              variant={securityStats.securityScore > 80 ? 'success' : 'warning'}
              className="px-3 py-1"
            >
              Security Score: {securityStats.securityScore}%
            </Badge>
            <Button
              variant="primary"
              icon={<Key className="h-4 w-4" />}
              onClick={() => setShowApiKeyModal(true)}
            >
              Manage API Keys
            </Button>
          </div>
        </div>

        {/* Security Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <SecurityMetricCard
            title="Active Threats"
            value={securityStats.activeThreats}
            icon={<AlertTriangle className="h-6 w-6" />}
            color="red"
            status="critical"
          />
          <SecurityMetricCard
            title="Blocked IPs"
            value={securityStats.blockedIPs}
            icon={<Ban className="h-6 w-6" />}
            color="yellow"
            status="warning"
          />
          <SecurityMetricCard
            title="Active Users"
            value={securityStats.activeUsers}
            icon={<Users className="h-6 w-6" />}
            color="green"
            status="good"
          />
          <SecurityMetricCard
            title="Failed Logins"
            value={securityStats.failedLogins}
            icon={<UserX className="h-6 w-6" />}
            color="red"
            status="warning"
          />
        </div>

        {/* Security Status */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Security Score */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Security Score
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
                      stroke={securityStats.securityScore > 80 ? '#10B981' : '#F59E0B'}
                      strokeWidth="8"
                      fill="none"
                      strokeLinecap="round"
                      initial={{ strokeDasharray: 0, strokeDashoffset: 0 }}
                      animate={{
                        strokeDasharray: 2 * Math.PI * 40,
                        strokeDashoffset: 2 * Math.PI * 40 * (1 - securityStats.securityScore / 100)
                      }}
                      transition={{ duration: 1, ease: 'easeOut' }}
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-2xl font-bold">{securityStats.securityScore}%</span>
                  </div>
                </div>
                <Badge 
                  variant={securityStats.securityScore > 80 ? 'success' : 'warning'}
                  className="capitalize"
                >
                  {securityStats.securityScore > 80 ? 'Secure' : 'Needs Attention'}
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Security Checklist */}
          <Card>
            <CardHeader>
              <CardTitle>Security Checklist</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <SecurityCheckItem
                  label="TLS Encryption"
                  status="enabled"
                  description="All traffic encrypted"
                />
                <SecurityCheckItem
                  label="API Authentication"
                  status="enabled"
                  description="API keys required"
                />
                <SecurityCheckItem
                  label="Rate Limiting"
                  status="enabled"
                  description="1000 req/min limit"
                />
                <SecurityCheckItem
                  label="Access Logging"
                  status="enabled"
                  description="All access logged"
                />
                <SecurityCheckItem
                  label="Vulnerability Scanning"
                  status="warning"
                  description="Last scan: 2 days ago"
                />
              </div>
            </CardContent>
          </Card>

          {/* Threat Detection */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Threat Detection
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Intrusion Detection</span>
                  <Badge variant="success">Active</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Anomaly Detection</span>
                  <Badge variant="success">Active</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Behavioral Analysis</span>
                  <Badge variant="warning">Limited</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Threat Intelligence</span>
                  <Badge variant="success">Updated</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Security Alerts */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5" />
                Security Alerts ({filteredAlerts.length})
              </span>
              <select
                className="input w-auto"
                value={alertFilter}
                onChange={(e) => setAlertFilter(e.target.value)}
              >
                <option value="all">All Alerts</option>
                <option value="critical">Critical</option>
                <option value="warning">Warning</option>
                <option value="info">Info</option>
              </select>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {filteredAlerts.map((alert, index) => (
                <SecurityAlertCard
                  key={alert.id}
                  alert={alert}
                  index={index}
                  onSelect={() => setSelectedAlert(alert)}
                />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Access Logs */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Recent Access Logs
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {accessLogs.map((log, index) => (
                <AccessLogItem key={log.id} log={log} index={index} />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Alert Details Modal */}
        <Modal
          isOpen={!!selectedAlert}
          onClose={() => setSelectedAlert(null)}
          title={selectedAlert?.title}
          size="lg"
        >
          {selectedAlert && <SecurityAlertDetails alert={selectedAlert} />}
        </Modal>

        {/* API Key Management Modal */}
        <Modal
          isOpen={showApiKeyModal}
          onClose={() => setShowApiKeyModal(false)}
          title="API Key Management"
          size="lg"
        >
          <ApiKeyManagement onClose={() => setShowApiKeyModal(false)} />
        </Modal>
      </div>
    </Layout>
  );
}

function SecurityMetricCard({ title, value, icon, color, status }: {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: 'red' | 'yellow' | 'green' | 'blue';
  status: 'critical' | 'warning' | 'good';
}) {
  const colorClasses = {
    red: 'text-red-400',
    yellow: 'text-yellow-400',
    green: 'text-green-400',
    blue: 'text-blue-400'
  };

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="label text-xs">{title}</p>
            <p className="text-xl font-bold mt-1">{value}</p>
          </div>
          <div className={colorClasses[color]}>
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SecurityCheckItem({ label, status, description }: {
  label: string;
  status: 'enabled' | 'disabled' | 'warning';
  description: string;
}) {
  const statusConfig = {
    enabled: { icon: CheckCircle, color: 'text-green-400', badge: 'success' },
    disabled: { icon: XCircle, color: 'text-red-400', badge: 'danger' },
    warning: { icon: AlertTriangle, color: 'text-yellow-400', badge: 'warning' }
  };

  const config = statusConfig[status];

  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <config.icon className={`h-4 w-4 ${config.color}`} />
        <div>
          <p className="text-sm font-medium">{label}</p>
          <p className="text-xs opacity-60">{description}</p>
        </div>
      </div>
      <Badge variant={config.badge as any} size="sm">
        {status}
      </Badge>
    </div>
  );
}

function SecurityAlertCard({ alert, index, onSelect }: {
  alert: typeof securityAlerts[0];
  index: number;
  onSelect: () => void;
}) {
  const severityConfig = {
    critical: { icon: XCircle, color: 'text-red-400', bg: 'bg-red-400/10', badge: 'danger' },
    warning: { icon: AlertTriangle, color: 'text-yellow-400', bg: 'bg-yellow-400/10', badge: 'warning' },
    info: { icon: CheckCircle, color: 'text-blue-400', bg: 'bg-blue-400/10', badge: 'info' }
  };

  const config = severityConfig[alert.severity as keyof typeof severityConfig];

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      className={`p-4 rounded-lg border border-white/10 dark:border-red-900/40 cursor-pointer hover:bg-white/10 dark:hover:bg-black/30 ${config.bg}`}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <config.icon className={`h-5 w-5 mt-0.5 ${config.color}`} />
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h3 className="font-medium text-sm">{alert.title}</h3>
              <Badge variant={config.badge as any} size="sm">
                {alert.severity}
              </Badge>
            </div>
            <p className="text-sm opacity-80 mb-2">{alert.description}</p>
            <div className="flex items-center gap-4 text-xs opacity-60">
              <span>Source: {alert.source}</span>
              <span>{alert.timestamp.toLocaleTimeString()}</span>
              <span>Status: {alert.status}</span>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function AccessLogItem({ log, index }: {
  log: typeof accessLogs[0];
  index: number;
}) {
  const statusConfig = {
    success: { color: 'text-green-400', bg: 'bg-green-400/10' },
    blocked: { color: 'text-red-400', bg: 'bg-red-400/10' },
    failed: { color: 'text-yellow-400', bg: 'bg-yellow-400/10' }
  };

  const config = statusConfig[log.status as keyof typeof statusConfig];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, delay: index * 0.05 }}
      className={`p-3 rounded-lg ${config.bg} border border-white/10 dark:border-red-900/40`}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-1">
            <p className="text-sm font-medium">{log.action}</p>
            <Badge variant={log.status === 'success' ? 'success' : 'danger'} size="sm">
              {log.status}
            </Badge>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs opacity-70">
            <span>User: {log.user}</span>
            <span>Resource: {log.resource}</span>
            <span>IP: {log.ip}</span>
            <span>{log.timestamp.toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function SecurityAlertDetails({ alert }: { alert: typeof securityAlerts[0] }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="label text-sm mb-3">Alert Information</h3>
          <div className="space-y-3">
            <div>
              <p className="label text-xs">Alert ID</p>
              <p className="font-mono text-sm">{alert.id}</p>
            </div>
            <div>
              <p className="label text-xs">Severity</p>
              <Badge variant="danger" size="sm">{alert.severity}</Badge>
            </div>
            <div>
              <p className="label text-xs">Source</p>
              <p className="text-sm">{alert.source}</p>
            </div>
            <div>
              <p className="label text-xs">Status</p>
              <Badge variant="warning" size="sm">{alert.status}</Badge>
            </div>
          </div>
        </div>

        <div>
          <h3 className="label text-sm mb-3">Recommended Actions</h3>
          <div className="space-y-2">
            {alert.actions.map((action, index) => (
              <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-white/5 dark:bg-black/20">
                <span className="text-sm">{action}</span>
                <Button size="sm" variant="ghost">
                  Execute
                </Button>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="flex gap-3">
        <Button variant="primary" icon={<CheckCircle className="h-4 w-4" />}>
          Mark Resolved
        </Button>
        <Button variant="secondary" icon={<Eye className="h-4 w-4" />}>
          Investigate
        </Button>
        <Button variant="danger" icon={<Ban className="h-4 w-4" />}>
          Block Source
        </Button>
      </div>
    </div>
  );
}

function ApiKeyManagement({ onClose }: { onClose: () => void }) {
  const [showKeys, setShowKeys] = useState(false);

  const apiKeys = [
    { id: 'key-001', name: 'Production API', key: 'sk-prod-abc123...', created: '2024-01-10', lastUsed: '2 min ago', status: 'active' },
    { id: 'key-002', name: 'Development API', key: 'sk-dev-def456...', created: '2024-01-08', lastUsed: '1 hour ago', status: 'active' },
    { id: 'key-003', name: 'Analytics API', key: 'sk-analytics-ghi789...', created: '2024-01-05', lastUsed: 'Never', status: 'inactive' }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="label text-sm">API Keys ({apiKeys.length})</h3>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            icon={showKeys ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            onClick={() => setShowKeys(!showKeys)}
          >
            {showKeys ? 'Hide' : 'Show'} Keys
          </Button>
          <Button variant="primary" size="sm" icon={<Key className="h-4 w-4" />}>
            Generate New
          </Button>
        </div>
      </div>

      <div className="space-y-3">
        {apiKeys.map((key) => (
          <div key={key.id} className="p-4 rounded-lg bg-white/5 dark:bg-black/20 border border-white/10 dark:border-red-900/40">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <h4 className="font-medium text-sm">{key.name}</h4>
                  <Badge variant={key.status === 'active' ? 'success' : 'warning'} size="sm">
                    {key.status}
                  </Badge>
                </div>
                <div className="grid grid-cols-3 gap-4 text-xs opacity-70">
                  <span>Created: {key.created}</span>
                  <span>Last used: {key.lastUsed}</span>
                  <span>Key: {showKeys ? key.key : '••••••••••••••••'}</span>
                </div>
              </div>
              <div className="flex gap-2">
                <Button size="sm" variant="ghost" icon={<Settings className="h-3 w-3" />} />
                <Button size="sm" variant="ghost" icon={<Trash2 className="h-3 w-3" />} />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="flex gap-3 pt-4">
        <Button variant="ghost" onClick={onClose}>
          Close
        </Button>
      </div>
    </div>
  );
}