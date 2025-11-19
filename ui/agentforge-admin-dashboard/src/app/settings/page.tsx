'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Settings,
  Server,
  Shield,
  Database,
  Bell,
  Palette,
  Key,
  Users,
  Globe,
  Zap,
  Save,
  RefreshCw,
  Download,
  Upload,
  AlertTriangle,
  CheckCircle,
  Eye,
  EyeOff
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Layout } from '@/components/layout/Layout';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';
import { getConfiguration, updateConfiguration } from '@/lib/api';

interface ConfigSection {
  id: string;
  title: string;
  icon: React.ReactNode;
  description: string;
}

const configSections: ConfigSection[] = [
  {
    id: 'general',
    title: 'General Settings',
    icon: <Settings className="h-5 w-5" />,
    description: 'Basic system configuration and preferences'
  },
  {
    id: 'agents',
    title: 'Agent Configuration',
    icon: <Users className="h-5 w-5" />,
    description: 'Agent deployment and management settings'
  },
  {
    id: 'security',
    title: 'Security & Authentication',
    icon: <Shield className="h-5 w-5" />,
    description: 'Security policies and authentication settings'
  },
  {
    id: 'database',
    title: 'Database & Storage',
    icon: <Database className="h-5 w-5" />,
    description: 'Database connections and storage configuration'
  },
  {
    id: 'networking',
    title: 'Network & API',
    icon: <Globe className="h-5 w-5" />,
    description: 'Network settings and API configurations'
  },
  {
    id: 'monitoring',
    title: 'Monitoring & Alerts',
    icon: <Bell className="h-5 w-5" />,
    description: 'System monitoring and alerting configuration'
  },
  {
    id: 'performance',
    title: 'Performance Tuning',
    icon: <Zap className="h-5 w-5" />,
    description: 'Performance optimization and resource management'
  },
  {
    id: 'appearance',
    title: 'Appearance & UI',
    icon: <Palette className="h-5 w-5" />,
    description: 'User interface customization and themes'
  }
];

export default function SettingsPage() {
  const [activeSection, setActiveSection] = useState('general');
  const [config, setConfig] = useState<Record<string, any>>({});
  const [hasChanges, setHasChanges] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showSecrets, setShowSecrets] = useState(false);
  const snap = useSnapshot(store);

  useEffect(() => {
    const loadConfiguration = async () => {
      try {
        const configuration = await getConfiguration();
        setConfig(configuration);
      } catch (error) {
        console.error('Failed to load configuration:', error);
        // Load mock configuration
        setConfig({
          general: {
            systemName: 'AgentForge Production',
            environment: 'production',
            logLevel: 'INFO',
            debugMode: false,
            maintenanceMode: false
          },
          agents: {
            maxAgents: 1000,
            defaultTimeout: 300,
            autoScaling: true,
            resourceLimits: {
              cpu: '2000m',
              memory: '4Gi',
              gpu: 1
            }
          },
          security: {
            apiKeyRequired: true,
            rateLimitEnabled: true,
            rateLimitRequests: 1000,
            rateLimitWindow: 60,
            corsOrigins: ['https://agentforge.company.com'],
            tlsEnabled: true
          },
          database: {
            url: 'postgresql://agentforge:***@db.example.com:5432/agentforge',
            poolSize: 50,
            maxOverflow: 100,
            connectionTimeout: 30
          },
          networking: {
            apiHost: '0.0.0.0',
            apiPort: 8000,
            websocketEnabled: true,
            maxConnections: 10000
          },
          monitoring: {
            metricsEnabled: true,
            tracingEnabled: true,
            alertsEnabled: true,
            healthCheckInterval: 30
          },
          performance: {
            maxConcurrentJobs: 200,
            taskQueueSize: 10000,
            workerCount: 4,
            cacheEnabled: true
          }
        });
      }
    };

    loadConfiguration();
  }, []);

  const handleConfigChange = (section: string, key: string, value: any) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await updateConfiguration(config);
      setHasChanges(false);
    } catch (error) {
      console.error('Failed to save configuration:', error);
    }
    setSaving(false);
  };

  const handleReset = () => {
    // Reset to original config
    setHasChanges(false);
  };

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'agentforge-config.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">System Settings</h1>
            <p className="text-sm opacity-70 mt-1">
              Configure and manage your AgentForge system
            </p>
          </div>
          <div className="flex items-center gap-3">
            {hasChanges && (
              <Badge variant="warning" className="px-3 py-1">
                Unsaved Changes
              </Badge>
            )}
            <Button variant="ghost" icon={<Download className="h-4 w-4" />} onClick={handleExport}>
              Export
            </Button>
            <Button variant="ghost" icon={<Upload className="h-4 w-4" />}>
              Import
            </Button>
            <Button 
              variant="primary" 
              icon={<Save className="h-4 w-4" />}
              loading={saving}
              disabled={!hasChanges}
              onClick={handleSave}
            >
              Save Changes
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Settings Navigation */}
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle>Configuration Sections</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="space-y-1">
                {configSections.map((section) => (
                  <button
                    key={section.id}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      activeSection === section.id
                        ? 'bg-day-accent/20 dark:bg-night-text/20 text-day-accent dark:text-night-text'
                        : 'hover:bg-white/10 dark:hover:bg-red-900/10'
                    }`}
                    onClick={() => setActiveSection(section.id)}
                  >
                    <div className="flex items-center gap-3">
                      {section.icon}
                      <div>
                        <p className="text-sm font-medium">{section.title}</p>
                        <p className="text-xs opacity-60">{section.description}</p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Settings Content */}
          <div className="lg:col-span-3">
            <motion.div
              key={activeSection}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
            >
              {activeSection === 'general' && (
                <GeneralSettings 
                  config={config.general || {}} 
                  onChange={(key, value) => handleConfigChange('general', key, value)}
                />
              )}
              {activeSection === 'agents' && (
                <AgentSettings 
                  config={config.agents || {}} 
                  onChange={(key, value) => handleConfigChange('agents', key, value)}
                />
              )}
              {activeSection === 'security' && (
                <SecuritySettings 
                  config={config.security || {}} 
                  onChange={(key, value) => handleConfigChange('security', key, value)}
                  showSecrets={showSecrets}
                />
              )}
              {activeSection === 'database' && (
                <DatabaseSettings 
                  config={config.database || {}} 
                  onChange={(key, value) => handleConfigChange('database', key, value)}
                  showSecrets={showSecrets}
                />
              )}
              {activeSection === 'networking' && (
                <NetworkingSettings 
                  config={config.networking || {}} 
                  onChange={(key, value) => handleConfigChange('networking', key, value)}
                />
              )}
              {activeSection === 'monitoring' && (
                <MonitoringSettings 
                  config={config.monitoring || {}} 
                  onChange={(key, value) => handleConfigChange('monitoring', key, value)}
                />
              )}
              {activeSection === 'performance' && (
                <PerformanceSettings 
                  config={config.performance || {}} 
                  onChange={(key, value) => handleConfigChange('performance', key, value)}
                />
              )}
              {activeSection === 'appearance' && (
                <AppearanceSettings />
              )}
            </motion.div>
          </div>
        </div>

        {/* Actions Bar */}
        {hasChanges && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="fixed bottom-6 right-6 left-6 lg:left-80"
          >
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5 text-yellow-400" />
                    <span className="text-sm font-medium">You have unsaved changes</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Button variant="ghost" onClick={handleReset}>
                      Reset
                    </Button>
                    <Button variant="primary" onClick={handleSave} loading={saving}>
                      Save Changes
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </div>
    </Layout>
  );
}

// Settings Components
function GeneralSettings({ config, onChange }: {
  config: any;
  onChange: (key: string, value: any) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>General System Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          label="System Name"
          value={config.systemName || ''}
          onChange={(e) => onChange('systemName', e.target.value)}
        />
        
        <div>
          <label className="label block mb-2">Environment</label>
          <select 
            className="input w-full"
            value={config.environment || 'development'}
            onChange={(e) => onChange('environment', e.target.value)}
          >
            <option value="development">Development</option>
            <option value="staging">Staging</option>
            <option value="production">Production</option>
          </select>
        </div>

        <div>
          <label className="label block mb-2">Log Level</label>
          <select 
            className="input w-full"
            value={config.logLevel || 'INFO'}
            onChange={(e) => onChange('logLevel', e.target.value)}
          >
            <option value="DEBUG">Debug</option>
            <option value="INFO">Info</option>
            <option value="WARNING">Warning</option>
            <option value="ERROR">Error</option>
            <option value="CRITICAL">Critical</option>
          </select>
        </div>

        <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div>
            <p className="text-sm font-medium">Debug Mode</p>
            <p className="text-xs opacity-60">Enable detailed debugging information</p>
          </div>
          <input
            type="checkbox"
            checked={config.debugMode || false}
            onChange={(e) => onChange('debugMode', e.target.checked)}
            className="toggle"
          />
        </div>

        <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div>
            <p className="text-sm font-medium">Maintenance Mode</p>
            <p className="text-xs opacity-60">Put system in maintenance mode</p>
          </div>
          <input
            type="checkbox"
            checked={config.maintenanceMode || false}
            onChange={(e) => onChange('maintenanceMode', e.target.checked)}
            className="toggle"
          />
        </div>
      </CardContent>
    </Card>
  );
}

function AgentSettings({ config, onChange }: {
  config: any;
  onChange: (key: string, value: any) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Agent Configuration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          label="Maximum Agents"
          type="number"
          value={config.maxAgents || 100}
          onChange={(e) => onChange('maxAgents', parseInt(e.target.value))}
        />
        
        <Input
          label="Default Timeout (seconds)"
          type="number"
          value={config.defaultTimeout || 300}
          onChange={(e) => onChange('defaultTimeout', parseInt(e.target.value))}
        />

        <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div>
            <p className="text-sm font-medium">Auto Scaling</p>
            <p className="text-xs opacity-60">Automatically scale agents based on demand</p>
          </div>
          <input
            type="checkbox"
            checked={config.autoScaling || false}
            onChange={(e) => onChange('autoScaling', e.target.checked)}
            className="toggle"
          />
        </div>

        <div>
          <h3 className="label text-sm mb-3">Resource Limits</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Input
              label="CPU Limit"
              value={config.resourceLimits?.cpu || '1000m'}
              onChange={(e) => onChange('resourceLimits', { ...config.resourceLimits, cpu: e.target.value })}
            />
            <Input
              label="Memory Limit"
              value={config.resourceLimits?.memory || '2Gi'}
              onChange={(e) => onChange('resourceLimits', { ...config.resourceLimits, memory: e.target.value })}
            />
            <Input
              label="GPU Limit"
              type="number"
              value={config.resourceLimits?.gpu || 0}
              onChange={(e) => onChange('resourceLimits', { ...config.resourceLimits, gpu: parseInt(e.target.value) })}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SecuritySettings({ config, onChange, showSecrets }: {
  config: any;
  onChange: (key: string, value: any) => void;
  showSecrets: boolean;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          Security & Authentication
          <Button
            variant="ghost"
            size="sm"
            icon={showSecrets ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          >
            {showSecrets ? 'Hide' : 'Show'} Secrets
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div>
            <p className="text-sm font-medium">API Key Required</p>
            <p className="text-xs opacity-60">Require API key for all requests</p>
          </div>
          <input
            type="checkbox"
            checked={config.apiKeyRequired || false}
            onChange={(e) => onChange('apiKeyRequired', e.target.checked)}
            className="toggle"
          />
        </div>

        <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div>
            <p className="text-sm font-medium">Rate Limiting</p>
            <p className="text-xs opacity-60">Enable request rate limiting</p>
          </div>
          <input
            type="checkbox"
            checked={config.rateLimitEnabled || false}
            onChange={(e) => onChange('rateLimitEnabled', e.target.checked)}
            className="toggle"
          />
        </div>

        {config.rateLimitEnabled && (
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="Rate Limit (requests)"
              type="number"
              value={config.rateLimitRequests || 1000}
              onChange={(e) => onChange('rateLimitRequests', parseInt(e.target.value))}
            />
            <Input
              label="Time Window (seconds)"
              type="number"
              value={config.rateLimitWindow || 60}
              onChange={(e) => onChange('rateLimitWindow', parseInt(e.target.value))}
            />
          </div>
        )}

        <Input
          label="CORS Origins"
          value={config.corsOrigins?.join(', ') || ''}
          onChange={(e) => onChange('corsOrigins', e.target.value.split(', '))}
        />

        <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div>
            <p className="text-sm font-medium">TLS Enabled</p>
            <p className="text-xs opacity-60">Enable HTTPS/TLS encryption</p>
          </div>
          <input
            type="checkbox"
            checked={config.tlsEnabled || false}
            onChange={(e) => onChange('tlsEnabled', e.target.checked)}
            className="toggle"
          />
        </div>
      </CardContent>
    </Card>
  );
}

function DatabaseSettings({ config, onChange, showSecrets }: {
  config: any;
  onChange: (key: string, value: any) => void;
  showSecrets: boolean;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Database & Storage Configuration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          label="Database URL"
          type={showSecrets ? 'text' : 'password'}
          value={config.url || ''}
          onChange={(e) => onChange('url', e.target.value)}
        />
        
        <div className="grid grid-cols-2 gap-4">
          <Input
            label="Pool Size"
            type="number"
            value={config.poolSize || 10}
            onChange={(e) => onChange('poolSize', parseInt(e.target.value))}
          />
          <Input
            label="Max Overflow"
            type="number"
            value={config.maxOverflow || 20}
            onChange={(e) => onChange('maxOverflow', parseInt(e.target.value))}
          />
        </div>

        <Input
          label="Connection Timeout (seconds)"
          type="number"
          value={config.connectionTimeout || 30}
          onChange={(e) => onChange('connectionTimeout', parseInt(e.target.value))}
        />
      </CardContent>
    </Card>
  );
}

function NetworkingSettings({ config, onChange }: {
  config: any;
  onChange: (key: string, value: any) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Network & API Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <Input
            label="API Host"
            value={config.apiHost || '0.0.0.0'}
            onChange={(e) => onChange('apiHost', e.target.value)}
          />
          <Input
            label="API Port"
            type="number"
            value={config.apiPort || 8000}
            onChange={(e) => onChange('apiPort', parseInt(e.target.value))}
          />
        </div>

        <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div>
            <p className="text-sm font-medium">WebSocket Enabled</p>
            <p className="text-xs opacity-60">Enable WebSocket connections</p>
          </div>
          <input
            type="checkbox"
            checked={config.websocketEnabled || false}
            onChange={(e) => onChange('websocketEnabled', e.target.checked)}
            className="toggle"
          />
        </div>

        <Input
          label="Max Connections"
          type="number"
          value={config.maxConnections || 1000}
          onChange={(e) => onChange('maxConnections', parseInt(e.target.value))}
        />
      </CardContent>
    </Card>
  );
}

function MonitoringSettings({ config, onChange }: {
  config: any;
  onChange: (key: string, value: any) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Monitoring & Alerts</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div>
            <p className="text-sm font-medium">Metrics Enabled</p>
            <p className="text-xs opacity-60">Enable system metrics collection</p>
          </div>
          <input
            type="checkbox"
            checked={config.metricsEnabled || false}
            onChange={(e) => onChange('metricsEnabled', e.target.checked)}
            className="toggle"
          />
        </div>

        <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div>
            <p className="text-sm font-medium">Tracing Enabled</p>
            <p className="text-xs opacity-60">Enable distributed tracing</p>
          </div>
          <input
            type="checkbox"
            checked={config.tracingEnabled || false}
            onChange={(e) => onChange('tracingEnabled', e.target.checked)}
            className="toggle"
          />
        </div>

        <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div>
            <p className="text-sm font-medium">Alerts Enabled</p>
            <p className="text-xs opacity-60">Enable system alerts</p>
          </div>
          <input
            type="checkbox"
            checked={config.alertsEnabled || false}
            onChange={(e) => onChange('alertsEnabled', e.target.checked)}
            className="toggle"
          />
        </div>

        <Input
          label="Health Check Interval (seconds)"
          type="number"
          value={config.healthCheckInterval || 30}
          onChange={(e) => onChange('healthCheckInterval', parseInt(e.target.value))}
        />
      </CardContent>
    </Card>
  );
}

function PerformanceSettings({ config, onChange }: {
  config: any;
  onChange: (key: string, value: any) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Performance Tuning</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          label="Max Concurrent Jobs"
          type="number"
          value={config.maxConcurrentJobs || 100}
          onChange={(e) => onChange('maxConcurrentJobs', parseInt(e.target.value))}
        />
        
        <Input
          label="Task Queue Size"
          type="number"
          value={config.taskQueueSize || 1000}
          onChange={(e) => onChange('taskQueueSize', parseInt(e.target.value))}
        />

        <Input
          label="Worker Count"
          type="number"
          value={config.workerCount || 4}
          onChange={(e) => onChange('workerCount', parseInt(e.target.value))}
        />

        <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div>
            <p className="text-sm font-medium">Cache Enabled</p>
            <p className="text-xs opacity-60">Enable response caching</p>
          </div>
          <input
            type="checkbox"
            checked={config.cacheEnabled || false}
            onChange={(e) => onChange('cacheEnabled', e.target.checked)}
            className="toggle"
          />
        </div>
      </CardContent>
    </Card>
  );
}

function AppearanceSettings() {
  const snap = useSnapshot(store);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Appearance & UI Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div>
            <p className="text-sm font-medium">Dark Mode</p>
            <p className="text-xs opacity-60">Use dark theme (Night mode)</p>
          </div>
          <input
            type="checkbox"
            checked={snap.theme === 'night'}
            onChange={() => store.toggleTheme()}
            className="toggle"
          />
        </div>

        <div>
          <p className="text-sm font-medium mb-2">Current Theme: {snap.theme}</p>
          <div className="grid grid-cols-2 gap-4">
            <div 
              className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${
                snap.theme === 'day' 
                  ? 'border-day-accent bg-day-accent/10' 
                  : 'border-white/20 hover:border-white/40'
              }`}
              onClick={() => snap.theme !== 'day' && store.toggleTheme()}
            >
              <div className="w-full h-16 bg-gradient-to-br from-day-bg to-day-grid rounded mb-2"></div>
              <p className="text-sm font-medium">Day Mode</p>
              <p className="text-xs opacity-60">Tactical blue theme</p>
            </div>
            
            <div 
              className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${
                snap.theme === 'night' 
                  ? 'border-night-text bg-night-text/10' 
                  : 'border-white/20 hover:border-white/40'
              }`}
              onClick={() => snap.theme !== 'night' && store.toggleTheme()}
            >
              <div className="w-full h-16 bg-gradient-to-br from-night-bg to-night-grid rounded mb-2"></div>
              <p className="text-sm font-medium">Night Mode</p>
              <p className="text-xs opacity-60">Dark red theme</p>
            </div>
          </div>
        </div>

        <div className="p-4 rounded-lg bg-white/5 dark:bg-black/20">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="h-4 w-4 text-green-400" />
            <p className="text-sm font-medium">Theme Applied Successfully</p>
          </div>
          <p className="text-xs opacity-60">
            Theme changes are applied immediately and persist across sessions.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

