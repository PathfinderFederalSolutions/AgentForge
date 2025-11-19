'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  BookOpen,
  Home,
  Bot,
  Briefcase,
  Users,
  Brain,
  Zap,
  Activity,
  BarChart3,
  Network,
  Terminal,
  Database,
  Shield,
  Settings,
  ChevronRight,
  Play,
  Lightbulb,
  Target,
  Rocket,
  CheckCircle
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Layout } from '@/components/layout/Layout';

const instructionSections = [
  {
    id: 'overview',
    title: 'Platform Overview',
    icon: <Home className="h-5 w-5" />,
    description: 'Understanding AgentForge and its capabilities'
  },
  {
    id: 'dashboard',
    title: 'Mission Control Dashboard',
    icon: <Home className="h-5 w-5" />,
    description: 'Your central command center for monitoring and control'
  },
  {
    id: 'agents',
    title: 'Agent Management',
    icon: <Bot className="h-5 w-5" />,
    description: 'Deploy, monitor, and control your AI agents'
  },
  {
    id: 'jobs',
    title: 'Job Management',
    icon: <Briefcase className="h-5 w-5" />,
    description: 'Create, track, and manage computational tasks'
  },
  {
    id: 'swarm',
    title: 'Swarm Control',
    icon: <Users className="h-5 w-5" />,
    description: 'Orchestrate distributed agent clusters'
  },
  {
    id: 'neural-mesh',
    title: 'Neural Mesh',
    icon: <Brain className="h-5 w-5" />,
    description: 'Advanced AI memory and knowledge management'
  },
  {
    id: 'quantum',
    title: 'Quantum Scheduler',
    icon: <Zap className="h-5 w-5" />,
    description: 'Million-scale agent coordination using quantum principles'
  },
  {
    id: 'monitoring',
    title: 'System Monitoring',
    icon: <Activity className="h-5 w-5" />,
    description: 'Real-time system health and performance monitoring'
  },
  {
    id: 'analytics',
    title: 'Analytics & Insights',
    icon: <BarChart3 className="h-5 w-5" />,
    description: 'Performance analytics and operational insights'
  },
  {
    id: 'network',
    title: 'Network Topology',
    icon: <Network className="h-5 w-5" />,
    description: 'Monitor and manage network infrastructure'
  },
  {
    id: 'data',
    title: 'Data Management',
    icon: <Database className="h-5 w-5" />,
    description: 'Manage data sources and processing pipelines'
  },
  {
    id: 'security',
    title: 'Security Center',
    icon: <Shield className="h-5 w-5" />,
    description: 'Monitor threats and manage security policies'
  },
  {
    id: 'terminal',
    title: 'System Terminal',
    icon: <Terminal className="h-5 w-5" />,
    description: 'Command-line interface for advanced operations'
  },
  {
    id: 'settings',
    title: 'System Settings',
    icon: <Settings className="h-5 w-5" />,
    description: 'Configure and customize your AgentForge system'
  }
];

const quickStartGuide = [
  {
    step: 1,
    title: 'Connect to Your System',
    description: 'Ensure your WebSocket connection is active (green indicator in header)',
    action: 'Check connection status in the header'
  },
  {
    step: 2,
    title: 'Deploy Your First Agent',
    description: 'Navigate to Agents page and deploy a neural mesh agent',
    action: 'Click "Deploy Agent" button'
  },
  {
    step: 3,
    title: 'Create a Job',
    description: 'Go to Jobs page and create your first computational task',
    action: 'Click "Create Job" and fill in details'
  },
  {
    step: 4,
    title: 'Monitor Performance',
    description: 'Use the Dashboard and Monitoring pages to track system performance',
    action: 'Review metrics and alerts'
  },
  {
    step: 5,
    title: 'Scale Your Swarm',
    description: 'Use Swarm Control to scale your agent clusters based on demand',
    action: 'Adjust cluster sizes as needed'
  }
];

export default function InstructionsPage() {
  const [activeSection, setActiveSection] = useState('overview');

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <BookOpen className="h-8 w-8 text-blue-400" />
              User Guide & Instructions
            </h1>
            <p className="text-sm opacity-70 mt-1">
              Complete guide to using AgentForge for maximum productivity
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="info" className="px-3 py-1">
              Version 2.1.0
            </Badge>
            <Button variant="primary" icon={<Rocket className="h-4 w-4" />}>
              Quick Start
            </Button>
          </div>
        </div>

        {/* Quick Start Guide */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Rocket className="h-5 w-5" />
              Quick Start Guide
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
              {quickStartGuide.map((step, index) => (
                <QuickStartStep key={step.step} step={step} index={index} />
              ))}
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Navigation */}
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle>Sections</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="space-y-1">
                {instructionSections.map((section) => (
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

          {/* Content */}
          <div className="lg:col-span-3">
            <motion.div
              key={activeSection}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
            >
              {activeSection === 'overview' && <OverviewSection />}
              {activeSection === 'dashboard' && <DashboardSection />}
              {activeSection === 'agents' && <AgentsSection />}
              {activeSection === 'jobs' && <JobsSection />}
              {activeSection === 'swarm' && <SwarmSection />}
              {activeSection === 'neural-mesh' && <NeuralMeshSection />}
              {activeSection === 'quantum' && <QuantumSection />}
              {activeSection === 'monitoring' && <MonitoringSection />}
              {activeSection === 'analytics' && <AnalyticsSection />}
              {activeSection === 'network' && <NetworkSection />}
              {activeSection === 'data' && <DataSection />}
              {activeSection === 'security' && <SecuritySection />}
              {activeSection === 'terminal' && <TerminalSection />}
              {activeSection === 'settings' && <SettingsSection />}
            </motion.div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

function QuickStartStep({ step, index }: {
  step: typeof quickStartGuide[0];
  index: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      className="text-center p-4 rounded-lg bg-white/5 dark:bg-black/20 border border-white/10 dark:border-red-900/40"
    >
      <div className="w-8 h-8 rounded-full bg-day-accent dark:bg-night-text text-white dark:text-black flex items-center justify-center font-bold text-sm mx-auto mb-3">
        {step.step}
      </div>
      <h3 className="font-medium text-sm mb-2">{step.title}</h3>
      <p className="text-xs opacity-70 mb-3">{step.description}</p>
      <Badge variant="info" size="sm" className="text-xs">
        {step.action}
      </Badge>
    </motion.div>
  );
}

// Instruction sections
function OverviewSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Target className="h-5 w-5" />
          AgentForge Platform Overview
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="font-semibold text-lg mb-3">What is AgentForge?</h3>
          <p className="text-sm opacity-80 leading-relaxed">
            AgentForge is a cutting-edge AI agent orchestration platform that enables you to deploy, 
            manage, and coordinate thousands of AI agents across distributed infrastructure. It combines 
            advanced neural mesh memory systems, quantum-inspired scheduling, and real-time monitoring 
            to provide unparalleled AI computational capabilities.
          </p>
        </div>

        <div>
          <h3 className="font-semibold text-lg mb-3">Key Capabilities</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FeatureCard
              icon={<Bot className="h-5 w-5" />}
              title="AI Agent Management"
              description="Deploy and manage thousands of specialized AI agents"
            />
            <FeatureCard
              icon={<Brain className="h-5 w-5" />}
              title="Neural Mesh Memory"
              description="Advanced AI memory system with 15,000+ interconnected nodes"
            />
            <FeatureCard
              icon={<Zap className="h-5 w-5" />}
              title="Quantum Scheduling"
              description="Million-scale task coordination using quantum computing principles"
            />
            <FeatureCard
              icon={<Activity className="h-5 w-5" />}
              title="Real-time Monitoring"
              description="Comprehensive system monitoring and performance analytics"
            />
          </div>
        </div>

        <div className="p-4 rounded-lg bg-day-accent/10 dark:bg-night-text/10 border border-day-accent/20 dark:border-night-text/20">
          <div className="flex items-start gap-3">
            <Lightbulb className="h-5 w-5 text-day-accent dark:text-night-text mt-0.5" />
            <div>
              <h4 className="font-medium text-sm mb-1">Pro Tip</h4>
              <p className="text-sm opacity-80">
                Start with the Dashboard to get an overview of your system, then explore specific 
                features based on your needs. The system is designed to be intuitive - most 
                actions are just a click away!
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function DashboardSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Home className="h-5 w-5" />
          Mission Control Dashboard
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="font-semibold text-lg mb-3">Dashboard Features</h3>
          <div className="space-y-3">
            <InstructionItem
              title="System Health Monitoring"
              description="Real-time health indicators show the status of all system components including Neural Mesh, Quantum Scheduler, and Agent Swarm."
              tips={['Green indicators mean systems are operational', 'Yellow indicates degraded performance', 'Red means immediate attention required']}
            />
            <InstructionItem
              title="Resource Usage Tracking"
              description="Monitor CPU, Memory, GPU, Network, and Storage usage across your infrastructure."
              tips={['Usage bars change color based on thresholds', 'Click on metrics for detailed views', 'Set up alerts for high usage']}
            />
            <InstructionItem
              title="Quick Actions Panel"
              description="Access frequently used functions like deploying agents, creating jobs, and accessing specialized interfaces."
              tips={['Use quick actions for common tasks', 'Each action opens the relevant specialized interface', 'Actions are context-aware based on system state']}
            />
          </div>
        </div>

        <div className="p-4 rounded-lg bg-green-400/10 border border-green-400/20">
          <div className="flex items-start gap-3">
            <CheckCircle className="h-5 w-5 text-green-400 mt-0.5" />
            <div>
              <h4 className="font-medium text-sm mb-1">Best Practice</h4>
              <p className="text-sm opacity-80">
                Check the dashboard daily to ensure all systems are running optimally. 
                Pay special attention to the health indicators and resource usage metrics.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function AgentsSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bot className="h-5 w-5" />
          Agent Management Guide
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="font-semibold text-lg mb-3">Managing Your AI Agents</h3>
          <div className="space-y-3">
            <InstructionItem
              title="Deploying New Agents"
              description="Click 'Deploy Agent' to create new AI agents. Choose from Neural Mesh, Quantum Scheduler, Universal I/O, Security, or Analytics agents."
              tips={['Select agent type based on your workload', 'Consider resource requirements', 'Deploy in appropriate regions for latency']}
            />
            <InstructionItem
              title="Monitoring Agent Performance"
              description="Each agent card shows real-time CPU, memory, and GPU usage, plus current tasks and completion statistics."
              tips={['Red metrics indicate high usage', 'Click on agents for detailed information', 'Monitor uptime for reliability']}
            />
            <InstructionItem
              title="Agent Actions"
              description="Use the action buttons to start, stop, configure, or terminate agents as needed."
              tips={['Always pause before terminating', 'Check for active tasks before stopping', 'Use configuration for performance tuning']}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function JobsSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Briefcase className="h-5 w-5" />
          Job Management Guide
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="font-semibold text-lg mb-3">Creating and Managing Jobs</h3>
          <div className="space-y-3">
            <InstructionItem
              title="Creating Jobs"
              description="Click 'Create Job' to define new computational tasks. Specify priority, agent type, and resource requirements."
              tips={['Use descriptive titles for easy identification', 'Set appropriate priority levels', 'Add tags for better organization']}
            />
            <InstructionItem
              title="Tracking Progress"
              description="Monitor job progress with real-time status updates, progress bars, and resource usage metrics."
              tips={['Progress bars show completion percentage', 'Status badges indicate current state', 'Click jobs for detailed logs']}
            />
            <InstructionItem
              title="Job Control"
              description="Pause, resume, restart, or cancel jobs using the action buttons in the job details view."
              tips={['Pause jobs to free up resources', 'Restart failed jobs after fixing issues', 'Cancel jobs that are no longer needed']}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function NeuralMeshSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          Neural Mesh Memory System
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="font-semibold text-lg mb-3">AI Memory Management</h3>
          <div className="space-y-3">
            <InstructionItem
              title="Memory Search"
              description="Use the search interface to query the neural mesh for specific memories, patterns, or knowledge."
              tips={['Use natural language queries', 'Search results show relevance scores', 'Explore connected memories for deeper insights']}
            />
            <InstructionItem
              title="Memory Types"
              description="The system manages Episodic (experiences), Semantic (facts), Procedural (skills), and Working (temporary) memories."
              tips={['Different memory types serve different purposes', 'Episodic memories capture experiences', 'Semantic memories store factual knowledge']}
            />
            <InstructionItem
              title="Network Visualization"
              description="View the animated network to see how memories are interconnected and how knowledge flows through the system."
              tips={['Larger nodes indicate more important memories', 'Connections show memory relationships', 'Animation shows active memory access']}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function QuantumSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-5 w-5" />
          Quantum Scheduler Guide
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="font-semibold text-lg mb-3">Million-Scale Coordination</h3>
          <div className="space-y-3">
            <InstructionItem
              title="Creating Quantum Tasks"
              description="Design tasks that coordinate thousands of agents simultaneously using quantum computing principles."
              tips={['Set appropriate coherence levels', 'Consider target agent counts', 'Monitor quantum efficiency metrics']}
            />
            <InstructionItem
              title="Quantum Metrics"
              description="Track quantum efficiency, entanglement strength, and decoherence rates to optimize performance."
              tips={['Higher coherence = better coordination', 'Monitor decoherence rates', 'Optimize for quantum efficiency']}
            />
            <InstructionItem
              title="Visualization"
              description="The quantum state visualization shows active quantum processes and agent entanglement patterns."
              tips={['Animated particles represent quantum states', 'Colors indicate different quantum properties', 'Rotation shows quantum evolution']}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function MonitoringSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          System Monitoring Guide
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="font-semibold text-lg mb-3">Real-time System Monitoring</h3>
          <div className="space-y-3">
            <InstructionItem
              title="Health Monitoring"
              description="The circular health indicator shows overall system health as a percentage with color-coded status."
              tips={['Green = healthy system (>90%)', 'Yellow = needs attention (70-90%)', 'Red = critical issues (<70%)']}
            />
            <InstructionItem
              title="Performance Metrics"
              description="Monitor CPU, memory, network, and disk usage with trend indicators showing performance changes."
              tips={['Trend arrows show performance direction', 'Set up alerts for threshold breaches', 'Use historical data for capacity planning']}
            />
            <InstructionItem
              title="Alert Management"
              description="View and manage system alerts with severity levels, timestamps, and recommended actions."
              tips={['Acknowledge alerts to clear them', 'Critical alerts require immediate attention', 'Use alert patterns to identify systemic issues']}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function TerminalSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Terminal className="h-5 w-5" />
          Terminal Interface Guide
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="font-semibold text-lg mb-3">Command-Line Operations</h3>
          <div className="space-y-3">
            <InstructionItem
              title="Basic Commands"
              description="Use commands like 'status', 'agents', 'jobs', and 'help' to get system information and perform operations."
              tips={['Type "help" for all available commands', 'Use arrow keys for command history', 'Commands are case-insensitive']}
            />
            <InstructionItem
              title="Advanced Operations"
              description="Deploy agents, stop processes, view logs, and access specialized interfaces using command-line syntax."
              tips={['Use "deploy <type>" to create agents', 'Use "logs <id>" to view detailed logs', 'Use "stop <id>" to halt processes']}
            />
            <InstructionItem
              title="Terminal Features"
              description="Export logs, use fullscreen mode, and access command history for efficient operations."
              tips={['Press Tab for command completion', 'Use fullscreen for extended sessions', 'Export logs for external analysis']}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Helper components
function FeatureCard({ icon, title, description }: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="p-4 rounded-lg bg-white/5 dark:bg-black/20 border border-white/10 dark:border-red-900/40">
      <div className="flex items-start gap-3">
        <div className="text-day-accent dark:text-night-text">
          {icon}
        </div>
        <div>
          <h4 className="font-medium text-sm mb-1">{title}</h4>
          <p className="text-xs opacity-70">{description}</p>
        </div>
      </div>
    </div>
  );
}

function InstructionItem({ title, description, tips }: {
  title: string;
  description: string;
  tips: string[];
}) {
  return (
    <div className="p-4 rounded-lg bg-white/5 dark:bg-black/20 border border-white/10 dark:border-red-900/40">
      <h4 className="font-medium text-sm mb-2">{title}</h4>
      <p className="text-sm opacity-80 mb-3">{description}</p>
      <div className="space-y-1">
        <p className="text-xs font-medium opacity-70">Tips:</p>
        {tips.map((tip, index) => (
          <div key={index} className="flex items-start gap-2">
            <ChevronRight className="h-3 w-3 text-day-accent dark:text-night-text mt-0.5 flex-shrink-0" />
            <p className="text-xs opacity-70">{tip}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// Placeholder sections for other features
function SwarmSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Swarm Control Instructions</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm opacity-80">
          Detailed instructions for managing distributed agent clusters, scaling operations, 
          and coordinating multi-agent workflows will be displayed here.
        </p>
      </CardContent>
    </Card>
  );
}

function AnalyticsSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Analytics Instructions</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm opacity-80">
          Guide for interpreting performance analytics, setting up custom dashboards, 
          and using insights to optimize your agent operations.
        </p>
      </CardContent>
    </Card>
  );
}

function NetworkSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Network Management Instructions</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm opacity-80">
          Instructions for monitoring network topology, managing connections, 
          and optimizing network performance for distributed operations.
        </p>
      </CardContent>
    </Card>
  );
}

function DataSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Data Management Instructions</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm opacity-80">
          Guide for managing data sources, setting up processing pipelines, 
          and ensuring data security and compliance.
        </p>
      </CardContent>
    </Card>
  );
}

function SecuritySection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Security Center Instructions</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm opacity-80">
          Comprehensive guide for managing security policies, monitoring threats, 
          handling alerts, and maintaining system security posture.
        </p>
      </CardContent>
    </Card>
  );
}

function SettingsSection() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Settings Configuration Guide</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm opacity-80">
          Instructions for configuring system settings, managing user preferences, 
          and customizing the platform for your specific needs.
        </p>
      </CardContent>
    </Card>
  );
}