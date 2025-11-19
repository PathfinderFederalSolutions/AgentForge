'use client';

import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  Terminal as TerminalIcon,
  Send,
  History,
  Download,
  Trash2,
  Settings,
  Maximize2,
  Minimize2
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Layout } from '@/components/layout/Layout';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';

interface TerminalLine {
  id: string;
  type: 'command' | 'output' | 'error';
  content: string;
  timestamp: Date;
}

export default function TerminalPage() {
  const [command, setCommand] = useState('');
  const [lines, setLines] = useState<TerminalLine[]>([
    {
      id: '1',
      type: 'output',
      content: 'AgentForge Terminal v2.1.0',
      timestamp: new Date()
    },
    {
      id: '2',
      type: 'output',
      content: 'Type "help" for available commands.',
      timestamp: new Date()
    }
  ]);
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const terminalRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const snap = useSnapshot(store);

  const commands = {
    help: () => [
      'Available commands:',
      '  help              - Show this help message',
      '  status            - Show system status',
      '  agents            - List active agents',
      '  jobs              - List current jobs',
      '  deploy <type>     - Deploy a new agent',
      '  stop <id>         - Stop an agent or job',
      '  logs <id>         - Show logs for agent/job',
      '  neural-mesh       - Access neural mesh interface',
      '  quantum           - Access quantum scheduler',
      '  monitoring        - Show system monitoring',
      '  config            - Show system configuration',
      '  clear             - Clear terminal',
      '  exit              - Close terminal session'
    ],
    status: () => [
      `System Status: ${snap.connected ? 'ONLINE' : 'OFFLINE'}`,
      `Active Nodes: ${snap.meta.nodes || 0}`,
      `Queue Depth: ${snap.meta.queueDepth || 0}`,
      `Requests/sec: ${snap.meta.rps || 0}`,
      `Total Jobs: ${snap.jobs.length}`,
      `WebSocket: ${snap.connected ? 'Connected' : 'Disconnected'}`
    ],
    agents: () => [
      'Active Agents:',
      '  agent-001  Neural Processor Alpha    [ACTIVE]',
      '  agent-002  Quantum Scheduler Beta    [ACTIVE]',
      '  agent-003  Data Processor Gamma      [IDLE]',
      '  agent-004  Security Monitor Delta    [WARNING]',
      '  agent-005  Analytics Engine Epsilon  [ERROR]'
    ],
    jobs: () => {
      if (snap.jobs.length === 0) {
        return ['No active jobs.'];
      }
      return [
        'Current Jobs:',
        ...snap.jobs.slice(0, 10).map(job => 
          `  ${job.id}  ${job.status.toUpperCase()}  ${job.owner || 'Unknown'}`
        )
      ];
    },
    'neural-mesh': () => [
      'Neural Mesh Status:',
      '  Total Nodes: 15,847',
      '  Active Connections: 89,432',
      '  Memory Utilization: 67%',
      '  Coherence Score: 94%',
      '  Last Sync: 2 minutes ago'
    ],
    quantum: () => [
      'Quantum Scheduler Status:',
      '  Total Clusters: 12',
      '  Entangled Agents: 2,847',
      '  Coherence Score: 91%',
      '  Quantum Volume: 64',
      '  Fidelity: 95%'
    ],
    monitoring: () => [
      'System Monitoring:',
      '  CPU Usage: 75%',
      '  Memory Usage: 62%',
      '  Network I/O: 34 MB/s',
      '  Disk I/O: 45 MB/s',
      '  Active Alerts: 3',
      '  System Health: 95%'
    ],
    config: () => [
      'System Configuration:',
      '  Environment: Production',
      '  Log Level: INFO',
      '  Debug Mode: Disabled',
      '  API Host: 0.0.0.0:8000',
      '  WebSocket: Enabled',
      '  TLS: Enabled',
      '  Rate Limiting: Enabled'
    ],
    clear: () => {
      setLines([]);
      return [];
    }
  };

  useEffect(() => {
    // Auto-scroll to bottom
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  useEffect(() => {
    // Focus input when component mounts
    inputRef.current?.focus();
  }, []);

  const executeCommand = (cmd: string) => {
    const trimmedCmd = cmd.trim().toLowerCase();
    
    // Add command to history
    if (trimmedCmd) {
      setCommandHistory(prev => [...prev, trimmedCmd]);
      setHistoryIndex(-1);
    }

    // Add command line to terminal
    const commandLine: TerminalLine = {
      id: Date.now().toString(),
      type: 'command',
      content: `$ ${cmd}`,
      timestamp: new Date()
    };

    let output: string[] = [];

    if (trimmedCmd === '') {
      // Empty command, just show prompt
      setLines(prev => [...prev, commandLine]);
      return;
    }

    // Parse command and arguments
    const [baseCmd, ...args] = trimmedCmd.split(' ');

    if (commands[baseCmd as keyof typeof commands]) {
      output = commands[baseCmd as keyof typeof commands]();
    } else if (baseCmd.startsWith('deploy')) {
      const agentType = args[0] || 'neural-mesh';
      output = [
        `Deploying ${agentType} agent...`,
        'Allocating resources...',
        'Initializing agent framework...',
        'Establishing neural connections...',
        `Agent deployed successfully with ID: agent-${Math.random().toString(36).substr(2, 9)}`
      ];
    } else if (baseCmd.startsWith('stop')) {
      const targetId = args[0] || 'unknown';
      output = [
        `Stopping ${targetId}...`,
        'Gracefully shutting down processes...',
        'Cleaning up resources...',
        `${targetId} stopped successfully.`
      ];
    } else if (baseCmd.startsWith('logs')) {
      const targetId = args[0] || 'system';
      output = [
        `Logs for ${targetId}:`,
        '[2024-01-15 10:30:15] INFO: Process started',
        '[2024-01-15 10:30:16] INFO: Initializing components',
        '[2024-01-15 10:30:17] INFO: Ready to accept requests',
        '[2024-01-15 10:30:18] INFO: Processing task batch',
        '[2024-01-15 10:30:19] INFO: Task completed successfully'
      ];
    } else if (trimmedCmd === 'exit') {
      output = ['Terminal session ended. Goodbye!'];
    } else {
      output = [`Command not found: ${baseCmd}`, 'Type "help" for available commands.'];
    }

    const outputLines: TerminalLine[] = output.map((line, index) => ({
      id: `${Date.now()}-${index}`,
      type: line.includes('ERROR') || line.includes('failed') ? 'error' : 'output',
      content: line,
      timestamp: new Date()
    }));

    setLines(prev => [...prev, commandLine, ...outputLines]);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      executeCommand(command);
      setCommand('');
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (commandHistory.length > 0) {
        const newIndex = historyIndex === -1 ? commandHistory.length - 1 : Math.max(0, historyIndex - 1);
        setHistoryIndex(newIndex);
        setCommand(commandHistory[newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex !== -1) {
        const newIndex = historyIndex + 1;
        if (newIndex >= commandHistory.length) {
          setHistoryIndex(-1);
          setCommand('');
        } else {
          setHistoryIndex(newIndex);
          setCommand(commandHistory[newIndex]);
        }
      }
    }
  };

  const exportLogs = () => {
    const logContent = lines.map(line => 
      `[${line.timestamp.toISOString()}] ${line.type.toUpperCase()}: ${line.content}`
    ).join('\n');
    
    const blob = new Blob([logContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `agentforge-terminal-${new Date().toISOString().split('T')[0]}.log`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <TerminalIcon className="h-8 w-8 text-green-400" />
              System Terminal
            </h1>
            <p className="text-sm opacity-70 mt-1">
              Command-line interface for AgentForge system management
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge 
              variant={snap.connected ? 'success' : 'danger'}
              className="px-3 py-1"
            >
              {snap.connected ? 'CONNECTED' : 'DISCONNECTED'}
            </Badge>
            <Button
              variant="ghost"
              icon={<History className="h-4 w-4" />}
              onClick={() => console.log('Command history:', commandHistory)}
            >
              History ({commandHistory.length})
            </Button>
            <Button
              variant="ghost"
              icon={<Download className="h-4 w-4" />}
              onClick={exportLogs}
            >
              Export
            </Button>
            <Button
              variant="ghost"
              icon={isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
              onClick={() => setIsFullscreen(!isFullscreen)}
            >
              {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
            </Button>
          </div>
        </div>

        {/* Terminal */}
        <Card className={isFullscreen ? 'fixed inset-4 z-50' : ''}>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <TerminalIcon className="h-5 w-5" />
                AgentForge Terminal
              </CardTitle>
              <div className="flex items-center gap-2">
                <div className="flex gap-1">
                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                  <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  icon={<Trash2 className="h-4 w-4" />}
                  onClick={() => setLines([])}
                >
                  Clear
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {/* Terminal Output */}
            <div 
              ref={terminalRef}
              className={`bg-black/90 font-mono text-sm p-4 overflow-y-auto ${
                isFullscreen ? 'h-[calc(100vh-12rem)]' : 'h-96'
              }`}
            >
              {lines.map((line, index) => (
                <motion.div
                  key={line.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.1, delay: index * 0.02 }}
                  className={`mb-1 ${
                    line.type === 'command' 
                      ? 'text-green-400' 
                      : line.type === 'error'
                      ? 'text-red-400'
                      : 'text-gray-300'
                  }`}
                >
                  {line.content}
                </motion.div>
              ))}
              
              {/* Command Input */}
              <div className="flex items-center gap-2 mt-2">
                <span className="text-green-400">$</span>
                <input
                  ref={inputRef}
                  type="text"
                  value={command}
                  onChange={(e) => setCommand(e.target.value)}
                  onKeyDown={handleKeyPress}
                  className="flex-1 bg-transparent text-green-400 outline-none"
                  placeholder="Enter command..."
                  autoFocus
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Quick Commands */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Commands</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {Object.keys(commands).slice(0, 12).map((cmd) => (
                <Button
                  key={cmd}
                  variant="ghost"
                  size="sm"
                  className="justify-start font-mono text-xs"
                  onClick={() => {
                    setCommand(cmd);
                    executeCommand(cmd);
                    setCommand('');
                  }}
                >
                  {cmd}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Command Reference */}
        <Card>
          <CardHeader>
            <CardTitle>Command Reference</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="label text-sm mb-3">System Commands</h3>
                <div className="space-y-2 font-mono text-sm">
                  <div><span className="text-green-400">status</span> - Show system status</div>
                  <div><span className="text-green-400">monitoring</span> - System monitoring</div>
                  <div><span className="text-green-400">config</span> - Show configuration</div>
                  <div><span className="text-green-400">clear</span> - Clear terminal</div>
                </div>
              </div>
              <div>
                <h3 className="label text-sm mb-3">Agent Commands</h3>
                <div className="space-y-2 font-mono text-sm">
                  <div><span className="text-green-400">agents</span> - List agents</div>
                  <div><span className="text-green-400">deploy &lt;type&gt;</span> - Deploy agent</div>
                  <div><span className="text-green-400">stop &lt;id&gt;</span> - Stop agent/job</div>
                  <div><span className="text-green-400">logs &lt;id&gt;</span> - Show logs</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}

