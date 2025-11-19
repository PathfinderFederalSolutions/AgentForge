'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { 
  Brain, 
  Network, 
  Zap, 
  Users, 
  BookOpen, 
  TrendingUp, 
  Settings, 
  Play,
  MessageSquare,
  FileText,
  BarChart3,
  Cpu,
  Database,
  Globe,
  Shield,
  Lightbulb
} from 'lucide-react';

interface ReasoningTrace {
  trace_id: string;
  pattern: string;
  steps: any[];
  confidence: number;
  execution_time: number;
  success: boolean;
}

interface KnowledgeQuery {
  query: string;
  response: string;
  source_documents: any[];
  confidence: number;
  processing_time: number;
}

export default function AICapabilitiesPage() {
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [availableCapabilities, setAvailableCapabilities] = useState<any>(null);
  const [reasoningPatterns, setReasoningPatterns] = useState<any>(null);
  const [learningAnalytics, setLearningAnalytics] = useState<any>(null);
  
  // Demo states
  const [demoResults, setDemoResults] = useState<any[]>([]);
  const [isRunningDemo, setIsRunningDemo] = useState(false);
  const [demoInput, setDemoInput] = useState('');
  const [selectedReasoningPattern, setSelectedReasoningPattern] = useState('chain_of_thought');
  const [knowledgeQuery, setKnowledgeQuery] = useState('');
  const [knowledgeResults, setKnowledgeResults] = useState<KnowledgeQuery[]>([]);

  useEffect(() => {
    loadAllData();
    
    // Refresh every 30 seconds
    const interval = setInterval(loadAllData, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadAllData = async () => {
    try {
      // Load system status
      const statusResponse = await fetch('http://localhost:8001/v1/ai/status');
      const status = await statusResponse.json();
      setSystemStatus(status);

      // Load available capabilities
      const capabilitiesResponse = await fetch('http://localhost:8001/v1/ai/capabilities/available');
      const capabilities = await capabilitiesResponse.json();
      setAvailableCapabilities(capabilities);

      // Load reasoning patterns
      const reasoningResponse = await fetch('http://localhost:8001/v1/ai/reasoning/patterns');
      const reasoning = await reasoningResponse.json();
      setReasoningPatterns(reasoning);

      // Load learning analytics
      const learningResponse = await fetch('http://localhost:8001/v1/ai/learning/analytics');
      const learning = await learningResponse.json();
      setLearningAnalytics(learning);

    } catch (error) {
      console.error('Failed to load AI capabilities data:', error);
    }
  };

  const runIntelligentDemo = async (demoType: string) => {
    setIsRunningDemo(true);
    try {
      let demoRequest = '';
      
      switch (demoType) {
        case 'security_analysis':
          demoRequest = 'Analyze system security vulnerabilities and recommend improvements';
          break;
        case 'performance_optimization':
          demoRequest = 'Identify performance bottlenecks and optimization opportunities';
          break;
        case 'code_review':
          demoRequest = 'Review codebase for quality, security, and maintainability issues';
          break;
        case 'data_analysis':
          demoRequest = 'Analyze data patterns and generate insights';
          break;
        default:
          demoRequest = demoInput || 'Demonstrate AI capabilities';
      }

      const response = await fetch('http://localhost:8001/v1/ai/demo/intelligent-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysis_request: demoRequest,
          use_swarm: true,
          agent_count: 5
        })
      });

      const result = await response.json();
      
      // Add to demo results
      setDemoResults(prev => [{
        id: Date.now(),
        type: demoType,
        request: demoRequest,
        result: result,
        timestamp: new Date().toISOString()
      }, ...prev.slice(0, 9)]); // Keep last 10 results

    } catch (error) {
      console.error('Demo failed:', error);
      alert('Demo failed: ' + error);
    } finally {
      setIsRunningDemo(false);
    }
  };

  const queryKnowledgeBase = async () => {
    if (!knowledgeQuery.trim()) return;
    
    try {
      const response = await fetch('http://localhost:8001/v1/ai/knowledge/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: knowledgeQuery,
          agent_id: 'demo_agent',
          max_context_docs: 5
        })
      });

      const result = await response.json();
      
      setKnowledgeResults(prev => [result, ...prev.slice(0, 4)]); // Keep last 5 results
      setKnowledgeQuery('');

    } catch (error) {
      console.error('Knowledge query failed:', error);
      alert('Knowledge query failed');
    }
  };

  const testReasoningPattern = async () => {
    if (!demoInput.trim()) return;
    
    try {
      const response = await fetch('http://localhost:8001/v1/ai/tasks/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          description: demoInput,
          task_type: 'reasoning_demo',
          priority: 'normal',
          reasoning_pattern: selectedReasoningPattern,
          required_capabilities: ['reasoning', 'analysis']
        })
      });

      const result = await response.json();
      
      setDemoResults(prev => [{
        id: Date.now(),
        type: 'reasoning_pattern',
        pattern: selectedReasoningPattern,
        request: demoInput,
        result: result,
        timestamp: new Date().toISOString()
      }, ...prev.slice(0, 9)]);
      
      setDemoInput('');

    } catch (error) {
      console.error('Reasoning test failed:', error);
      alert('Reasoning test failed');
    }
  };

  if (!systemStatus) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Cpu className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p>Loading AI Capabilities...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Brain className="h-8 w-8" />
          AI Capabilities Dashboard
        </h1>
        <div className="flex gap-2">
          <Badge variant={systemStatus.enhanced_ai_available ? "success" : "destructive"}>
            Enhanced AI: {systemStatus.enhanced_ai_available ? "Online" : "Offline"}
          </Badge>
          <Badge variant={systemStatus.neural_mesh_available ? "success" : "destructive"}>
            Neural Mesh: {systemStatus.neural_mesh_available ? "Online" : "Offline"}
          </Badge>
        </div>
      </div>

      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="p-6 text-center">
          <Brain className="h-8 w-8 mx-auto mb-2 text-blue-600" />
          <div className="text-2xl font-bold text-blue-600">
            {availableCapabilities?.total_capabilities || 0}
          </div>
          <div className="text-sm text-gray-600">AI Capabilities</div>
        </Card>

        <Card className="p-6 text-center">
          <Network className="h-8 w-8 mx-auto mb-2 text-green-600" />
          <div className="text-2xl font-bold text-green-600">
            {systemStatus.systems?.neural_mesh?.active_swarms || 0}
          </div>
          <div className="text-sm text-gray-600">Active Swarms</div>
        </Card>

        <Card className="p-6 text-center">
          <Users className="h-8 w-8 mx-auto mb-2 text-purple-600" />
          <div className="text-2xl font-bold text-purple-600">
            {systemStatus.systems?.neural_mesh?.neural_mesh_status?.active_agents || 0}
          </div>
          <div className="text-sm text-gray-600">Intelligent Agents</div>
        </Card>

        <Card className="p-6 text-center">
          <TrendingUp className="h-8 w-8 mx-auto mb-2 text-orange-600" />
          <div className="text-2xl font-bold text-orange-600">
            {systemStatus.systems?.neural_mesh?.collective_intelligence?.intelligence_amplification_factor?.toFixed(1) || '1.0'}x
          </div>
          <div className="text-sm text-gray-600">Intelligence Amplification</div>
        </Card>
      </div>

      {/* Interactive Demo Section */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Play className="h-5 w-5" />
          Interactive AI Demonstrations
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Reasoning Pattern Demo */}
          <div className="space-y-4">
            <h3 className="font-medium">Test Reasoning Patterns</h3>
            <div className="space-y-2">
              <Input
                value={demoInput}
                onChange={(e) => setDemoInput(e.target.value)}
                placeholder="Enter a problem to solve..."
              />
              <select 
                value={selectedReasoningPattern}
                onChange={(e) => setSelectedReasoningPattern(e.target.value)}
                className="w-full p-2 border rounded"
              >
                <option value="chain_of_thought">Chain-of-Thought</option>
                <option value="react">ReAct (Reasoning + Acting)</option>
                <option value="tree_of_thoughts">Tree-of-Thoughts</option>
              </select>
              <Button onClick={testReasoningPattern} disabled={!demoInput.trim()}>
                Test Reasoning Pattern
              </Button>
            </div>
          </div>

          {/* Knowledge Query Demo */}
          <div className="space-y-4">
            <h3 className="font-medium">Query Knowledge Base</h3>
            <div className="space-y-2">
              <Input
                value={knowledgeQuery}
                onChange={(e) => setKnowledgeQuery(e.target.value)}
                placeholder="Ask a question about the system..."
              />
              <Button onClick={queryKnowledgeBase} disabled={!knowledgeQuery.trim()}>
                Query Knowledge Base
              </Button>
            </div>
          </div>
        </div>
      </Card>

      {/* Pre-built Demos */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Pre-built AI Demonstrations</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Button 
            onClick={() => runIntelligentDemo('security_analysis')}
            disabled={isRunningDemo}
            className="h-20 flex flex-col items-center justify-center gap-2"
          >
            <Shield className="h-6 w-6" />
            Security Analysis
          </Button>
          
          <Button 
            onClick={() => runIntelligentDemo('performance_optimization')}
            disabled={isRunningDemo}
            className="h-20 flex flex-col items-center justify-center gap-2"
          >
            <TrendingUp className="h-6 w-6" />
            Performance Optimization
          </Button>
          
          <Button 
            onClick={() => runIntelligentDemo('code_review')}
            disabled={isRunningDemo}
            className="h-20 flex flex-col items-center justify-center gap-2"
          >
            <FileText className="h-6 w-6" />
            Code Review
          </Button>
          
          <Button 
            onClick={() => runIntelligentDemo('data_analysis')}
            disabled={isRunningDemo}
            className="h-20 flex flex-col items-center justify-center gap-2"
          >
            <BarChart3 className="h-6 w-6" />
            Data Analysis
          </Button>
        </div>
        
        {isRunningDemo && (
          <div className="mt-4 text-center">
            <div className="inline-flex items-center gap-2 text-blue-600">
              <Cpu className="h-4 w-4 animate-spin" />
              Running intelligent analysis with agent swarm...
            </div>
          </div>
        )}
      </Card>

      {/* Knowledge Query Results */}
      {knowledgeResults.length > 0 && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            Knowledge Base Queries
          </h2>
          <div className="space-y-4">
            {knowledgeResults.map((result, index) => (
              <div key={index} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">Query: {result.query}</span>
                  <div className="flex gap-2">
                    <Badge variant="info">
                      Confidence: {(result.confidence * 100).toFixed(1)}%
                    </Badge>
                    <Badge variant="outline">
                      {result.processing_time.toFixed(2)}s
                    </Badge>
                  </div>
                </div>
                <div className="text-sm bg-gray-50 p-3 rounded mb-2">
                  {result.response}
                </div>
                <div className="text-xs text-gray-600">
                  Sources: {result.source_documents.length} documents
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Demo Results */}
      {demoResults.length > 0 && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Lightbulb className="h-5 w-5" />
            Demo Results
          </h2>
          <div className="space-y-4">
            {demoResults.map((demo) => (
              <div key={demo.id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{demo.type.replace('_', ' ').toUpperCase()}</span>
                  <Badge variant="success">
                    {demo.result.demo_type || 'Completed'}
                  </Badge>
                </div>
                <div className="text-sm text-gray-600 mb-2">
                  <strong>Request:</strong> {demo.request}
                </div>
                <div className="text-sm bg-blue-50 p-3 rounded">
                  {demo.result.swarm_deployed && (
                    <div className="mb-2">
                      <strong>Swarm Deployed:</strong> {demo.result.agents_count} agents
                    </div>
                  )}
                  {demo.result.intelligence_amplification && (
                    <div className="mb-2">
                      <strong>Intelligence Amplification:</strong> {demo.result.intelligence_amplification}x
                    </div>
                  )}
                  {demo.result.collective_confidence && (
                    <div>
                      <strong>Collective Confidence:</strong> {(demo.result.collective_confidence * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
                <div className="text-xs text-gray-500 mt-2">
                  {demo.timestamp}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Available Capabilities */}
      {availableCapabilities && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Available AI Capabilities ({availableCapabilities.total_capabilities})
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(availableCapabilities.capability_types).map(([type, count]) => (
              <div key={type} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium capitalize">{type.replace('_', ' ')}</span>
                  <Badge variant="outline">{count as number} capabilities</Badge>
                </div>
                <div className="text-sm text-gray-600">
                  {type === 'computation' && 'Mathematical and logical operations'}
                  {type === 'data_processing' && 'Data analysis and transformation'}
                  {type === 'communication' && 'Inter-agent messaging and coordination'}
                  {type === 'file_operations' && 'File reading, writing, and processing'}
                  {type === 'web_scraping' && 'Web content extraction and analysis'}
                  {type === 'api_integration' && 'External API integration and calls'}
                  {type === 'code_execution' && 'Safe code execution and analysis'}
                  {type === 'analysis' && 'Advanced analysis and pattern recognition'}
                  {type === 'generation' && 'Content and code generation'}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Reasoning Patterns */}
      {reasoningPatterns && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Reasoning Patterns & Analytics
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {reasoningPatterns.available_patterns.map((pattern: any) => (
              <div key={pattern.pattern} className="border rounded-lg p-4">
                <div className="font-medium mb-2 capitalize">
                  {pattern.pattern.replace('_', ' ')}
                </div>
                <div className="text-sm text-gray-600 mb-3">
                  {pattern.description}
                </div>
                <div className="space-y-1 text-xs">
                  <div>Average Time: {pattern.average_time}</div>
                  <div>Accuracy: {pattern.accuracy}</div>
                  <div>Use Cases: {pattern.use_cases.join(', ')}</div>
                </div>
              </div>
            ))}
          </div>
          
          {reasoningPatterns.usage_analytics && (
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium mb-2">Usage Analytics</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <div className="font-medium">Total Traces</div>
                  <div>{reasoningPatterns.usage_analytics.total_traces}</div>
                </div>
                <div>
                  <div className="font-medium">Success Rate</div>
                  <div>{(reasoningPatterns.usage_analytics.success_rate * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <div className="font-medium">Avg Time</div>
                  <div>{reasoningPatterns.usage_analytics.avg_execution_time.toFixed(2)}s</div>
                </div>
                <div>
                  <div className="font-medium">Avg Confidence</div>
                  <div>{(reasoningPatterns.usage_analytics.avg_confidence * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Learning Analytics */}
      {learningAnalytics && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Learning & Improvement Analytics
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3">System Learning Metrics</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Total Feedback Records</span>
                  <span>{learningAnalytics.total_feedback_records}</span>
                </div>
                <div className="flex justify-between">
                  <span>Active A/B Tests</span>
                  <span>{learningAnalytics.active_ab_tests}</span>
                </div>
                <div className="flex justify-between">
                  <span>Learning Events</span>
                  <span>{learningAnalytics.learning_analytics?.learning_summary?.learning_velocity || 0}</span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-medium mb-3">Recent Feedback</h4>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {learningAnalytics.recent_feedback?.slice(0, 5).map((feedback: any, index: number) => (
                  <div key={index} className="text-sm border-l-2 border-blue-200 pl-3">
                    <div className="flex justify-between">
                      <span>Agent: {feedback.agent_id}</span>
                      <Badge variant={feedback.rating > 0.7 ? "success" : feedback.rating > 0.4 ? "warning" : "destructive"}>
                        {(feedback.rating * 100).toFixed(0)}%
                      </Badge>
                    </div>
                    <div className="text-gray-600">Type: {feedback.feedback_type}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Interactive Testing */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Custom Demo Input */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Custom AI Demo</h3>
          <div className="space-y-4">
            <Input
              value={demoInput}
              onChange={(e) => setDemoInput(e.target.value)}
              placeholder="Describe what you want the AI to analyze or solve..."
            />
            <div className="flex gap-2">
              <Button 
                onClick={() => runIntelligentDemo('custom')}
                disabled={isRunningDemo || !demoInput.trim()}
                className="flex-1"
              >
                {isRunningDemo ? 'Running...' : 'Run AI Analysis'}
              </Button>
            </div>
          </div>
        </Card>

        {/* Knowledge Base Query */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Knowledge Base Query</h3>
          <div className="space-y-4">
            <Input
              value={knowledgeQuery}
              onChange={(e) => setKnowledgeQuery(e.target.value)}
              placeholder="Ask a question about the system knowledge..."
            />
            <Button 
              onClick={queryKnowledgeBase}
              disabled={!knowledgeQuery.trim()}
              className="w-full"
            >
              Query Knowledge Base
            </Button>
          </div>
        </Card>
      </div>

      {/* System Health Indicators */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5" />
          AI System Health
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <CheckCircle className="h-6 w-6 mx-auto mb-2 text-green-600" />
            <div className="font-medium">LLM Integration</div>
            <div className="text-sm text-gray-600">
              {systemStatus.systems?.llm_integration?.total_available || 0} Providers
            </div>
          </div>
          
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <Database className="h-6 w-6 mx-auto mb-2 text-blue-600" />
            <div className="font-medium">Neural Mesh</div>
            <div className="text-sm text-gray-600">
              {systemStatus.systems?.neural_mesh?.neural_mesh_status?.system_health ? 
                (systemStatus.systems.neural_mesh.neural_mesh_status.system_health * 100).toFixed(1) + '% Health' : 
                'Status Unknown'
              }
            </div>
          </div>
          
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <Globe className="h-6 w-6 mx-auto mb-2 text-purple-600" />
            <div className="font-medium">Distributed Memory</div>
            <div className="text-sm text-gray-600">
              {systemStatus.systems?.neural_mesh?.neural_mesh_status?.total_memories || 0} Memories
            </div>
          </div>
          
          <div className="text-center p-4 bg-orange-50 rounded-lg">
            <Zap className="h-6 w-6 mx-auto mb-2 text-orange-600" />
            <div className="font-medium">Collective Intelligence</div>
            <div className="text-sm text-gray-600">
              {systemStatus.systems?.neural_mesh?.collective_intelligence?.emergence_score ? 
                (systemStatus.systems.neural_mesh.collective_intelligence.emergence_score * 100).toFixed(1) + '% Emergence' : 
                'Monitoring'
              }
            </div>
          </div>
        </div>
      </Card>

      {/* Real-time Metrics */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Real-time AI Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-blue-600">
              {systemStatus.systems?.llm_integration?.total_requests || 0}
            </div>
            <div className="text-sm text-gray-600">LLM Requests</div>
          </div>
          
          <div>
            <div className="text-2xl font-bold text-green-600">
              {systemStatus.systems?.capabilities?.total_executions || 0}
            </div>
            <div className="text-sm text-gray-600">Capability Executions</div>
          </div>
          
          <div>
            <div className="text-2xl font-bold text-purple-600">
              {systemStatus.systems?.neural_mesh?.neural_mesh_status?.sync_operations_per_second?.toFixed(1) || '0.0'}
            </div>
            <div className="text-sm text-gray-600">Sync Ops/sec</div>
          </div>
          
          <div>
            <div className="text-2xl font-bold text-orange-600">
              {systemStatus.systems?.learning?.learning_summary?.average_feedback_rating ? 
                (systemStatus.systems.learning.learning_summary.average_feedback_rating * 100).toFixed(0) + '%' : 
                'N/A'
              }
            </div>
            <div className="text-sm text-gray-600">Learning Score</div>
          </div>
        </div>
      </Card>
    </div>
  );
}
