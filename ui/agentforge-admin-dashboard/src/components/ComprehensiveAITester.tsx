'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { enhancedAIClient } from '@/lib/enhancedAIClient';
import { 
  Play, 
  Brain, 
  Network, 
  Users, 
  BookOpen, 
  Zap, 
  MessageSquare,
  TrendingUp,
  CheckCircle,
  Clock,
  AlertCircle,
  Loader,
  FileText,
  Database
} from 'lucide-react';

interface TestResult {
  id: string;
  test_type: string;
  description: string;
  result: any;
  success: boolean;
  execution_time: number;
  timestamp: string;
}

export default function ComprehensiveAITester() {
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [isRunningTest, setIsRunningTest] = useState(false);
  const [currentTest, setCurrentTest] = useState('');
  
  // Test inputs
  const [customPrompt, setCustomPrompt] = useState('');
  const [knowledgeQuery, setKnowledgeQuery] = useState('');
  const [collaborationObjective, setCollaborationObjective] = useState('');
  const [reasoningProblem, setReasoningProblem] = useState('');
  
  // System status
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [activeAgents, setActiveAgents] = useState<any[]>([]);

  useEffect(() => {
    loadSystemStatus();
    loadActiveAgents();
  }, []);

  const loadSystemStatus = async () => {
    try {
      const status = await enhancedAIClient.getAISystemStatus();
      setSystemStatus(status);
    } catch (error) {
      console.error('Failed to load system status:', error);
    }
  };

  const loadActiveAgents = async () => {
    try {
      const agentsData = await enhancedAIClient.listActiveAgents();
      setActiveAgents(agentsData.agents || []);
    } catch (error) {
      console.error('Failed to load active agents:', error);
    }
  };

  const runTest = async (testType: string, testFunction: () => Promise<any>, description: string) => {
    setIsRunningTest(true);
    setCurrentTest(testType);
    
    const startTime = Date.now();
    
    try {
      const result = await testFunction();
      const executionTime = Date.now() - startTime;
      
      const testResult: TestResult = {
        id: `test_${Date.now()}`,
        test_type: testType,
        description,
        result,
        success: true,
        execution_time: executionTime,
        timestamp: new Date().toISOString()
      };
      
      setTestResults(prev => [testResult, ...prev.slice(0, 9)]); // Keep last 10 results
      
    } catch (error) {
      const executionTime = Date.now() - startTime;
      
      const testResult: TestResult = {
        id: `test_${Date.now()}`,
        test_type: testType,
        description,
        result: { error: error instanceof Error ? error.message : 'Unknown error' },
        success: false,
        execution_time: executionTime,
        timestamp: new Date().toISOString()
      };
      
      setTestResults(prev => [testResult, ...prev.slice(0, 9)]);
      console.error(`Test ${testType} failed:`, error);
    } finally {
      setIsRunningTest(false);
      setCurrentTest('');
    }
  };

  const testCreateAgent = () => runTest(
    'create_agent',
    () => enhancedAIClient.createEnhancedAgent('specialist', ['analysis', 'research']),
    'Create enhanced agent with analysis and research specializations'
  );

  const testDeploySwarm = () => runTest(
    'deploy_swarm',
    () => enhancedAIClient.deployIntelligentSwarm(
      'Comprehensive system analysis and optimization recommendations',
      ['analysis', 'optimization', 'security', 'performance'],
      ['system_architecture', 'performance_engineering', 'security'],
      5,
      'collective'
    ),
    'Deploy intelligent swarm with collective intelligence'
  );

  const testCollectiveReasoning = async () => {
    if (activeAgents.length === 0) {
      alert('No active agents available. Please create agents first.');
      return;
    }
    
    // First deploy a swarm if none exists
    const swarm = await enhancedAIClient.deployIntelligentSwarm(
      'Test collective reasoning capabilities',
      ['reasoning', 'analysis'],
      ['logic', 'problem_solving'],
      3,
      'collective'
    );
    
    // Wait for swarm initialization
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    return runTest(
      'collective_reasoning',
      () => enhancedAIClient.coordinateCollectiveReasoning(
        swarm.swarm_id,
        reasoningProblem || 'How can we optimize AI system performance while maintaining security?',
        'collective_chain_of_thought'
      ),
      'Coordinate collective reasoning across agent swarm'
    );
  };

  const testKnowledgeQuery = () => runTest(
    'knowledge_query',
    () => enhancedAIClient.queryKnowledgeBase(
      knowledgeQuery || 'What are the best practices for AI system architecture?',
      'test_agent'
    ),
    'Query knowledge base using RAG (Retrieval Augmented Generation)'
  );

  const testKnowledgeSynthesis = () => runTest(
    'knowledge_synthesis',
    () => enhancedAIClient.synthesizeCollectiveKnowledge(
      'artificial_intelligence_best_practices'
    ),
    'Synthesize collective knowledge from multiple agents'
  );

  const testCapabilityExecution = () => runTest(
    'capability_execution',
    () => enhancedAIClient.executeCapability(
      activeAgents[0]?.agent_id || 'test_agent',
      'process_text',
      { text: 'This is a test of the text processing capability', operation: 'analyze' }
    ),
    'Execute agent capability with sandboxed execution'
  );

  const testNeuralMeshMemory = () => runTest(
    'neural_mesh_memory',
    async () => {
      // Store memory
      const storeResult = await enhancedAIClient.storeMemoryInNeuralMesh(
        'test_agent',
        'test_memory',
        { test_data: 'This is test memory content', confidence: 0.9 },
        'L2',
        { test: true }
      );
      
      // Retrieve memory
      const retrieveResult = await enhancedAIClient.retrieveAgentMemories(
        'test_agent',
        'test_data',
        'hybrid',
        5
      );
      
      return { store_result: storeResult, retrieve_result: retrieveResult };
    },
    'Store and retrieve memory in neural mesh'
  );

  const runComprehensiveTest = () => runTest(
    'comprehensive_demo',
    () => enhancedAIClient.runComprehensiveDemo(),
    'Run comprehensive demo of all AI capabilities'
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Play className="h-6 w-6" />
          Comprehensive AI Testing Interface
        </h2>
        {systemStatus && (
          <div className="flex gap-2">
            <Badge variant={systemStatus.enhanced_ai_available ? "success" : "destructive"}>
              Enhanced AI: {systemStatus.enhanced_ai_available ? "Online" : "Offline"}
            </Badge>
            <Badge variant={systemStatus.neural_mesh_available ? "success" : "destructive"}>
              Neural Mesh: {systemStatus.neural_mesh_available ? "Online" : "Offline"}
            </Badge>
          </div>
        )}
      </div>

      {/* Quick Tests */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Quick AI Capability Tests</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Button 
            onClick={testCreateAgent}
            disabled={isRunningTest}
            className="h-20 flex flex-col items-center justify-center gap-2"
          >
            <Users className="h-5 w-5" />
            Create Agent
          </Button>
          
          <Button 
            onClick={testDeploySwarm}
            disabled={isRunningTest}
            className="h-20 flex flex-col items-center justify-center gap-2"
          >
            <Network className="h-5 w-5" />
            Deploy Swarm
          </Button>
          
          <Button 
            onClick={testCollectiveReasoning}
            disabled={isRunningTest}
            className="h-20 flex flex-col items-center justify-center gap-2"
          >
            <Brain className="h-5 w-5" />
            Collective Reasoning
          </Button>
          
          <Button 
            onClick={testKnowledgeQuery}
            disabled={isRunningTest}
            className="h-20 flex flex-col items-center justify-center gap-2"
          >
            <BookOpen className="h-5 w-5" />
            Knowledge Query
          </Button>
        </div>
      </Card>

      {/* Advanced Tests */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Advanced AI Tests</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Button 
            onClick={testKnowledgeSynthesis}
            disabled={isRunningTest}
            className="h-16 flex items-center justify-center gap-2"
          >
            <Zap className="h-5 w-5" />
            Knowledge Synthesis
          </Button>
          
          <Button 
            onClick={testCapabilityExecution}
            disabled={isRunningTest}
            className="h-16 flex items-center justify-center gap-2"
          >
            <Settings className="h-5 w-5" />
            Capability Execution
          </Button>
          
          <Button 
            onClick={testNeuralMeshMemory}
            disabled={isRunningTest}
            className="h-16 flex items-center justify-center gap-2"
          >
            <Database className="h-5 w-5" />
            Neural Mesh Memory
          </Button>
        </div>
      </Card>

      {/* Custom Tests */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Custom AI Tests</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Custom Reasoning Problem</label>
              <Input
                value={reasoningProblem}
                onChange={(e) => setReasoningProblem(e.target.value)}
                placeholder="Enter a complex problem for collective reasoning..."
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Knowledge Query</label>
              <Input
                value={knowledgeQuery}
                onChange={(e) => setKnowledgeQuery(e.target.value)}
                placeholder="Ask a question about the system knowledge..."
              />
            </div>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Collaboration Objective</label>
              <Input
                value={collaborationObjective}
                onChange={(e) => setCollaborationObjective(e.target.value)}
                placeholder="Define an objective for agent collaboration..."
              />
            </div>
            
            <Button 
              onClick={runComprehensiveTest}
              disabled={isRunningTest}
              className="w-full h-12 flex items-center justify-center gap-2"
            >
              {isRunningTest ? (
                <>
                  <Loader className="h-5 w-5 animate-spin" />
                  Running Comprehensive Test...
                </>
              ) : (
                <>
                  <Play className="h-5 w-5" />
                  Run Comprehensive AI Demo
                </>
              )}
            </Button>
          </div>
        </div>
      </Card>

      {/* Current Test Status */}
      {isRunningTest && (
        <Card className="p-6 border-blue-200 bg-blue-50">
          <div className="flex items-center gap-3">
            <Loader className="h-5 w-5 animate-spin text-blue-600" />
            <div>
              <div className="font-medium text-blue-900">Running Test: {currentTest.replace('_', ' ').toUpperCase()}</div>
              <div className="text-sm text-blue-700">Please wait while the AI systems process your request...</div>
            </div>
          </div>
        </Card>
      )}

      {/* Test Results */}
      {testResults.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Test Results ({testResults.length})
          </h3>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {testResults.map((test) => (
              <div key={test.id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {test.success ? (
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-red-600" />
                    )}
                    <span className="font-medium">
                      {test.test_type.replace('_', ' ').toUpperCase()}
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <Badge variant={test.success ? "success" : "destructive"}>
                      {test.success ? "Success" : "Failed"}
                    </Badge>
                    <Badge variant="outline">
                      {test.execution_time}ms
                    </Badge>
                  </div>
                </div>
                
                <div className="text-sm text-gray-600 mb-2">
                  {test.description}
                </div>
                
                <div className="text-xs bg-gray-50 p-3 rounded">
                  {test.success ? (
                    <div>
                      {test.test_type === 'create_agent' && (
                        <div>
                          <strong>Agent Created:</strong> {test.result.agent_id}<br/>
                          <strong>Role:</strong> {test.result.role}<br/>
                          <strong>Capabilities:</strong> {test.result.capabilities_available}
                        </div>
                      )}
                      
                      {test.test_type === 'deploy_swarm' && (
                        <div>
                          <strong>Swarm ID:</strong> {test.result.swarm_id}<br/>
                          <strong>Agents Deployed:</strong> {test.result.agents_deployed}<br/>
                          <strong>Intelligence Mode:</strong> {test.result.intelligence_mode}<br/>
                          <strong>Amplification:</strong> {test.result.estimated_capability_amplification}x
                        </div>
                      )}
                      
                      {test.test_type === 'collective_reasoning' && (
                        <div>
                          <strong>Session ID:</strong> {test.result.reasoning_session_id}<br/>
                          <strong>Participants:</strong> {test.result.participating_agents}<br/>
                          <strong>Confidence:</strong> {(test.result.collective_confidence * 100).toFixed(1)}%<br/>
                          <strong>Amplification:</strong> {test.result.intelligence_amplification}x
                        </div>
                      )}
                      
                      {test.test_type === 'knowledge_query' && (
                        <div>
                          <strong>Query:</strong> {test.result.query}<br/>
                          <strong>Sources:</strong> {test.result.source_documents?.length || 0}<br/>
                          <strong>Confidence:</strong> {(test.result.confidence * 100).toFixed(1)}%<br/>
                          <strong>Response Preview:</strong> {test.result.response?.substring(0, 100)}...
                        </div>
                      )}
                      
                      {test.test_type === 'comprehensive_demo' && (
                        <div>
                          <strong>Agent Created:</strong> {test.result.agent_created?.agent_id}<br/>
                          <strong>Swarm Deployed:</strong> {test.result.swarm_deployed?.agents_deployed} agents<br/>
                          <strong>Reasoning Confidence:</strong> {(test.result.collective_reasoning?.collective_confidence * 100).toFixed(1)}%<br/>
                          <strong>Intelligence Amplification:</strong> {test.result.collective_reasoning?.intelligence_amplification}x
                        </div>
                      )}
                      
                      {!['create_agent', 'deploy_swarm', 'collective_reasoning', 'knowledge_query', 'comprehensive_demo'].includes(test.test_type) && (
                        <pre className="whitespace-pre-wrap">
                          {JSON.stringify(test.result, null, 2)}
                        </pre>
                      )}
                    </div>
                  ) : (
                    <div className="text-red-600">
                      <strong>Error:</strong> {test.result.error}
                    </div>
                  )}
                </div>
                
                <div className="text-xs text-gray-500 mt-2">
                  {test.timestamp}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* System Information */}
      {systemStatus && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">System Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-2">AI Systems Status</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Enhanced AI:</span>
                  <Badge variant={systemStatus.enhanced_ai_available ? "success" : "destructive"}>
                    {systemStatus.enhanced_ai_available ? "Available" : "Unavailable"}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span>Neural Mesh:</span>
                  <Badge variant={systemStatus.neural_mesh_available ? "success" : "destructive"}>
                    {systemStatus.neural_mesh_available ? "Available" : "Unavailable"}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span>Active Agents:</span>
                  <span>{activeAgents.length}</span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-medium mb-2">Available Features</h4>
              <div className="space-y-1 text-sm">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-3 w-3 text-green-600" />
                  <span>Multi-Provider LLM Integration</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-3 w-3 text-green-600" />
                  <span>Advanced Reasoning Patterns</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-3 w-3 text-green-600" />
                  <span>Intelligent Agent Swarms</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-3 w-3 text-green-600" />
                  <span>Collective Intelligence</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-3 w-3 text-green-600" />
                  <span>Neural Mesh Memory</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-3 w-3 text-green-600" />
                  <span>Continuous Learning</span>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Instructions */}
      <Card className="p-6 bg-blue-50 border-blue-200">
        <h3 className="text-lg font-semibold mb-4 text-blue-900">How to Test AI Capabilities</h3>
        <div className="space-y-3 text-sm text-blue-800">
          <div className="flex items-start gap-2">
            <div className="w-6 h-6 rounded-full bg-blue-200 text-blue-800 flex items-center justify-center text-xs font-bold">1</div>
            <div>
              <strong>Start with Quick Tests:</strong> Use the quick test buttons to create agents, deploy swarms, and test basic functionality.
            </div>
          </div>
          <div className="flex items-start gap-2">
            <div className="w-6 h-6 rounded-full bg-blue-200 text-blue-800 flex items-center justify-center text-xs font-bold">2</div>
            <div>
              <strong>Try Advanced Features:</strong> Test knowledge synthesis, capability execution, and neural mesh memory operations.
            </div>
          </div>
          <div className="flex items-start gap-2">
            <div className="w-6 h-6 rounded-full bg-blue-200 text-blue-800 flex items-center justify-center text-xs font-bold">3</div>
            <div>
              <strong>Run Comprehensive Demo:</strong> Execute the full AI capabilities demonstration to see all systems working together.
            </div>
          </div>
          <div className="flex items-start gap-2">
            <div className="w-6 h-6 rounded-full bg-blue-200 text-blue-800 flex items-center justify-center text-xs font-bold">4</div>
            <div>
              <strong>Monitor Results:</strong> Watch the test results panel to see detailed information about each operation.
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
