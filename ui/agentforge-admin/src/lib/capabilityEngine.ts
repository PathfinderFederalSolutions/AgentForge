/**
 * Capability Engine - Phase 1 Implementation
 * Intelligent capability discovery and suggestions for chat interface
 */

import { DataSource, Message } from './store';

export interface CapabilitySuggestion {
  id: string;
  type: 'input' | 'output' | 'processing' | 'optimization';
  icon: string;
  title: string;
  description: string;
  examples: string[];
  action?: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
}

export interface CapabilityMatch {
  capability: string;
  confidence: number;
  reasoning: string;
  suggestedAgentTypes: string[];
  estimatedComplexity: number;
}

export interface InputAnalysis {
  detectedIntents: string[];
  suggestedCapabilities: CapabilityMatch[];
  recommendedActions: CapabilitySuggestion[];
  dataRequirements: string[];
  outputPredictions: string[];
}

class CapabilityEngine {
  private capabilities: Map<string, CapabilitySuggestion> = new Map();
  private intentPatterns: Map<string, RegExp[]> = new Map();
  private contextHistory: string[] = [];

  constructor() {
    this.initializeCapabilities();
    this.initializeIntentPatterns();
  }

  private initializeCapabilities() {
    const capabilities: CapabilitySuggestion[] = [
      {
        id: 'universal_input_processing',
        type: 'input',
        icon: '📁',
        title: 'Universal Input Processing',
        description: 'Process any input type with specialized AI agents',
        examples: [
          'Upload documents (PDF, Word, Excel, PowerPoint)',
          'Process images and videos with computer vision',
          'Analyze audio files and transcribe speech',
          'Connect to real-time data streams and APIs',
          'Import databases and structured data'
        ],
        action: 'showUploadModal',
        priority: 'high',
        confidence: 0.95
      },
      {
        id: 'neural_mesh_analysis',
        type: 'processing',
        icon: '🧠',
        title: 'Neural Mesh Intelligence',
        description: 'Deep pattern analysis using 4-tier memory system',
        examples: [
          'Analyze complex patterns across multiple data sources',
          'Generate insights from historical conversation context',
          'Cross-reference organizational knowledge base',
          'Detect emergent patterns in user behavior',
          'Build comprehensive knowledge graphs'
        ],
        action: 'enableNeuralMeshMode',
        priority: 'high',
        confidence: 0.92
      },
      {
        id: 'quantum_coordination',
        type: 'processing',
        icon: '⚡',
        title: 'Quantum Agent Coordination',
        description: 'Million-scale agent coordination for complex problems',
        examples: [
          'Deploy 1000+ specialized agents for enterprise analysis',
          'Coordinate parallel processing across agent clusters',
          'Optimize resource allocation using quantum algorithms',
          'Handle complex multi-domain problem solving',
          'Scale automatically based on problem complexity'
        ],
        action: 'enableQuantumMode',
        priority: 'medium',
        confidence: 0.88
      },
      {
        id: 'universal_output_generation',
        type: 'output',
        icon: '🛠️',
        title: 'Universal Output Generation',
        description: 'Generate any output format from natural language',
        examples: [
          'Build complete web and mobile applications',
          'Create professional reports and presentations',
          'Generate images, videos, and multimedia content',
          'Design interactive dashboards and visualizations',
          'Produce automation scripts and workflows'
        ],
        action: 'showOutputOptions',
        priority: 'high',
        confidence: 0.90
      },
      {
        id: 'real_time_processing',
        type: 'processing',
        icon: '📡',
        title: 'Real-Time Stream Processing',
        description: 'Monitor and analyze live data streams',
        examples: [
          'Monitor IoT sensors and device telemetry',
          'Analyze social media feeds and news streams',
          'Process financial market data in real-time',
          'Detect anomalies in system logs and metrics',
          'Track user behavior and engagement patterns'
        ],
        action: 'enableStreamingMode',
        priority: 'medium',
        confidence: 0.85
      },
      {
        id: 'predictive_intelligence',
        type: 'processing',
        icon: '🔮',
        title: 'Predictive Intelligence',
        description: 'Forecast trends and predict outcomes',
        examples: [
          'Build machine learning models for forecasting',
          'Predict customer behavior and preferences',
          'Analyze market trends and business metrics',
          'Identify potential risks and opportunities',
          'Generate scenario planning and what-if analysis'
        ],
        action: 'enablePredictiveMode',
        priority: 'medium',
        confidence: 0.87
      },
      {
        id: 'creative_generation',
        type: 'output',
        icon: '🎨',
        title: 'Creative Content Generation',
        description: 'Generate creative and artistic content',
        examples: [
          'Create original images and artwork',
          'Compose music and audio content',
          'Write stories, scripts, and creative content',
          'Design logos, graphics, and visual assets',
          'Generate video content and animations'
        ],
        action: 'enableCreativeMode',
        priority: 'medium',
        confidence: 0.83
      },
      {
        id: 'automation_workflows',
        type: 'output',
        icon: '🔄',
        title: 'Process Automation',
        description: 'Automate complex business processes',
        examples: [
          'Create RPA bots for repetitive tasks',
          'Design workflow automation systems',
          'Build API integrations and data pipelines',
          'Generate testing and deployment scripts',
          'Automate reporting and monitoring systems'
        ],
        action: 'showAutomationOptions',
        priority: 'medium',
        confidence: 0.86
      }
    ];

    capabilities.forEach(cap => this.capabilities.set(cap.id, cap));
  }

  private initializeIntentPatterns() {
    this.intentPatterns.set('analyze', [
      /\b(analyz|pattern|insight|understand|examine|investigate|study)\w*\b/gi,
      /\b(what does|how does|why does|explain|breakdown|dissect)\b/gi,
      /\b(trends|correlation|relationship|connection|link)\b/gi
    ]);

    this.intentPatterns.set('create', [
      /\b(creat|build|generat|mak|produc|design|develop)\w*\b/gi,
      /\b(app|application|website|dashboard|report|document)\b/gi,
      /\b(I need|I want|can you|please)\b.*\b(creat|build|mak)\w*\b/gi
    ]);

    this.intentPatterns.set('optimize', [
      /\b(optim|improv|enhanc|better|faster|efficient|performance)\w*\b/gi,
      /\b(speed up|make better|increase|reduce|minimize|maximize)\b/gi,
      /\b(bottleneck|slow|issue|problem|fix)\b/gi
    ]);

    this.intentPatterns.set('predict', [
      /\b(predict|forecast|trend|future|expect|anticipat|project)\w*\b/gi,
      /\b(what will|when will|how will|likely|probable|chance)\b/gi,
      /\b(model|algorithm|machine learning|AI|neural)\b/gi
    ]);

    this.intentPatterns.set('monitor', [
      /\b(monitor|watch|track|observ|alert|notif|detect)\w*\b/gi,
      /\b(real.?time|live|streaming|continuous|ongoing)\b/gi,
      /\b(anomal|unusual|strange|error|issue|problem)\b/gi
    ]);

    this.intentPatterns.set('upload', [
      /\b(upload|import|load|add|attach|file|data|document)\b/gi,
      /\b(csv|json|pdf|excel|image|video|audio)\b/gi,
      /\b(I have|here is|attached|included)\b/gi
    ]);
  }

  analyzeInput(
    input: string, 
    context: {
      conversationHistory: Message[];
      dataSources: DataSource[];
      userPreferences?: Record<string, any>;
    }
  ): InputAnalysis {
    const detectedIntents = this.detectIntents(input);
    const suggestedCapabilities = this.matchCapabilities(input, detectedIntents, context);
    const recommendedActions = this.generateRecommendations(input, suggestedCapabilities, context);
    const dataRequirements = this.identifyDataRequirements(input, context);
    const outputPredictions = this.predictOutputTypes(input, suggestedCapabilities);

    // Store in context history for learning
    this.contextHistory.push(input);
    if (this.contextHistory.length > 100) {
      this.contextHistory.shift();
    }

    return {
      detectedIntents,
      suggestedCapabilities,
      recommendedActions,
      dataRequirements,
      outputPredictions
    };
  }

  private detectIntents(input: string): string[] {
    const detectedIntents: string[] = [];
    const lowerInput = input.toLowerCase();

    for (const [intent, patterns] of this.intentPatterns.entries()) {
      for (const pattern of patterns) {
        if (pattern.test(input)) {
          detectedIntents.push(intent);
          break;
        }
      }
    }

    // Additional context-based intent detection
    if (lowerInput.includes('help') || lowerInput.includes('how') || lowerInput.includes('what can')) {
      detectedIntents.push('help');
    }

    if (lowerInput.includes('complex') || lowerInput.includes('enterprise') || lowerInput.includes('scale')) {
      detectedIntents.push('enterprise');
    }

    return [...new Set(detectedIntents)];
  }

  private matchCapabilities(
    input: string, 
    intents: string[], 
    context: { conversationHistory: Message[]; dataSources: DataSource[] }
  ): CapabilityMatch[] {
    const matches: CapabilityMatch[] = [];
    const lowerInput = input.toLowerCase();

    // Intent-based capability matching
    const intentCapabilityMap: Record<string, string[]> = {
      'analyze': ['neural_mesh_analysis', 'predictive_intelligence'],
      'create': ['universal_output_generation', 'creative_generation'],
      'optimize': ['quantum_coordination', 'automation_workflows'],
      'predict': ['predictive_intelligence', 'neural_mesh_analysis'],
      'monitor': ['real_time_processing', 'neural_mesh_analysis'],
      'upload': ['universal_input_processing'],
      'enterprise': ['quantum_coordination', 'neural_mesh_analysis']
    };

    for (const intent of intents) {
      const capabilityIds = intentCapabilityMap[intent] || [];
      for (const capId of capabilityIds) {
        const capability = this.capabilities.get(capId);
        if (capability && !matches.find(m => m.capability === capId)) {
          matches.push({
            capability: capId,
            confidence: this.calculateConfidence(input, intent, capability, context),
            reasoning: this.generateReasoning(intent, capability, context),
            suggestedAgentTypes: this.suggestAgentTypes(capId, intent),
            estimatedComplexity: this.estimateComplexity(input, intent, context)
          });
        }
      }
    }

    // Keyword-based matching
    const keywordMatches = this.matchByKeywords(lowerInput);
    matches.push(...keywordMatches.filter(km => !matches.find(m => m.capability === km.capability)));

    // Context-based matching
    if (context.dataSources.length > 0) {
      const dataMatch: CapabilityMatch = {
        capability: 'universal_input_processing',
        confidence: 0.9,
        reasoning: `You have ${context.dataSources.length} data sources that can be processed`,
        suggestedAgentTypes: ['data-processor', 'multi-modal-analyzer'],
        estimatedComplexity: Math.min(context.dataSources.length * 0.3, 2.0)
      };
      if (!matches.find(m => m.capability === 'universal_input_processing')) {
        matches.push(dataMatch);
      }
    }

    return matches.sort((a, b) => b.confidence - a.confidence);
  }

  private matchByKeywords(input: string): CapabilityMatch[] {
    const matches: CapabilityMatch[] = [];
    
    const keywordMap: Record<string, { capability: string; keywords: string[]; confidence: number }> = {
      'app_creation': {
        capability: 'universal_output_generation',
        keywords: ['app', 'application', 'website', 'mobile', 'web', 'frontend', 'backend'],
        confidence: 0.85
      },
      'data_analysis': {
        capability: 'neural_mesh_analysis',
        keywords: ['data', 'analysis', 'pattern', 'insight', 'statistics', 'correlation'],
        confidence: 0.88
      },
      'real_time': {
        capability: 'real_time_processing',
        keywords: ['real-time', 'live', 'streaming', 'monitor', 'continuous'],
        confidence: 0.82
      },
      'prediction': {
        capability: 'predictive_intelligence',
        keywords: ['predict', 'forecast', 'trend', 'future', 'model', 'ml', 'ai'],
        confidence: 0.86
      },
      'creative': {
        capability: 'creative_generation',
        keywords: ['image', 'video', 'art', 'creative', 'design', 'generate', 'music'],
        confidence: 0.80
      }
    };

    for (const [key, config] of Object.entries(keywordMap)) {
      const keywordCount = config.keywords.filter(keyword => input.includes(keyword)).length;
      if (keywordCount > 0) {
        const confidence = config.confidence * (keywordCount / config.keywords.length);
        matches.push({
          capability: config.capability,
          confidence,
          reasoning: `Detected ${keywordCount} relevant keywords: ${config.keywords.filter(k => input.includes(k)).join(', ')}`,
          suggestedAgentTypes: this.suggestAgentTypes(config.capability, key),
          estimatedComplexity: keywordCount * 0.3
        });
      }
    }

    return matches;
  }

  private calculateConfidence(
    input: string, 
    intent: string, 
    capability: CapabilitySuggestion,
    context: { conversationHistory: Message[]; dataSources: DataSource[] }
  ): number {
    let confidence = capability.confidence;

    // Boost confidence based on explicit matches
    if (capability.examples.some(example => 
      example.toLowerCase().split(' ').some(word => input.toLowerCase().includes(word))
    )) {
      confidence += 0.1;
    }

    // Context boosts
    if (context.dataSources.length > 0 && capability.type === 'input') {
      confidence += 0.05;
    }

    if (context.conversationHistory.length > 3 && capability.id === 'neural_mesh_analysis') {
      confidence += 0.05;
    }

    // Intent alignment
    const intentBoosts: Record<string, Record<string, number>> = {
      'analyze': { 'neural_mesh_analysis': 0.1, 'predictive_intelligence': 0.05 },
      'create': { 'universal_output_generation': 0.1, 'creative_generation': 0.05 },
      'enterprise': { 'quantum_coordination': 0.15 }
    };

    if (intentBoosts[intent] && intentBoosts[intent][capability.id]) {
      confidence += intentBoosts[intent][capability.id];
    }

    return Math.min(confidence, 0.99);
  }

  private generateReasoning(
    intent: string, 
    capability: CapabilitySuggestion,
    context: { conversationHistory: Message[]; dataSources: DataSource[] }
  ): string {
    const reasons: string[] = [];

    if (intent === 'analyze' && capability.id === 'neural_mesh_analysis') {
      reasons.push('Your request involves analysis, which is perfect for neural mesh intelligence');
    }

    if (intent === 'create' && capability.id === 'universal_output_generation') {
      reasons.push('You want to create something, which matches universal output generation');
    }

    if (context.dataSources.length > 0) {
      reasons.push(`You have ${context.dataSources.length} data sources that can enhance this capability`);
    }

    if (context.conversationHistory.length > 5) {
      reasons.push('Your conversation history provides rich context for enhanced processing');
    }

    return reasons.join('. ') || `This capability aligns well with your ${intent} intent`;
  }

  private suggestAgentTypes(capabilityId: string, intent: string): string[] {
    const agentTypeMap: Record<string, string[]> = {
      'universal_input_processing': ['data-processor', 'multi-modal-analyzer', 'format-converter'],
      'neural_mesh_analysis': ['neural-mesh', 'pattern-detector', 'knowledge-synthesizer'],
      'quantum_coordination': ['quantum-scheduler', 'cluster-coordinator', 'resource-optimizer'],
      'universal_output_generation': ['output-generator', 'code-generator', 'content-creator'],
      'real_time_processing': ['stream-processor', 'real-time-analyzer', 'anomaly-detector'],
      'predictive_intelligence': ['ml-trainer', 'predictor', 'trend-analyzer'],
      'creative_generation': ['creative-agent', 'media-generator', 'design-assistant'],
      'automation_workflows': ['workflow-builder', 'automation-agent', 'integration-specialist']
    };

    return agentTypeMap[capabilityId] || ['general-intelligence'];
  }

  private estimateComplexity(input: string, intent: string, context: { dataSources: DataSource[] }): number {
    let complexity = 1.0;

    // Base complexity by intent
    const intentComplexity: Record<string, number> = {
      'analyze': 1.5,
      'create': 2.0,
      'optimize': 1.8,
      'predict': 2.2,
      'monitor': 1.3,
      'enterprise': 3.0
    };

    complexity += intentComplexity[intent] || 0.5;

    // Data source complexity
    complexity += context.dataSources.length * 0.3;

    // Input length and complexity indicators
    const complexityKeywords = ['complex', 'enterprise', 'scale', 'multiple', 'advanced', 'sophisticated'];
    const keywordCount = complexityKeywords.filter(keyword => input.toLowerCase().includes(keyword)).length;
    complexity += keywordCount * 0.4;

    return Math.min(complexity, 5.0);
  }

  private identifyDataRequirements(input: string, context: { dataSources: DataSource[] }): string[] {
    const requirements: string[] = [];

    if (input.toLowerCase().includes('analyz') && context.dataSources.length === 0) {
      requirements.push('Structured data for analysis (CSV, JSON, database)');
    }

    if (input.toLowerCase().includes('image') || input.toLowerCase().includes('visual')) {
      requirements.push('Image files or visual data');
    }

    if (input.toLowerCase().includes('predict') || input.toLowerCase().includes('forecast')) {
      requirements.push('Historical data for training predictive models');
    }

    if (input.toLowerCase().includes('real-time') || input.toLowerCase().includes('monitor')) {
      requirements.push('Live data streams or API endpoints');
    }

    return requirements;
  }

  private predictOutputTypes(input: string, capabilities: CapabilityMatch[]): string[] {
    const outputs: string[] = [];

    for (const capability of capabilities) {
      const cap = this.capabilities.get(capability.capability);
      if (cap) {
        if (cap.id === 'universal_output_generation') {
          if (input.toLowerCase().includes('app')) outputs.push('Complete application with UI and backend');
          if (input.toLowerCase().includes('report')) outputs.push('Professional report with insights and visualizations');
          if (input.toLowerCase().includes('dashboard')) outputs.push('Interactive dashboard with real-time data');
        }
        
        if (cap.id === 'neural_mesh_analysis') {
          outputs.push('Deep insights and pattern analysis');
          outputs.push('Knowledge graph connections');
        }
        
        if (cap.id === 'predictive_intelligence') {
          outputs.push('Predictive models and forecasts');
          outputs.push('Trend analysis and recommendations');
        }
      }
    }

    return [...new Set(outputs)];
  }

  private generateRecommendations(
    input: string,
    capabilities: CapabilityMatch[],
    context: { conversationHistory: Message[]; dataSources: DataSource[] }
  ): CapabilitySuggestion[] {
    const recommendations: CapabilitySuggestion[] = [];

    // Add top capability matches as recommendations
    for (const match of capabilities.slice(0, 3)) {
      const capability = this.capabilities.get(match.capability);
      if (capability) {
        recommendations.push({
          ...capability,
          confidence: match.confidence,
          priority: match.confidence > 0.8 ? 'high' : match.confidence > 0.6 ? 'medium' : 'low'
        });
      }
    }

    // Add contextual recommendations
    if (context.dataSources.length === 0 && capabilities.some(c => c.capability === 'neural_mesh_analysis')) {
      recommendations.push({
        id: 'upload_data_suggestion',
        type: 'input',
        icon: '📊',
        title: 'Upload Data for Better Analysis',
        description: 'Adding data sources will significantly enhance analysis capabilities',
        examples: ['Upload CSV files with your data', 'Connect to your database', 'Import documents for analysis'],
        action: 'showUploadModal',
        priority: 'high',
        confidence: 0.9
      });
    }

    if (capabilities.some(c => c.estimatedComplexity > 2.0)) {
      recommendations.push({
        id: 'quantum_scaling_suggestion',
        type: 'optimization',
        icon: '⚡',
        title: 'Scale with Quantum Coordination',
        description: 'Your request is complex - I can deploy quantum-coordinated agent swarms for optimal results',
        examples: ['Deploy 100+ specialized agents', 'Use quantum superposition for parallel processing', 'Coordinate million-scale agent clusters'],
        action: 'enableQuantumMode',
        priority: 'medium',
        confidence: 0.85
      });
    }

    return recommendations;
  }

  getCapabilityById(id: string): CapabilitySuggestion | undefined {
    return this.capabilities.get(id);
  }

  getAllCapabilities(): CapabilitySuggestion[] {
    return Array.from(this.capabilities.values());
  }

  getCapabilitiesByType(type: CapabilitySuggestion['type']): CapabilitySuggestion[] {
    return Array.from(this.capabilities.values()).filter(cap => cap.type === type);
  }

  // Real-time suggestions as user types
  getRealtimeSuggestions(partialInput: string): CapabilitySuggestion[] {
    if (partialInput.length < 3) return [];

    const suggestions: CapabilitySuggestion[] = [];
    const lowerInput = partialInput.toLowerCase();

    // Quick keyword matching for real-time suggestions
    if (lowerInput.includes('uplo') || lowerInput.includes('file') || lowerInput.includes('data')) {
      suggestions.push(this.capabilities.get('universal_input_processing')!);
    }

    if (lowerInput.includes('creat') || lowerInput.includes('build') || lowerInput.includes('app')) {
      suggestions.push(this.capabilities.get('universal_output_generation')!);
    }

    if (lowerInput.includes('analy') || lowerInput.includes('pattern') || lowerInput.includes('insight')) {
      suggestions.push(this.capabilities.get('neural_mesh_analysis')!);
    }

    return suggestions.filter(Boolean).slice(0, 3);
  }
}

export default CapabilityEngine;
