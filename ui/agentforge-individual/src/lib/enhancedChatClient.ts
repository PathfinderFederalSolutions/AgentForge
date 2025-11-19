/**
 * Enhanced Chat Client for Natural AI Conversations
 * Seamlessly integrates enhanced AI capabilities while maintaining natural conversation flow
 */

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    agentsWorking?: number;
    processingTime?: number;
    confidence?: number;
    capabilitiesUsed?: string[];
    intelligenceAmplification?: number;
  };
}

export interface ChatContext {
  userId: string;
  sessionId: string;
  conversationHistory: ChatMessage[];
  dataSources: any[];
  userPreferences: any;
}

export interface ChatResponse {
  response: string;
  agentsDeployed?: number;
  swarmActivity?: any[];
  confidence?: number;
  processingTime?: number;
  capabilitiesUsed?: string[];
  intelligenceAmplification?: number;
  enhanced?: boolean;
}

class EnhancedChatClient {
  private mainApiUrl: string = 'http://localhost:8000';
  private enhancedApiUrl: string = 'http://localhost:8001';

  async sendMessage(message: string, context: ChatContext): Promise<ChatResponse> {
    try {
      // Always start with natural conversation through main API
      const mainResponse = await this.getMainApiResponse(message, context);
      
      // Check if we should enhance with advanced AI (background process)
      const shouldEnhance = this.shouldEnhanceRequest(message);
      
      if (shouldEnhance) {
        // Enhance in background without blocking main response
        this.enhanceInBackground(message, mainResponse, context);
        
        // Return main response immediately for natural feel
        return {
          ...mainResponse,
          enhanced: true
        };
      }
      
      return mainResponse;
      
    } catch (error) {
      console.error('Error in enhanced chat:', error);
      return {
        response: "I apologize, but I encountered an error. Please try again.",
        confidence: 0.1
      };
    }
  }

  private async getMainApiResponse(message: string, context: ChatContext): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.mainApiUrl}/v1/chat/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, context })
      });

      if (!response.ok) {
        throw new Error(`Main API error: ${response.status}`);
      }

      const data = await response.json();
      
      return {
        response: data.response || "I'm here to help! What would you like to know?",
        agentsDeployed: data.agentsDeployed || 1,
        confidence: data.confidence || 0.8,
        processingTime: data.processingTime || 1000,
        capabilitiesUsed: data.capabilitiesUsed || []
      };

    } catch (error) {
      console.error('Main API error:', error);
      
      // Fallback response
      return {
        response: this.generateFallbackResponse(message),
        confidence: 0.7,
        agentsDeployed: 1
      };
    }
  }

  private shouldEnhanceRequest(message: string): boolean {
    const complexityIndicators = [
      'analyze', 'research', 'investigate', 'optimize', 'design', 'create',
      'comprehensive', 'detailed', 'security', 'performance', 'architecture',
      'system', 'enterprise', 'complex', 'advanced'
    ];
    
    const messageWords = message.toLowerCase().split(/\s+/);
    const indicatorCount = messageWords.filter(word => 
      complexityIndicators.some(indicator => word.includes(indicator))
    ).length;
    
    // Enhance if complexity indicators present and message is substantial
    return indicatorCount >= 2 && message.length > 50;
  }

  private async enhanceInBackground(message: string, mainResponse: ChatResponse, context: ChatContext): Promise<void> {
    try {
      // Check enhanced AI availability
      const healthResponse = await fetch(`${this.enhancedApiUrl}/v1/ai/health`);
      if (!healthResponse.ok) return;

      console.log('🧠 Enhancing response with advanced AI...');

      // Deploy intelligent swarm
      const swarmResponse = await fetch(`${this.enhancedApiUrl}/v1/ai/swarms/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          objective: message,
          capabilities: this.getRecommendedCapabilities(message),
          specializations: ['analysis', 'research', 'problem_solving'],
          max_agents: 6,
          intelligence_mode: 'collective'
        })
      });

      if (swarmResponse.ok) {
        const swarmResult = await swarmResponse.json();
        console.log(`🚀 Deployed ${swarmResult.agents_deployed} agents for enhanced analysis`);

        // Wait for swarm initialization
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Coordinate collective reasoning
        const reasoningResponse = await fetch(`${this.enhancedApiUrl}/v1/ai/reasoning/collective`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            swarm_id: swarmResult.swarm_id,
            reasoning_objective: message,
            reasoning_pattern: 'collective_chain_of_thought'
          })
        });

        if (reasoningResponse.ok) {
          const reasoningResult = await reasoningResponse.json();
          console.log(`🧠 Collective reasoning complete with ${reasoningResult.intelligence_amplification}x amplification`);

          // Trigger a follow-up message event (this would be handled by the store)
          if (typeof window !== 'undefined') {
            window.dispatchEvent(new CustomEvent('enhancedAnalysisComplete', {
              detail: {
                originalMessage: message,
                enhancedResponse: reasoningResult.collective_reasoning,
                intelligenceAmplification: reasoningResult.intelligence_amplification,
                collectiveConfidence: reasoningResult.collective_confidence,
                agentsInvolved: reasoningResult.participating_agents
              }
            }));
          }
        }
      }

    } catch (error) {
      console.log('Background enhancement failed gracefully:', error);
      // Fail silently to maintain natural conversation flow
    }
  }

  private getRecommendedCapabilities(message: string): string[] {
    const capabilities = [];
    const messageLower = message.toLowerCase();
    
    if (messageLower.includes('security') || messageLower.includes('vulnerability')) {
      capabilities.push('security_analysis', 'threat_detection');
    }
    if (messageLower.includes('performance') || messageLower.includes('optimization')) {
      capabilities.push('performance_analysis', 'optimization');
    }
    if (messageLower.includes('data') || messageLower.includes('analysis')) {
      capabilities.push('data_processing', 'pattern_recognition');
    }
    if (messageLower.includes('research') || messageLower.includes('investigate')) {
      capabilities.push('research', 'investigation');
    }
    
    return capabilities.length > 0 ? capabilities : ['reasoning', 'analysis'];
  }

  private generateFallbackResponse(message: string): string {
    if (message.toLowerCase().includes('hello') || message.toLowerCase().includes('hi')) {
      return "Hello! I'm AgentForge AI, powered by advanced agent intelligence and collective reasoning capabilities. I can help you with analysis, research, optimization, and complex problem-solving. What would you like to explore?";
    }
    
    if (this.shouldEnhanceRequest(message)) {
      return `I understand you're looking for ${this.getRequestType(message)}. I'm coordinating with my intelligent agent network to provide you with comprehensive analysis. Let me work on this for you.`;
    }
    
    return "I'm here to help! I can assist with analysis, research, problem-solving, and much more. What specific task can I help you with?";
  }

  private getRequestType(message: string): string {
    const messageLower = message.toLowerCase();
    
    if (messageLower.includes('security')) return 'security analysis';
    if (messageLower.includes('performance')) return 'performance optimization';
    if (messageLower.includes('research')) return 'research and analysis';
    if (messageLower.includes('design')) return 'system design';
    if (messageLower.includes('analyze')) return 'comprehensive analysis';
    
    return 'assistance with your request';
  }

  async testConnection(): Promise<boolean> {
    try {
      const mainHealth = await fetch(`${this.mainApiUrl}/health`);
      const enhancedHealth = await fetch(`${this.enhancedApiUrl}/v1/ai/health`);
      
      return mainHealth.ok && enhancedHealth.ok;
    } catch (error) {
      return false;
    }
  }
}

export const enhancedChatClient = new EnhancedChatClient();
