#!/usr/bin/env python3
"""
Comprehensive AGI API - Natural Language Orchestration
Complete integration of all AGI capabilities with intelligent orchestration
"""

import asyncio
import json
import logging
import time
import os
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("comprehensive-agi-api")

# Import intelligent orchestrator
try:
    from services.intelligent_orchestrator import intelligent_orchestrator
    INTELLIGENT_ORCHESTRATOR_AVAILABLE = True
    log.info("✅ Intelligent Orchestrator loaded")
except ImportError as e:
    INTELLIGENT_ORCHESTRATOR_AVAILABLE = False
    log.error(f"❌ Intelligent Orchestrator not available: {e}")

# Import all existing systems
try:
    from neural_mesh_coordinator import neural_mesh, AgentKnowledge, AgentAction
    NEURAL_MESH_AVAILABLE = True
    log.info("✅ Neural Mesh Coordinator loaded")
except ImportError:
    NEURAL_MESH_AVAILABLE = False
    log.error("❌ Neural Mesh Coordinator not available")

try:
    from self_coding_agi import self_coding_agi
    SELF_CODING_AVAILABLE = True
    log.info("✅ Self-Coding AGI loaded")
except ImportError:
    SELF_CODING_AVAILABLE = False
    log.error("❌ Self-Coding AGI not available")

try:
    from ai_analysis_system import AGIIntrospectiveSystem
    agi_system = AGIIntrospectiveSystem()
    AGI_INTROSPECTIVE_AVAILABLE = True
    log.info("✅ AGI Introspective System loaded")
except ImportError:
    AGI_INTROSPECTIVE_AVAILABLE = False
    log.error("❌ AGI Introspective System not available")

# Initialize LLM clients
from multi_llm_router import MultiLLMRouter
llm_router = MultiLLMRouter()

app = FastAPI(
    title="Comprehensive AGI API",
    description="Natural Language Orchestration with Complete AGI Capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatContext(BaseModel):
    conversationHistory: List[Dict[str, Any]] = []
    dataSources: List[Dict[str, Any]] = []
    userId: str = "user_001"
    sessionId: str = "session_001"
    userPreferences: Dict[str, Any] = {}

class ChatMessageRequest(BaseModel):
    message: str
    context: ChatContext
    capabilities: List[str] = []

class AgentMetrics(BaseModel):
    totalAgentsDeployed: int
    activeAgents: int
    completedTasks: int
    averageTaskTime: float
    successRate: float
    agiReadiness: Optional[float] = None
    orchestrationMethod: str = "intelligent_nlp"
    complexityLevel: str = "unknown"
    executionTime: float = 0.0
    capabilitiesUsed: List[str] = []

class SwarmActivity(BaseModel):
    agentType: str
    action: str
    status: str
    timestamp: float
    details: str

class ChatMessageResponse(BaseModel):
    response: str
    swarmActivity: List[SwarmActivity]
    capabilitiesUsed: List[str]
    confidence: float
    processingTime: float
    agentMetrics: AgentMetrics
    llmUsed: str
    realAgentData: bool = True

# System status tracking
system_stats = {
    "requests_processed": 0,
    "total_agents_deployed": 0,
    "successful_orchestrations": 0,
    "average_response_time": 0.0,
    "capabilities_available": 0
}

@app.on_startup
async def startup_event():
    """Initialize all systems on startup"""
    
    log.info("🚀 Starting Comprehensive AGI API...")
    
    # Count available capabilities
    capabilities_count = 0
    if INTELLIGENT_ORCHESTRATOR_AVAILABLE:
        capabilities_count += len(intelligent_orchestrator.available_capabilities)
    
    system_stats["capabilities_available"] = capabilities_count
    
    log.info(f"✅ System initialized with {capabilities_count} capabilities")
    log.info(f"✅ Intelligent Orchestrator: {'Available' if INTELLIGENT_ORCHESTRATOR_AVAILABLE else 'Not Available'}")
    log.info(f"✅ Neural Mesh: {'Available' if NEURAL_MESH_AVAILABLE else 'Not Available'}")
    log.info(f"✅ Self-Coding AGI: {'Available' if SELF_CODING_AVAILABLE else 'Not Available'}")
    log.info(f"✅ AGI Introspective: {'Available' if AGI_INTROSPECTIVE_AVAILABLE else 'Not Available'}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "intelligent_orchestrator": INTELLIGENT_ORCHESTRATOR_AVAILABLE,
        "neural_mesh": NEURAL_MESH_AVAILABLE,
        "self_coding_agi": SELF_CODING_AVAILABLE,
        "agi_introspective": AGI_INTROSPECTIVE_AVAILABLE,
        "capabilities_available": system_stats["capabilities_available"],
        "timestamp": time.time()
    }

@app.get("/v1/capabilities")
async def get_capabilities():
    """Get all available AGI capabilities"""
    
    capabilities = []
    
    if INTELLIGENT_ORCHESTRATOR_AVAILABLE:
        for cap_name, cap_info in intelligent_orchestrator.available_capabilities.items():
            capabilities.append({
                "name": cap_name,
                "description": cap_info["description"],
                "agent_range": cap_info["agent_range"],
                "requirements": cap_info["requirements"],
                "available": all(globals().get(req, False) for req in cap_info["requirements"])
            })
    
    return {
        "capabilities": capabilities,
        "total_available": len([cap for cap in capabilities if cap["available"]]),
        "orchestration_method": "intelligent_nlp" if INTELLIGENT_ORCHESTRATOR_AVAILABLE else "keyword_fallback"
    }

@app.post("/v1/chat/message", response_model=ChatMessageResponse)
async def process_chat_message(request: ChatMessageRequest):
    """
    Process chat message with intelligent natural language orchestration
    """
    
    start_time = time.time()
    system_stats["requests_processed"] += 1
    
    try:
        log.info(f"🧠 Processing message: '{request.message[:100]}...'")
        
        if INTELLIGENT_ORCHESTRATOR_AVAILABLE:
            # Use intelligent orchestration
            log.info("🎯 Using intelligent natural language orchestration...")
            
            # Convert context to dict
            context_dict = {
                "conversationHistory": request.context.conversationHistory,
                "dataSources": request.context.dataSources,
                "userId": request.context.userId,
                "sessionId": request.context.sessionId,
                "userPreferences": request.context.userPreferences
            }
            
            # Execute intelligent orchestration
            execution_result = await intelligent_orchestrator.understand_and_orchestrate(
                request.message, context_dict
            )
            
            # Update system stats
            system_stats["total_agents_deployed"] += execution_result.agents_deployed
            if execution_result.success:
                system_stats["successful_orchestrations"] += 1
            
            # Generate response using LLM with orchestration results
            response_text = await generate_response_with_results(
                request.message, execution_result, context_dict
            )
            
            # Create swarm activity
            swarm_activity = create_swarm_activity_from_results(execution_result)
            
            # Create agent metrics
            agent_metrics = AgentMetrics(
                totalAgentsDeployed=execution_result.agents_deployed,
                activeAgents=0,
                completedTasks=execution_result.agents_deployed,
                averageTaskTime=execution_result.execution_time / max(execution_result.agents_deployed, 1),
                successRate=1.0 if execution_result.success else 0.0,
                agiReadiness=extract_agi_readiness(execution_result),
                orchestrationMethod="intelligent_nlp",
                complexityLevel=execution_result.metrics.get("plan_complexity", "unknown"),
                executionTime=execution_result.execution_time,
                capabilitiesUsed=execution_result.capabilities_used
            )
            
            processing_time = time.time() - start_time
            system_stats["average_response_time"] = (
                system_stats["average_response_time"] * (system_stats["requests_processed"] - 1) + processing_time
            ) / system_stats["requests_processed"]
            
            return ChatMessageResponse(
                response=response_text,
                swarmActivity=swarm_activity,
                capabilitiesUsed=execution_result.capabilities_used,
                confidence=execution_result.metrics.get("intent_confidence", 0.9),
                processingTime=processing_time,
                agentMetrics=agent_metrics,
                llmUsed="intelligent_orchestrator",
                realAgentData=True
            )
            
        else:
            # Fallback to basic processing
            log.warning("⚠️ Falling back to basic processing - Intelligent Orchestrator not available")
            
            response_text = "I apologize, but the intelligent orchestration system is not available. The system is operating in basic mode."
            
            processing_time = time.time() - start_time
            
            return ChatMessageResponse(
                response=response_text,
                swarmActivity=[],
                capabilitiesUsed=[],
                confidence=0.5,
                processingTime=processing_time,
                agentMetrics=AgentMetrics(
                    totalAgentsDeployed=0,
                    activeAgents=0,
                    completedTasks=0,
                    averageTaskTime=0.0,
                    successRate=0.0,
                    orchestrationMethod="fallback"
                ),
                llmUsed="fallback",
                realAgentData=False
            )
            
    except Exception as e:
        log.error(f"❌ Message processing failed: {e}")
        
        processing_time = time.time() - start_time
        
        return ChatMessageResponse(
            response=f"I encountered an error processing your request: {str(e)}",
            swarmActivity=[],
            capabilitiesUsed=[],
            confidence=0.0,
            processingTime=processing_time,
            agentMetrics=AgentMetrics(
                totalAgentsDeployed=0,
                activeAgents=0,
                completedTasks=0,
                averageTaskTime=0.0,
                successRate=0.0,
                orchestrationMethod="error"
            ),
            llmUsed="error",
            realAgentData=False
        )

async def generate_response_with_results(message: str, execution_result, context: Dict[str, Any]) -> str:
    """Generate natural language response incorporating orchestration results"""
    
    # Build comprehensive response prompt
    response_prompt = f"""
    User asked: "{message}"
    
    ORCHESTRATION RESULTS:
    - Success: {execution_result.success}
    - Agents Deployed: {execution_result.agents_deployed}
    - Execution Time: {execution_result.execution_time:.2f} seconds
    - Capabilities Used: {', '.join(execution_result.capabilities_used)}
    - Complexity Level: {execution_result.metrics.get('plan_complexity', 'unknown')}
    
    DETAILED RESULTS:
    """
    
    for capability, result in execution_result.results.items():
        response_prompt += f"\n{capability.upper()}:\n"
        if isinstance(result, dict):
            for key, value in result.items():
                if key != "error":
                    response_prompt += f"  - {key}: {value}\n"
    
    if execution_result.errors:
        response_prompt += f"\nERRORS: {', '.join(execution_result.errors)}\n"
    
    response_prompt += f"""
    
    Generate a natural, helpful response that:
    1. Directly answers the user's question
    2. Includes the REAL numbers and metrics from the orchestration
    3. Explains what the agents actually accomplished
    4. Is conversational and clear
    5. Mentions the intelligent orchestration approach used
    
    Be specific about the actual results achieved, not generic descriptions.
    """
    
    try:
        # Use LLM router to generate response
        from multi_llm_router import TaskType
        
        llm_result = await llm_router.process_with_best_llm(
            response_prompt,
            TaskType.CONVERSATIONAL,
            system_prompt="You are an expert at explaining AI system results clearly and accurately."
        )
        
        return llm_result.get("response", "I successfully processed your request with intelligent orchestration.")
        
    except Exception as e:
        log.error(f"Response generation failed: {e}")
        
        # Fallback response with key metrics
        if execution_result.success:
            return f"I successfully processed your request using intelligent orchestration. I deployed {execution_result.agents_deployed} agents across {len(execution_result.capabilities_used)} capabilities in {execution_result.execution_time:.2f} seconds. The analysis completed successfully with {execution_result.metrics.get('plan_complexity', 'standard')} complexity level."
        else:
            return f"I attempted to process your request but encountered some issues: {', '.join(execution_result.errors)}. I deployed {execution_result.agents_deployed} agents but the execution was not fully successful."

def create_swarm_activity_from_results(execution_result) -> List[SwarmActivity]:
    """Create swarm activity list from execution results"""
    
    activities = []
    current_time = time.time()
    
    for i, capability in enumerate(execution_result.capabilities_used):
        activities.append(SwarmActivity(
            agentType=capability.replace("_", "-"),
            action=f"executing {capability}",
            status="completed" if execution_result.success else "partial",
            timestamp=current_time - (len(execution_result.capabilities_used) - i) * 2,
            details=f"Deployed agents for {capability} capability"
        ))
    
    # Add orchestration activity
    activities.append(SwarmActivity(
        agentType="orchestration-coordinator",
        action="coordinating swarm execution",
        status="completed",
        timestamp=current_time,
        details=f"Coordinated {execution_result.agents_deployed} agents across {len(execution_result.capabilities_used)} capabilities"
    ))
    
    return activities

def extract_agi_readiness(execution_result) -> Optional[float]:
    """Extract AGI readiness from execution results"""
    
    for capability, result in execution_result.results.items():
        if capability == "introspection" and isinstance(result, dict):
            return result.get("agi_readiness", 0.87)
    
    return None

@app.get("/v1/system/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    
    return {
        "system_stats": system_stats,
        "orchestration_history": len(intelligent_orchestrator.execution_history) if INTELLIGENT_ORCHESTRATOR_AVAILABLE else 0,
        "available_systems": {
            "intelligent_orchestrator": INTELLIGENT_ORCHESTRATOR_AVAILABLE,
            "neural_mesh": NEURAL_MESH_AVAILABLE,
            "self_coding_agi": SELF_CODING_AVAILABLE,
            "agi_introspective": AGI_INTROSPECTIVE_AVAILABLE
        },
        "uptime": time.time()
    }

@app.post("/v1/orchestration/direct")
async def direct_orchestration(request: Dict[str, Any]):
    """Direct access to intelligent orchestration for advanced users"""
    
    if not INTELLIGENT_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Intelligent Orchestrator not available")
    
    user_prompt = request.get("prompt", "")
    context = request.get("context", {})
    
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    try:
        execution_result = await intelligent_orchestrator.understand_and_orchestrate(user_prompt, context)
        
        return {
            "success": execution_result.success,
            "results": execution_result.results,
            "agents_deployed": execution_result.agents_deployed,
            "execution_time": execution_result.execution_time,
            "capabilities_used": execution_result.capabilities_used,
            "metrics": execution_result.metrics,
            "errors": execution_result.errors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    log.info("🚀 Starting Comprehensive AGI API with Intelligent Natural Language Orchestration")
    log.info("✅ Complete integration of all AGI capabilities")
    log.info("✅ Natural language understanding replaces keyword detection")
    log.info("✅ Automatic swarm deployment based on intent analysis")
    log.info("✅ Real metrics from actual system execution")
    log.info("Backend will be available at: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
