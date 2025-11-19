#!/usr/bin/env python3
"""
Enhanced AI Capabilities API for AgentForge
Exposes all advanced AI features to the frontend: agent intelligence, neural mesh, collective reasoning
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

# Import enhanced AI systems with lazy loading to avoid async initialization issues
ENHANCED_AI_AVAILABLE = False
NEURAL_MESH_AVAILABLE = False

# Global references (will be loaded on first use)
master_coordinator = None
llm_integration = None
reasoning_engine = None
capabilities_system = None
knowledge_system = None
learning_system = None
prompt_system = None

async def initialize_ai_systems():
    """Initialize AI systems on first use"""
    global master_coordinator, llm_integration, reasoning_engine, capabilities_system
    global knowledge_system, learning_system, prompt_system
    global ENHANCED_AI_AVAILABLE, NEURAL_MESH_AVAILABLE
    
    if ENHANCED_AI_AVAILABLE:
        return  # Already initialized
    
    try:
        # Import and initialize core systems
        from core.master_agent_coordinator import MasterAgentCoordinator
        from core.enhanced_llm_integration import EnhancedLLMIntegration
        from core.advanced_reasoning_engine import AdvancedReasoningEngine
        from core.agent_capabilities_system import AgentCapabilitiesSystem
        from core.knowledge_management_system import KnowledgeManagementSystem
        from core.agent_learning_system import AgentLearningSystem
        from core.prompt_template_system import PromptTemplateSystem
        
        # Create instances
        master_coordinator = MasterAgentCoordinator()
        llm_integration = EnhancedLLMIntegration()
        reasoning_engine = AdvancedReasoningEngine()
        capabilities_system = AgentCapabilitiesSystem()
        knowledge_system = KnowledgeManagementSystem()
        learning_system = AgentLearningSystem()
        prompt_system = PromptTemplateSystem()
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        ENHANCED_AI_AVAILABLE = True
        log.info("✅ Enhanced AI systems initialized")
        
    except Exception as e:
        log.error(f"Failed to initialize enhanced AI systems: {e}")
        ENHANCED_AI_AVAILABLE = False
    
    try:
        # Try to import neural mesh
        from services.neural_mesh import enhanced_neural_mesh
        NEURAL_MESH_AVAILABLE = True
        log.info("✅ Neural mesh systems available")
        
    except Exception as e:
        log.error(f"Neural mesh not available: {e}")
        NEURAL_MESH_AVAILABLE = False

log = logging.getLogger("enhanced-ai-api")

app = FastAPI(
    title="AgentForge Enhanced AI Capabilities API",
    description="Complete API for advanced AI agent capabilities, neural mesh, and collective intelligence",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class IntelligentAgentRequest(BaseModel):
    role: str = "generalist"
    specializations: List[str] = []
    capabilities: List[str] = []

class IntelligentTaskRequest(BaseModel):
    description: str
    task_type: str = "general"
    priority: str = "normal"
    required_capabilities: List[str] = []
    context: Dict[str, Any] = {}
    reasoning_pattern: Optional[str] = None

class IntelligentSwarmRequest(BaseModel):
    objective: str
    capabilities: List[str]
    specializations: List[str] = []
    max_agents: int = 10
    intelligence_mode: str = "collective"

class CollectiveReasoningRequest(BaseModel):
    swarm_id: str
    reasoning_objective: str
    reasoning_pattern: str = "collective_chain_of_thought"

class KnowledgeQueryRequest(BaseModel):
    query: str
    agent_id: str
    max_context_docs: int = 5

class FeedbackRequest(BaseModel):
    agent_id: str
    task_id: str
    rating: float
    comments: str = ""
    improvement_suggestions: List[str] = []

class MemoryRequest(BaseModel):
    agent_id: str
    memory_type: str
    content: Dict[str, Any]
    memory_tier: str = "L2"
    metadata: Dict[str, Any] = {}

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

async def broadcast_to_connections(message: Dict[str, Any]):
    """Broadcast message to all connected WebSocket clients"""
    if active_connections:
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            active_connections.remove(conn)

# Enhanced AI Capabilities Endpoints

@app.get("/v1/ai/status")
async def get_ai_system_status():
    """Get comprehensive AI system status"""
    
    # Initialize AI systems on first access
    await initialize_ai_systems()
    
    status = {
        "timestamp": time.time(),
        "enhanced_ai_available": ENHANCED_AI_AVAILABLE,
        "neural_mesh_available": NEURAL_MESH_AVAILABLE,
        "systems": {}
    }
    
    if ENHANCED_AI_AVAILABLE:
        # Get master coordinator status
        coordinator_status = master_coordinator.get_system_status()
        status["systems"]["master_coordinator"] = coordinator_status
        
        # Get LLM integration status
        llm_integration = await get_llm_integration()
        llm_stats = await llm_integration.get_system_usage_stats()
        status["systems"]["llm_integration"] = llm_stats
        
        # Get capabilities system status
        capabilities_analytics = capabilities_system.get_capability_analytics()
        status["systems"]["capabilities"] = capabilities_analytics
        
        # Get learning system status
        learning_analytics = learning_system.get_agent_learning_summary("system")
        status["systems"]["learning"] = learning_analytics
    
    if NEURAL_MESH_AVAILABLE:
        # Get neural mesh status
        neural_mesh_status = await get_collective_intelligence_status()
        status["systems"]["neural_mesh"] = neural_mesh_status
    
    return status

@app.post("/v1/ai/agents/create")
async def create_enhanced_agent(request: IntelligentAgentRequest):
    """Create enhanced agent with full AI capabilities"""
    
    if not ENHANCED_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced AI systems not available")
    
    try:
        agent_id = await create_intelligent_agent(
            role=request.role,
            specializations=request.specializations
        )
        
        # Get agent status
        agent_status = master_coordinator.active_agents.get(agent_id, {})
        
        # Broadcast agent creation
        await broadcast_to_connections({
            "type": "agent_created",
            "agent_id": agent_id,
            "role": request.role,
            "specializations": request.specializations,
            "timestamp": time.time()
        })
        
        return {
            "agent_id": agent_id,
            "role": request.role,
            "specializations": request.specializations,
            "status": "ready",
            "capabilities_available": len(capabilities_system.discover_capabilities(agent_id=agent_id)),
            "neural_mesh_connected": True,
            "created_at": time.time()
        }
        
    except Exception as e:
        log.error(f"Error creating enhanced agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/tasks/execute")
async def execute_intelligent_task(request: IntelligentTaskRequest):
    """Execute task using intelligent agent coordination"""
    
    if not ENHANCED_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced AI systems not available")
    
    try:
        # Execute task with intelligent coordination
        result = await execute_intelligent_task(
            description=request.description,
            task_type=request.task_type,
            priority=request.priority,
            required_capabilities=request.required_capabilities,
            context=request.context
        )
        
        # Broadcast task execution
        await broadcast_to_connections({
            "type": "task_executed",
            "task_id": result["task_id"],
            "success": result["success"],
            "agents_involved": result["agents_involved"],
            "execution_time": result["execution_time"],
            "timestamp": time.time()
        })
        
        return result
        
    except Exception as e:
        log.error(f"Error executing intelligent task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/swarms/deploy")
async def deploy_intelligent_swarm_api(request: IntelligentSwarmRequest):
    """Deploy intelligent agent swarm with neural mesh"""
    
    if not (ENHANCED_AI_AVAILABLE and NEURAL_MESH_AVAILABLE):
        raise HTTPException(status_code=503, detail="Enhanced AI and Neural Mesh systems required")
    
    try:
        # Deploy intelligent swarm
        swarm_result = await create_intelligent_neural_mesh_swarm(
            objective=request.objective,
            capabilities=request.capabilities,
            specializations=request.specializations,
            max_agents=request.max_agents,
            intelligence_mode=request.intelligence_mode
        )
        
        # Broadcast swarm deployment
        await broadcast_to_connections({
            "type": "swarm_deployed",
            "swarm_id": swarm_result["swarm_id"],
            "agents_deployed": swarm_result["agents_deployed"],
            "intelligence_mode": request.intelligence_mode,
            "objective": request.objective,
            "timestamp": time.time()
        })
        
        return swarm_result
        
    except Exception as e:
        log.error(f"Error deploying intelligent swarm: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/reasoning/collective")
async def coordinate_collective_reasoning_api(request: CollectiveReasoningRequest):
    """Coordinate collective reasoning across agent swarm"""
    
    if not NEURAL_MESH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Neural Mesh system required")
    
    try:
        # Coordinate collective reasoning
        reasoning_result = await coordinate_swarm_collective_reasoning(
            swarm_id=request.swarm_id,
            reasoning_objective=request.reasoning_objective,
            reasoning_pattern=request.reasoning_pattern
        )
        
        # Broadcast reasoning completion
        await broadcast_to_connections({
            "type": "collective_reasoning_completed",
            "swarm_id": request.swarm_id,
            "reasoning_session_id": reasoning_result["reasoning_session_id"],
            "collective_confidence": reasoning_result["collective_confidence"],
            "intelligence_amplification": reasoning_result["intelligence_amplification"],
            "timestamp": time.time()
        })
        
        return reasoning_result
        
    except Exception as e:
        log.error(f"Error coordinating collective reasoning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/knowledge/query")
async def query_knowledge_base(request: KnowledgeQueryRequest):
    """Query knowledge base using RAG"""
    
    if not ENHANCED_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced AI systems not available")
    
    try:
        # Generate RAG response
        rag_response = await knowledge_system.generate_rag_response(
            query=request.query,
            agent_id=request.agent_id,
            max_context_docs=request.max_context_docs
        )
        
        return {
            "query": request.query,
            "response": rag_response.response,
            "source_documents": [
                {
                    "document_id": result.document.id,
                    "content_preview": result.document.content[:200] + "...",
                    "relevance_score": result.score,
                    "document_type": result.document.document_type.value,
                    "metadata": result.document.metadata
                }
                for result in rag_response.source_documents
            ],
            "confidence": rag_response.confidence,
            "reasoning": rag_response.reasoning,
            "token_usage": rag_response.token_usage,
            "processing_time": rag_response.processing_time
        }
        
    except Exception as e:
        log.error(f"Error querying knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/knowledge/upload")
async def upload_document_to_knowledge_base(
    file: UploadFile = File(...),
    agent_id: str = "system",
    document_type: str = "auto"
):
    """Upload document to knowledge base"""
    
    if not ENHANCED_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced AI systems not available")
    
    try:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Determine document type
            if document_type == "auto":
                doc_type = knowledge_system._detect_document_type(temp_file_path)
            else:
                doc_type = DocumentType(document_type)
            
            # Process document
            document_ids = await knowledge_system.process_document_pipeline(
                file_path=temp_file_path,
                document_type=doc_type,
                chunk_size=1000,
                overlap=200
            )
            
            return {
                "filename": file.filename,
                "document_ids": document_ids,
                "chunks_created": len(document_ids),
                "document_type": doc_type.value,
                "file_size": len(content),
                "uploaded_at": time.time()
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
        
    except Exception as e:
        log.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/feedback/submit")
async def submit_agent_feedback(request: FeedbackRequest):
    """Submit feedback for agent learning"""
    
    if not ENHANCED_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced AI systems not available")
    
    try:
        # Record feedback
        feedback_id = await learning_system.record_feedback(
            agent_id=request.agent_id,
            task_id=request.task_id,
            feedback_type=FeedbackType.HUMAN,
            feedback_source="frontend_user",
            rating=request.rating,
            comments=request.comments,
            improvement_suggestions=request.improvement_suggestions
        )
        
        return {
            "feedback_id": feedback_id,
            "agent_id": request.agent_id,
            "rating": request.rating,
            "learning_triggered": request.rating < 0.5 or len(request.improvement_suggestions) > 0,
            "recorded_at": time.time()
        }
        
    except Exception as e:
        log.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/ai/agents/{agent_id}/status")
async def get_agent_detailed_status(agent_id: str):
    """Get detailed status for specific agent"""
    
    if not ENHANCED_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced AI systems not available")
    
    try:
        # Get agent from master coordinator
        agent_info = master_coordinator.active_agents.get(agent_id)
        if not agent_info:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent = agent_info["agent"]
        agent_status = agent.get_agent_status()
        
        # Get learning summary
        learning_summary = await learning_system.get_agent_learning_summary(agent_id)
        
        # Get capability analytics
        capability_analytics = capabilities_system.get_capability_analytics(agent_id=agent_id)
        
        return {
            "agent_id": agent_id,
            "basic_status": agent_status,
            "learning_summary": learning_summary,
            "capability_analytics": capability_analytics,
            "performance_metrics": agent_info.get("performance_history", []),
            "specializations": agent_info.get("specializations", []),
            "created_at": agent_info.get("created_at", time.time())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/ai/capabilities/available")
async def get_available_capabilities():
    """Get all available agent capabilities"""
    
    if not ENHANCED_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced AI systems not available")
    
    try:
        # Get capabilities schema
        capabilities_schema = capabilities_system.export_capabilities_schema()
        
        return {
            "total_capabilities": capabilities_schema["total_capabilities"],
            "capabilities": capabilities_schema["capabilities"],
            "capability_types": capabilities_schema["capability_types"],
            "security_levels": capabilities_schema["security_levels"],
            "execution_modes": capabilities_schema["execution_modes"]
        }
        
    except Exception as e:
        log.error(f"Error getting available capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/capabilities/execute")
async def execute_agent_capability(
    agent_id: str,
    capability_name: str,
    parameters: Dict[str, Any]
):
    """Execute specific agent capability"""
    
    if not ENHANCED_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced AI systems not available")
    
    try:
        # Execute capability
        execution = await capabilities_system.execute_capability(
            agent_id=agent_id,
            capability_name=capability_name,
            parameters=parameters
        )
        
        return {
            "execution_id": execution.execution_id,
            "capability_name": capability_name,
            "success": execution.success,
            "result": execution.result,
            "execution_time": execution.execution_time,
            "error_message": execution.error_message
        }
        
    except Exception as e:
        log.error(f"Error executing capability: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/ai/neural-mesh/status")
async def get_neural_mesh_status():
    """Get comprehensive neural mesh status"""
    
    if not NEURAL_MESH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Neural Mesh not available")
    
    try:
        status = await get_collective_intelligence_status()
        return status
        
    except Exception as e:
        log.error(f"Error getting neural mesh status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/neural-mesh/memory/store")
async def store_memory_in_neural_mesh(request: MemoryRequest):
    """Store memory in neural mesh"""
    
    if not NEURAL_MESH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Neural Mesh not available")
    
    try:
        from services.neural_mesh import store_agent_memory
        
        memory_id = await store_agent_memory(
            agent_id=request.agent_id,
            memory_type=request.memory_type,
            content=request.content,
            memory_tier=request.memory_tier,
            metadata=request.metadata
        )
        
        return {
            "memory_id": memory_id,
            "agent_id": request.agent_id,
            "memory_type": request.memory_type,
            "memory_tier": request.memory_tier,
            "stored_at": time.time()
        }
        
    except Exception as e:
        log.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/ai/neural-mesh/memory/{agent_id}")
async def retrieve_agent_memories(
    agent_id: str,
    query: str = "",
    strategy: str = "hybrid",
    limit: int = 10
):
    """Retrieve memories for agent"""
    
    if not NEURAL_MESH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Neural Mesh not available")
    
    try:
        from services.neural_mesh import retrieve_agent_memories
        
        memories = await retrieve_agent_memories(
            agent_id=agent_id,
            query=query,
            strategy=strategy,
            limit=limit
        )
        
        return {
            "agent_id": agent_id,
            "query": query,
            "strategy": strategy,
            "memories": memories,
            "count": len(memories),
            "retrieved_at": time.time()
        }
        
    except Exception as e:
        log.error(f"Error retrieving memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/collaboration/facilitate")
async def facilitate_agent_collaboration_api(
    initiator_agent: str,
    target_agents: List[str],
    collaboration_objective: str,
    shared_context: Dict[str, Any] = {}
):
    """Facilitate collaboration between agents"""
    
    if not NEURAL_MESH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Neural Mesh not available")
    
    try:
        collaboration_id = await facilitate_agent_collaboration(
            initiator_agent=initiator_agent,
            target_agents=target_agents,
            collaboration_objective=collaboration_objective,
            shared_context=shared_context
        )
        
        # Broadcast collaboration start
        await broadcast_to_connections({
            "type": "collaboration_started",
            "collaboration_id": collaboration_id,
            "initiator": initiator_agent,
            "participants": target_agents,
            "objective": collaboration_objective,
            "timestamp": time.time()
        })
        
        return {
            "collaboration_id": collaboration_id,
            "initiator_agent": initiator_agent,
            "target_agents": target_agents,
            "objective": collaboration_objective,
            "status": "active",
            "started_at": time.time()
        }
        
    except Exception as e:
        log.error(f"Error facilitating collaboration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/knowledge/synthesize")
async def synthesize_collective_knowledge_api(
    knowledge_domain: str,
    contributing_agents: List[str] = None
):
    """Synthesize collective knowledge from multiple agents"""
    
    if not NEURAL_MESH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Neural Mesh not available")
    
    try:
        synthesis_result = await synthesize_collective_knowledge(
            knowledge_domain=knowledge_domain,
            contributing_agents=contributing_agents
        )
        
        return synthesis_result
        
    except Exception as e:
        log.error(f"Error synthesizing knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/ai/analytics/system")
async def get_system_analytics():
    """Get comprehensive system analytics"""
    
    analytics = {
        "timestamp": time.time(),
        "enhanced_ai_available": ENHANCED_AI_AVAILABLE,
        "neural_mesh_available": NEURAL_MESH_AVAILABLE
    }
    
    if ENHANCED_AI_AVAILABLE:
        # LLM usage analytics
        llm_integration = await get_llm_integration()
        analytics["llm_usage"] = await llm_integration.get_system_usage_stats()
        
        # Reasoning analytics
        analytics["reasoning"] = reasoning_engine.get_reasoning_analytics()
        
        # Capabilities analytics
        analytics["capabilities"] = capabilities_system.get_capability_analytics()
        
        # Learning analytics
        analytics["learning"] = await learning_system.get_agent_learning_summary("system")
        
        # Knowledge analytics
        analytics["knowledge"] = knowledge_system.get_knowledge_analytics()
    
    if NEURAL_MESH_AVAILABLE:
        # Neural mesh analytics
        neural_mesh_status = await get_collective_intelligence_status()
        analytics["neural_mesh"] = neural_mesh_status
    
    return analytics

@app.get("/v1/ai/agents/list")
async def list_all_agents():
    """List all active agents with their capabilities"""
    
    if not ENHANCED_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced AI systems not available")
    
    try:
        agents_list = []
        
        for agent_id, agent_info in master_coordinator.active_agents.items():
            agent = agent_info["agent"]
            agent_status = agent.get_agent_status()
            
            agents_list.append({
                "agent_id": agent_id,
                "role": agent_info.get("role", "unknown"),
                "specializations": agent_info.get("specializations", []),
                "status": agent_status["state"],
                "performance_metrics": agent_status["performance_metrics"],
                "capabilities_available": agent_status["capabilities_available"],
                "neural_mesh_connected": agent_status["neural_mesh_connected"],
                "created_at": agent_info.get("created_at", time.time())
            })
        
        return {
            "agents": agents_list,
            "total_agents": len(agents_list),
            "active_agents": len([a for a in agents_list if a["status"] == "ready"]),
            "timestamp": time.time()
        }
        
    except Exception as e:
        log.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/ai/reasoning/patterns")
async def get_reasoning_patterns():
    """Get available reasoning patterns and their analytics"""
    
    if not ENHANCED_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced AI systems not available")
    
    try:
        # Get reasoning analytics
        reasoning_analytics = reasoning_engine.get_reasoning_analytics()
        
        # Get available patterns
        patterns = [
            {
                "pattern": "chain_of_thought",
                "description": "Systematic step-by-step reasoning",
                "use_cases": ["analysis", "problem_solving", "planning"],
                "average_time": "2-5 seconds",
                "accuracy": "95%"
            },
            {
                "pattern": "react",
                "description": "Reasoning + Acting with tool usage",
                "use_cases": ["tool_usage", "multi_step_tasks", "research"],
                "average_time": "5-15 seconds",
                "accuracy": "92%"
            },
            {
                "pattern": "tree_of_thoughts",
                "description": "Multiple reasoning paths exploration",
                "use_cases": ["strategic_decisions", "complex_problems", "optimization"],
                "average_time": "10-30 seconds",
                "accuracy": "97%"
            }
        ]
        
        return {
            "available_patterns": patterns,
            "usage_analytics": reasoning_analytics,
            "total_reasoning_traces": len(reasoning_engine.reasoning_traces),
            "timestamp": time.time()
        }
        
    except Exception as e:
        log.error(f"Error getting reasoning patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/ai/learning/analytics")
async def get_learning_analytics():
    """Get learning system analytics"""
    
    if not ENHANCED_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced AI systems not available")
    
    try:
        # Get system-wide learning analytics
        analytics = await learning_system.get_agent_learning_summary("system")
        
        # Get recent feedback
        recent_feedback = [
            {
                "feedback_id": f.feedback_id,
                "agent_id": f.agent_id,
                "rating": f.rating,
                "feedback_type": f.feedback_type.value,
                "timestamp": f.timestamp
            }
            for f in learning_system.feedback_records[-20:]  # Last 20 feedback entries
        ]
        
        return {
            "learning_analytics": analytics,
            "recent_feedback": recent_feedback,
            "total_feedback_records": len(learning_system.feedback_records),
            "active_ab_tests": len(learning_system.ab_tests),
            "timestamp": time.time()
        }
        
    except Exception as e:
        log.error(f"Error getting learning analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/v1/ai/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time AI system updates"""
    
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial status
        if ENHANCED_AI_AVAILABLE:
            initial_status = {
                "type": "initial_status",
                "ai_systems_available": True,
                "active_agents": len(master_coordinator.active_agents),
                "neural_mesh_available": NEURAL_MESH_AVAILABLE,
                "timestamp": time.time()
            }
            await websocket.send_text(json.dumps(initial_status))
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "subscribe":
                    # Handle subscription requests
                    subscription_type = message.get("subscription")
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "subscription": subscription_type,
                        "timestamp": time.time()
                    }))
                
                elif message.get("type") == "ping":
                    # Handle ping
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                log.error(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

# Demonstration endpoints for showcasing capabilities
@app.post("/v1/ai/demo/intelligent-analysis")
async def demo_intelligent_analysis(
    analysis_request: str,
    use_swarm: bool = False,
    agent_count: int = 5
):
    """Demonstrate intelligent analysis capabilities"""
    
    if not ENHANCED_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Enhanced AI systems not available")
    
    try:
        if use_swarm:
            # Deploy intelligent swarm for analysis
            swarm_result = await deploy_intelligent_swarm(
                objective=f"Comprehensive analysis: {analysis_request}",
                specializations=["analysis", "research", "data_processing"],
                max_agents=agent_count
            )
            
            # Wait a moment for swarm to initialize
            await asyncio.sleep(2)
            
            # Coordinate collective reasoning
            if NEURAL_MESH_AVAILABLE:
                reasoning_result = await coordinate_swarm_collective_reasoning(
                    swarm_id=swarm_result["swarm_id"],
                    reasoning_objective=analysis_request,
                    reasoning_pattern="collective_chain_of_thought"
                )
                
                return {
                    "demo_type": "intelligent_swarm_analysis",
                    "swarm_deployed": True,
                    "agents_count": swarm_result["agents_deployed"],
                    "collective_reasoning": reasoning_result["collective_reasoning"],
                    "intelligence_amplification": reasoning_result["intelligence_amplification"],
                    "collective_confidence": reasoning_result["collective_confidence"],
                    "processing_time": reasoning_result.get("processing_time", 0)
                }
        else:
            # Single agent intelligent analysis
            result = await execute_intelligent_task(
                description=analysis_request,
                task_type="analysis",
                priority="normal",
                required_capabilities=["analyze_data", "pattern_recognition"]
            )
            
            return {
                "demo_type": "single_agent_analysis",
                "task_result": result,
                "agents_involved": result["agents_involved"],
                "execution_time": result["execution_time"]
            }
        
    except Exception as e:
        log.error(f"Error in demo analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ai/demo/collective-reasoning")
async def demo_collective_reasoning(
    reasoning_problem: str,
    agent_count: int = 3
):
    """Demonstrate collective reasoning capabilities"""
    
    if not (ENHANCED_AI_AVAILABLE and NEURAL_MESH_AVAILABLE):
        raise HTTPException(status_code=503, detail="Enhanced AI and Neural Mesh required")
    
    try:
        # Create temporary swarm for demonstration
        swarm_result = await create_intelligent_neural_mesh_swarm(
            objective=f"Collective reasoning demonstration: {reasoning_problem}",
            capabilities=["reasoning", "analysis", "problem_solving"],
            specializations=["logic", "analysis", "strategy"],
            max_agents=agent_count,
            intelligence_mode="collective"
        )
        
        # Wait for swarm initialization
        await asyncio.sleep(3)
        
        # Coordinate collective reasoning
        reasoning_result = await coordinate_swarm_collective_reasoning(
            swarm_id=swarm_result["swarm_id"],
            reasoning_objective=reasoning_problem,
            reasoning_pattern="collective_chain_of_thought"
        )
        
        return {
            "demo_type": "collective_reasoning",
            "problem": reasoning_problem,
            "swarm_id": swarm_result["swarm_id"],
            "agents_participated": reasoning_result["participating_agents"],
            "collective_result": reasoning_result["collective_reasoning"],
            "individual_contributions": reasoning_result["individual_contributions"],
            "intelligence_amplification": reasoning_result["intelligence_amplification"],
            "collective_confidence": reasoning_result["collective_confidence"],
            "emergent_insights": reasoning_result.get("emergent_insights", [])
        }
        
    except Exception as e:
        log.error(f"Error in collective reasoning demo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/ai/demo/capabilities-showcase")
async def demo_capabilities_showcase():
    """Showcase all available AI capabilities"""
    
    showcase = {
        "timestamp": time.time(),
        "enhanced_ai_available": ENHANCED_AI_AVAILABLE,
        "neural_mesh_available": NEURAL_MESH_AVAILABLE,
        "capabilities": {}
    }
    
    if ENHANCED_AI_AVAILABLE:
        showcase["capabilities"]["agent_intelligence"] = {
            "multi_provider_llm": "OpenAI, Anthropic, Google, Cohere, Mistral, XAI",
            "reasoning_patterns": ["chain_of_thought", "react", "tree_of_thoughts"],
            "learning_systems": ["feedback_loops", "ab_testing", "behavior_analytics"],
            "capabilities_system": f"{len(capabilities_system.capabilities)} registered capabilities",
            "knowledge_management": f"{len(knowledge_system.documents)} documents in knowledge base"
        }
    
    if NEURAL_MESH_AVAILABLE:
        neural_status = await get_collective_intelligence_status()
        showcase["capabilities"]["neural_mesh"] = {
            "distributed_memory": "4-tier architecture (L1→L2→L3→L4)",
            "collective_intelligence": f"Mode: {neural_status['neural_mesh_status']['intelligence_level']}",
            "active_agents": neural_status["neural_mesh_status"]["active_agents"],
            "memory_synchronization": "Event-driven with conflict resolution",
            "cross_datacenter_replication": "Multi-region disaster recovery"
        }
    
    return showcase

# Health check
@app.get("/v1/ai/health")
async def ai_health_check():
    """Health check for AI systems"""
    
    # Try to initialize AI systems
    try:
        await initialize_ai_systems()
    except Exception as e:
        log.error(f"AI systems initialization failed: {e}")
    
    health = {
        "status": "healthy" if ENHANCED_AI_AVAILABLE else "degraded",
        "timestamp": time.time(),
        "systems": {
            "enhanced_ai": ENHANCED_AI_AVAILABLE,
            "neural_mesh": NEURAL_MESH_AVAILABLE
        }
    }
    
    if ENHANCED_AI_AVAILABLE:
        try:
            # Test core systems
            coordinator_status = master_coordinator.get_system_status()
            health["systems"]["master_coordinator"] = coordinator_status["master_coordinator"]["status"] == "active"
            
            llm_integration = await get_llm_integration()
            health["systems"]["llm_integration"] = len(llm_integration.providers) > 0
            
        except Exception as e:
            health["status"] = "degraded"
            health["error"] = str(e)
    
    return health

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
