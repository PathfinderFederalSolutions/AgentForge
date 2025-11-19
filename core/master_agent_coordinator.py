#!/usr/bin/env python3
"""
Master Agent Intelligence Coordinator for AgentForge
Integrates all AI systems: LLM, reasoning, capabilities, knowledge, learning, neural mesh, and orchestrator
"""

import asyncio
import json
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger("master-agent-coordinator")

class CoordinationMode(Enum):
    """Agent coordination modes"""
    SOLO = "solo"                    # Single agent operation
    COLLABORATIVE = "collaborative"  # Multiple agents working together
    HIERARCHICAL = "hierarchical"   # Coordinated by lead agent
    SWARM = "swarm"                 # Large-scale swarm coordination

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10

@dataclass
class MasterTask:
    """Comprehensive task definition"""
    task_id: str
    description: str
    task_type: str
    priority: TaskPriority
    complexity_score: float
    required_capabilities: List[str] = field(default_factory=list)
    coordination_mode: CoordinationMode = CoordinationMode.SOLO
    deadline: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    approval_required: bool = False
    created_at: float = field(default_factory=time.time)

@dataclass
class AgentAssignment:
    """Agent assignment for a task"""
    agent_id: str
    role: str
    specializations: List[str]
    assigned_capabilities: List[str]
    estimated_completion_time: float
    confidence: float

@dataclass
class CoordinationResult:
    """Result of agent coordination"""
    task_id: str
    coordination_mode: CoordinationMode
    agents_involved: List[str]
    execution_time: float
    success: bool
    results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    lessons_learned: List[str] = field(default_factory=list)

class MasterAgentCoordinator:
    """Master coordinator for all agent intelligence systems"""
    
    def __init__(self):
        # Core AI systems
        self.llm_integration = None
        self.reasoning_engine = None
        self.capabilities_system = None
        self.knowledge_system = None
        self.learning_system = None
        self.prompt_system = None
        
        # Integration systems
        self.neural_mesh = None
        self.orchestrator = None
        self.communications_gateway = None
        self.hitl_service = None
        
        # Active agents and tasks
        self.active_agents: Dict[str, Any] = {}
        self.active_tasks: Dict[str, MasterTask] = {}
        self.task_assignments: Dict[str, List[AgentAssignment]] = {}
        
        # Coordination history
        self.coordination_history: List[CoordinationResult] = []
        
        # Initialize
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize all systems"""
        try:
            # Import and initialize core AI systems
            from core.enhanced_llm_integration import get_llm_integration
            from core.advanced_reasoning_engine import reasoning_engine
            from core.agent_capabilities_system import capabilities_system
            from core.knowledge_management_system import knowledge_system
            from core.agent_learning_system import learning_system
            from core.prompt_template_system import prompt_system
            
            self.llm_integration = await get_llm_integration()
            self.reasoning_engine = reasoning_engine
            self.capabilities_system = capabilities_system
            self.knowledge_system = knowledge_system
            self.learning_system = learning_system
            self.prompt_system = prompt_system
            
            # Initialize integration systems
            await self._initialize_integration_systems()
            
            # Start coordination processes
            asyncio.create_task(self._coordination_monitor())
            asyncio.create_task(self._performance_monitor())
            
            log.info("✅ Master Agent Coordinator initialized successfully")
            
        except Exception as e:
            log.error(f"Failed to initialize Master Agent Coordinator: {e}")
    
    async def _initialize_integration_systems(self):
        """Initialize integration with other AgentForge systems"""
        
        # Neural mesh integration
        try:
            from services.neural_mesh.core.enhanced_memory import EnhancedNeuralMesh
            self.neural_mesh = EnhancedNeuralMesh()
            await self.neural_mesh.initialize()
            log.info("✅ Neural mesh integrated with master coordinator")
        except ImportError:
            log.warning("Neural mesh not available")
        
        # Orchestrator integration
        try:
            from services.unified_orchestrator.orchestrator import UnifiedQuantumOrchestrator
            self.orchestrator = UnifiedQuantumOrchestrator()
            await self.orchestrator.initialize()
            log.info("✅ Unified orchestrator integrated with master coordinator")
        except ImportError:
            log.warning("Unified orchestrator not available")
        
        # Communications gateway integration
        try:
            from services.comms_gateway.integrations.swarm_bridge import SwarmBridge
            self.communications_gateway = SwarmBridge()
            await self.communications_gateway.initialize()
            log.info("✅ Communications gateway integrated with master coordinator")
        except ImportError:
            log.warning("Communications gateway not available")
        
        # HITL service integration
        try:
            from services.hitl.integrations.swarm_approvals import SwarmApprovalBridge
            self.hitl_service = SwarmApprovalBridge()
            await self.hitl_service.initialize()
            log.info("✅ HITL service integrated with master coordinator")
        except ImportError:
            log.warning("HITL service not available")
    
    async def create_enhanced_agent(
        self,
        role: str = "generalist",
        specializations: List[str] = None,
        agent_id: Optional[str] = None
    ) -> str:
        """Create a new enhanced agent with full AI capabilities"""
        
        if agent_id is None:
            agent_id = f"agent_{role}_{str(uuid.uuid4())[:8]}"
        
        try:
            # Import and create enhanced agent
            from core.enhanced_agent_intelligence import create_enhanced_agent, AgentRole
            
            # Map role string to enum
            role_mapping = {
                "coordinator": AgentRole.COORDINATOR,
                "specialist": AgentRole.SPECIALIST,
                "generalist": AgentRole.GENERALIST,
                "analyzer": AgentRole.ANALYZER,
                "executor": AgentRole.EXECUTOR,
                "validator": AgentRole.VALIDATOR
            }
            
            agent_role = role_mapping.get(role, AgentRole.GENERALIST)
            
            # Create agent
            agent = await create_enhanced_agent(
                role=agent_role,
                specializations=specializations,
                agent_id=agent_id
            )
            
            # Register with coordinator
            self.active_agents[agent_id] = {
                "agent": agent,
                "role": role,
                "specializations": specializations or [],
                "created_at": time.time(),
                "status": "ready",
                "current_task": None,
                "performance_history": []
            }
            
            # Register with communications gateway
            if self.communications_gateway:
                await self.communications_gateway.register_agent(
                    agent_id=agent_id,
                    agent_type=role,
                    capabilities=specializations or []
                )
            
            # Store agent creation in neural mesh
            if self.neural_mesh:
                await self.neural_mesh.store_knowledge(
                    agent_id="master_coordinator",
                    knowledge_type="agent_creation",
                    data={
                        "agent_id": agent_id,
                        "role": role,
                        "specializations": specializations,
                        "created_by": "master_coordinator",
                        "timestamp": time.time()
                    },
                    memory_tier="L4"
                )
            
            log.info(f"Created enhanced agent {agent_id} with role {role}")
            return agent_id
            
        except Exception as e:
            log.error(f"Error creating enhanced agent: {e}")
            raise
    
    async def coordinate_task_execution(
        self,
        task: MasterTask,
        preferred_agents: List[str] = None
    ) -> CoordinationResult:
        """Coordinate task execution across agents"""
        
        start_time = time.time()
        
        try:
            # Analyze task requirements
            task_analysis = await self._analyze_task_requirements(task)
            
            # Determine coordination strategy
            coordination_mode = await self._determine_coordination_mode(task, task_analysis)
            task.coordination_mode = coordination_mode
            
            # Select and assign agents
            agent_assignments = await self._select_and_assign_agents(
                task, task_analysis, preferred_agents
            )
            
            # Request approval if required
            if task.approval_required and self.hitl_service:
                approval_result = await self._request_task_approval(task, agent_assignments)
                if not approval_result.get("approved", False):
                    raise Exception(f"Task approval denied: {approval_result.get('reason', 'Unknown')}")
            
            # Execute coordination
            if coordination_mode == CoordinationMode.SOLO:
                results = await self._execute_solo_task(task, agent_assignments[0])
            elif coordination_mode == CoordinationMode.COLLABORATIVE:
                results = await self._execute_collaborative_task(task, agent_assignments)
            elif coordination_mode == CoordinationMode.HIERARCHICAL:
                results = await self._execute_hierarchical_task(task, agent_assignments)
            elif coordination_mode == CoordinationMode.SWARM:
                results = await self._execute_swarm_task(task, agent_assignments)
            else:
                raise ValueError(f"Unsupported coordination mode: {coordination_mode}")
            
            # Create coordination result
            coordination_result = CoordinationResult(
                task_id=task.task_id,
                coordination_mode=coordination_mode,
                agents_involved=[a.agent_id for a in agent_assignments],
                execution_time=time.time() - start_time,
                success=results.get("success", False),
                results=results,
                performance_metrics=self._calculate_coordination_metrics(results, agent_assignments)
            )
            
            # Store results and learn
            await self._store_coordination_results(coordination_result)
            await self._learn_from_coordination(coordination_result)
            
            return coordination_result
            
        except Exception as e:
            log.error(f"Error coordinating task {task.task_id}: {e}")
            
            return CoordinationResult(
                task_id=task.task_id,
                coordination_mode=task.coordination_mode,
                agents_involved=[],
                execution_time=time.time() - start_time,
                success=False,
                results={"error": str(e)},
                performance_metrics={}
            )
    
    async def initiate_intelligent_swarm(
        self,
        objective: str,
        complexity_score: float,
        required_specializations: List[str] = None,
        max_agents: int = 100
    ) -> Dict[str, Any]:
        """Initiate intelligent agent swarm with full AI capabilities"""
        
        try:
            # Analyze swarm requirements
            swarm_analysis = await self._analyze_swarm_requirements(
                objective, complexity_score, required_specializations
            )
            
            # Determine optimal agent count and composition
            agent_composition = await self._determine_agent_composition(
                swarm_analysis, max_agents
            )
            
            # Create specialized agents for swarm
            swarm_agents = []
            for agent_spec in agent_composition:
                agent_id = await self.create_enhanced_agent(
                    role=agent_spec["role"],
                    specializations=agent_spec["specializations"]
                )
                swarm_agents.append(agent_id)
            
            # Initialize swarm coordination through orchestrator
            if self.orchestrator:
                swarm_deployment = await self.orchestrator.deploy_agent_swarm({
                    "objective": objective,
                    "agents": swarm_agents,
                    "coordination_pattern": "intelligent_mesh",
                    "neural_mesh_enabled": True,
                    "learning_enabled": True
                })
            else:
                # Fallback coordination
                swarm_deployment = {
                    "swarm_id": f"swarm_{str(uuid.uuid4())[:8]}",
                    "agents": swarm_agents,
                    "status": "deployed"
                }
            
            # Set up inter-agent communication
            if self.communications_gateway:
                for agent_id in swarm_agents:
                    await self.communications_gateway.register_agent(
                        agent_id=agent_id,
                        agent_type="swarm_member",
                        capabilities=required_specializations or []
                    )
            
            # Store swarm information in neural mesh
            if self.neural_mesh:
                await self.neural_mesh.store_knowledge(
                    agent_id="master_coordinator",
                    knowledge_type="swarm_deployment",
                    data={
                        "swarm_id": swarm_deployment["swarm_id"],
                        "objective": objective,
                        "agents": swarm_agents,
                        "composition": agent_composition,
                        "complexity_score": complexity_score,
                        "timestamp": time.time()
                    },
                    memory_tier="L3"
                )
            
            log.info(f"Initiated intelligent swarm: {swarm_deployment['swarm_id']} with {len(swarm_agents)} agents")
            
            return {
                "swarm_id": swarm_deployment["swarm_id"],
                "agents_deployed": len(swarm_agents),
                "agent_composition": agent_composition,
                "coordination_enabled": True,
                "neural_mesh_enabled": self.neural_mesh is not None,
                "learning_enabled": True,
                "estimated_capability": swarm_analysis.get("estimated_capability", 0.8)
            }
            
        except Exception as e:
            log.error(f"Error initiating intelligent swarm: {e}")
            return {"error": str(e), "success": False}
    
    async def _analyze_task_requirements(self, task: MasterTask) -> Dict[str, Any]:
        """Analyze task requirements using AI"""
        
        analysis_prompt = f"""Analyze this task to determine optimal execution strategy:

Task: {task.description}
Type: {task.task_type}
Priority: {task.priority.name}
Complexity Score: {task.complexity_score}
Required Capabilities: {task.required_capabilities}
Constraints: {task.constraints}
Success Criteria: {task.success_criteria}

Please analyze:
1. Task complexity and difficulty level
2. Required expertise and specializations
3. Estimated execution time and resources
4. Potential risks and challenges
5. Optimal coordination approach
6. Success probability assessment

Analysis:"""
        
        from core.enhanced_llm_integration import LLMRequest
        
        request = LLMRequest(
            agent_id="master_coordinator",
            task_type="task_analysis",
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        response = await self.llm_integration.generate_response(request)
        
        # Parse analysis (simplified - would use more sophisticated parsing)
        return {
            "analysis": response.content,
            "estimated_difficulty": task.complexity_score,
            "estimated_duration": 1800,  # Default 30 minutes
            "risk_level": "medium",
            "recommended_agents": max(1, int(task.complexity_score * 5)),
            "specializations_needed": task.required_capabilities
        }
    
    async def _determine_coordination_mode(
        self,
        task: MasterTask,
        analysis: Dict[str, Any]
    ) -> CoordinationMode:
        """Determine optimal coordination mode for task"""
        
        # Simple heuristics for coordination mode selection
        if task.complexity_score < 0.3:
            return CoordinationMode.SOLO
        elif task.complexity_score < 0.6:
            return CoordinationMode.COLLABORATIVE
        elif task.complexity_score < 0.8:
            return CoordinationMode.HIERARCHICAL
        else:
            return CoordinationMode.SWARM
    
    async def _select_and_assign_agents(
        self,
        task: MasterTask,
        analysis: Dict[str, Any],
        preferred_agents: List[str] = None
    ) -> List[AgentAssignment]:
        """Select and assign optimal agents for task"""
        
        assignments = []
        
        try:
            # Get available agents
            available_agents = list(self.active_agents.keys())
            
            # Filter by preferences
            if preferred_agents:
                available_agents = [a for a in available_agents if a in preferred_agents]
            
            # Determine number of agents needed
            num_agents = min(
                analysis.get("recommended_agents", 1),
                len(available_agents),
                10  # Maximum agents for coordination
            )
            
            # Select best agents for task
            agent_scores = {}
            for agent_id in available_agents:
                agent_info = self.active_agents[agent_id]
                score = await self._calculate_agent_suitability(agent_info, task, analysis)
                agent_scores[agent_id] = score
            
            # Sort by suitability and select top agents
            sorted_agents = sorted(agent_scores.keys(), key=lambda a: agent_scores[a], reverse=True)
            selected_agents = sorted_agents[:num_agents]
            
            # Create assignments
            for i, agent_id in enumerate(selected_agents):
                agent_info = self.active_agents[agent_id]
                
                assignment = AgentAssignment(
                    agent_id=agent_id,
                    role="lead" if i == 0 else "member",
                    specializations=agent_info["specializations"],
                    assigned_capabilities=self._assign_capabilities_to_agent(task, agent_info),
                    estimated_completion_time=analysis.get("estimated_duration", 1800),
                    confidence=agent_scores[agent_id]
                )
                
                assignments.append(assignment)
            
            return assignments
            
        except Exception as e:
            log.error(f"Error selecting agents: {e}")
            return []
    
    async def _execute_solo_task(
        self,
        task: MasterTask,
        assignment: AgentAssignment
    ) -> Dict[str, Any]:
        """Execute task with single agent"""
        
        try:
            agent_info = self.active_agents[assignment.agent_id]
            agent = agent_info["agent"]
            
            # Create agent task
            from core.enhanced_agent_intelligence import AgentTask
            
            agent_task = AgentTask(
                task_id=task.task_id,
                task_type=task.task_type,
                description=task.description,
                parameters=task.context,
                priority=task.priority.value,
                deadline=task.deadline,
                required_capabilities=assignment.assigned_capabilities,
                approval_required=task.approval_required
            )
            
            # Execute task
            response = await agent.process_task(agent_task)
            
            return {
                "success": response.confidence > 0.5,
                "response": response.content,
                "confidence": response.confidence,
                "execution_time": response.execution_time,
                "capabilities_used": response.capabilities_used,
                "reasoning_trace": response.reasoning_trace,
                "token_usage": response.token_usage,
                "cost": response.cost
            }
            
        except Exception as e:
            log.error(f"Error in solo task execution: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_collaborative_task(
        self,
        task: MasterTask,
        assignments: List[AgentAssignment]
    ) -> Dict[str, Any]:
        """Execute task with collaborative agents"""
        
        try:
            # Execute task with all agents in parallel
            agent_tasks = []
            
            for assignment in assignments:
                agent_info = self.active_agents[assignment.agent_id]
                agent = agent_info["agent"]
                
                from core.enhanced_agent_intelligence import AgentTask
                
                agent_task = AgentTask(
                    task_id=f"{task.task_id}_{assignment.agent_id}",
                    task_type=task.task_type,
                    description=task.description,
                    parameters=task.context,
                    required_capabilities=assignment.assigned_capabilities,
                    collaboration_required=True
                )
                
                task_coroutine = agent.process_task(agent_task)
                agent_tasks.append((assignment.agent_id, task_coroutine))
            
            # Execute all tasks
            results = {}
            for agent_id, task_coro in agent_tasks:
                try:
                    response = await task_coro
                    results[agent_id] = {
                        "success": response.confidence > 0.5,
                        "response": response.content,
                        "confidence": response.confidence,
                        "capabilities_used": response.capabilities_used
                    }
                except Exception as e:
                    results[agent_id] = {"success": False, "error": str(e)}
            
            # Aggregate results
            successful_agents = [a for a, r in results.items() if r.get("success", False)]
            overall_success = len(successful_agents) > len(assignments) / 2
            
            # Combine responses using knowledge system
            combined_response = await self._combine_agent_responses(
                [r["response"] for r in results.values() if r.get("success", False)]
            )
            
            return {
                "success": overall_success,
                "combined_response": combined_response,
                "individual_results": results,
                "successful_agents": len(successful_agents),
                "total_agents": len(assignments),
                "collaboration_effectiveness": len(successful_agents) / len(assignments)
            }
            
        except Exception as e:
            log.error(f"Error in collaborative task execution: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_swarm_task(
        self,
        task: MasterTask,
        assignments: List[AgentAssignment]
    ) -> Dict[str, Any]:
        """Execute task with agent swarm"""
        
        try:
            # Use orchestrator for swarm coordination
            if not self.orchestrator:
                raise Exception("Orchestrator required for swarm execution")
            
            # Create swarm deployment request
            swarm_request = {
                "task_id": task.task_id,
                "objective": task.description,
                "agents": [a.agent_id for a in assignments],
                "coordination_pattern": "neural_mesh_swarm",
                "performance_requirements": {
                    "min_confidence": 0.7,
                    "max_execution_time": task.deadline - time.time() if task.deadline else 3600
                }
            }
            
            # Deploy swarm
            swarm_result = await self.orchestrator.execute_swarm_task(swarm_request)
            
            return {
                "success": swarm_result.get("success", False),
                "swarm_response": swarm_result.get("response", ""),
                "agents_participated": swarm_result.get("agents_participated", 0),
                "coordination_efficiency": swarm_result.get("coordination_efficiency", 0.0),
                "collective_confidence": swarm_result.get("collective_confidence", 0.0),
                "emergent_insights": swarm_result.get("emergent_insights", [])
            }
            
        except Exception as e:
            log.error(f"Error in swarm task execution: {e}")
            return {"success": False, "error": str(e)}
    
    async def _combine_agent_responses(self, responses: List[str]) -> str:
        """Combine multiple agent responses into coherent result"""
        
        if not responses:
            return "No successful responses to combine"
        
        if len(responses) == 1:
            return responses[0]
        
        # Use LLM to combine responses
        combination_prompt = f"""I need to combine these responses from multiple AI agents into a single, coherent response:

Agent Responses:
{chr(10).join(f'Agent {i+1}: {response}' for i, response in enumerate(responses))}

Please create a combined response that:
1. Integrates the best insights from each agent
2. Resolves any contradictions or conflicts
3. Provides a comprehensive and coherent answer
4. Maintains accuracy and completeness

Combined Response:"""
        
        from core.enhanced_llm_integration import LLMRequest
        
        request = LLMRequest(
            agent_id="master_coordinator",
            task_type="response_combination",
            messages=[{"role": "user", "content": combination_prompt}]
        )
        
        response = await self.llm_integration.generate_response(request)
        return response.content
    
    async def _calculate_agent_suitability(
        self,
        agent_info: Dict[str, Any],
        task: MasterTask,
        analysis: Dict[str, Any]
    ) -> float:
        """Calculate how suitable an agent is for a task"""
        
        score = 0.0
        
        # Specialization match
        agent_specializations = set(agent_info.get("specializations", []))
        required_specializations = set(analysis.get("specializations_needed", []))
        
        if required_specializations:
            specialization_match = len(agent_specializations.intersection(required_specializations)) / len(required_specializations)
            score += specialization_match * 0.4
        else:
            score += 0.4  # No specific requirements
        
        # Performance history
        performance_history = agent_info.get("performance_history", [])
        if performance_history:
            avg_performance = sum(p.get("success_rate", 0.5) for p in performance_history) / len(performance_history)
            score += avg_performance * 0.3
        else:
            score += 0.15  # Default for new agents
        
        # Current availability
        if agent_info.get("current_task") is None:
            score += 0.2  # Available
        else:
            score += 0.05  # Busy but could multitask
        
        # Role suitability
        agent_role = agent_info.get("role", "generalist")
        if task.task_type in ["coordination", "management"] and agent_role == "coordinator":
            score += 0.1
        elif task.task_type in ["analysis", "research"] and agent_role == "analyzer":
            score += 0.1
        
        return min(score, 1.0)
    
    def _assign_capabilities_to_agent(
        self,
        task: MasterTask,
        agent_info: Dict[str, Any]
    ) -> List[str]:
        """Assign specific capabilities to agent for task"""
        
        # Get agent's available capabilities
        agent_capabilities = self.capabilities_system.discover_capabilities(
            agent_id=agent_info["agent"]["agent_id"]
        )
        
        available_cap_names = [cap.name for cap in agent_capabilities]
        
        # Match with task requirements
        assigned = []
        for required_cap in task.required_capabilities:
            if required_cap in available_cap_names:
                assigned.append(required_cap)
        
        # Add role-specific capabilities
        role = agent_info.get("role", "generalist")
        role_capabilities = {
            "coordinator": ["coordinate_agents", "manage_resources"],
            "analyzer": ["analyze_data", "pattern_recognition"],
            "executor": ["execute_tasks", "workflow_management"]
        }
        
        for cap in role_capabilities.get(role, []):
            if cap in available_cap_names and cap not in assigned:
                assigned.append(cap)
        
        return assigned
    
    async def _coordination_monitor(self):
        """Monitor coordination effectiveness"""
        
        while True:
            try:
                # Check active tasks
                for task_id, task in self.active_tasks.items():
                    # Monitor task progress
                    await self._monitor_task_progress(task)
                
                # Check agent health
                for agent_id, agent_info in self.active_agents.items():
                    await self._monitor_agent_health(agent_id, agent_info)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in coordination monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _performance_monitor(self):
        """Monitor system performance and trigger improvements"""
        
        while True:
            try:
                # Analyze system-wide performance
                system_metrics = await self._calculate_system_metrics()
                
                # Check for performance issues
                if system_metrics.get("average_success_rate", 1.0) < 0.7:
                    await self._trigger_system_improvement()
                
                # Update neural mesh with system state
                if self.neural_mesh:
                    await self.neural_mesh.store_knowledge(
                        agent_id="master_coordinator",
                        knowledge_type="system_metrics",
                        data=system_metrics,
                        memory_tier="L2"
                    )
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_system_metrics(self) -> Dict[str, Any]:
        """Calculate system-wide performance metrics"""
        
        # Recent coordination results
        recent_results = [
            r for r in self.coordination_history
            if r.timestamp > time.time() - 3600  # Last hour
        ]
        
        if not recent_results:
            return {"no_recent_activity": True}
        
        # Calculate metrics
        success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
        avg_execution_time = sum(r.execution_time for r in recent_results) / len(recent_results)
        total_agents_used = sum(len(r.agents_involved) for r in recent_results)
        
        return {
            "total_coordinations": len(recent_results),
            "average_success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "total_agents_utilized": total_agents_used,
            "active_agents": len(self.active_agents),
            "system_load": min(total_agents_used / max(len(self.active_agents), 1), 1.0),
            "timestamp": time.time()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "master_coordinator": {
                "status": "active",
                "active_agents": len(self.active_agents),
                "active_tasks": len(self.active_tasks),
                "coordination_history": len(self.coordination_history)
            },
            "ai_systems": {
                "llm_integration": self.llm_integration is not None,
                "reasoning_engine": self.reasoning_engine is not None,
                "capabilities_system": self.capabilities_system is not None,
                "knowledge_system": self.knowledge_system is not None,
                "learning_system": self.learning_system is not None,
                "prompt_system": self.prompt_system is not None
            },
            "integration_systems": {
                "neural_mesh": self.neural_mesh is not None,
                "orchestrator": self.orchestrator is not None,
                "communications_gateway": self.communications_gateway is not None,
                "hitl_service": self.hitl_service is not None
            },
            "capabilities": {
                "total_registered": len(self.capabilities_system.capabilities) if self.capabilities_system else 0,
                "knowledge_documents": len(self.knowledge_system.documents) if self.knowledge_system else 0,
                "learning_records": len(self.learning_system.feedback_records) if self.learning_system else 0
            }
        }

# Global instance
master_coordinator = MasterAgentCoordinator()

# Convenience functions for external use
async def create_intelligent_agent(
    role: str = "generalist",
    specializations: List[str] = None
) -> str:
    """Create an intelligent agent with full AI capabilities"""
    return await master_coordinator.create_enhanced_agent(role, specializations)

async def execute_intelligent_task(
    description: str,
    task_type: str = "general",
    priority: str = "normal",
    required_capabilities: List[str] = None,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Execute a task using intelligent agent coordination"""
    
    # Map priority string to enum
    priority_mapping = {
        "low": TaskPriority.LOW,
        "normal": TaskPriority.NORMAL,
        "high": TaskPriority.HIGH,
        "critical": TaskPriority.CRITICAL
    }
    
    task = MasterTask(
        task_id=str(uuid.uuid4()),
        description=description,
        task_type=task_type,
        priority=priority_mapping.get(priority, TaskPriority.NORMAL),
        complexity_score=0.5,  # Default complexity
        required_capabilities=required_capabilities or [],
        context=context or {}
    )
    
    result = await master_coordinator.coordinate_task_execution(task)
    
    return {
        "task_id": task.task_id,
        "success": result.success,
        "results": result.results,
        "agents_involved": result.agents_involved,
        "execution_time": result.execution_time,
        "coordination_mode": result.coordination_mode.value
    }

async def deploy_intelligent_swarm(
    objective: str,
    specializations: List[str] = None,
    max_agents: int = 10
) -> Dict[str, Any]:
    """Deploy intelligent agent swarm for complex objectives"""
    
    complexity_score = min(len(specializations or []) * 0.2 + 0.3, 1.0)
    
    return await master_coordinator.initiate_intelligent_swarm(
        objective=objective,
        complexity_score=complexity_score,
        required_specializations=specializations,
        max_agents=max_agents
    )
