#!/usr/bin/env python3
"""
Enhanced Agent Intelligence System for AgentForge
Integrates all AI capabilities: LLM integration, reasoning, capabilities, neural mesh, and orchestrator
"""

import asyncio
import json
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger("enhanced-agent-intelligence")

class AgentRole(Enum):
    """Agent role types"""
    COORDINATOR = "coordinator"      # Coordinates other agents
    SPECIALIST = "specialist"        # Specialized domain expert
    GENERALIST = "generalist"        # General purpose agent
    ANALYZER = "analyzer"           # Data analysis specialist
    EXECUTOR = "executor"           # Task execution specialist
    VALIDATOR = "validator"         # Quality assurance and validation

class AgentState(Enum):
    """Agent operational states"""
    INITIALIZING = "initializing"
    READY = "ready"
    THINKING = "thinking"
    EXECUTING = "executing"
    COLLABORATING = "collaborating"
    LEARNING = "learning"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class AgentTask:
    """Task assigned to an agent"""
    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    deadline: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: List[str] = field(default_factory=list)
    collaboration_required: bool = False
    approval_required: bool = False

@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_id: str
    task_id: str
    response_type: str
    content: Any
    confidence: float
    reasoning_trace: Optional[Dict[str, Any]] = None
    capabilities_used: List[str] = field(default_factory=list)
    collaboration_requests: List[str] = field(default_factory=list)
    follow_up_tasks: List[AgentTask] = field(default_factory=list)
    execution_time: float = 0.0
    token_usage: int = 0
    cost: float = 0.0

class EnhancedAgentIntelligence:
    """Enhanced agent with full AI capabilities integration"""
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole = AgentRole.GENERALIST,
        specializations: List[str] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.specializations = specializations or []
        self.state = AgentState.INITIALIZING
        
        # Core AI systems
        self.llm_integration = None
        self.reasoning_engine = None
        self.capabilities_system = None
        self.prompt_system = None
        
        # Neural mesh and orchestrator
        self.neural_mesh = None
        self.orchestrator = None
        
        # Agent memory and context
        self.working_memory = {}
        self.task_history = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "success_rate": 1.0,
            "avg_response_time": 0.0,
            "total_cost": 0.0,
            "learning_events": 0
        }
        
        # Initialize
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize all AI systems"""
        try:
            # Import and initialize core systems
            from core.enhanced_llm_integration import get_llm_integration
            from core.advanced_reasoning_engine import reasoning_engine
            from core.agent_capabilities_system import capabilities_system
            from core.prompt_template_system import prompt_system
            
            self.llm_integration = await get_llm_integration()
            self.reasoning_engine = reasoning_engine
            self.capabilities_system = capabilities_system
            self.prompt_system = prompt_system
            
            # Neural mesh integration
            try:
                from services.neural_mesh.core.enhanced_memory import EnhancedNeuralMesh
                self.neural_mesh = EnhancedNeuralMesh()
                await self.neural_mesh.initialize()
            except ImportError:
                log.warning(f"Neural mesh not available for agent {self.agent_id}")
            
            # Orchestrator integration
            try:
                from services.unified_orchestrator.orchestrator import UnifiedQuantumOrchestrator
                self.orchestrator = UnifiedQuantumOrchestrator()
                await self.orchestrator.initialize()
            except ImportError:
                log.warning(f"Orchestrator not available for agent {self.agent_id}")
            
            # Set up agent permissions
            await self._setup_agent_permissions()
            
            # Load agent context from neural mesh
            await self._load_agent_context()
            
            self.state = AgentState.READY
            log.info(f"✅ Enhanced agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            log.error(f"Failed to initialize agent {self.agent_id}: {e}")
            self.state = AgentState.ERROR
    
    async def process_task(
        self,
        task: AgentTask,
        reasoning_pattern: Optional[str] = None
    ) -> AgentResponse:
        """Process a task using full AI capabilities"""
        
        self.state = AgentState.THINKING
        start_time = time.time()
        
        try:
            # Get relevant context from neural mesh
            context = await self._get_task_context(task)
            
            # Determine reasoning approach
            if reasoning_pattern is None:
                reasoning_pattern = await self._select_reasoning_pattern(task)
            
            # Execute reasoning
            reasoning_trace = await self._execute_reasoning(task, reasoning_pattern, context)
            
            # Execute required capabilities
            self.state = AgentState.EXECUTING
            capability_results = await self._execute_task_capabilities(task, reasoning_trace)
            
            # Generate response
            response_content = await self._generate_task_response(
                task, reasoning_trace, capability_results
            )
            
            # Self-reflection and validation
            if task.task_type in ["critical", "high_risk"]:
                reflection = await self._perform_self_reflection(
                    task, response_content
                )
                if reflection.identified_errors:
                    response_content = reflection.corrected_answer
            
            # Create response
            response = AgentResponse(
                agent_id=self.agent_id,
                task_id=task.task_id,
                response_type=task.task_type,
                content=response_content,
                confidence=reasoning_trace.confidence if reasoning_trace else 0.8,
                reasoning_trace=self._serialize_reasoning_trace(reasoning_trace),
                capabilities_used=[r["capability"] for r in capability_results if r["success"]],
                execution_time=time.time() - start_time,
                token_usage=reasoning_trace.token_usage if reasoning_trace else 0
            )
            
            # Check for collaboration needs
            if task.collaboration_required or self._needs_collaboration(task, response):
                collaboration_requests = await self._identify_collaboration_needs(task, response)
                response.collaboration_requests = collaboration_requests
            
            # Identify follow-up tasks
            follow_up_tasks = await self._identify_follow_up_tasks(task, response)
            response.follow_up_tasks = follow_up_tasks
            
            # Store learning
            await self._store_task_learning(task, response)
            
            # Update performance metrics
            self._update_performance_metrics(response)
            
            self.state = AgentState.READY
            return response
            
        except Exception as e:
            log.error(f"Error processing task {task.task_id}: {e}")
            
            self.state = AgentState.ERROR
            return AgentResponse(
                agent_id=self.agent_id,
                task_id=task.task_id,
                response_type="error",
                content=f"Task processing failed: {str(e)}",
                confidence=0.0,
                execution_time=time.time() - start_time
            )
    
    async def initiate_swarm(
        self,
        swarm_objective: str,
        required_agents: int,
        specializations: List[str] = None,
        coordination_pattern: str = "hierarchical"
    ) -> Dict[str, Any]:
        """Initiate agent swarm for complex tasks"""
        
        if not self.orchestrator:
            raise Exception("Orchestrator not available for swarm initiation")
        
        try:
            # Analyze swarm requirements
            swarm_analysis = await self._analyze_swarm_requirements(
                swarm_objective, required_agents, specializations
            )
            
            # Request swarm deployment from orchestrator
            swarm_request = {
                "initiator_agent": self.agent_id,
                "objective": swarm_objective,
                "required_agents": required_agents,
                "specializations": specializations or [],
                "coordination_pattern": coordination_pattern,
                "analysis": swarm_analysis,
                "priority": "normal"
            }
            
            # Submit to orchestrator
            swarm_deployment = await self.orchestrator.deploy_agent_swarm(swarm_request)
            
            # Store swarm initiation in neural mesh
            if self.neural_mesh:
                await self.neural_mesh.store_knowledge(
                    agent_id=self.agent_id,
                    knowledge_type="swarm_initiation",
                    data={
                        "swarm_id": swarm_deployment.get("swarm_id"),
                        "objective": swarm_objective,
                        "agents_deployed": swarm_deployment.get("agents_deployed", 0),
                        "coordination_pattern": coordination_pattern,
                        "timestamp": time.time()
                    },
                    memory_tier="L3"
                )
            
            log.info(f"Agent {self.agent_id} initiated swarm: {swarm_deployment.get('swarm_id')}")
            return swarm_deployment
            
        except Exception as e:
            log.error(f"Error initiating swarm: {e}")
            return {"error": str(e), "success": False}
    
    async def collaborate_with_agents(
        self,
        target_agents: List[str],
        collaboration_type: str,
        shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collaborate with other agents through neural mesh"""
        
        if not self.neural_mesh:
            log.warning("Neural mesh not available for collaboration")
            return {"error": "Neural mesh not available", "success": False}
        
        try:
            # Store collaboration request in neural mesh
            collaboration_id = str(uuid.uuid4())
            
            await self.neural_mesh.store_knowledge(
                agent_id=self.agent_id,
                knowledge_type="collaboration_request",
                data={
                    "collaboration_id": collaboration_id,
                    "target_agents": target_agents,
                    "collaboration_type": collaboration_type,
                    "shared_context": shared_context,
                    "initiator": self.agent_id,
                    "status": "active",
                    "timestamp": time.time()
                },
                memory_tier="L2"
            )
            
            # Notify target agents (through communications gateway)
            notifications_sent = await self._notify_collaboration_targets(
                target_agents, collaboration_id, collaboration_type
            )
            
            return {
                "collaboration_id": collaboration_id,
                "target_agents": target_agents,
                "notifications_sent": notifications_sent,
                "success": True
            }
            
        except Exception as e:
            log.error(f"Error in collaboration: {e}")
            return {"error": str(e), "success": False}
    
    async def learn_from_feedback(
        self,
        task_id: str,
        feedback: Dict[str, Any],
        feedback_source: str = "human"
    ):
        """Learn from feedback and update performance"""
        
        try:
            # Store feedback in neural mesh
            if self.neural_mesh:
                await self.neural_mesh.store_knowledge(
                    agent_id=self.agent_id,
                    knowledge_type="feedback",
                    data={
                        "task_id": task_id,
                        "feedback": feedback,
                        "feedback_source": feedback_source,
                        "agent_id": self.agent_id,
                        "timestamp": time.time()
                    },
                    memory_tier="L3"
                )
            
            # Update performance metrics
            if "performance_score" in feedback:
                self._update_learning_metrics(feedback["performance_score"])
            
            # Identify improvement areas
            if feedback.get("improvement_areas"):
                await self._process_improvement_suggestions(feedback["improvement_areas"])
            
            log.info(f"Agent {self.agent_id} learned from feedback for task {task_id}")
            
        except Exception as e:
            log.error(f"Error learning from feedback: {e}")
    
    async def _setup_agent_permissions(self):
        """Set up agent permissions based on role and specializations"""
        
        # Base capabilities for all agents
        base_capabilities = [
            "process_text",
            "compute_math",
            "analyze_data"
        ]
        
        # Role-specific capabilities
        role_capabilities = {
            AgentRole.COORDINATOR: ["initiate_swarm", "coordinate_agents", "manage_resources"],
            AgentRole.SPECIALIST: ["deep_analysis", "domain_expertise", "specialized_tools"],
            AgentRole.GENERALIST: ["general_reasoning", "multi_domain", "adaptation"],
            AgentRole.ANALYZER: ["data_processing", "statistical_analysis", "pattern_recognition"],
            AgentRole.EXECUTOR: ["task_execution", "workflow_management", "automation"],
            AgentRole.VALIDATOR: ["quality_assurance", "validation", "testing"]
        }
        
        # Specialization-specific capabilities
        specialization_capabilities = {
            "security": ["security_analysis", "threat_detection", "vulnerability_scanning"],
            "data_science": ["ml_modeling", "data_visualization", "statistical_analysis"],
            "software": ["code_analysis", "debugging", "testing", "deployment"],
            "research": ["literature_review", "hypothesis_testing", "experiment_design"]
        }
        
        # Combine capabilities
        allowed_capabilities = base_capabilities.copy()
        allowed_capabilities.extend(role_capabilities.get(self.role, []))
        
        for spec in self.specializations:
            allowed_capabilities.extend(specialization_capabilities.get(spec, []))
        
        # Determine security clearance
        security_clearance = SecurityLevel.SAFE
        if self.role == AgentRole.COORDINATOR:
            security_clearance = SecurityLevel.MEDIUM_RISK
        if "security" in self.specializations:
            security_clearance = SecurityLevel.HIGH_RISK
        
        # Set permissions
        self.capabilities_system.set_agent_permissions(
            agent_id=self.agent_id,
            allowed_capabilities=allowed_capabilities,
            security_clearance=security_clearance,
            resource_limits={
                "max_execution_time": 600,
                "max_memory_mb": 2048,
                "max_file_size_mb": 100
            }
        )
    
    async def _get_task_context(self, task: AgentTask) -> Dict[str, Any]:
        """Get relevant context for task from neural mesh"""
        
        if not self.neural_mesh:
            return {}
        
        try:
            # Query neural mesh for relevant context
            context_query = f"task_type:{task.task_type} agent_role:{self.role.value}"
            
            context = await self.neural_mesh.get_context(
                agent_id=self.agent_id,
                query=context_query,
                memory_tiers=["L1", "L2", "L3"]
            )
            
            return context or {}
            
        except Exception as e:
            log.error(f"Error getting task context: {e}")
            return {}
    
    async def _select_reasoning_pattern(self, task: AgentTask) -> str:
        """Select optimal reasoning pattern for task"""
        
        # Simple heuristics for pattern selection
        if task.task_type in ["analysis", "research", "investigation"]:
            return "chain_of_thought"
        elif task.required_capabilities:
            return "react"  # Need to use tools
        elif task.task_type in ["planning", "strategy", "complex_problem"]:
            return "tree_of_thoughts"
        else:
            return "chain_of_thought"  # Default
    
    async def _execute_reasoning(
        self,
        task: AgentTask,
        reasoning_pattern: str,
        context: Dict[str, Any]
    ):
        """Execute reasoning for the task"""
        
        from core.advanced_reasoning_engine import ReasoningPattern
        
        pattern_map = {
            "chain_of_thought": ReasoningPattern.CHAIN_OF_THOUGHT,
            "react": ReasoningPattern.REACT,
            "tree_of_thoughts": ReasoningPattern.TREE_OF_THOUGHTS
        }
        
        pattern = pattern_map.get(reasoning_pattern, ReasoningPattern.CHAIN_OF_THOUGHT)
        
        # Get available tools for ReAct
        tools = []
        if pattern == ReasoningPattern.REACT:
            available_capabilities = self.capabilities_system.discover_capabilities(
                agent_id=self.agent_id
            )
            tools = [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "parameters": cap.parameters
                }
                for cap in available_capabilities
            ]
        
        # Execute reasoning
        reasoning_trace = await self.reasoning_engine.execute_reasoning_with_pattern(
            agent_id=self.agent_id,
            problem=task.description,
            pattern=pattern,
            context=context,
            tools=tools
        )
        
        return reasoning_trace
    
    async def _execute_task_capabilities(
        self,
        task: AgentTask,
        reasoning_trace
    ) -> List[Dict[str, Any]]:
        """Execute required capabilities for the task"""
        
        results = []
        
        for capability_name in task.required_capabilities:
            try:
                # Get capability parameters from reasoning or task
                parameters = task.parameters.get(capability_name, {})
                
                # Execute capability
                execution = await self.capabilities_system.execute_capability(
                    agent_id=self.agent_id,
                    capability_name=capability_name,
                    parameters=parameters
                )
                
                results.append({
                    "capability": capability_name,
                    "success": execution.success,
                    "result": execution.result,
                    "execution_time": execution.execution_time,
                    "error": execution.error_message
                })
                
            except Exception as e:
                log.error(f"Error executing capability {capability_name}: {e}")
                results.append({
                    "capability": capability_name,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    async def _generate_task_response(
        self,
        task: AgentTask,
        reasoning_trace,
        capability_results: List[Dict[str, Any]]
    ) -> str:
        """Generate final response for the task"""
        
        # Create response generation prompt
        response_prompt = f"""Based on my analysis and execution, I need to provide a comprehensive response:

Original Task: {task.description}
Task Type: {task.task_type}

My Reasoning: {reasoning_trace.final_answer if reasoning_trace else "No reasoning trace available"}

Capability Executions:
{json.dumps(capability_results, indent=2)}

Please provide a clear, comprehensive response that:
1. Directly addresses the original task
2. Incorporates insights from my reasoning
3. Includes relevant results from capability executions
4. Provides actionable recommendations if appropriate
5. Indicates confidence level and any limitations

Response:"""
        
        from core.enhanced_llm_integration import LLMRequest
        
        request = LLMRequest(
            agent_id=self.agent_id,
            task_type="response_generation",
            messages=[{"role": "user", "content": response_prompt}],
            temperature=0.3  # Lower temperature for more focused responses
        )
        
        response = await self.llm_integration.generate_response(request)
        return response.content
    
    async def _perform_self_reflection(
        self,
        task: AgentTask,
        response_content: str
    ):
        """Perform self-reflection on the response"""
        
        return await self.reasoning_engine.self_reflect_and_correct(
            agent_id=self.agent_id,
            original_answer=response_content,
            problem=task.description,
            context=task.context
        )
    
    async def _analyze_swarm_requirements(
        self,
        objective: str,
        required_agents: int,
        specializations: List[str]
    ) -> Dict[str, Any]:
        """Analyze requirements for swarm deployment"""
        
        analysis_prompt = f"""I need to analyze the requirements for deploying an agent swarm:

Objective: {objective}
Required Agents: {required_agents}
Specializations: {specializations}

Analysis needed:
1. Task Complexity Assessment
2. Resource Requirements
3. Coordination Strategy
4. Success Metrics
5. Risk Assessment

Please provide detailed analysis:"""
        
        from core.enhanced_llm_integration import LLMRequest
        
        request = LLMRequest(
            agent_id=self.agent_id,
            task_type="swarm_analysis",
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        response = await self.llm_integration.generate_response(request)
        
        return {
            "analysis": response.content,
            "complexity_score": 0.7,  # Would be calculated based on analysis
            "estimated_duration": 1800,  # Would be estimated
            "resource_requirements": {
                "cpu_cores": required_agents * 2,
                "memory_gb": required_agents * 4,
                "storage_gb": required_agents * 10
            }
        }
    
    async def _load_agent_context(self):
        """Load agent context and history from neural mesh"""
        
        if not self.neural_mesh:
            return
        
        try:
            # Get agent's historical context
            context = await self.neural_mesh.get_context(
                agent_id=self.agent_id,
                query=f"agent:{self.agent_id}",
                memory_tiers=["L2", "L3", "L4"]
            )
            
            if context:
                # Load performance history
                if "performance_metrics" in context:
                    self.performance_metrics.update(context["performance_metrics"])
                
                # Load specialization knowledge
                if "specialization_knowledge" in context:
                    self.working_memory["specializations"] = context["specialization_knowledge"]
                
                log.info(f"Loaded context for agent {self.agent_id}")
            
        except Exception as e:
            log.error(f"Error loading agent context: {e}")
    
    async def _store_task_learning(self, task: AgentTask, response: AgentResponse):
        """Store task execution learning in neural mesh"""
        
        if not self.neural_mesh:
            return
        
        try:
            learning_data = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "agent_role": self.role.value,
                "specializations": self.specializations,
                "success": response.confidence > 0.7,
                "performance_metrics": {
                    "execution_time": response.execution_time,
                    "token_usage": response.token_usage,
                    "cost": response.cost,
                    "confidence": response.confidence
                },
                "capabilities_used": response.capabilities_used,
                "reasoning_pattern": response.reasoning_trace.get("pattern") if response.reasoning_trace else None,
                "timestamp": time.time()
            }
            
            await self.neural_mesh.store_knowledge(
                agent_id=self.agent_id,
                knowledge_type="task_learning",
                data=learning_data,
                memory_tier="L3"
            )
            
        except Exception as e:
            log.error(f"Error storing task learning: {e}")
    
    def _needs_collaboration(self, task: AgentTask, response: AgentResponse) -> bool:
        """Determine if task needs collaboration with other agents"""
        
        # Check if confidence is low
        if response.confidence < 0.6:
            return True
        
        # Check if task is complex
        if task.task_type in ["research", "analysis", "planning"]:
            return True
        
        # Check if multiple specializations needed
        if len(task.required_capabilities) > 3:
            return True
        
        return False
    
    async def _identify_collaboration_needs(
        self,
        task: AgentTask,
        response: AgentResponse
    ) -> List[str]:
        """Identify what type of collaboration is needed"""
        
        collaboration_prompt = f"""Based on this task and my response, what collaboration might be helpful?

Task: {task.description}
My Response: {response.content}
My Confidence: {response.confidence}

What types of agents or expertise would be valuable to:
1. Validate my response
2. Provide additional insights
3. Handle aspects I might have missed
4. Improve the overall solution

Collaboration Needs:"""
        
        from core.enhanced_llm_integration import LLMRequest
        
        request = LLMRequest(
            agent_id=self.agent_id,
            task_type="collaboration_analysis",
            messages=[{"role": "user", "content": collaboration_prompt}]
        )
        
        llm_response = await self.llm_integration.generate_response(request)
        
        # Parse collaboration needs (simplified)
        needs = []
        content = llm_response.content.lower()
        
        if "security" in content:
            needs.append("security_specialist")
        if "data" in content or "analysis" in content:
            needs.append("data_analyst")
        if "validation" in content or "review" in content:
            needs.append("validator")
        if "research" in content:
            needs.append("researcher")
        
        return needs
    
    async def _identify_follow_up_tasks(
        self,
        task: AgentTask,
        response: AgentResponse
    ) -> List[AgentTask]:
        """Identify follow-up tasks based on response"""
        
        follow_up_tasks = []
        
        # Check if response suggests additional work
        if "follow up" in response.content.lower() or "next steps" in response.content.lower():
            # Generate follow-up task suggestions
            follow_up_prompt = f"""Based on my response to this task, what follow-up tasks would be valuable?

Original Task: {task.description}
My Response: {response.content}

What specific follow-up tasks would:
1. Build on this work
2. Validate the results
3. Implement recommendations
4. Address any gaps

Follow-up Tasks:"""
            
            from core.enhanced_llm_integration import LLMRequest
            
            request = LLMRequest(
                agent_id=self.agent_id,
                task_type="follow_up_identification",
                messages=[{"role": "user", "content": follow_up_prompt}]
            )
            
            llm_response = await self.llm_integration.generate_response(request)
            
            # Parse follow-up tasks (simplified)
            # In production, would use more sophisticated parsing
            if "validation" in llm_response.content.lower():
                follow_up_tasks.append(AgentTask(
                    task_id=str(uuid.uuid4()),
                    task_type="validation",
                    description=f"Validate results from task {task.task_id}",
                    context={"original_task": task.task_id, "original_response": response.content}
                ))
        
        return follow_up_tasks
    
    async def _notify_collaboration_targets(
        self,
        target_agents: List[str],
        collaboration_id: str,
        collaboration_type: str
    ) -> int:
        """Notify target agents about collaboration request"""
        
        # This would integrate with the communications gateway
        # For now, return success count
        return len(target_agents)
    
    def _update_performance_metrics(self, response: AgentResponse):
        """Update agent performance metrics"""
        
        self.performance_metrics["tasks_completed"] += 1
        
        # Update success rate
        if response.confidence > 0.7:
            success_count = self.performance_metrics["tasks_completed"] * self.performance_metrics["success_rate"]
            success_count += 1
            self.performance_metrics["success_rate"] = success_count / self.performance_metrics["tasks_completed"]
        
        # Update average response time
        total_time = (self.performance_metrics["avg_response_time"] * 
                     (self.performance_metrics["tasks_completed"] - 1) + 
                     response.execution_time)
        self.performance_metrics["avg_response_time"] = total_time / self.performance_metrics["tasks_completed"]
        
        # Update total cost
        self.performance_metrics["total_cost"] += response.cost
    
    def _update_learning_metrics(self, performance_score: float):
        """Update learning-related metrics"""
        
        self.performance_metrics["learning_events"] += 1
        
        # Adjust confidence based on feedback
        if performance_score > 0.8:
            # Positive feedback - increase confidence in similar tasks
            pass
        elif performance_score < 0.5:
            # Negative feedback - decrease confidence and trigger learning
            pass
    
    async def _process_improvement_suggestions(self, suggestions: List[str]):
        """Process improvement suggestions from feedback"""
        
        for suggestion in suggestions:
            # Store improvement suggestion in neural mesh
            if self.neural_mesh:
                await self.neural_mesh.store_knowledge(
                    agent_id=self.agent_id,
                    knowledge_type="improvement_suggestion",
                    data={
                        "suggestion": suggestion,
                        "agent_id": self.agent_id,
                        "timestamp": time.time()
                    },
                    memory_tier="L3"
                )
    
    def _serialize_reasoning_trace(self, reasoning_trace) -> Optional[Dict[str, Any]]:
        """Serialize reasoning trace for response"""
        
        if not reasoning_trace:
            return None
        
        return {
            "trace_id": reasoning_trace.trace_id,
            "pattern": reasoning_trace.reasoning_pattern.value,
            "steps": reasoning_trace.steps,
            "confidence": reasoning_trace.confidence,
            "execution_time": reasoning_trace.execution_time,
            "token_usage": reasoning_trace.token_usage,
            "success": reasoning_trace.success
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "specializations": self.specializations,
            "state": self.state.value,
            "performance_metrics": self.performance_metrics,
            "capabilities_available": len(self.capabilities_system.discover_capabilities(agent_id=self.agent_id)),
            "neural_mesh_connected": self.neural_mesh is not None,
            "orchestrator_connected": self.orchestrator is not None,
            "working_memory_size": len(self.working_memory),
            "task_history_length": len(self.task_history)
        }
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        
        self.state = AgentState.SHUTDOWN
        
        # Store final state in neural mesh
        if self.neural_mesh:
            await self.neural_mesh.store_knowledge(
                agent_id=self.agent_id,
                knowledge_type="agent_shutdown",
                data={
                    "agent_id": self.agent_id,
                    "final_metrics": self.performance_metrics,
                    "shutdown_time": time.time()
                },
                memory_tier="L4"
            )
        
        log.info(f"Agent {self.agent_id} shutdown completed")

# Factory function for creating enhanced agents
async def create_enhanced_agent(
    role: AgentRole = AgentRole.GENERALIST,
    specializations: List[str] = None,
    agent_id: Optional[str] = None
) -> EnhancedAgentIntelligence:
    """Create an enhanced agent with full AI capabilities"""
    
    if agent_id is None:
        agent_id = f"{role.value}_{str(uuid.uuid4())[:8]}"
    
    agent = EnhancedAgentIntelligence(
        agent_id=agent_id,
        role=role,
        specializations=specializations
    )
    
    # Wait for initialization
    while agent.state == AgentState.INITIALIZING:
        await asyncio.sleep(0.1)
    
    if agent.state == AgentState.ERROR:
        raise Exception(f"Failed to initialize agent {agent_id}")
    
    return agent
