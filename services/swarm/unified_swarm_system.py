"""
Unified Swarm System - Consolidated Production-Ready Intelligence Swarm
Integrates mega-swarm coordination, swarm processing, and worker execution
with perfect neural mesh and unified orchestrator integration
"""

import asyncio
import time
import json
import logging
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# Core AgentForge imports
try:
    from ...unified_orchestrator.core.quantum_orchestrator import (
        UnifiedQuantumOrchestrator, UnifiedTask, TaskPriority
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    # Mock classes
    class UnifiedQuantumOrchestrator: pass
    class UnifiedTask: pass
    class TaskPriority: pass

try:
    from ...neural_mesh.production_neural_mesh import ProductionNeuralMesh
    NEURAL_MESH_AVAILABLE = True
except ImportError:
    NEURAL_MESH_AVAILABLE = False
    class ProductionNeuralMesh: pass
from .fusion.production_fusion_system import (
    ProductionFusionSystem, IntelligenceFusionRequest,
    IntelligenceDomain, FusionQualityLevel, create_production_fusion_system
)

# Enhanced fusion capabilities
from .fusion import (
    StreamingFusionProcessor, DistributedCoordinator, StreamProcessingMode,
    CoordinationStrategy, FusionPriority, ClassificationLevel
)

# Legacy components
from .capability_registry import CapabilityRegistry
from .agent_factory import AgentFactory

log = logging.getLogger("unified-swarm-system")

class SwarmScale(Enum):
    """Scale levels for swarm operations"""
    SMALL = "small"          # 1-1K agents
    MEDIUM = "medium"        # 1K-10K agents  
    LARGE = "large"          # 10K-100K agents
    MEGA = "mega"            # 100K-1M agents
    GIGA = "giga"            # 1M+ agents
    QUANTUM = "quantum"      # 1M+ agents with quantum coordination

class SwarmObjective(Enum):
    """High-level objectives for swarm operations"""
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_LATENCY = "minimize_latency"
    OPTIMIZE_QUALITY = "optimize_quality"
    BALANCE_RESOURCES = "balance_resources"
    ADAPTIVE_LEARNING = "adaptive_learning"
    INTELLIGENCE_FUSION = "intelligence_fusion"
    NEURAL_MESH_SYNC = "neural_mesh_sync"

class SwarmMode(Enum):
    """Swarm operation modes"""
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    CLUSTER_SWARM = "cluster_swarm"
    MEGA_SWARM = "mega_swarm"
    QUANTUM_SWARM = "quantum_swarm"
    NEURAL_MESH_SWARM = "neural_mesh_swarm"

@dataclass
class UnifiedGoal:
    """Unified goal for swarm execution"""
    goal_id: str
    description: str
    objective: SwarmObjective
    mode: SwarmMode
    scale: SwarmScale
    intelligence_domain: IntelligenceDomain = IntelligenceDomain.REAL_TIME_OPERATIONS
    classification_level: ClassificationLevel = ClassificationLevel.CONFIDENTIAL
    requirements: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 50
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_unified_task(self) -> UnifiedTask:
        """Convert to unified orchestrator task"""
        task = UnifiedTask(
            task_id=self.goal_id,
            description=self.description,
            priority=TaskPriority.NORMAL if self.priority <= 50 else TaskPriority.HIGH
        )
        
        # Set additional fields
        task.required_agents = self.requirements.get("required_agents", 1)
        task.max_agents = self.requirements.get("max_agents", 1000)
        task.resource_constraints = self.constraints
        task.metadata.update({
            **self.metadata,
            "swarm_scale": self.scale.value,
            "swarm_mode": self.mode.value,
            "intelligence_domain": self.intelligence_domain.value,
            "classification_level": self.classification_level.value
        })
        
        return task

@dataclass
class SwarmAgent:
    """Enhanced swarm agent with neural mesh integration"""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    current_load: float = 0.0
    performance_score: float = 1.0
    neural_mesh_connection: Optional[str] = None
    cluster_id: Optional[str] = None
    quantum_state: Optional[Dict[str, Any]] = None
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_available(self) -> bool:
        """Check if agent is available for tasks"""
        return (
            self.current_load < 0.8 and
            (time.time() - self.last_heartbeat) < 60.0  # 1 minute timeout
        )
    
    def get_capacity(self) -> float:
        """Get available capacity"""
        return max(0.0, 1.0 - self.current_load)

@dataclass
class SwarmCluster:
    """Swarm cluster for hierarchical organization"""
    cluster_id: str
    cluster_type: str
    agents: Dict[str, SwarmAgent]
    specialization: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    neural_mesh_sync: bool = True
    quantum_entangled: bool = False
    
    def get_cluster_capacity(self) -> float:
        """Get total cluster capacity"""
        if not self.agents:
            return 0.0
        
        return sum(agent.get_capacity() for agent in self.agents.values())
    
    def get_available_agents(self) -> List[SwarmAgent]:
        """Get available agents in cluster"""
        return [agent for agent in self.agents.values() if agent.is_available()]

@dataclass
class SwarmExecutionResult:
    """Result of swarm execution"""
    goal_id: str
    success: bool
    result: Dict[str, Any]
    execution_metrics: Dict[str, Any]
    agents_used: int
    clusters_used: int
    execution_time: float
    confidence: float
    neural_mesh_updates: Dict[str, Any] = field(default_factory=dict)
    fusion_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedSwarmSystem:
    """
    Unified Swarm System - Complete Integration
    
    Consolidates mega-swarm coordination, swarm processing, and worker execution
    with perfect neural mesh and unified orchestrator integration.
    """
    
    def __init__(self, 
                 node_id: str,
                 neural_mesh: Optional[ProductionNeuralMesh] = None,
                 orchestrator: Optional[UnifiedQuantumOrchestrator] = None):
        
        self.node_id = node_id
        self.neural_mesh = neural_mesh
        self.orchestrator = orchestrator
        
        # Core swarm components
        self.agents: Dict[str, SwarmAgent] = {}
        self.clusters: Dict[str, SwarmCluster] = {}
        self.active_goals: Dict[str, UnifiedGoal] = {}
        
        # Legacy component integration
        self.capability_registry = CapabilityRegistry()
        
        # Initialize agent registry and factory
        from .agent_factory import AgentRegistry
        self.agent_registry = AgentRegistry()
        self.agent_factory = AgentFactory(self.agent_registry)
        
        # Production fusion system
        self.fusion_system: Optional[ProductionFusionSystem] = None
        
        # Streaming and coordination
        self.streaming_processor: Optional[StreamingFusionProcessor] = None
        self.distributed_coordinator: Optional[DistributedCoordinator] = None
        
        # Performance tracking
        self.swarm_metrics = {
            "goals_executed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_agents_deployed": 0,
            "peak_concurrent_agents": 0,
            "average_execution_time": 0.0,
            "neural_mesh_syncs": 0,
            "quantum_coherence": 1.0
        }
        
        # Execution state
        self.execution_pipeline: Dict[str, List[Dict[str, Any]]] = {}
        self.result_cache: Dict[str, SwarmExecutionResult] = {}
        
        # Control flags
        self._initialized = False
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
        log.info(f"Unified swarm system created for node {node_id}")
    
    async def initialize_system(self) -> bool:
        """Initialize the unified swarm system"""
        
        try:
            start_time = time.time()
            log.info("Initializing unified swarm system...")
            
            # Initialize neural mesh if not provided
            if not self.neural_mesh:
                from ..neural_mesh.production_neural_mesh import create_production_neural_mesh
                self.neural_mesh = await create_production_neural_mesh(
                    node_id=self.node_id,
                    enable_l4_memory=True
                )
            
            # Initialize orchestrator if not provided
            if not self.orchestrator:
                self.orchestrator = UnifiedQuantumOrchestrator(
                    node_id=self.node_id,
                    max_agents=1000000,
                    enable_security=True
                )
                await self.orchestrator.initialize()
            
            # Initialize production fusion system
            self.fusion_system = await create_production_fusion_system(
                node_id=self.node_id,
                intelligence_domain=IntelligenceDomain.REAL_TIME_OPERATIONS
            )
            
            # Initialize streaming components
            from .fusion.streaming_fusion import create_high_performance_fusion_system
            self.streaming_processor, self.distributed_coordinator = await create_high_performance_fusion_system(
                node_id=self.node_id,
                processing_mode=StreamProcessingMode.REAL_TIME,
                coordination_strategy=CoordinationStrategy.QUANTUM_DISTRIBUTED
            )
            
            # Setup integration bridges
            await self._setup_neural_mesh_integration()
            await self._setup_orchestrator_integration()
            await self._setup_fusion_integration()
            
            # Register core capabilities
            await self._register_core_capabilities()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._initialized = True
            self._running = True
            
            initialization_time = (time.time() - start_time) * 1000
            log.info(f"Unified swarm system initialized in {initialization_time:.2f}ms")
            
            return True
            
        except Exception as e:
            log.error(f"Unified swarm system initialization failed: {e}")
            return False
    
    async def execute_unified_goal(self, goal: UnifiedGoal) -> SwarmExecutionResult:
        """Execute unified goal with full system integration"""
        
        if not self._initialized:
            raise RuntimeError("Unified swarm system not initialized")
        
        start_time = time.time()
        self.active_goals[goal.goal_id] = goal
        
        try:
            log.info(f"Executing unified goal {goal.goal_id} in {goal.mode.value} mode")
            
            # Store goal in neural mesh
            await self.neural_mesh.store(
                f"goal:{goal.goal_id}",
                goal.description,
                context={
                    "type": "swarm_goal",
                    "mode": goal.mode.value,
                    "scale": goal.scale.value,
                    "objective": goal.objective.value
                },
                metadata=goal.metadata
            )
            
            # Execute based on swarm mode
            if goal.mode == SwarmMode.QUANTUM_SWARM:
                result = await self._execute_quantum_swarm(goal)
            elif goal.mode == SwarmMode.NEURAL_MESH_SWARM:
                result = await self._execute_neural_mesh_swarm(goal)
            elif goal.mode == SwarmMode.MEGA_SWARM:
                result = await self._execute_mega_swarm(goal)
            elif goal.mode == SwarmMode.CLUSTER_SWARM:
                result = await self._execute_cluster_swarm(goal)
            elif goal.mode == SwarmMode.MULTI_AGENT:
                result = await self._execute_multi_agent(goal)
            else:
                result = await self._execute_single_agent(goal)
            
            # Update neural mesh with results
            await self._update_neural_mesh_with_results(goal, result)
            
            # Update system metrics
            self._update_swarm_metrics(result, True)
            
            log.info(f"Goal {goal.goal_id} executed successfully with {result.agents_used} agents")
            
            return result
            
        except Exception as e:
            log.error(f"Goal execution failed for {goal.goal_id}: {e}")
            
            # Create error result
            error_result = SwarmExecutionResult(
                goal_id=goal.goal_id,
                success=False,
                result={"error": str(e)},
                execution_metrics={"error": str(e)},
                agents_used=0,
                clusters_used=0,
                execution_time=time.time() - start_time,
                confidence=0.0
            )
            
            self._update_swarm_metrics(error_result, False)
            return error_result
            
        finally:
            # Clean up active goal
            if goal.goal_id in self.active_goals:
                del self.active_goals[goal.goal_id]
    
    async def _execute_quantum_swarm(self, goal: UnifiedGoal) -> SwarmExecutionResult:
        """Execute goal using quantum swarm coordination"""
        
        start_time = time.time()
        
        try:
            # Convert to unified task for orchestrator
            unified_task = goal.to_unified_task()
            
            # Execute through quantum orchestrator
            orchestration_result = await self.orchestrator.execute_task(unified_task)
            
            # Process fusion requirements if applicable
            fusion_results = []
            if goal.objective == SwarmObjective.INTELLIGENCE_FUSION:
                fusion_request = IntelligenceFusionRequest(
                    request_id=goal.goal_id,
                    domain=goal.intelligence_domain,
                    sensor_data=goal.requirements.get("sensor_data", {}),
                    quality_requirement=FusionQualityLevel.OPERATIONAL_GRADE,
                    classification_level=goal.classification_level,
                    priority=FusionPriority.HIGH
                )
                
                fusion_result = await self.fusion_system.process_intelligence_fusion(fusion_request)
                fusion_results.append(fusion_result.fusion_result)
            
            return SwarmExecutionResult(
                goal_id=goal.goal_id,
                success=orchestration_result.success,
                result=orchestration_result.result,
                execution_metrics=orchestration_result.execution_metrics,
                agents_used=orchestration_result.agents_used,
                clusters_used=len(orchestration_result.cluster_assignments),
                execution_time=time.time() - start_time,
                confidence=orchestration_result.confidence,
                fusion_results=fusion_results,
                metadata={
                    "execution_mode": "quantum_swarm",
                    "quantum_coherence": orchestration_result.quantum_coherence
                }
            )
            
        except Exception as e:
            log.error(f"Quantum swarm execution failed: {e}")
            raise
    
    async def _execute_neural_mesh_swarm(self, goal: UnifiedGoal) -> SwarmExecutionResult:
        """Execute goal using neural mesh swarm coordination"""
        
        start_time = time.time()
        
        try:
            # Query neural mesh for relevant knowledge
            relevant_knowledge = await self.neural_mesh.query(
                goal.description,
                context={"type": "goal_execution", "domain": goal.intelligence_domain.value},
                limit=10
            )
            
            # Enhance goal with neural mesh insights
            enhanced_requirements = {
                **goal.requirements,
                "neural_mesh_insights": [item.content for item in relevant_knowledge.results],
                "knowledge_confidence": relevant_knowledge.confidence
            }
            
            # Create enhanced goal
            enhanced_goal = UnifiedGoal(
                goal_id=f"{goal.goal_id}_enhanced",
                description=goal.description,
                objective=goal.objective,
                mode=SwarmMode.MEGA_SWARM,  # Execute as mega swarm
                scale=goal.scale,
                intelligence_domain=goal.intelligence_domain,
                classification_level=goal.classification_level,
                requirements=enhanced_requirements,
                constraints=goal.constraints,
                priority=goal.priority,
                deadline=goal.deadline,
                metadata={**goal.metadata, "neural_mesh_enhanced": True}
            )
            
            # Execute enhanced goal
            result = await self._execute_mega_swarm(enhanced_goal)
            
            # Update neural mesh with execution results
            await self.neural_mesh.store(
                f"execution_result:{goal.goal_id}",
                json.dumps(result.result),
                context={
                    "type": "execution_result",
                    "success": result.success,
                    "confidence": result.confidence
                },
                metadata=result.metadata
            )
            
            result.metadata["execution_mode"] = "neural_mesh_swarm"
            result.metadata["neural_mesh_enhanced"] = True
            
            return result
            
        except Exception as e:
            log.error(f"Neural mesh swarm execution failed: {e}")
            raise
    
    async def _execute_mega_swarm(self, goal: UnifiedGoal) -> SwarmExecutionResult:
        """Execute goal using mega-swarm coordination"""
        
        start_time = time.time()
        
        try:
            # Decompose goal into cluster tasks
            cluster_tasks = await self._decompose_goal_to_clusters(goal)
            
            # Assign tasks to clusters
            cluster_assignments = await self._assign_tasks_to_clusters(cluster_tasks)
            
            # Execute cluster tasks in parallel
            cluster_results = await self._execute_cluster_tasks_parallel(cluster_assignments)
            
            # Aggregate results using quantum-inspired fusion
            aggregated_result = await self._quantum_aggregate_results(cluster_results)
            
            # Calculate execution metrics
            total_agents = sum(len(assignment.get("agents", [])) for assignment in cluster_assignments)
            
            return SwarmExecutionResult(
                goal_id=goal.goal_id,
                success=len([r for r in cluster_results if r.get("success", False)]) > len(cluster_results) * 0.7,
                result=aggregated_result,
                execution_metrics={
                    "cluster_tasks": len(cluster_tasks),
                    "cluster_assignments": len(cluster_assignments),
                    "successful_clusters": len([r for r in cluster_results if r.get("success", False)]),
                    "total_processing_time": sum(r.get("processing_time", 0) for r in cluster_results)
                },
                agents_used=total_agents,
                clusters_used=len(cluster_assignments),
                execution_time=time.time() - start_time,
                confidence=aggregated_result.get("confidence", 0.8),
                metadata={"execution_mode": "mega_swarm"}
            )
            
        except Exception as e:
            log.error(f"Mega swarm execution failed: {e}")
            raise
    
    async def _execute_cluster_swarm(self, goal: UnifiedGoal) -> SwarmExecutionResult:
        """Execute goal using cluster swarm coordination"""
        
        start_time = time.time()
        
        try:
            # Select appropriate clusters
            selected_clusters = await self._select_clusters_for_goal(goal)
            
            # Distribute goal across selected clusters
            cluster_tasks = []
            for cluster in selected_clusters:
                cluster_task = {
                    "cluster_id": cluster.cluster_id,
                    "task_description": goal.description,
                    "requirements": goal.requirements,
                    "agents": list(cluster.get_available_agents())
                }
                cluster_tasks.append(cluster_task)
            
            # Execute cluster tasks
            cluster_results = []
            for task in cluster_tasks:
                result = await self._execute_cluster_task(task)
                cluster_results.append(result)
            
            # Aggregate results
            final_result = await self._aggregate_cluster_results(cluster_results)
            
            return SwarmExecutionResult(
                goal_id=goal.goal_id,
                success=len([r for r in cluster_results if r.get("success", False)]) > 0,
                result=final_result,
                execution_metrics={"cluster_results": cluster_results},
                agents_used=sum(len(task["agents"]) for task in cluster_tasks),
                clusters_used=len(selected_clusters),
                execution_time=time.time() - start_time,
                confidence=final_result.get("confidence", 0.7),
                metadata={"execution_mode": "cluster_swarm"}
            )
            
        except Exception as e:
            log.error(f"Cluster swarm execution failed: {e}")
            raise
    
    async def _execute_multi_agent(self, goal: UnifiedGoal) -> SwarmExecutionResult:
        """Execute goal using multi-agent coordination"""
        
        start_time = time.time()
        
        try:
            # Select agents for goal
            selected_agents = await self._select_agents_for_goal(goal, max_agents=10)
            
            # Create agent tasks
            agent_tasks = []
            for agent in selected_agents:
                agent_task = {
                    "agent_id": agent.agent_id,
                    "task_description": goal.description,
                    "requirements": goal.requirements,
                    "capabilities": agent.capabilities
                }
                agent_tasks.append(agent_task)
            
            # Execute agent tasks in parallel
            agent_results = await asyncio.gather(
                *[self._execute_agent_task(task) for task in agent_tasks],
                return_exceptions=True
            )
            
            # Process results
            successful_results = [r for r in agent_results if not isinstance(r, Exception) and r.get("success", False)]
            
            # Aggregate results
            final_result = await self._aggregate_agent_results(successful_results)
            
            return SwarmExecutionResult(
                goal_id=goal.goal_id,
                success=len(successful_results) > 0,
                result=final_result,
                execution_metrics={"agent_results": len(agent_results), "successful_agents": len(successful_results)},
                agents_used=len(selected_agents),
                clusters_used=0,
                execution_time=time.time() - start_time,
                confidence=final_result.get("confidence", 0.6),
                metadata={"execution_mode": "multi_agent"}
            )
            
        except Exception as e:
            log.error(f"Multi-agent execution failed: {e}")
            raise
    
    async def _execute_single_agent(self, goal: UnifiedGoal) -> SwarmExecutionResult:
        """Execute goal using single agent"""
        
        start_time = time.time()
        
        try:
            # Select best agent for goal
            selected_agents = await self._select_agents_for_goal(goal, max_agents=1)
            
            if not selected_agents:
                raise RuntimeError("No suitable agent available")
            
            agent = selected_agents[0]
            
            # Execute task
            agent_task = {
                "agent_id": agent.agent_id,
                "task_description": goal.description,
                "requirements": goal.requirements,
                "capabilities": agent.capabilities
            }
            
            result = await self._execute_agent_task(agent_task)
            
            return SwarmExecutionResult(
                goal_id=goal.goal_id,
                success=result.get("success", False),
                result=result,
                execution_metrics={"single_agent_execution": True},
                agents_used=1,
                clusters_used=0,
                execution_time=time.time() - start_time,
                confidence=result.get("confidence", 0.5),
                metadata={"execution_mode": "single_agent", "agent_id": agent.agent_id}
            )
            
        except Exception as e:
            log.error(f"Single agent execution failed: {e}")
            raise
    
    async def _setup_neural_mesh_integration(self):
        """Setup neural mesh integration"""
        
        # Register swarm system with neural mesh
        await self.neural_mesh.store(
            f"swarm_system:{self.node_id}",
            "Unified Swarm System",
            context={"type": "system_registration", "node_id": self.node_id},
            metadata={"capabilities": ["mega_swarm", "quantum_coordination", "intelligence_fusion"]}
        )
        
        log.info("Neural mesh integration setup complete")
    
    async def _setup_orchestrator_integration(self):
        """Setup orchestrator integration"""
        
        # Register swarm system as orchestrator component
        if hasattr(self.orchestrator, 'register_component'):
            await self.orchestrator.register_component(
                component_id=f"swarm_{self.node_id}",
                component_type="unified_swarm",
                capabilities=["million_scale_coordination", "quantum_processing", "neural_mesh_sync"]
            )
        
        log.info("Orchestrator integration setup complete")
    
    async def _setup_fusion_integration(self):
        """Setup fusion system integration"""
        
        # Register fusion result callback
        async def fusion_result_callback(fusion_result: Dict[str, Any]):
            """Handle fusion results and update neural mesh"""
            try:
                await self.neural_mesh.store(
                    f"fusion_result:{fusion_result.get('task_id', 'unknown')}",
                    json.dumps(fusion_result),
                    context={"type": "fusion_result", "confidence": fusion_result.get("confidence", 0.0)},
                    metadata={"processing_time": fusion_result.get("processing_time_ms", 0)}
                )
                
                self.swarm_metrics["neural_mesh_syncs"] += 1
                
            except Exception as e:
                log.warning(f"Fusion result neural mesh update failed: {e}")
        
        if self.streaming_processor:
            self.streaming_processor.register_result_callback("neural_mesh_sync", fusion_result_callback)
        
        log.info("Fusion integration setup complete")
    
    async def _register_core_capabilities(self):
        """Register core swarm capabilities"""
        
        # Register mega-swarm capability
        self.capability_registry.register_capability(
            name="mega_swarm_coordination",
            handler=self._mega_swarm_handler,
            provides=["million_scale_coordination", "quantum_processing"],
            tags=["swarm", "coordination", "quantum"]
        )
        
        # Register neural mesh capability
        self.capability_registry.register_capability(
            name="neural_mesh_swarm",
            handler=self._neural_mesh_handler,
            provides=["neural_mesh_coordination", "belief_revision"],
            tags=["neural_mesh", "swarm", "intelligence"]
        )
        
        # Register fusion capability
        self.capability_registry.register_capability(
            name="intelligence_fusion",
            handler=self._fusion_handler,
            provides=["multi_sensor_fusion", "bayesian_processing"],
            tags=["fusion", "intelligence", "sensors"]
        )
        
        log.info("Core capabilities registered")
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        
        # Agent heartbeat monitoring
        self._background_tasks.append(
            asyncio.create_task(self._agent_heartbeat_monitor())
        )
        
        # Cluster health monitoring
        self._background_tasks.append(
            asyncio.create_task(self._cluster_health_monitor())
        )
        
        # Neural mesh synchronization
        self._background_tasks.append(
            asyncio.create_task(self._neural_mesh_sync_task())
        )
        
        # Metrics collection
        self._background_tasks.append(
            asyncio.create_task(self._metrics_collection_task())
        )
        
        log.info("Background tasks started")
    
    async def _agent_heartbeat_monitor(self):
        """Monitor agent heartbeats and remove inactive agents"""
        
        while self._running:
            try:
                current_time = time.time()
                inactive_agents = []
                
                for agent_id, agent in self.agents.items():
                    if current_time - agent.last_heartbeat > 120.0:  # 2 minutes timeout
                        inactive_agents.append(agent_id)
                
                # Remove inactive agents
                for agent_id in inactive_agents:
                    await self._remove_agent(agent_id)
                    log.info(f"Removed inactive agent: {agent_id}")
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                log.error(f"Agent heartbeat monitoring failed: {e}")
                await asyncio.sleep(60.0)
    
    async def _cluster_health_monitor(self):
        """Monitor cluster health and performance"""
        
        while self._running:
            try:
                for cluster_id, cluster in self.clusters.items():
                    # Update cluster performance metrics
                    cluster.performance_metrics.update({
                        "available_agents": len(cluster.get_available_agents()),
                        "total_capacity": cluster.get_cluster_capacity(),
                        "last_health_check": time.time()
                    })
                
                await asyncio.sleep(60.0)  # Check every minute
                
            except Exception as e:
                log.error(f"Cluster health monitoring failed: {e}")
                await asyncio.sleep(120.0)
    
    async def _neural_mesh_sync_task(self):
        """Periodic neural mesh synchronization"""
        
        while self._running:
            try:
                # Sync swarm metrics to neural mesh
                await self.neural_mesh.store(
                    f"swarm_metrics:{self.node_id}",
                    json.dumps(self.swarm_metrics),
                    context={"type": "system_metrics", "node_id": self.node_id},
                    metadata={"timestamp": time.time()}
                )
                
                # Sync active goals
                for goal_id, goal in self.active_goals.items():
                    await self.neural_mesh.store(
                        f"active_goal:{goal_id}",
                        goal.description,
                        context={"type": "active_goal", "status": "executing"},
                        metadata=goal.metadata
                    )
                
                self.swarm_metrics["neural_mesh_syncs"] += 1
                
                await asyncio.sleep(300.0)  # Sync every 5 minutes
                
            except Exception as e:
                log.error(f"Neural mesh sync failed: {e}")
                await asyncio.sleep(600.0)
    
    async def _metrics_collection_task(self):
        """Collect and update system metrics"""
        
        while self._running:
            try:
                # Update system metrics
                self.swarm_metrics.update({
                    "active_agents": len([a for a in self.agents.values() if a.is_available()]),
                    "active_clusters": len(self.clusters),
                    "active_goals": len(self.active_goals),
                    "system_uptime": time.time(),
                    "neural_mesh_connected": self.neural_mesh is not None,
                    "orchestrator_connected": self.orchestrator is not None
                })
                
                await asyncio.sleep(30.0)  # Update every 30 seconds
                
            except Exception as e:
                log.error(f"Metrics collection failed: {e}")
                await asyncio.sleep(60.0)
    
    # Capability handlers
    async def _mega_swarm_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for mega-swarm capability"""
        try:
            goal_data = kwargs.get("goal", {})
            goal = UnifiedGoal(
                goal_id=goal_data.get("goal_id", f"mega_{uuid.uuid4().hex[:8]}"),
                description=goal_data.get("description", "Mega-swarm execution"),
                objective=SwarmObjective(goal_data.get("objective", "maximize_throughput")),
                mode=SwarmMode.MEGA_SWARM,
                scale=SwarmScale(goal_data.get("scale", "mega")),
                requirements=goal_data.get("requirements", {})
            )
            
            result = await self.execute_unified_goal(goal)
            return result.result
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _neural_mesh_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for neural mesh capability"""
        try:
            goal_data = kwargs.get("goal", {})
            goal = UnifiedGoal(
                goal_id=goal_data.get("goal_id", f"neural_{uuid.uuid4().hex[:8]}"),
                description=goal_data.get("description", "Neural mesh swarm execution"),
                objective=SwarmObjective(goal_data.get("objective", "adaptive_learning")),
                mode=SwarmMode.NEURAL_MESH_SWARM,
                scale=SwarmScale(goal_data.get("scale", "large")),
                requirements=goal_data.get("requirements", {})
            )
            
            result = await self.execute_unified_goal(goal)
            return result.result
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _fusion_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for fusion capability"""
        try:
            sensor_data = kwargs.get("sensor_data", {})
            domain = kwargs.get("domain", "real_time_operations")
            
            fusion_request = IntelligenceFusionRequest(
                request_id=f"fusion_{uuid.uuid4().hex[:8]}",
                domain=IntelligenceDomain(domain),
                sensor_data=sensor_data,
                quality_requirement=FusionQualityLevel.OPERATIONAL_GRADE,
                classification_level=ClassificationLevel.CONFIDENTIAL
            )
            
            result = await self.fusion_system.process_intelligence_fusion(fusion_request)
            return result.fusion_result
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    # Utility methods
    def _update_swarm_metrics(self, result: SwarmExecutionResult, success: bool):
        """Update swarm performance metrics"""
        
        self.swarm_metrics["goals_executed"] += 1
        
        if success:
            self.swarm_metrics["successful_executions"] += 1
        else:
            self.swarm_metrics["failed_executions"] += 1
        
        # Update average execution time
        current_avg = self.swarm_metrics["average_execution_time"]
        total_goals = self.swarm_metrics["goals_executed"]
        
        self.swarm_metrics["average_execution_time"] = (
            (current_avg * (total_goals - 1) + result.execution_time) / total_goals
        )
        
        # Update peak agents
        if result.agents_used > self.swarm_metrics["peak_concurrent_agents"]:
            self.swarm_metrics["peak_concurrent_agents"] = result.agents_used
        
        self.swarm_metrics["total_agents_deployed"] += result.agents_used
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        try:
            status = {
                "timestamp": time.time(),
                "node_id": self.node_id,
                "initialized": self._initialized,
                "running": self._running,
                "swarm_metrics": self.swarm_metrics.copy(),
                "active_agents": len(self.agents),
                "active_clusters": len(self.clusters),
                "active_goals": len(self.active_goals),
                "background_tasks": len(self._background_tasks)
            }
            
            if self._initialized:
                # Get component statuses
                status["component_status"] = {}
                
                if self.neural_mesh:
                    status["component_status"]["neural_mesh"] = await self.neural_mesh.get_status()
                
                if self.orchestrator:
                    status["component_status"]["orchestrator"] = self.orchestrator.get_orchestration_stats()
                
                if self.fusion_system:
                    status["component_status"]["fusion_system"] = await self.fusion_system.get_system_status()
                
                if self.streaming_processor:
                    status["component_status"]["streaming_processor"] = self.streaming_processor.get_performance_statistics()
            
            return status
            
        except Exception as e:
            log.error(f"System status generation failed: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    async def shutdown_system(self):
        """Gracefully shutdown the unified swarm system"""
        
        log.info("Shutting down unified swarm system...")
        
        try:
            self._running = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for background tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Shutdown components
            if self.fusion_system:
                await self.fusion_system.shutdown_system()
            
            if self.streaming_processor:
                await self.streaming_processor.stop_processing()
            
            if self.orchestrator:
                await self.orchestrator.shutdown()
            
            # Final neural mesh update
            if self.neural_mesh:
                await self.neural_mesh.store(
                    f"system_shutdown:{self.node_id}",
                    "Unified swarm system shutdown",
                    context={"type": "system_event", "event": "shutdown"},
                    metadata={"final_metrics": self.swarm_metrics}
                )
            
            log.info("Unified swarm system shutdown complete")
            
        except Exception as e:
            log.error(f"System shutdown error: {e}")

# Factory function for creating unified swarm systems
async def create_unified_swarm_system(
    node_id: str,
    neural_mesh: Optional[ProductionNeuralMesh] = None,
    orchestrator: Optional[UnifiedQuantumOrchestrator] = None
) -> UnifiedSwarmSystem:
    """Create and initialize unified swarm system"""
    
    system = UnifiedSwarmSystem(node_id, neural_mesh, orchestrator)
    
    if await system.initialize_system():
        log.info(f"Unified swarm system ready for node {node_id}")
        return system
    else:
        raise RuntimeError("Failed to initialize unified swarm system")

# Backwards compatibility exports
__all__ = [
    "UnifiedSwarmSystem",
    "UnifiedGoal", 
    "SwarmAgent",
    "SwarmCluster",
    "SwarmExecutionResult",
    "SwarmScale",
    "SwarmObjective", 
    "SwarmMode",
    "create_unified_swarm_system"
]
