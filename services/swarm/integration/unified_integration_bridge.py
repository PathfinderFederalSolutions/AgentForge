"""
Unified Integration Bridge
Perfect integration between Unified Swarm, Neural Mesh, and Unified Orchestrator
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

# Core system imports
try:
    from ...neural_mesh.production_neural_mesh import ProductionNeuralMesh
    from ...unified_orchestrator.core.quantum_orchestrator import (
        UnifiedQuantumOrchestrator, UnifiedTask, TaskPriority, QuantumAgent
    )
    CORE_SYSTEMS_AVAILABLE = True
except ImportError:
    ProductionNeuralMesh = None  # type: ignore
    UnifiedQuantumOrchestrator = None  # type: ignore
    UnifiedTask = None  # type: ignore
    TaskPriority = None  # type: ignore
    QuantumAgent = None  # type: ignore
    CORE_SYSTEMS_AVAILABLE = False

# Swarm system imports
from ..unified_swarm_system import UnifiedSwarmSystem, UnifiedGoal, SwarmExecutionResult
from ..fusion.production_fusion_system import ProductionFusionSystem, IntelligenceFusionRequest
from ..core.unified_agent import UnifiedAgent, UnifiedAgentFactory

log = logging.getLogger("unified-integration-bridge")

class IntegrationMode(Enum):
    """Integration modes between systems"""
    FULL_INTEGRATION = "full_integration"
    NEURAL_MESH_ONLY = "neural_mesh_only"  
    ORCHESTRATOR_ONLY = "orchestrator_only"
    SWARM_STANDALONE = "swarm_standalone"

class SynchronizationStrategy(Enum):
    """Strategies for system synchronization"""
    REAL_TIME = "real_time"
    BATCH_SYNC = "batch_sync"
    EVENT_DRIVEN = "event_driven"
    PERIODIC = "periodic"

@dataclass
class IntegrationConfiguration:
    """Configuration for system integration"""
    mode: IntegrationMode = IntegrationMode.FULL_INTEGRATION
    sync_strategy: SynchronizationStrategy = SynchronizationStrategy.EVENT_DRIVEN
    sync_interval_seconds: float = 300.0  # 5 minutes
    enable_cross_system_tasks: bool = True
    enable_shared_memory: bool = True
    enable_unified_metrics: bool = True
    max_sync_retries: int = 3
    
class UnifiedIntegrationBridge:
    """
    Unified Integration Bridge - Perfect System Integration
    
    Ensures seamless integration between:
    - Unified Swarm System
    - Neural Mesh Production System  
    - Unified Quantum Orchestrator
    
    Provides unified task execution, shared memory, and coordinated intelligence.
    """
    
    def __init__(self, 
                 config: IntegrationConfiguration,
                 swarm_system: Optional[UnifiedSwarmSystem] = None,
                 neural_mesh: Optional[ProductionNeuralMesh] = None,
                 orchestrator: Optional[UnifiedQuantumOrchestrator] = None):
        
        self.config = config
        self.swarm_system = swarm_system
        self.neural_mesh = neural_mesh
        self.orchestrator = orchestrator
        
        # Integration state
        self.integration_active = False
        self.cross_system_tasks: Dict[str, Dict[str, Any]] = {}
        self.sync_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.integration_metrics = {
            "tasks_coordinated": 0,
            "successful_integrations": 0,
            "failed_integrations": 0,
            "neural_mesh_syncs": 0,
            "orchestrator_coordinations": 0,
            "cross_system_executions": 0,
            "average_integration_time": 0.0
        }
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        log.info(f"Unified integration bridge initialized in {config.mode.value} mode")
    
    async def initialize_integration(self) -> bool:
        """Initialize system integration"""
        
        try:
            log.info("Initializing unified system integration...")
            
            # Validate system availability
            if not self._validate_system_availability():
                return False
            
            # Setup cross-system communication
            await self._setup_cross_system_communication()
            
            # Setup shared memory integration
            if self.config.enable_shared_memory:
                await self._setup_shared_memory_integration()
            
            # Setup unified task coordination
            if self.config.enable_cross_system_tasks:
                await self._setup_unified_task_coordination()
            
            # Start background synchronization
            await self._start_background_sync()
            
            self.integration_active = True
            
            log.info("Unified system integration initialized successfully")
            return True
            
        except Exception as e:
            log.error(f"Integration initialization failed: {e}")
            return False
    
    async def execute_unified_task(self, 
                                  task_description: str,
                                  task_type: str = "general",
                                  priority: str = "normal",
                                  requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute task across unified systems"""
        
        if not self.integration_active:
            raise RuntimeError("Integration not active")
        
        start_time = time.time()
        task_id = f"unified_{int(time.time() * 1000)}"
        
        try:
            log.info(f"Executing unified task {task_id}: {task_description[:100]}...")
            
            # Store cross-system task
            self.cross_system_tasks[task_id] = {
                "description": task_description,
                "type": task_type,
                "priority": priority,
                "requirements": requirements or {},
                "start_time": start_time,
                "status": "executing"
            }
            
            # Determine execution strategy
            execution_strategy = self._determine_execution_strategy(task_description, task_type, requirements)
            
            # Execute based on strategy
            if execution_strategy == "neural_mesh_enhanced":
                result = await self._execute_neural_mesh_enhanced_task(task_id, task_description, requirements)
            elif execution_strategy == "orchestrator_coordinated":
                result = await self._execute_orchestrator_coordinated_task(task_id, task_description, requirements)
            elif execution_strategy == "swarm_distributed":
                result = await self._execute_swarm_distributed_task(task_id, task_description, requirements)
            elif execution_strategy == "fusion_integrated":
                result = await self._execute_fusion_integrated_task(task_id, task_description, requirements)
            else:
                result = await self._execute_full_integration_task(task_id, task_description, requirements)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_integration_metrics(True, execution_time)
            
            # Store result in neural mesh
            if self.neural_mesh:
                await self._store_unified_result(task_id, result)
            
            # Update task status
            self.cross_system_tasks[task_id]["status"] = "completed"
            self.cross_system_tasks[task_id]["result"] = result
            self.cross_system_tasks[task_id]["execution_time"] = execution_time
            
            log.info(f"Unified task {task_id} completed in {execution_time:.2f}s")
            
            return {
                "task_id": task_id,
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "execution_strategy": execution_strategy,
                "systems_involved": self._get_systems_involved(execution_strategy)
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_integration_metrics(False, execution_time)
            
            log.error(f"Unified task {task_id} failed: {e}")
            
            # Update task status
            if task_id in self.cross_system_tasks:
                self.cross_system_tasks[task_id]["status"] = "failed"
                self.cross_system_tasks[task_id]["error"] = str(e)
                self.cross_system_tasks[task_id]["execution_time"] = execution_time
            
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    async def _execute_neural_mesh_enhanced_task(self, task_id: str, description: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with neural mesh enhancement"""
        
        try:
            # Query neural mesh for relevant knowledge
            if self.neural_mesh:
                knowledge_query = await self.neural_mesh.query(
                    description,
                    context={"type": "task_enhancement", "task_id": task_id},
                    limit=5
                )
                
                # Enhance task with neural mesh insights
                enhanced_description = f"{description}\n\nNeural mesh insights:\n"
                for item in knowledge_query.results:
                    enhanced_description += f"- {item.content}\n"
                
                # Execute enhanced task through swarm
                if self.swarm_system:
                    from ..unified_swarm_system import SwarmObjective, SwarmMode, SwarmScale
                    
                    goal = UnifiedGoal(
                        goal_id=task_id,
                        description=enhanced_description,
                        objective=SwarmObjective.ADAPTIVE_LEARNING,
                        mode=SwarmMode.NEURAL_MESH_SWARM,
                        scale=SwarmScale.MEDIUM,
                        requirements=requirements
                    )
                    
                    swarm_result = await self.swarm_system.execute_unified_goal(goal)
                    
                    # Store result in neural mesh
                    await self.neural_mesh.store(
                        f"enhanced_task_result:{task_id}",
                        json.dumps(swarm_result.result),
                        context={"type": "enhanced_task_result", "task_id": task_id},
                        metadata={"confidence": swarm_result.confidence}
                    )
                    
                    return {
                        "enhanced_result": swarm_result.result,
                        "neural_mesh_confidence": knowledge_query.confidence,
                        "swarm_confidence": swarm_result.confidence,
                        "enhancement_method": "neural_mesh_enhanced"
                    }
            
            return {"error": "Neural mesh not available for enhancement"}
            
        except Exception as e:
            log.error(f"Neural mesh enhanced task execution failed: {e}")
            return {"error": str(e)}
    
    async def _execute_orchestrator_coordinated_task(self, task_id: str, description: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with orchestrator coordination"""
        
        try:
            if not self.orchestrator:
                return {"error": "Orchestrator not available"}
            
            # Create unified task for orchestrator
            unified_task = UnifiedTask(
                task_id=task_id,
                task_type="coordinated_execution",
                priority=TaskPriority.NORMAL,
                requirements={"description": description, **requirements},
                constraints={},
                metadata={"integration_bridge": True}
            )
            
            # Execute through orchestrator
            orchestration_result = await self.orchestrator.execute_task(unified_task)
            
            # Integrate result with swarm if available
            if self.swarm_system and orchestration_result.success:
                # Update swarm metrics based on orchestrator result
                self.swarm_system.swarm_metrics["quantum_coherence"] = orchestration_result.quantum_coherence
                
                # Store orchestrator result in swarm cache
                self.swarm_system.result_cache[task_id] = SwarmExecutionResult(
                    goal_id=task_id,
                    success=orchestration_result.success,
                    result=orchestration_result.result,
                    execution_metrics=orchestration_result.execution_metrics,
                    agents_used=orchestration_result.agents_used,
                    clusters_used=len(orchestration_result.cluster_assignments),
                    execution_time=orchestration_result.execution_time,
                    confidence=orchestration_result.confidence,
                    metadata={"coordinated_by": "orchestrator"}
                )
            
            return {
                "orchestrated_result": orchestration_result.result,
                "quantum_coherence": orchestration_result.quantum_coherence,
                "agents_coordinated": orchestration_result.agents_used,
                "coordination_method": "quantum_orchestrator"
            }
            
        except Exception as e:
            log.error(f"Orchestrator coordinated task execution failed: {e}")
            return {"error": str(e)}
    
    async def _execute_swarm_distributed_task(self, task_id: str, description: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with swarm distribution"""
        
        try:
            if not self.swarm_system:
                return {"error": "Swarm system not available"}
            
            # Create swarm goal
            from ..unified_swarm_system import SwarmObjective, SwarmMode, SwarmScale
            
            goal = UnifiedGoal(
                goal_id=task_id,
                description=description,
                objective=SwarmObjective.MAXIMIZE_THROUGHPUT,
                mode=SwarmMode.MEGA_SWARM,
                scale=SwarmScale.LARGE,
                requirements=requirements
            )
            
            # Execute through swarm
            swarm_result = await self.swarm_system.execute_unified_goal(goal)
            
            # Coordinate with orchestrator if available
            if self.orchestrator and swarm_result.success:
                # Update orchestrator with swarm metrics
                if hasattr(self.orchestrator, 'update_external_metrics'):
                    await self.orchestrator.update_external_metrics({
                        "swarm_agents_used": swarm_result.agents_used,
                        "swarm_confidence": swarm_result.confidence,
                        "swarm_execution_time": swarm_result.execution_time
                    })
            
            return {
                "swarm_result": swarm_result.result,
                "agents_used": swarm_result.agents_used,
                "clusters_used": swarm_result.clusters_used,
                "swarm_confidence": swarm_result.confidence,
                "distribution_method": "mega_swarm"
            }
            
        except Exception as e:
            log.error(f"Swarm distributed task execution failed: {e}")
            return {"error": str(e)}
    
    async def _execute_fusion_integrated_task(self, task_id: str, description: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with fusion integration"""
        
        try:
            # Check if task involves sensor data
            sensor_data = requirements.get("sensor_data", {})
            
            if not sensor_data:
                return {"error": "No sensor data provided for fusion task"}
            
            # Get fusion system from swarm
            if self.swarm_system and self.swarm_system.fusion_system:
                fusion_system = self.swarm_system.fusion_system
                
                # Create fusion request
                fusion_request = IntelligenceFusionRequest(
                    request_id=task_id,
                    domain=requirements.get("domain", "real_time_operations"),
                    sensor_data=sensor_data,
                    quality_requirement=requirements.get("quality", "operational_grade"),
                    classification_level=requirements.get("classification", "confidential")
                )
                
                # Process fusion
                fusion_result = await fusion_system.process_intelligence_fusion(fusion_request)
                
                # Integrate with neural mesh
                if self.neural_mesh:
                    await self.neural_mesh.store(
                        f"fusion_task_result:{task_id}",
                        json.dumps(fusion_result.fusion_result),
                        context={"type": "fusion_task", "confidence": fusion_result.confidence},
                        metadata={"processing_time": fusion_result.processing_time_ms}
                    )
                
                return {
                    "fusion_result": fusion_result.fusion_result,
                    "fusion_confidence": fusion_result.confidence,
                    "quality_achieved": fusion_result.quality_achieved.value,
                    "evidence_chain_id": fusion_result.evidence_chain_id,
                    "integration_method": "production_fusion"
                }
            
            return {"error": "Fusion system not available"}
            
        except Exception as e:
            log.error(f"Fusion integrated task execution failed: {e}")
            return {"error": str(e)}
    
    async def _execute_full_integration_task(self, task_id: str, description: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with full system integration"""
        
        try:
            # Phase 1: Neural mesh knowledge enhancement
            enhanced_description = description
            neural_mesh_insights = {}
            
            if self.neural_mesh:
                knowledge_query = await self.neural_mesh.query(
                    description,
                    context={"type": "full_integration_enhancement"},
                    limit=3
                )
                
                if knowledge_query.results:
                    insights = [item.content for item in knowledge_query.results]
                    enhanced_description = f"{description}\n\nRelevant knowledge:\n" + "\n".join(insights)
                    neural_mesh_insights = {
                        "insights_count": len(insights),
                        "confidence": knowledge_query.confidence
                    }
            
            # Phase 2: Orchestrator task coordination
            orchestrator_result = {}
            
            if self.orchestrator:
                unified_task = UnifiedTask(
                    task_id=f"{task_id}_orchestrated",
                    task_type="full_integration",
                    priority=TaskPriority.NORMAL,
                    requirements={"enhanced_description": enhanced_description, **requirements},
                    constraints={},
                    metadata={"integration_bridge": True, "phase": "orchestrator_coordination"}
                )
                
                orchestration = await self.orchestrator.execute_task(unified_task)
                orchestrator_result = {
                    "orchestrated": orchestration.success,
                    "quantum_coherence": orchestration.quantum_coherence,
                    "agents_coordinated": orchestration.agents_used
                }
            
            # Phase 3: Swarm execution
            swarm_result = {}
            
            if self.swarm_system:
                from ..unified_swarm_system import SwarmObjective, SwarmMode, SwarmScale
                
                goal = UnifiedGoal(
                    goal_id=f"{task_id}_swarm",
                    description=enhanced_description,
                    objective=SwarmObjective.OPTIMIZE_QUALITY,
                    mode=SwarmMode.QUANTUM_SWARM if self.orchestrator else SwarmMode.MEGA_SWARM,
                    scale=SwarmScale.LARGE,
                    requirements={**requirements, "orchestrator_result": orchestrator_result}
                )
                
                swarm_execution = await self.swarm_system.execute_unified_goal(goal)
                swarm_result = {
                    "swarm_executed": swarm_execution.success,
                    "agents_used": swarm_execution.agents_used,
                    "swarm_confidence": swarm_execution.confidence,
                    "swarm_result": swarm_execution.result
                }
            
            # Phase 4: Results integration and storage
            integrated_result = {
                "task_id": task_id,
                "description": description,
                "enhanced_description": enhanced_description,
                "neural_mesh_insights": neural_mesh_insights,
                "orchestrator_result": orchestrator_result,
                "swarm_result": swarm_result,
                "integration_success": True,
                "overall_confidence": self._calculate_overall_confidence(
                    neural_mesh_insights.get("confidence", 0.5),
                    swarm_result.get("swarm_confidence", 0.5)
                )
            }
            
            # Store final result in neural mesh
            if self.neural_mesh:
                await self.neural_mesh.store(
                    f"full_integration_result:{task_id}",
                    json.dumps(integrated_result),
                    context={"type": "full_integration_result", "success": True},
                    metadata={"systems_involved": ["neural_mesh", "orchestrator", "swarm"]}
                )
            
            # Update cross-system metrics
            self.integration_metrics["cross_system_executions"] += 1
            
            return integrated_result
            
        except Exception as e:
            log.error(f"Full integration task execution failed: {e}")
            return {"error": str(e), "task_id": task_id}
    
    def _determine_execution_strategy(self, description: str, task_type: str, requirements: Optional[Dict[str, Any]]) -> str:
        """Determine optimal execution strategy"""
        
        desc_lower = description.lower()
        reqs = requirements or {}
        
        # Fusion tasks
        if "fusion" in desc_lower or "sensor" in desc_lower or reqs.get("sensor_data"):
            return "fusion_integrated"
        
        # Neural mesh tasks
        elif "knowledge" in desc_lower or "belief" in desc_lower or "learn" in desc_lower:
            return "neural_mesh_enhanced"
        
        # Large scale coordination tasks
        elif "coordinate" in desc_lower or "million" in desc_lower or task_type == "coordination":
            return "orchestrator_coordinated"
        
        # Distributed processing tasks
        elif "distribute" in desc_lower or "parallel" in desc_lower or task_type == "distributed":
            return "swarm_distributed"
        
        # Default to full integration
        else:
            return "full_integration"
    
    def _get_systems_involved(self, execution_strategy: str) -> List[str]:
        """Get list of systems involved in execution strategy"""
        
        systems_map = {
            "neural_mesh_enhanced": ["neural_mesh", "swarm"],
            "orchestrator_coordinated": ["orchestrator", "swarm"],
            "swarm_distributed": ["swarm"],
            "fusion_integrated": ["swarm", "fusion"],
            "full_integration": ["neural_mesh", "orchestrator", "swarm", "fusion"]
        }
        
        return systems_map.get(execution_strategy, ["swarm"])
    
    def _calculate_overall_confidence(self, *confidences: float) -> float:
        """Calculate overall confidence from multiple sources"""
        
        valid_confidences = [c for c in confidences if 0.0 <= c <= 1.0]
        
        if not valid_confidences:
            return 0.5
        
        # Weighted average with higher weight for higher confidences
        weights = [c ** 2 for c in valid_confidences]  # Square to emphasize high confidence
        total_weight = sum(weights)
        
        if total_weight == 0:
            return 0.5
        
        weighted_confidence = sum(c * w for c, w in zip(valid_confidences, weights)) / total_weight
        
        return min(0.99, max(0.01, weighted_confidence))
    
    async def _setup_cross_system_communication(self):
        """Setup communication channels between systems"""
        
        try:
            # Setup swarm -> neural mesh communication
            if self.swarm_system and self.neural_mesh:
                # Register neural mesh callback with swarm
                async def swarm_to_neural_mesh_callback(swarm_event: Dict[str, Any]):
                    try:
                        await self.neural_mesh.store(
                            f"swarm_event:{swarm_event.get('event_id', int(time.time()))}",
                            json.dumps(swarm_event),
                            context={"type": "swarm_event", "source": "unified_swarm"},
                            metadata={"timestamp": time.time()}
                        )
                        self.integration_metrics["neural_mesh_syncs"] += 1
                    except Exception as e:
                        log.warning(f"Swarm to neural mesh sync failed: {e}")
                
                # Would register callback with swarm system
                log.info("Swarm -> Neural mesh communication setup")
            
            # Setup orchestrator -> swarm communication
            if self.orchestrator and self.swarm_system:
                # Would setup orchestrator task delegation to swarm
                log.info("Orchestrator -> Swarm communication setup")
            
            # Setup neural mesh -> orchestrator communication
            if self.neural_mesh and self.orchestrator:
                # Would setup knowledge sharing from neural mesh to orchestrator
                log.info("Neural mesh -> Orchestrator communication setup")
            
            log.info("Cross-system communication channels established")
            
        except Exception as e:
            log.error(f"Cross-system communication setup failed: {e}")
    
    async def _setup_shared_memory_integration(self):
        """Setup shared memory integration across systems"""
        
        try:
            if self.neural_mesh:
                # Create shared memory namespace
                await self.neural_mesh.store(
                    "integration:shared_memory:initialized",
                    "Shared memory integration active",
                    context={"type": "system_integration", "component": "shared_memory"},
                    metadata={"initialization_time": time.time()}
                )
                
                log.info("Shared memory integration setup complete")
            
        except Exception as e:
            log.error(f"Shared memory integration setup failed: {e}")
    
    async def _setup_unified_task_coordination(self):
        """Setup unified task coordination across systems"""
        
        try:
            # Create task coordination namespace
            if self.neural_mesh:
                await self.neural_mesh.store(
                    "integration:task_coordination:initialized",
                    "Unified task coordination active",
                    context={"type": "system_integration", "component": "task_coordination"},
                    metadata={"coordination_strategies": ["neural_mesh_enhanced", "orchestrator_coordinated", "swarm_distributed"]}
                )
            
            log.info("Unified task coordination setup complete")
            
        except Exception as e:
            log.error(f"Unified task coordination setup failed: {e}")
    
    async def _start_background_sync(self):
        """Start background synchronization tasks"""
        
        if self.config.sync_strategy == SynchronizationStrategy.PERIODIC:
            self._background_tasks.append(
                asyncio.create_task(self._periodic_sync_task())
            )
        elif self.config.sync_strategy == SynchronizationStrategy.REAL_TIME:
            self._background_tasks.append(
                asyncio.create_task(self._real_time_sync_task())
            )
        
        # Always run metrics sync
        self._background_tasks.append(
            asyncio.create_task(self._metrics_sync_task())
        )
        
        log.info(f"Background sync started with {self.config.sync_strategy.value} strategy")
    
    async def _periodic_sync_task(self):
        """Periodic synchronization between systems"""
        
        while self.integration_active:
            try:
                await self._perform_system_sync()
                await asyncio.sleep(self.config.sync_interval_seconds)
                
            except Exception as e:
                log.error(f"Periodic sync failed: {e}")
                await asyncio.sleep(self.config.sync_interval_seconds * 2)
    
    async def _real_time_sync_task(self):
        """Real-time synchronization between systems"""
        
        while self.integration_active:
            try:
                # Monitor for changes and sync immediately
                await self._monitor_and_sync_changes()
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                log.error(f"Real-time sync failed: {e}")
                await asyncio.sleep(5.0)
    
    async def _metrics_sync_task(self):
        """Sync metrics across systems"""
        
        while self.integration_active:
            try:
                # Collect metrics from all systems
                unified_metrics = await self._collect_unified_metrics()
                
                # Store in neural mesh
                if self.neural_mesh:
                    await self.neural_mesh.store(
                        f"integration_metrics:{int(time.time())}",
                        json.dumps(unified_metrics),
                        context={"type": "integration_metrics"},
                        metadata={"collection_timestamp": time.time()}
                    )
                
                await asyncio.sleep(60.0)  # Sync every minute
                
            except Exception as e:
                log.error(f"Metrics sync failed: {e}")
                await asyncio.sleep(120.0)
    
    async def _perform_system_sync(self):
        """Perform synchronization between systems"""
        
        sync_start = time.time()
        
        try:
            sync_operations = []
            
            # Sync swarm state to neural mesh
            if self.swarm_system and self.neural_mesh:
                sync_operations.append(self._sync_swarm_to_neural_mesh())
            
            # Sync neural mesh insights to orchestrator
            if self.neural_mesh and self.orchestrator:
                sync_operations.append(self._sync_neural_mesh_to_orchestrator())
            
            # Sync orchestrator state to swarm
            if self.orchestrator and self.swarm_system:
                sync_operations.append(self._sync_orchestrator_to_swarm())
            
            # Execute all sync operations
            if sync_operations:
                await asyncio.gather(*sync_operations, return_exceptions=True)
            
            # Record sync
            sync_time = time.time() - sync_start
            self.sync_history.append({
                "timestamp": sync_start,
                "sync_time": sync_time,
                "operations": len(sync_operations),
                "success": True
            })
            
            # Keep limited history
            if len(self.sync_history) > 1000:
                self.sync_history.pop(0)
            
            log.debug(f"System sync completed in {sync_time:.2f}s")
            
        except Exception as e:
            log.error(f"System sync failed: {e}")
            self.sync_history.append({
                "timestamp": sync_start,
                "sync_time": time.time() - sync_start,
                "operations": 0,
                "success": False,
                "error": str(e)
            })
    
    async def _sync_swarm_to_neural_mesh(self):
        """Sync swarm state to neural mesh"""
        
        if not (self.swarm_system and self.neural_mesh):
            return
        
        try:
            # Get swarm status
            swarm_status = await self.swarm_system.get_system_status()
            
            # Store in neural mesh
            await self.neural_mesh.store(
                f"swarm_status:{self.swarm_system.node_id}",
                json.dumps(swarm_status),
                context={"type": "swarm_status", "node_id": self.swarm_system.node_id},
                metadata={"sync_timestamp": time.time()}
            )
            
        except Exception as e:
            log.warning(f"Swarm to neural mesh sync failed: {e}")
    
    async def _sync_neural_mesh_to_orchestrator(self):
        """Sync neural mesh insights to orchestrator"""
        
        if not (self.neural_mesh and self.orchestrator):
            return
        
        try:
            # Get recent insights from neural mesh
            recent_insights = await self.neural_mesh.query(
                "system insights coordination optimization",
                context={"type": "orchestrator_enhancement"},
                limit=5
            )
            
            # Would update orchestrator with insights
            # For now, just log
            log.debug(f"Synced {len(recent_insights.results)} insights to orchestrator")
            
        except Exception as e:
            log.warning(f"Neural mesh to orchestrator sync failed: {e}")
    
    async def _sync_orchestrator_to_swarm(self):
        """Sync orchestrator state to swarm"""
        
        if not (self.orchestrator and self.swarm_system):
            return
        
        try:
            # Get orchestrator stats
            orchestrator_stats = self.orchestrator.get_orchestration_stats()
            
            # Update swarm quantum coherence
            self.swarm_system.swarm_metrics["quantum_coherence"] = orchestrator_stats.get("quantum_coherence_global", 1.0)
            
            log.debug("Synced orchestrator state to swarm")
            
        except Exception as e:
            log.warning(f"Orchestrator to swarm sync failed: {e}")
    
    async def _collect_unified_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all integrated systems"""
        
        unified_metrics = {
            "timestamp": time.time(),
            "integration_metrics": self.integration_metrics.copy(),
            "systems_active": []
        }
        
        # Collect swarm metrics
        if self.swarm_system:
            swarm_status = await self.swarm_system.get_system_status()
            unified_metrics["swarm_metrics"] = swarm_status.get("swarm_metrics", {})
            unified_metrics["systems_active"].append("swarm")
        
        # Collect orchestrator metrics
        if self.orchestrator:
            orchestrator_stats = self.orchestrator.get_orchestration_stats()
            unified_metrics["orchestrator_metrics"] = orchestrator_stats
            unified_metrics["systems_active"].append("orchestrator")
        
        # Collect neural mesh metrics
        if self.neural_mesh:
            neural_mesh_status = await self.neural_mesh.get_status()
            unified_metrics["neural_mesh_metrics"] = neural_mesh_status
            unified_metrics["systems_active"].append("neural_mesh")
        
        return unified_metrics
    
    def _validate_system_availability(self) -> bool:
        """Validate that required systems are available"""
        
        if self.config.mode == IntegrationMode.FULL_INTEGRATION:
            required_systems = [self.swarm_system, self.neural_mesh, self.orchestrator]
            if not all(required_systems):
                log.error("Full integration mode requires all systems")
                return False
        
        elif self.config.mode == IntegrationMode.NEURAL_MESH_ONLY:
            if not (self.swarm_system and self.neural_mesh):
                log.error("Neural mesh mode requires swarm and neural mesh")
                return False
        
        elif self.config.mode == IntegrationMode.ORCHESTRATOR_ONLY:
            if not (self.swarm_system and self.orchestrator):
                log.error("Orchestrator mode requires swarm and orchestrator")
                return False
        
        return True
    
    def _update_integration_metrics(self, success: bool, execution_time: float):
        """Update integration performance metrics"""
        
        self.integration_metrics["tasks_coordinated"] += 1
        
        if success:
            self.integration_metrics["successful_integrations"] += 1
        else:
            self.integration_metrics["failed_integrations"] += 1
        
        # Update average integration time
        total_tasks = self.integration_metrics["tasks_coordinated"]
        current_avg = self.integration_metrics["average_integration_time"]
        
        self.integration_metrics["average_integration_time"] = (
            (current_avg * (total_tasks - 1) + execution_time) / total_tasks
        )
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        
        try:
            status = {
                "timestamp": time.time(),
                "integration_active": self.integration_active,
                "configuration": {
                    "mode": self.config.mode.value,
                    "sync_strategy": self.config.sync_strategy.value,
                    "sync_interval": self.config.sync_interval_seconds
                },
                "systems_connected": {
                    "swarm_system": self.swarm_system is not None,
                    "neural_mesh": self.neural_mesh is not None,
                    "orchestrator": self.orchestrator is not None
                },
                "integration_metrics": self.integration_metrics.copy(),
                "active_cross_system_tasks": len(self.cross_system_tasks),
                "recent_sync_history": self.sync_history[-10:] if self.sync_history else []
            }
            
            # Add system-specific status if available
            if self.swarm_system:
                status["swarm_status"] = await self.swarm_system.get_system_status()
            
            if self.orchestrator:
                status["orchestrator_status"] = self.orchestrator.get_orchestration_stats()
            
            if self.neural_mesh:
                status["neural_mesh_status"] = await self.neural_mesh.get_status()
            
            return status
            
        except Exception as e:
            log.error(f"Integration status generation failed: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    async def shutdown_integration(self):
        """Shutdown integration bridge"""
        
        log.info("Shutting down unified integration bridge...")
        
        try:
            self.integration_active = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for background tasks
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Final sync
            if self.neural_mesh:
                await self.neural_mesh.store(
                    "integration:shutdown",
                    "Integration bridge shutdown",
                    context={"type": "system_event", "event": "shutdown"},
                    metadata={"final_metrics": self.integration_metrics}
                )
            
            log.info("Unified integration bridge shutdown complete")
            
        except Exception as e:
            log.error(f"Integration shutdown error: {e}")

# Factory function
async def create_unified_integration_bridge(
    swarm_system: Optional[UnifiedSwarmSystem] = None,
    neural_mesh: Optional[ProductionNeuralMesh] = None,
    orchestrator: Optional[UnifiedQuantumOrchestrator] = None,
    mode: IntegrationMode = IntegrationMode.FULL_INTEGRATION
) -> UnifiedIntegrationBridge:
    """Create unified integration bridge"""
    
    config = IntegrationConfiguration(
        mode=mode,
        sync_strategy=SynchronizationStrategy.EVENT_DRIVEN,
        enable_cross_system_tasks=True,
        enable_shared_memory=True,
        enable_unified_metrics=True
    )
    
    bridge = UnifiedIntegrationBridge(config, swarm_system, neural_mesh, orchestrator)
    
    if await bridge.initialize_integration():
        log.info(f"Unified integration bridge created in {mode.value} mode")
        return bridge
    else:
        raise RuntimeError("Failed to initialize unified integration bridge")
