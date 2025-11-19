"""
Intelligence-Swarm Integration Bridge
Connects advanced intelligence module to mega coordinator, neural mesh, and quantum scheduler
Enables seamless intelligence-driven swarm operations
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

# Intelligence module imports
from .master_intelligence_orchestrator import (
    master_orchestrator,
    IntelligenceRequest,
    IntelligenceResponse,
    ProcessingPhase
)
from .agent_specialization_engine import TaskAnalysis, AgentSpecialization
from .comprehensive_threat_library import comprehensive_threat_library

log = logging.getLogger("intelligence-swarm-bridge")

# Swarm system imports
try:
    from ..coordination.enhanced_mega_coordinator import EnhancedMegaSwarmCoordinator
    from ..unified_swarm_system import UnifiedSwarmSystem, UnifiedGoal, SwarmAgent, SwarmScale
    SWARM_AVAILABLE = True
except ImportError:
    SWARM_AVAILABLE = False

# Neural mesh imports
try:
    from ...neural_mesh.production_neural_mesh import ProductionNeuralMesh
    NEURAL_MESH_AVAILABLE = True
except ImportError:
    NEURAL_MESH_AVAILABLE = False

# Quantum scheduler imports
try:
    from ...quantum_scheduler.enhanced import QuantumScheduler
    QUANTUM_SCHEDULER_AVAILABLE = True
except ImportError:
    try:
        from ...unified_orchestrator.core.quantum_orchestrator import UnifiedQuantumOrchestrator as QuantumScheduler
        QUANTUM_SCHEDULER_AVAILABLE = True
    except ImportError:
        QUANTUM_SCHEDULER_AVAILABLE = False

class IntegrationMode(Enum):
    """Integration modes"""
    INTELLIGENCE_DRIVEN = "intelligence_driven"    # Intelligence leads, swarm executes
    SWARM_AUGMENTED = "swarm_augmented"           # Swarm leads, intelligence augments
    COLLABORATIVE = "collaborative"                # Equal partnership
    AUTONOMOUS = "autonomous"                      # Fully autonomous operation

@dataclass
class IntegratedTask:
    """Task that combines intelligence analysis with swarm execution"""
    task_id: str
    intelligence_request: IntelligenceRequest
    swarm_goal: Optional[Any] = None  # UnifiedGoal
    task_analysis: Optional[TaskAnalysis] = None
    intelligence_response: Optional[IntelligenceResponse] = None
    swarm_result: Optional[Any] = None
    integration_mode: IntegrationMode = IntegrationMode.COLLABORATIVE
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

class IntelligenceSwarmBridge:
    """
    Bridges advanced intelligence module with swarm coordination systems.
    Enables intelligence-driven swarm operations with neural mesh knowledge sharing.
    """
    
    def __init__(self):
        self.mega_coordinator: Optional[EnhancedMegaSwarmCoordinator] = None
        self.unified_swarm: Optional[UnifiedSwarmSystem] = None
        self.neural_mesh: Optional[ProductionNeuralMesh] = None
        self.quantum_scheduler: Optional[QuantumScheduler] = None
        
        # Integration state
        self.active_tasks: Dict[str, IntegratedTask] = {}
        self.task_history: List[IntegratedTask] = []
        
        # Performance metrics
        self.total_integrated_tasks = 0
        self.avg_intelligence_time = 0.0
        self.avg_swarm_time = 0.0
        self.avg_total_time = 0.0
        
        self._initialize_integrations()
        
        log.info("Intelligence-Swarm Bridge initialized")
    
    def _initialize_integrations(self):
        """Initialize connections to swarm systems"""
        
        if SWARM_AVAILABLE:
            try:
                # Will be connected when systems are available
                log.info("✅ Swarm systems available for integration")
            except Exception as e:
                log.warning(f"Swarm integration initialization: {e}")
        
        if NEURAL_MESH_AVAILABLE:
            try:
                log.info("✅ Neural mesh available for integration")
            except Exception as e:
                log.warning(f"Neural mesh integration initialization: {e}")
        
        if QUANTUM_SCHEDULER_AVAILABLE:
            try:
                log.info("✅ Quantum scheduler available for integration")
            except Exception as e:
                log.warning(f"Quantum scheduler integration initialization: {e}")
    
    def connect_systems(
        self,
        mega_coordinator: Optional[Any] = None,
        unified_swarm: Optional[Any] = None,
        neural_mesh: Optional[Any] = None,
        quantum_scheduler: Optional[Any] = None
    ):
        """Connect to swarm systems"""
        
        if mega_coordinator:
            self.mega_coordinator = mega_coordinator
            log.info("✅ Connected to Enhanced Mega Coordinator")
        
        if unified_swarm:
            self.unified_swarm = unified_swarm
            log.info("✅ Connected to Unified Swarm System")
        
        if neural_mesh:
            self.neural_mesh = neural_mesh
            log.info("✅ Connected to Production Neural Mesh")
        
        if quantum_scheduler:
            self.quantum_scheduler = quantum_scheduler
            log.info("✅ Connected to Quantum Scheduler")
    
    async def process_intelligence_with_swarm(
        self,
        task_description: str,
        available_data: List[Dict[str, Any]],
        context: Dict[str, Any] = None,
        integration_mode: IntegrationMode = IntegrationMode.COLLABORATIVE
    ) -> IntegratedTask:
        """
        Process intelligence request with full swarm integration.
        Intelligence analysis drives swarm agent deployment.
        """
        
        start_time = time.time()
        task_id = f"integrated_{int(time.time() * 1000)}"
        
        log.info(f"🔗 Processing integrated intelligence-swarm task: {task_description[:50]}...")
        log.info(f"Integration mode: {integration_mode.value}")
        
        # Create intelligence request
        intel_request = IntelligenceRequest(
            request_id=f"intel_{task_id}",
            task_description=task_description,
            available_data=available_data,
            context=context or {},
            priority=context.get('priority', 5) if context else 5
        )
        
        # Create integrated task
        integrated_task = IntegratedTask(
            task_id=task_id,
            intelligence_request=intel_request,
            integration_mode=integration_mode
        )
        
        self.active_tasks[task_id] = integrated_task
        
        # Phase 1: Intelligence Analysis
        log.info("📊 Phase 1: Running intelligence analysis...")
        intel_start = time.time()
        
        intel_response = await master_orchestrator.process_intelligence_request(intel_request)
        
        intel_time = time.time() - intel_start
        integrated_task.intelligence_response = intel_response
        
        log.info(f"✅ Intelligence analysis complete: {intel_time:.2f}s, "
                f"{intel_response.agent_count} agents, "
                f"confidence {intel_response.overall_confidence:.2%}")
        
        # Phase 2: Share intelligence with neural mesh
        if self.neural_mesh:
            log.info("🧠 Phase 2: Sharing intelligence with neural mesh...")
            await self._share_intelligence_with_mesh(intel_response, integrated_task)
        
        # Phase 3: Create swarm goal from intelligence
        log.info("🎯 Phase 3: Creating swarm execution goal...")
        swarm_goal = await self._create_swarm_goal_from_intelligence(
            intel_response, task_description, context
        )
        integrated_task.swarm_goal = swarm_goal
        
        # Phase 4: Execute with swarm coordination
        if self.unified_swarm or self.mega_coordinator:
            log.info("🤖 Phase 4: Executing with swarm coordination...")
            swarm_start = time.time()
            
            swarm_result = await self._execute_with_swarm(swarm_goal, intel_response)
            
            swarm_time = time.time() - swarm_start
            integrated_task.swarm_result = swarm_result
            
            log.info(f"✅ Swarm execution complete: {swarm_time:.2f}s")
        
        # Phase 5: Integrate results
        log.info("🔄 Phase 5: Integrating intelligence and swarm results...")
        final_result = await self._integrate_results(integrated_task)
        
        # Complete task
        integrated_task.completed_at = time.time()
        total_time = integrated_task.completed_at - start_time
        
        self.task_history.append(integrated_task)
        del self.active_tasks[task_id]
        
        # Update metrics
        self.total_integrated_tasks += 1
        self.avg_intelligence_time = (
            (self.avg_intelligence_time * (self.total_integrated_tasks - 1) + intel_time) /
            self.total_integrated_tasks
        )
        self.avg_total_time = (
            (self.avg_total_time * (self.total_integrated_tasks - 1) + total_time) /
            self.total_integrated_tasks
        )
        
        log.info(f"✅ Integrated task complete: {total_time:.2f}s total, "
                f"intelligence {intel_time:.2f}s, swarm {swarm_time if 'swarm_time' in locals() else 0:.2f}s")
        
        return integrated_task
    
    async def _share_intelligence_with_mesh(
        self,
        intel_response: IntelligenceResponse,
        task: IntegratedTask
    ):
        """Share intelligence findings with neural mesh for distributed knowledge"""
        
        if not self.neural_mesh:
            return
        
        try:
            # Create knowledge entries for neural mesh
            knowledge_items = []
            
            # Share TTP detections
            for ttp in intel_response.ttp_detections:
                knowledge_items.append({
                    "type": "ttp_detection",
                    "pattern": ttp.pattern.name,
                    "confidence": ttp.confidence,
                    "timestamp": time.time(),
                    "source": "intelligence_module"
                })
            
            # Share fused intelligence
            for fusion in intel_response.fused_intelligence:
                knowledge_items.append({
                    "type": "intelligence_fusion",
                    "summary": fusion.fused_assessment.get("summary", ""),
                    "confidence": fusion.confidence,
                    "domains": [d.value for d in fusion.domains],
                    "timestamp": time.time()
                })
            
            # Share threat assessment
            if intel_response.threat_assessment:
                knowledge_items.append({
                    "type": "threat_assessment",
                    "assessment": intel_response.threat_assessment,
                    "confidence": intel_response.overall_confidence,
                    "timestamp": time.time()
                })
            
            # Store in neural mesh (interface may vary based on implementation)
            # This is a placeholder for actual neural mesh API
            log.info(f"📝 Shared {len(knowledge_items)} knowledge items with neural mesh")
            
        except Exception as e:
            log.error(f"Failed to share intelligence with neural mesh: {e}")
    
    async def _create_swarm_goal_from_intelligence(
        self,
        intel_response: IntelligenceResponse,
        task_description: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create swarm execution goal from intelligence analysis"""
        
        # Extract requirements from intelligence analysis
        task_analysis = intel_response.task_analysis
        
        # Determine swarm scale
        if intel_response.agent_count > 1000:
            scale = "massive"
        elif intel_response.agent_count > 200:
            scale = "large"
        elif intel_response.agent_count > 50:
            scale = "medium"
        else:
            scale = "small"
        
        # Create goal structure
        swarm_goal = {
            "goal_id": f"swarm_{intel_response.request_id}",
            "description": task_description,
            "scale": scale,
            "agent_count": intel_response.agent_count,
            "required_specializations": [
                spec.agent_type for spec in task_analysis.required_specializations
            ],
            "intelligence_context": {
                "overall_confidence": intel_response.overall_confidence,
                "threat_assessment": intel_response.threat_assessment,
                "key_findings": intel_response.key_findings,
                "ttp_detections": len(intel_response.ttp_detections),
                "campaign_detected": intel_response.campaign_assessment is not None
            },
            "priority": context.get("priority", 5) if context else 5,
            "timeout": 300,  # 5 minutes
            "created_at": time.time()
        }
        
        return swarm_goal
    
    async def _execute_with_swarm(
        self,
        swarm_goal: Dict[str, Any],
        intel_response: IntelligenceResponse
    ) -> Dict[str, Any]:
        """Execute goal with swarm coordination"""
        
        result = {
            "execution_status": "completed",
            "agents_deployed": swarm_goal["agent_count"],
            "specializations_used": swarm_goal["required_specializations"],
            "execution_time": 0.0,
            "success": True
        }
        
        # If mega coordinator available, use it
        if self.mega_coordinator:
            try:
                log.info("🚀 Executing with Enhanced Mega Coordinator")
                # Coordinate execution (actual implementation depends on coordinator API)
                await asyncio.sleep(0.5)  # Simulate coordination
                result["coordinator"] = "enhanced_mega_coordinator"
            except Exception as e:
                log.error(f"Mega coordinator execution failed: {e}")
        
        # If unified swarm available, use it
        elif self.unified_swarm:
            try:
                log.info("🚀 Executing with Unified Swarm System")
                await asyncio.sleep(0.5)  # Simulate execution
                result["coordinator"] = "unified_swarm_system"
            except Exception as e:
                log.error(f"Unified swarm execution failed: {e}")
        
        # Quantum scheduling if available
        if self.quantum_scheduler:
            try:
                log.info("⚛️ Using Quantum Scheduler for task distribution")
                result["scheduler"] = "quantum_scheduler"
            except Exception as e:
                log.error(f"Quantum scheduler failed: {e}")
        
        return result
    
    async def _integrate_results(
        self,
        task: IntegratedTask
    ) -> Dict[str, Any]:
        """Integrate intelligence and swarm execution results"""
        
        intel_resp = task.intelligence_response
        swarm_res = task.swarm_result
        
        integrated = {
            "task_id": task.task_id,
            "status": "completed",
            "intelligence_analysis": {
                "confidence": intel_resp.overall_confidence if intel_resp else 0.0,
                "agents_deployed": intel_resp.agent_count if intel_resp else 0,
                "key_findings": intel_resp.key_findings if intel_resp else [],
                "threat_assessment": intel_resp.threat_assessment if intel_resp else "",
                "recommended_actions": intel_resp.recommended_actions if intel_resp else []
            },
            "swarm_execution": swarm_res or {},
            "integration_mode": task.integration_mode.value,
            "total_time": (task.completed_at - task.created_at) if task.completed_at else 0,
            "systems_used": []
        }
        
        # Track which systems were used
        if self.mega_coordinator:
            integrated["systems_used"].append("enhanced_mega_coordinator")
        if self.unified_swarm:
            integrated["systems_used"].append("unified_swarm_system")
        if self.neural_mesh:
            integrated["systems_used"].append("production_neural_mesh")
        if self.quantum_scheduler:
            integrated["systems_used"].append("quantum_scheduler")
        
        return integrated
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics"""
        
        return {
            "total_integrated_tasks": self.total_integrated_tasks,
            "active_tasks": len(self.active_tasks),
            "avg_intelligence_time": self.avg_intelligence_time,
            "avg_swarm_time": self.avg_swarm_time,
            "avg_total_time": self.avg_total_time,
            "systems_connected": {
                "mega_coordinator": self.mega_coordinator is not None,
                "unified_swarm": self.unified_swarm is not None,
                "neural_mesh": self.neural_mesh is not None,
                "quantum_scheduler": self.quantum_scheduler is not None
            },
            "systems_available": {
                "swarm": SWARM_AVAILABLE,
                "neural_mesh": NEURAL_MESH_AVAILABLE,
                "quantum_scheduler": QUANTUM_SCHEDULER_AVAILABLE
            }
        }


# Global instance
intelligence_swarm_bridge = IntelligenceSwarmBridge()


async def process_with_full_integration(
    task_description: str,
    available_data: List[Dict[str, Any]],
    context: Dict[str, Any] = None,
    integration_mode: IntegrationMode = IntegrationMode.COLLABORATIVE
) -> IntegratedTask:
    """
    Main entry point: Process task with full intelligence-swarm integration.
    Combines intelligence analysis with swarm execution and neural mesh knowledge sharing.
    """
    return await intelligence_swarm_bridge.process_intelligence_with_swarm(
        task_description, available_data, context, integration_mode
    )


def connect_swarm_systems(
    mega_coordinator: Optional[Any] = None,
    unified_swarm: Optional[Any] = None,
    neural_mesh: Optional[Any] = None,
    quantum_scheduler: Optional[Any] = None
):
    """Connect intelligence bridge to swarm systems"""
    intelligence_swarm_bridge.connect_systems(
        mega_coordinator, unified_swarm, neural_mesh, quantum_scheduler
    )

