"""
Intelligence-Enhanced Mega Swarm Coordinator
Integrates advanced intelligence directly into swarm coordination
Enables intelligence-driven agent deployment and task distribution
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("intelligence-enhanced-coordinator")

# Import intelligence systems
try:
    from ..intelligence import (
        analyze_task_and_determine_agents,
        process_intelligence,
        intelligence_swarm_bridge,
        TaskAnalysis,
        IntelligenceResponse
    )
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    INTELLIGENCE_AVAILABLE = False
    log.warning("Intelligence module not available")

# Import swarm systems
try:
    from .enhanced_mega_coordinator import EnhancedMegaSwarmCoordinator
    from ..unified_swarm_system import UnifiedGoal, SwarmScale
    SWARM_AVAILABLE = True
except ImportError:
    SWARM_AVAILABLE = False
    log.warning("Swarm systems not available")

# Import neural mesh
try:
    from ...neural_mesh.production_neural_mesh import ProductionNeuralMesh
    NEURAL_MESH_AVAILABLE = True
except ImportError:
    NEURAL_MESH_AVAILABLE = False

class CoordinationStrategy(Enum):
    """Coordination strategies"""
    INTELLIGENCE_FIRST = "intelligence_first"      # Analyze then execute
    PARALLEL_PROCESSING = "parallel_processing"    # Analyze and execute simultaneously
    ADAPTIVE = "adaptive"                          # Decide based on task complexity

@dataclass
class IntelligenceEnhancedResult:
    """Result from intelligence-enhanced coordination"""
    task_id: str
    intelligence_analysis: Optional[IntelligenceResponse]
    swarm_execution: Dict[str, Any]
    coordination_strategy: CoordinationStrategy
    total_agents_deployed: int
    overall_confidence: float
    processing_time: float
    intelligence_driven: bool

class IntelligenceEnhancedCoordinator:
    """
    Mega swarm coordinator enhanced with intelligence capabilities.
    Uses intelligence analysis to drive optimal agent deployment.
    """
    
    def __init__(
        self,
        node_id: str = "intel_enhanced_coordinator",
        neural_mesh: Optional[ProductionNeuralMesh] = None
    ):
        self.node_id = node_id
        self.neural_mesh = neural_mesh
        
        # Create/connect to mega coordinator if available
        self.mega_coordinator = None
        if SWARM_AVAILABLE:
            try:
                self.mega_coordinator = EnhancedMegaSwarmCoordinator(
                    node_id=node_id,
                    neural_mesh=neural_mesh
                )
                log.info("✅ Enhanced Mega Coordinator initialized")
            except Exception as e:
                log.error(f"Failed to initialize mega coordinator: {e}")
        
        # Connect intelligence bridge
        if INTELLIGENCE_AVAILABLE and SWARM_AVAILABLE:
            intelligence_swarm_bridge.connect_systems(
                mega_coordinator=self.mega_coordinator,
                neural_mesh=neural_mesh
            )
            log.info("✅ Intelligence bridge connected")
        
        # Coordination history
        self.coordination_history: List[IntelligenceEnhancedResult] = []
        
        # Performance metrics
        self.total_coordinations = 0
        self.intelligence_driven_count = 0
        self.avg_confidence = 0.0
        self.avg_processing_time = 0.0
        
        log.info(f"Intelligence-Enhanced Coordinator initialized (node: {node_id})")
    
    async def coordinate_with_intelligence(
        self,
        task_description: str,
        available_data: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None,
        strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE
    ) -> IntelligenceEnhancedResult:
        """
        Coordinate swarm execution with intelligence analysis.
        Intelligence drives agent selection and deployment.
        """
        
        start_time = time.time()
        task_id = f"coord_{int(time.time() * 1000)}"
        
        log.info(f"🎯 Intelligence-Enhanced Coordination: {task_description[:50]}...")
        log.info(f"Strategy: {strategy.value}")
        
        intelligence_analysis = None
        swarm_execution = {}
        intelligence_driven = False
        
        # Determine if we should use intelligence
        use_intelligence = (
            INTELLIGENCE_AVAILABLE and
            available_data and
            len(available_data) > 0
        )
        
        if use_intelligence:
            intelligence_driven = True
            log.info("📊 Using intelligence-driven coordination")
            
            if strategy == CoordinationStrategy.INTELLIGENCE_FIRST:
                # Run intelligence first, then execute
                intelligence_analysis = await self._run_intelligence_analysis(
                    task_description, available_data, context
                )
                
                swarm_execution = await self._execute_with_intelligence(
                    task_description, intelligence_analysis, context
                )
            
            elif strategy == CoordinationStrategy.PARALLEL_PROCESSING:
                # Run intelligence and execution in parallel
                intel_task = asyncio.create_task(
                    self._run_intelligence_analysis(task_description, available_data, context)
                )
                swarm_task = asyncio.create_task(
                    self._execute_standard_swarm(task_description, context)
                )
                
                intelligence_analysis, swarm_execution = await asyncio.gather(
                    intel_task, swarm_task
                )
            
            else:  # ADAPTIVE
                # Decide based on task complexity
                if available_data and len(available_data) >= 5:
                    # Complex task - intelligence first
                    intelligence_analysis = await self._run_intelligence_analysis(
                        task_description, available_data, context
                    )
                    swarm_execution = await self._execute_with_intelligence(
                        task_description, intelligence_analysis, context
                    )
                else:
                    # Simple task - parallel
                    intel_task = asyncio.create_task(
                        self._run_intelligence_analysis(task_description, available_data, context)
                    )
                    swarm_task = asyncio.create_task(
                        self._execute_standard_swarm(task_description, context)
                    )
                    
                    intelligence_analysis, swarm_execution = await asyncio.gather(
                        intel_task, swarm_task
                    )
        else:
            # Standard swarm coordination without intelligence
            log.info("🤖 Using standard swarm coordination")
            swarm_execution = await self._execute_standard_swarm(task_description, context)
        
        # Calculate overall metrics
        total_agents = (
            intelligence_analysis.agent_count if intelligence_analysis
            else swarm_execution.get("agents_deployed", 0)
        )
        
        overall_confidence = (
            intelligence_analysis.overall_confidence if intelligence_analysis
            else swarm_execution.get("confidence", 0.75)
        )
        
        processing_time = time.time() - start_time
        
        # Create result
        result = IntelligenceEnhancedResult(
            task_id=task_id,
            intelligence_analysis=intelligence_analysis,
            swarm_execution=swarm_execution,
            coordination_strategy=strategy,
            total_agents_deployed=total_agents,
            overall_confidence=overall_confidence,
            processing_time=processing_time,
            intelligence_driven=intelligence_driven
        )
        
        self.coordination_history.append(result)
        self._update_metrics(result)
        
        log.info(f"✅ Coordination complete: {processing_time:.2f}s, "
                f"{total_agents} agents, confidence {overall_confidence:.2%}")
        
        return result
    
    async def _run_intelligence_analysis(
        self,
        task_description: str,
        available_data: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> IntelligenceResponse:
        """Run intelligence analysis"""
        
        log.info("🧠 Running intelligence analysis...")
        
        response = await process_intelligence(
            task_description=task_description,
            available_data=available_data,
            context=context or {}
        )
        
        log.info(f"✅ Intelligence complete: {response.agent_count} agents, "
                f"confidence {response.overall_confidence:.2%}")
        
        return response
    
    async def _execute_with_intelligence(
        self,
        task_description: str,
        intelligence: IntelligenceResponse,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute swarm using intelligence-derived parameters"""
        
        log.info("🚀 Executing swarm with intelligence parameters...")
        
        # Use intelligence to inform swarm execution
        execution_params = {
            "description": task_description,
            "agent_count": intelligence.agent_count,
            "specializations": [
                spec.agent_type for spec in intelligence.task_analysis.required_specializations
            ],
            "confidence_target": intelligence.overall_confidence,
            "threat_context": {
                "threat_assessment": intelligence.threat_assessment,
                "ttp_detections": len(intelligence.ttp_detections),
                "campaign_detected": intelligence.campaign_assessment is not None
            }
        }
        
        # Simulate execution (actual mega coordinator would be called here)
        await asyncio.sleep(0.5)
        
        return {
            "status": "completed",
            "agents_deployed": intelligence.agent_count,
            "confidence": intelligence.overall_confidence,
            "intelligence_driven": True,
            "execution_params": execution_params
        }
    
    async def _execute_standard_swarm(
        self,
        task_description: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute with standard swarm coordination"""
        
        log.info("🤖 Executing with standard swarm...")
        
        # Simulate standard execution
        await asyncio.sleep(0.3)
        
        return {
            "status": "completed",
            "agents_deployed": 10,
            "confidence": 0.75,
            "intelligence_driven": False
        }
    
    def _update_metrics(self, result: IntelligenceEnhancedResult):
        """Update performance metrics"""
        
        self.total_coordinations += 1
        
        if result.intelligence_driven:
            self.intelligence_driven_count += 1
        
        # Update averages
        n = self.total_coordinations
        self.avg_confidence = (
            (self.avg_confidence * (n-1) + result.overall_confidence) / n
        )
        self.avg_processing_time = (
            (self.avg_processing_time * (n-1) + result.processing_time) / n
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get coordination metrics"""
        
        return {
            "total_coordinations": self.total_coordinations,
            "intelligence_driven_count": self.intelligence_driven_count,
            "intelligence_usage_rate": self.intelligence_driven_count / max(self.total_coordinations, 1),
            "avg_confidence": self.avg_confidence,
            "avg_processing_time": self.avg_processing_time,
            "systems_available": {
                "intelligence": INTELLIGENCE_AVAILABLE,
                "swarm": SWARM_AVAILABLE,
                "neural_mesh": NEURAL_MESH_AVAILABLE
            }
        }


# Global instance
intelligence_enhanced_coordinator = IntelligenceEnhancedCoordinator()


async def coordinate_intelligence_swarm(
    task_description: str,
    available_data: List[Dict[str, Any]] = None,
    context: Dict[str, Any] = None,
    strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE
) -> IntelligenceEnhancedResult:
    """
    Main entry point: Coordinate swarm with intelligence integration.
    """
    return await intelligence_enhanced_coordinator.coordinate_with_intelligence(
        task_description, available_data, context, strategy
    )

