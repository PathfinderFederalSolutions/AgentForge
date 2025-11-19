"""
Unified Swarm Service - Main Entry Point
Consolidated mega-swarm, swarm, and swarm-worker with perfect neural mesh and orchestrator integration
"""

import asyncio
import logging
import os
import sys
from typing import Optional, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import unified system components
from .unified_swarm_system import UnifiedSwarmSystem, create_unified_swarm_system
from .coordination.enhanced_mega_coordinator import EnhancedMegaSwarmCoordinator, create_enhanced_mega_swarm_coordinator
from .workers.enhanced_million_scale_worker import EnhancedMillionScaleWorker, create_enhanced_million_scale_worker
from .integration.unified_integration_bridge import UnifiedIntegrationBridge, create_unified_integration_bridge, IntegrationMode

# Import neural mesh and orchestrator
try:
    from ..neural_mesh.production_neural_mesh import ProductionNeuralMesh, create_production_neural_mesh
    NEURAL_MESH_AVAILABLE = True
except ImportError:
    NEURAL_MESH_AVAILABLE = False

try:
    from ..unified_orchestrator.core.quantum_orchestrator import UnifiedQuantumOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

log = logging.getLogger("unified-swarm-service")

class UnifiedSwarmService:
    """
    Unified Swarm Service - Complete Integration
    
    Consolidates all swarm capabilities into a single service:
    - Mega-swarm coordination (million+ agents)
    - Enhanced agent processing (multi-LLM)
    - Million-scale worker processing
    - Production fusion system
    - Perfect neural mesh integration
    - Unified orchestrator coordination
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Core systems
        self.neural_mesh: Optional[ProductionNeuralMesh] = None
        self.orchestrator: Optional[UnifiedQuantumOrchestrator] = None
        self.unified_swarm: Optional[UnifiedSwarmSystem] = None
        
        # Enhanced components
        self.mega_coordinator: Optional[EnhancedMegaSwarmCoordinator] = None
        self.million_scale_worker: Optional[EnhancedMillionScaleWorker] = None
        self.integration_bridge: Optional[UnifiedIntegrationBridge] = None
        
        # Service state
        self.service_active = False
        self.initialization_time = 0.0
        
        log.info(f"Unified swarm service created for node {node_id}")
    
    async def initialize_service(self, 
                                enable_neural_mesh: bool = True,
                                enable_orchestrator: bool = True,
                                worker_type: str = "general") -> bool:
        """Initialize unified swarm service with all components"""
        
        start_time = time.time()
        
        try:
            log.info("Initializing unified swarm service...")
            
            # Initialize neural mesh if enabled and available
            if enable_neural_mesh and NEURAL_MESH_AVAILABLE:
                self.neural_mesh = await create_production_neural_mesh(
                    node_id=self.node_id,
                    enable_l4_memory=True
                )
                log.info("Neural mesh initialized")
            
            # Initialize orchestrator if enabled and available
            if enable_orchestrator and ORCHESTRATOR_AVAILABLE:
                self.orchestrator = UnifiedQuantumOrchestrator(
                    node_id=self.node_id,
                    max_agents=1000000,
                    enable_security=True
                )
                await self.orchestrator.initialize()
                log.info("Unified orchestrator initialized")
            
            # Initialize unified swarm system
            self.unified_swarm = await create_unified_swarm_system(
                node_id=self.node_id,
                neural_mesh=self.neural_mesh,
                orchestrator=self.orchestrator
            )
            log.info("Unified swarm system initialized")
            
            # Initialize enhanced mega-coordinator
            self.mega_coordinator = await create_enhanced_mega_swarm_coordinator(
                node_id=self.node_id,
                neural_mesh=self.neural_mesh,
                orchestrator=self.orchestrator,
                unified_swarm=self.unified_swarm
            )
            log.info("Enhanced mega-coordinator initialized")
            
            # Initialize enhanced million-scale worker
            self.million_scale_worker = await create_enhanced_million_scale_worker(
                worker_type=worker_type,
                unified_swarm=self.unified_swarm
            )
            log.info("Enhanced million-scale worker initialized")
            
            # Initialize integration bridge
            self.integration_bridge = await create_unified_integration_bridge(
                swarm_system=self.unified_swarm,
                neural_mesh=self.neural_mesh,
                orchestrator=self.orchestrator,
                mode=IntegrationMode.FULL_INTEGRATION
            )
            log.info("Unified integration bridge initialized")
            
            # Start all components
            await self._start_all_components()
            
            self.service_active = True
            self.initialization_time = time.time() - start_time
            
            log.info(f"Unified swarm service initialized successfully in {self.initialization_time:.2f}s")
            
            return True
            
        except Exception as e:
            log.error(f"Unified swarm service initialization failed: {e}")
            return False
    
    async def _start_all_components(self):
        """Start all service components"""
        
        start_tasks = []
        
        # Start million-scale worker
        if self.million_scale_worker:
            start_tasks.append(self.million_scale_worker.start_enhanced_processing())
        
        # Execute all start tasks
        if start_tasks:
            await asyncio.gather(*start_tasks, return_exceptions=True)
        
        log.info("All service components started")
    
    async def process_swarm_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process swarm request through unified system"""
        
        if not self.service_active:
            return {"error": "Service not active"}
        
        try:
            request_type = request.get("type", "general")
            
            if request_type == "mega_swarm_coordination":
                return await self._handle_mega_swarm_request(request)
            elif request_type == "neural_mesh_task":
                return await self._handle_neural_mesh_request(request)
            elif request_type == "fusion_processing":
                return await self._handle_fusion_request(request)
            elif request_type == "unified_task":
                return await self._handle_unified_task_request(request)
            else:
                return await self._handle_general_request(request)
                
        except Exception as e:
            log.error(f"Swarm request processing failed: {e}")
            return {"error": str(e), "success": False}
    
    async def _handle_mega_swarm_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mega-swarm coordination request"""
        
        if not self.mega_coordinator:
            return {"error": "Mega-coordinator not available"}
        
        try:
            # Create goal from request
            from .legacy.mega_swarm_coordinator import Goal, SwarmObjective, SwarmScale
            
            goal = Goal(
                goal_id=request.get("goal_id", f"mega_{int(time.time() * 1000)}"),
                description=request.get("description", "Mega-swarm coordination"),
                objective=SwarmObjective(request.get("objective", "maximize_throughput")),
                requirements=request.get("requirements", {}),
                constraints=request.get("constraints", {}),
                expected_scale=SwarmScale(request.get("scale", "mega"))
            )
            
            # Execute through enhanced coordinator
            result = await self.mega_coordinator.coordinate_enhanced_million_agents(goal)
            
            return {
                "success": result.success,
                "result": result.result,
                "agents_used": result.total_agents_used,
                "execution_time": result.total_execution_time,
                "confidence": result.confidence,
                "enhanced": True
            }
            
        except Exception as e:
            log.error(f"Mega-swarm request handling failed: {e}")
            return {"error": str(e)}
    
    async def _handle_unified_task_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unified task request through integration bridge"""
        
        if not self.integration_bridge:
            return {"error": "Integration bridge not available"}
        
        try:
            result = await self.integration_bridge.execute_unified_task(
                task_description=request.get("description", "Unified task"),
                task_type=request.get("task_type", "general"),
                priority=request.get("priority", "normal"),
                requirements=request.get("requirements", {})
            )
            
            return result
            
        except Exception as e:
            log.error(f"Unified task request handling failed: {e}")
            return {"error": str(e)}
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        
        try:
            status = {
                "service_active": self.service_active,
                "node_id": self.node_id,
                "initialization_time": self.initialization_time,
                "components": {}
            }
            
            # Get component statuses
            if self.unified_swarm:
                status["components"]["unified_swarm"] = await self.unified_swarm.get_system_status()
            
            if self.mega_coordinator:
                status["components"]["mega_coordinator"] = await self.mega_coordinator.get_enhanced_swarm_status()
            
            if self.million_scale_worker:
                status["components"]["million_scale_worker"] = self.million_scale_worker.get_enhanced_worker_status()
            
            if self.integration_bridge:
                status["components"]["integration_bridge"] = await self.integration_bridge.get_integration_status()
            
            # Add system-wide metrics
            status["system_metrics"] = {
                "total_components": len([c for c in [self.unified_swarm, self.mega_coordinator, 
                                                   self.million_scale_worker, self.integration_bridge] if c]),
                "neural_mesh_available": self.neural_mesh is not None,
                "orchestrator_available": self.orchestrator is not None,
                "full_integration_active": all([self.unified_swarm, self.neural_mesh, self.orchestrator, self.integration_bridge])
            }
            
            return status
            
        except Exception as e:
            log.error(f"Service status generation failed: {e}")
            return {"error": str(e), "service_active": self.service_active}
    
    async def shutdown_service(self):
        """Shutdown unified swarm service"""
        
        log.info("Shutting down unified swarm service...")
        
        try:
            # Shutdown components in reverse order
            if self.integration_bridge:
                await self.integration_bridge.shutdown_integration()
            
            if self.million_scale_worker:
                await self.million_scale_worker.stop_enhanced_processing()
            
            if self.mega_coordinator:
                await self.mega_coordinator.shutdown_enhanced_coordinator()
            
            if self.unified_swarm:
                await self.unified_swarm.shutdown_system()
            
            if self.orchestrator:
                await self.orchestrator.shutdown()
            
            self.service_active = False
            
            log.info("Unified swarm service shutdown complete")
            
        except Exception as e:
            log.error(f"Service shutdown error: {e}")

# Service factory function
async def create_unified_swarm_service(
    node_id: str,
    enable_neural_mesh: bool = True,
    enable_orchestrator: bool = True,
    worker_type: str = "general"
) -> UnifiedSwarmService:
    """Create and initialize unified swarm service"""
    
    service = UnifiedSwarmService(node_id)
    
    if await service.initialize_service(enable_neural_mesh, enable_orchestrator, worker_type):
        log.info(f"Unified swarm service ready for node {node_id}")
        return service
    else:
        raise RuntimeError("Failed to initialize unified swarm service")

# Main entry point
async def main():
    """Main entry point for unified swarm service"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration from environment
    node_id = os.getenv("SWARM_NODE_ID", f"swarm_node_{os.getpid()}")
    enable_neural_mesh = os.getenv("ENABLE_NEURAL_MESH", "true").lower() == "true"
    enable_orchestrator = os.getenv("ENABLE_ORCHESTRATOR", "true").lower() == "true"
    worker_type = os.getenv("WORKER_TYPE", "general")
    
    try:
        # Create and start service
        service = await create_unified_swarm_service(
            node_id=node_id,
            enable_neural_mesh=enable_neural_mesh,
            enable_orchestrator=enable_orchestrator,
            worker_type=worker_type
        )
        
        log.info(f"🚀 Unified Swarm Service started successfully!")
        log.info(f"   Node ID: {node_id}")
        log.info(f"   Neural Mesh: {'✅ Enabled' if enable_neural_mesh else '❌ Disabled'}")
        log.info(f"   Orchestrator: {'✅ Enabled' if enable_orchestrator else '❌ Disabled'}")
        log.info(f"   Worker Type: {worker_type}")
        
        # Get initial status
        status = await service.get_service_status()
        log.info(f"   Components: {status['system_metrics']['total_components']}")
        log.info(f"   Full Integration: {'✅ Active' if status['system_metrics']['full_integration_active'] else '❌ Partial'}")
        
        # Keep service running
        try:
            while True:
                await asyncio.sleep(60)
                
                # Periodic status check
                current_status = await service.get_service_status()
                if not current_status.get("service_active", False):
                    log.warning("Service became inactive, shutting down")
                    break
                    
        except KeyboardInterrupt:
            log.info("Shutdown signal received")
        
        # Graceful shutdown
        await service.shutdown_service()
        log.info("🛑 Unified Swarm Service shutdown complete")
        
    except Exception as e:
        log.error(f"Unified swarm service failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the unified swarm service
    asyncio.run(main())
