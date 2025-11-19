"""
Enhanced Million-Scale Worker - Integrated with Unified System
Preserves all worker capabilities while integrating with neural mesh and orchestrator
"""

import asyncio
import json
import logging
import os
import time
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

# Import unified system components
from ..unified_swarm_system import UnifiedSwarmSystem
from ..workers.unified_worker import UnifiedWorker, WorkerType, ProcessingMode, WorkerConfiguration

# Import original worker components (preserved)
from ..legacy.workers.million_scale_worker import MillionScaleWorker, WorkerMetrics, WORKER_CONFIGS
from ..legacy.workers.nats_worker import main as nats_worker_main
from ..legacy.workers.temporal_workflows import run_worker as temporal_worker_main

# Enhanced components
try:
    from ..enhanced_jetstream import get_enhanced_jetstream, EnhancedJetStream
    from ..backpressure_manager import get_backpressure_manager, BackpressureManager
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False

# Metrics imports
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

log = logging.getLogger("enhanced-million-scale-worker")

class EnhancedMillionScaleWorker:
    """
    Enhanced Million-Scale Worker - Production Integration
    
    Preserves all original worker capabilities while adding:
    - Unified system integration
    - Neural mesh synchronization
    - Orchestrator coordination
    - Enhanced performance monitoring
    """
    
    def __init__(self, 
                 worker_type: str = "general",
                 unified_swarm: Optional[UnifiedSwarmSystem] = None):
        
        self.worker_type = worker_type
        self.worker_id = os.getenv("WORKER_ID", f"enhanced_worker_{os.getpid()}")
        self.unified_swarm = unified_swarm
        
        # Initialize legacy worker
        self.legacy_worker = MillionScaleWorker(worker_type)
        
        # Initialize unified worker
        config = WorkerConfiguration(
            worker_type=WorkerType(worker_type),
            processing_mode=ProcessingMode.BATCH_PROCESSING,
            max_concurrent_jobs=int(os.getenv("MAX_CONCURRENT_JOBS", "100")),
            batch_size=int(os.getenv("BATCH_SIZE", "50"))
        )
        
        self.unified_worker = UnifiedWorker(
            config=config,
            neural_mesh=unified_swarm.neural_mesh if unified_swarm else None,
            fusion_system=unified_swarm.fusion_system if unified_swarm else None
        )
        
        # Enhanced state
        self.integration_active = False
        self.processing_mode = "hybrid"  # Can switch between legacy and unified
        
        # Enhanced metrics
        self.enhanced_metrics = {
            "legacy_messages_processed": 0,
            "unified_messages_processed": 0,
            "neural_mesh_syncs": 0,
            "fusion_operations": 0,
            "hybrid_mode_switches": 0,
            "performance_improvements": 0.0
        }
        
        log.info(f"Enhanced million-scale worker {self.worker_id} initialized")
    
    async def initialize_enhanced_worker(self) -> bool:
        """Initialize enhanced worker with all capabilities"""
        
        try:
            log.info("Initializing enhanced million-scale worker...")
            
            # Initialize legacy worker
            await self.legacy_worker.initialize()
            
            # Initialize unified worker
            await self.unified_worker.initialize()
            
            # Setup integration
            await self._setup_worker_integration()
            
            self.integration_active = True
            
            log.info("Enhanced million-scale worker initialized successfully")
            return True
            
        except Exception as e:
            log.error(f"Enhanced worker initialization failed: {e}")
            return False
    
    async def start_enhanced_processing(self):
        """Start enhanced processing with both legacy and unified capabilities"""
        
        if not self.integration_active:
            raise RuntimeError("Enhanced worker not initialized")
        
        try:
            # Start legacy worker
            await self.legacy_worker.start()
            
            # Start unified worker
            await self.unified_worker.start_processing()
            
            # Start enhanced processing loop
            asyncio.create_task(self._enhanced_processing_loop())
            
            log.info(f"Enhanced processing started for worker {self.worker_id}")
            
        except Exception as e:
            log.error(f"Enhanced processing startup failed: {e}")
            raise
    
    async def stop_enhanced_processing(self):
        """Stop enhanced processing"""
        
        try:
            # Stop unified worker
            await self.unified_worker.stop_processing()
            
            # Stop legacy worker
            await self.legacy_worker.stop()
            
            self.integration_active = False
            
            log.info(f"Enhanced processing stopped for worker {self.worker_id}")
            
        except Exception as e:
            log.error(f"Enhanced processing shutdown failed: {e}")
    
    async def _enhanced_processing_loop(self):
        """Enhanced processing loop that intelligently routes messages"""
        
        while self.integration_active:
            try:
                # Monitor performance and decide routing
                if await self._should_use_unified_processing():
                    self.processing_mode = "unified"
                else:
                    self.processing_mode = "legacy"
                
                # Process based on current mode
                if self.processing_mode == "unified":
                    await self._process_with_unified_system()
                else:
                    await self._process_with_legacy_system()
                
                await asyncio.sleep(0.1)  # Small delay
                
            except Exception as e:
                log.error(f"Enhanced processing loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _should_use_unified_processing(self) -> bool:
        """Determine whether to use unified or legacy processing"""
        
        try:
            # Use unified for complex tasks that benefit from integration
            if self.unified_swarm:
                # Check if neural mesh or fusion capabilities are needed
                pending_messages = self.unified_worker.message_queue.qsize()
                
                # Use unified for neural mesh tasks, fusion tasks, or when performance is better
                if (pending_messages > 0 and 
                    (self.worker_type in ["neural_mesh", "fusion"] or 
                     self.enhanced_metrics["performance_improvements"] > 0.1)):
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def _process_with_unified_system(self):
        """Process using unified system capabilities"""
        
        try:
            # The unified worker handles its own processing loop
            # We just track that we're using unified processing
            self.enhanced_metrics["unified_messages_processed"] += 1
            
        except Exception as e:
            log.warning(f"Unified processing error: {e}")
    
    async def _process_with_legacy_system(self):
        """Process using legacy system capabilities"""
        
        try:
            # The legacy worker handles its own processing
            # We just track that we're using legacy processing
            self.enhanced_metrics["legacy_messages_processed"] += 1
            
        except Exception as e:
            log.warning(f"Legacy processing error: {e}")
    
    async def _setup_worker_integration(self):
        """Setup integration between legacy and unified workers"""
        
        try:
            # Setup message routing between systems
            if self.unified_swarm and self.unified_swarm.neural_mesh:
                # Register worker with neural mesh
                await self.unified_swarm.neural_mesh.store(
                    f"worker_registration:{self.worker_id}",
                    f"Enhanced million-scale worker: {self.worker_type}",
                    context={"type": "worker_registration", "worker_type": self.worker_type},
                    metadata={"capabilities": ["legacy_processing", "unified_processing"]}
                )
            
            log.info("Worker integration setup complete")
            
        except Exception as e:
            log.warning(f"Worker integration setup failed: {e}")
    
    def get_enhanced_worker_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced worker status"""
        
        try:
            # Get legacy status
            legacy_status = {
                "legacy_metrics": self.legacy_worker.metrics.__dict__,
                "legacy_running": self.legacy_worker.running,
                "legacy_active_jobs": len(self.legacy_worker.active_jobs)
            }
            
            # Get unified status
            unified_status = self.unified_worker.get_worker_status()
            
            # Combine with enhanced metrics
            enhanced_status = {
                "worker_id": self.worker_id,
                "worker_type": self.worker_type,
                "integration_active": self.integration_active,
                "processing_mode": self.processing_mode,
                "enhanced_metrics": self.enhanced_metrics.copy(),
                "legacy_status": legacy_status,
                "unified_status": unified_status,
                "total_capabilities": len(legacy_status) + len(unified_status)
            }
            
            return enhanced_status
            
        except Exception as e:
            log.error(f"Enhanced worker status generation failed: {e}")
            return {"error": str(e), "worker_id": self.worker_id}

# Factory function
async def create_enhanced_million_scale_worker(
    worker_type: str = "general",
    unified_swarm: Optional[UnifiedSwarmSystem] = None
) -> EnhancedMillionScaleWorker:
    """Create enhanced million-scale worker"""
    
    worker = EnhancedMillionScaleWorker(worker_type, unified_swarm)
    
    if await worker.initialize_enhanced_worker():
        log.info(f"Enhanced million-scale worker created: {worker_type}")
        return worker
    else:
        raise RuntimeError("Failed to initialize enhanced million-scale worker")
