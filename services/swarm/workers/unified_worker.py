"""
Unified Worker System - Consolidated Worker Implementation
Integrates million-scale worker, NATS worker, and temporal workflows
with perfect neural mesh and orchestrator integration
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
import psutil

# Core imports with fallbacks
try:
    import nats
    from nats.js import JetStreamContext
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False

try:
    from temporalio import activity, workflow
    from temporalio.client import Client
    from temporalio.worker import Worker
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = lambda *args, **kwargs: None

# AgentForge imports
from ..core.unified_agent import UnifiedAgent, UnifiedAgentFactory, AgentType
from ..capability_registry import CapabilityRegistry
from ..fusion.production_fusion_system import ProductionFusionSystem

# Enhanced components
try:
    from ..enhanced_jetstream import EnhancedJetStream, get_enhanced_jetstream
    from ..backpressure_manager import BackpressureManager, get_backpressure_manager
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False

log = logging.getLogger("unified-worker")

class WorkerType(Enum):
    """Types of unified workers"""
    GENERAL = "general"
    NEURAL_MESH = "neural_mesh"
    FUSION = "fusion"
    LIFECYCLE = "lifecycle"
    RESULTS = "results"
    TEMPORAL = "temporal"
    STREAMING = "streaming"
    COORDINATOR = "coordinator"

class ProcessingMode(Enum):
    """Worker processing modes"""
    SINGLE_MESSAGE = "single_message"
    BATCH_PROCESSING = "batch_processing"
    STREAMING = "streaming"
    REAL_TIME = "real_time"
    TEMPORAL_WORKFLOW = "temporal_workflow"

@dataclass
class WorkerConfiguration:
    """Worker configuration"""
    worker_type: WorkerType
    processing_mode: ProcessingMode
    max_concurrent_jobs: int = 100
    batch_size: int = 50
    fetch_timeout: float = 5.0
    metrics_port: int = 8080
    stream_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default stream configuration"""
        if not self.stream_config:
            self.stream_config = self._get_default_stream_config()
    
    def _get_default_stream_config(self) -> Dict[str, Any]:
        """Get default stream configuration for worker type"""
        
        worker_id = os.getenv("WORKER_ID", f"worker-{os.getpid()}")
        
        configs = {
            WorkerType.GENERAL: {
                "stream": "agentforge_unified_general",
                "consumer": f"general_processor_{worker_id}",
                "subjects": ["jobs.>", "tasks.>"],
                "description": "General unified processing worker"
            },
            WorkerType.NEURAL_MESH: {
                "stream": "agentforge_neural_mesh_unified",
                "consumer": f"neural_mesh_worker_{worker_id}",
                "subjects": ["mesh.sync.>", "mesh.crdt.>", "mesh.belief.>"],
                "description": "Neural mesh unified worker"
            },
            WorkerType.FUSION: {
                "stream": "agentforge_fusion_unified",
                "consumer": f"fusion_worker_{worker_id}",
                "subjects": ["fusion.>", "sensors.>", "intelligence.>"],
                "description": "Intelligence fusion unified worker"
            },
            WorkerType.LIFECYCLE: {
                "stream": "agentforge_lifecycle_unified",
                "consumer": f"lifecycle_worker_{worker_id}",
                "subjects": ["agent.spawn.>", "agent.terminate.>", "swarm.scale.>"],
                "description": "Agent lifecycle unified worker"
            },
            WorkerType.RESULTS: {
                "stream": "agentforge_results_unified",
                "consumer": f"results_worker_{worker_id}",
                "subjects": ["results.>", "metrics.>"],
                "description": "Results processing unified worker"
            },
            WorkerType.TEMPORAL: {
                "stream": "agentforge_temporal_unified",
                "consumer": f"temporal_worker_{worker_id}",
                "subjects": ["workflows.>", "activities.>"],
                "description": "Temporal workflow unified worker"
            },
            WorkerType.STREAMING: {
                "stream": "agentforge_streaming_unified",
                "consumer": f"streaming_worker_{worker_id}",
                "subjects": ["stream.>", "realtime.>"],
                "description": "Real-time streaming unified worker"
            },
            WorkerType.COORDINATOR: {
                "stream": "agentforge_coordination_unified",
                "consumer": f"coordinator_worker_{worker_id}",
                "subjects": ["coord.>", "swarm.>", "quantum.>"],
                "description": "Swarm coordination unified worker"
            }
        }
        
        return configs.get(self.worker_type, configs[WorkerType.GENERAL])

@dataclass
class WorkerMetrics:
    """Comprehensive worker metrics"""
    messages_processed: int = 0
    messages_failed: int = 0
    total_processing_time: float = 0.0
    current_queue_depth: int = 0
    active_jobs: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    neural_mesh_syncs: int = 0
    fusion_operations: int = 0
    quantum_coordinations: int = 0
    last_update: float = field(default_factory=time.time)
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            process = psutil.Process()
            self.cpu_usage = process.cpu_percent()
            self.memory_usage = process.memory_percent()
            self.last_update = time.time()
        except Exception:
            pass  # Metrics update is non-critical

class UnifiedWorker:
    """
    Unified Worker System - Complete Integration
    
    Consolidates million-scale worker, NATS worker, and temporal workflows
    with neural mesh and orchestrator integration.
    """
    
    def __init__(self, 
                 config: WorkerConfiguration,
                 neural_mesh: Optional[Any] = None,
                 fusion_system: Optional[ProductionFusionSystem] = None):
        
        self.config = config
        self.worker_id = os.getenv("WORKER_ID", f"unified_worker_{uuid.uuid4().hex[:8]}")
        self.neural_mesh = neural_mesh
        self.fusion_system = fusion_system
        
        # Core components
        self.jetstream: Optional[Any] = None
        self.backpressure_manager: Optional[Any] = None
        self.agent_factory: Optional[UnifiedAgentFactory] = None
        self.capability_registry = CapabilityRegistry()
        
        # Worker state
        self.running = False
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.metrics = WorkerMetrics()
        self.processing_semaphore = asyncio.Semaphore(config.max_concurrent_jobs)
        
        # Message processing
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.result_callbacks: Dict[str, Callable] = {}
        
        # Temporal workflow support
        self.temporal_client: Optional[Any] = None
        self.temporal_worker: Optional[Any] = None
        
        # Prometheus metrics
        self._init_metrics()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        log.info(f"Unified worker {self.worker_id} ({config.worker_type.value}) initialized")
    
    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        
        if not METRICS_AVAILABLE:
            return
        
        self.messages_processed_counter = Counter(
            'unified_worker_messages_processed_total',
            'Total messages processed by unified worker',
            ['worker_id', 'worker_type', 'processing_mode', 'status']
        )
        
        self.processing_latency_histogram = Histogram(
            'unified_worker_processing_latency_seconds',
            'Message processing latency',
            ['worker_id', 'worker_type', 'processing_mode']
        )
        
        self.active_jobs_gauge = Gauge(
            'unified_worker_active_jobs',
            'Currently active jobs',
            ['worker_id', 'worker_type']
        )
        
        self.queue_depth_gauge = Gauge(
            'unified_worker_queue_depth',
            'Current message queue depth',
            ['worker_id', 'worker_type']
        )
        
        self.neural_mesh_syncs_counter = Counter(
            'unified_worker_neural_mesh_syncs_total',
            'Total neural mesh synchronizations',
            ['worker_id', 'worker_type']
        )
        
        self.fusion_operations_counter = Counter(
            'unified_worker_fusion_operations_total',
            'Total fusion operations',
            ['worker_id', 'worker_type']
        )
    
    async def initialize(self) -> bool:
        """Initialize unified worker"""
        
        try:
            log.info(f"Initializing unified worker {self.worker_id}...")
            
            # Initialize NATS JetStream if available
            if NATS_AVAILABLE and ENHANCED_COMPONENTS_AVAILABLE:
                self.jetstream = await get_enhanced_jetstream()
                self.backpressure_manager = await get_backpressure_manager()
                log.info("Enhanced JetStream and backpressure manager initialized")
            
            # Initialize agent factory
            self.agent_factory = UnifiedAgentFactory(self.neural_mesh)
            
            # Initialize temporal client if needed
            if self.config.worker_type == WorkerType.TEMPORAL and TEMPORAL_AVAILABLE:
                await self._init_temporal_client()
            
            # Register core capabilities
            await self._register_worker_capabilities()
            
            # Start metrics server
            if METRICS_AVAILABLE:
                start_http_server(self.config.metrics_port)
                log.info(f"Metrics server started on port {self.config.metrics_port}")
            
            log.info(f"Unified worker {self.worker_id} initialized successfully")
            return True
            
        except Exception as e:
            log.error(f"Worker initialization failed: {e}")
            return False
    
    async def start_processing(self):
        """Start worker processing"""
        
        if self.running:
            log.warning("Worker already running")
            return
        
        self.running = True
        
        # Start processing based on mode
        if self.config.processing_mode == ProcessingMode.SINGLE_MESSAGE:
            self._background_tasks.append(
                asyncio.create_task(self._single_message_processor())
            )
        elif self.config.processing_mode == ProcessingMode.BATCH_PROCESSING:
            self._background_tasks.append(
                asyncio.create_task(self._batch_processor())
            )
        elif self.config.processing_mode == ProcessingMode.STREAMING:
            self._background_tasks.append(
                asyncio.create_task(self._streaming_processor())
            )
        elif self.config.processing_mode == ProcessingMode.REAL_TIME:
            self._background_tasks.append(
                asyncio.create_task(self._real_time_processor())
            )
        elif self.config.processing_mode == ProcessingMode.TEMPORAL_WORKFLOW:
            await self._start_temporal_worker()
        
        # Start background monitoring
        self._background_tasks.extend([
            asyncio.create_task(self._metrics_collection_task()),
            asyncio.create_task(self._health_monitoring_task()),
            asyncio.create_task(self._neural_mesh_sync_task())
        ])
        
        # Start agent factory
        if self.agent_factory:
            await self.agent_factory.start_all_agents()
        
        log.info(f"Unified worker {self.worker_id} processing started")
    
    async def stop_processing(self):
        """Stop worker processing"""
        
        self.running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Stop agent factory
        if self.agent_factory:
            await self.agent_factory.stop_all_agents()
        
        # Stop temporal worker if running
        if self.temporal_worker:
            await self.temporal_worker.shutdown()
        
        log.info(f"Unified worker {self.worker_id} processing stopped")
    
    async def _single_message_processor(self):
        """Single message processing loop"""
        
        while self.running:
            try:
                # Get message from queue
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process message
                async with self.processing_semaphore:
                    await self._process_message(message)
                
            except Exception as e:
                log.error(f"Single message processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _batch_processor(self):
        """Batch processing loop"""
        
        batch = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                # Collect batch
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                    batch.append(message)
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if conditions met
                current_time = time.time()
                batch_timeout = (current_time - last_batch_time) * 1000 > 100  # 100ms timeout
                
                if len(batch) >= self.config.batch_size or (batch and batch_timeout):
                    await self._process_message_batch(batch)
                    batch.clear()
                    last_batch_time = current_time
                
                await asyncio.sleep(0.001)
                
            except Exception as e:
                log.error(f"Batch processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _streaming_processor(self):
        """Streaming processing loop"""
        
        while self.running:
            try:
                # Process all available messages immediately
                messages_to_process = []
                
                # Drain queue up to batch size
                for _ in range(self.config.batch_size):
                    try:
                        message = self.message_queue.get_nowait()
                        messages_to_process.append(message)
                    except asyncio.QueueEmpty:
                        break
                
                if messages_to_process:
                    # Process in parallel
                    await asyncio.gather(
                        *[self._process_message(msg) for msg in messages_to_process],
                        return_exceptions=True
                    )
                
                await asyncio.sleep(0.01)  # Small delay
                
            except Exception as e:
                log.error(f"Streaming processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _real_time_processor(self):
        """Real-time processing loop with priority handling"""
        
        while self.running:
            try:
                # Get message with priority handling
                message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                
                # Immediate processing for real-time
                asyncio.create_task(self._process_message(message))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Real-time processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual message"""
        
        start_time = time.time()
        job_id = message.get("job_id", f"job_{int(time.time() * 1000)}")
        
        try:
            self.metrics.active_jobs += 1
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.active_jobs_gauge.labels(
                    worker_id=self.worker_id,
                    worker_type=self.config.worker_type.value
                ).set(self.metrics.active_jobs)
            
            # Process based on worker type
            if self.config.worker_type == WorkerType.NEURAL_MESH:
                result = await self._process_neural_mesh_message(message)
            elif self.config.worker_type == WorkerType.FUSION:
                result = await self._process_fusion_message(message)
            elif self.config.worker_type == WorkerType.LIFECYCLE:
                result = await self._process_lifecycle_message(message)
            elif self.config.worker_type == WorkerType.RESULTS:
                result = await self._process_results_message(message)
            elif self.config.worker_type == WorkerType.COORDINATOR:
                result = await self._process_coordination_message(message)
            else:
                result = await self._process_general_message(message)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.messages_processed += 1
            self.metrics.total_processing_time += processing_time
            
            if METRICS_AVAILABLE:
                self.messages_processed_counter.labels(
                    worker_id=self.worker_id,
                    worker_type=self.config.worker_type.value,
                    processing_mode=self.config.processing_mode.value,
                    status="success"
                ).inc()
                
                self.processing_latency_histogram.labels(
                    worker_id=self.worker_id,
                    worker_type=self.config.worker_type.value,
                    processing_mode=self.config.processing_mode.value
                ).observe(processing_time)
            
            # Call result callbacks
            for callback_name, callback in self.result_callbacks.items():
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)
                    else:
                        callback(result)
                except Exception as e:
                    log.warning(f"Result callback {callback_name} failed: {e}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.messages_failed += 1
            
            log.error(f"Message processing failed for job {job_id}: {e}")
            
            if METRICS_AVAILABLE:
                self.messages_processed_counter.labels(
                    worker_id=self.worker_id,
                    worker_type=self.config.worker_type.value,
                    processing_mode=self.config.processing_mode.value,
                    status="error"
                ).inc()
            
            return {"error": str(e), "job_id": job_id}
            
        finally:
            self.metrics.active_jobs = max(0, self.metrics.active_jobs - 1)
    
    async def _process_message_batch(self, batch: List[Dict[str, Any]]):
        """Process batch of messages"""
        
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Process batch in parallel
            results = await asyncio.gather(
                *[self._process_message(msg) for msg in batch],
                return_exceptions=True
            )
            
            # Update batch metrics
            successful = sum(1 for r in results if not isinstance(r, Exception) and not r.get("error"))
            batch_processing_time = time.time() - start_time
            
            log.debug(f"Processed batch of {len(batch)} messages: {successful} successful in {batch_processing_time:.2f}s")
            
        except Exception as e:
            log.error(f"Batch processing failed: {e}")
    
    async def _process_neural_mesh_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process neural mesh specific message"""
        
        try:
            operation = message.get("operation", "unknown")
            
            if operation == "sync":
                return await self._handle_neural_mesh_sync(message)
            elif operation == "crdt":
                return await self._handle_crdt_operation(message)
            elif operation == "belief_revision":
                return await self._handle_belief_revision(message)
            else:
                return {"error": f"Unknown neural mesh operation: {operation}"}
                
        except Exception as e:
            log.error(f"Neural mesh message processing failed: {e}")
            return {"error": str(e)}
    
    async def _process_fusion_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process fusion specific message"""
        
        try:
            if not self.fusion_system:
                return {"error": "Fusion system not available"}
            
            operation = message.get("operation", "fuse")
            
            if operation == "fuse":
                return await self._handle_fusion_operation(message)
            elif operation == "calibrate":
                return await self._handle_calibration_operation(message)
            elif operation == "assess_quality":
                return await self._handle_quality_assessment(message)
            else:
                return {"error": f"Unknown fusion operation: {operation}"}
                
        except Exception as e:
            log.error(f"Fusion message processing failed: {e}")
            return {"error": str(e)}
    
    async def _process_lifecycle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent lifecycle message"""
        
        try:
            operation = message.get("operation", "unknown")
            
            if operation == "spawn":
                return await self._handle_agent_spawn(message)
            elif operation == "terminate":
                return await self._handle_agent_terminate(message)
            elif operation == "scale":
                return await self._handle_swarm_scale(message)
            else:
                return {"error": f"Unknown lifecycle operation: {operation}"}
                
        except Exception as e:
            log.error(f"Lifecycle message processing failed: {e}")
            return {"error": str(e)}
    
    async def _process_results_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process results message"""
        
        try:
            result_type = message.get("type", "general")
            
            # Store result based on type
            if result_type == "fusion_result":
                await self._store_fusion_result(message)
            elif result_type == "agent_result":
                await self._store_agent_result(message)
            elif result_type == "swarm_result":
                await self._store_swarm_result(message)
            
            return {"stored": True, "result_type": result_type}
            
        except Exception as e:
            log.error(f"Results message processing failed: {e}")
            return {"error": str(e)}
    
    async def _process_coordination_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process coordination message"""
        
        try:
            coord_type = message.get("type", "unknown")
            
            if coord_type == "swarm_coordination":
                return await self._handle_swarm_coordination(message)
            elif coord_type == "quantum_coordination":
                return await self._handle_quantum_coordination(message)
            elif coord_type == "cluster_management":
                return await self._handle_cluster_management(message)
            else:
                return {"error": f"Unknown coordination type: {coord_type}"}
                
        except Exception as e:
            log.error(f"Coordination message processing failed: {e}")
            return {"error": str(e)}
    
    async def _process_general_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process general message using agent"""
        
        try:
            # Create task from message
            task_data = message.get("task", message)
            
            from ..forge_types import Task
            task = Task(
                id=task_data.get("id", f"task_{int(time.time() * 1000)}"),
                description=task_data.get("description", "General processing task"),
                memory_scopes=task_data.get("memory_scopes", []),
                budget=task_data.get("budget", 1000),
                tools=task_data.get("tools", []),
                priority=task_data.get("priority", 1)
            )
            
            # Get or create appropriate agent
            agent = await self._get_or_create_agent_for_task(task)
            
            # Process task
            result = await agent.process_task(task)
            
            return {
                "task_id": task.id,
                "result": result,
                "agent_id": agent.agent_id,
                "processing_worker": self.worker_id
            }
            
        except Exception as e:
            log.error(f"General message processing failed: {e}")
            return {"error": str(e)}
    
    async def _get_or_create_agent_for_task(self, task) -> UnifiedAgent:
        """Get or create appropriate agent for task"""
        
        if not self.agent_factory:
            raise RuntimeError("Agent factory not initialized")
        
        # Determine agent type needed
        task_desc = task.description.lower()
        
        if "evaluate" in task_desc or "judge" in task_desc:
            agent_type = AgentType.CRITIC
        elif "fusion" in task_desc or "sensor" in task_desc:
            agent_type = AgentType.FUSION
        elif "neural mesh" in task_desc or "belief" in task_desc:
            agent_type = AgentType.NEURAL_MESH
        elif "quantum" in task_desc or "coordinate" in task_desc:
            agent_type = AgentType.QUANTUM
        else:
            agent_type = AgentType.STANDARD
        
        # Get existing agent or create new one
        agent_key = f"{agent_type.value}_agent"
        
        if agent_key in self.agent_factory.created_agents:
            return list(self.agent_factory.created_agents.values())[0]  # Use first available
        else:
            return self.agent_factory.create_agent(
                name=f"{agent_type.value}_worker_agent",
                agent_type=agent_type
            )
    
    async def _register_worker_capabilities(self):
        """Register worker-specific capabilities"""
        
        # Register unified processing capability
        self.capability_registry.register_capability(
            name="unified_processing",
            handler=self._unified_processing_handler,
            provides=["task_processing", "message_handling"],
            tags=["worker", "unified", "processing"]
        )
        
        # Register neural mesh capability if available
        if self.neural_mesh:
            self.capability_registry.register_capability(
                name="neural_mesh_worker",
                handler=self._neural_mesh_handler,
                provides=["neural_mesh_sync", "belief_processing"],
                tags=["neural_mesh", "worker", "sync"]
            )
        
        # Register fusion capability if available
        if self.fusion_system:
            self.capability_registry.register_capability(
                name="fusion_worker",
                handler=self._fusion_handler,
                provides=["sensor_fusion", "intelligence_processing"],
                tags=["fusion", "worker", "intelligence"]
            )
        
        log.info("Worker capabilities registered")
    
    # Capability handlers
    async def _unified_processing_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for unified processing capability"""
        
        try:
            message = kwargs.get("message", {})
            await self.message_queue.put(message)
            return {"queued": True, "worker_id": self.worker_id}
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _neural_mesh_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for neural mesh capability"""
        
        try:
            operation = kwargs.get("operation", "sync")
            data = kwargs.get("data", {})
            
            if operation == "sync" and self.neural_mesh:
                await self.neural_mesh.store(
                    kwargs.get("key", f"worker_data_{int(time.time())}"),
                    json.dumps(data),
                    context={"type": "worker_sync", "worker_id": self.worker_id},
                    metadata={"timestamp": time.time()}
                )
                
                self.metrics.neural_mesh_syncs += 1
                return {"synced": True, "worker_id": self.worker_id}
            
            return {"error": f"Unknown neural mesh operation: {operation}"}
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _fusion_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for fusion capability"""
        
        try:
            if not self.fusion_system:
                return {"error": "Fusion system not available"}
            
            # Create fusion request from kwargs
            from ..fusion.production_fusion_system import (
                IntelligenceFusionRequest, IntelligenceDomain, FusionQualityLevel, ClassificationLevel
            )
            
            sensor_data = kwargs.get("sensor_data", {})
            domain = kwargs.get("domain", "real_time_operations")
            
            fusion_request = IntelligenceFusionRequest(
                request_id=f"worker_fusion_{int(time.time() * 1000)}",
                domain=IntelligenceDomain(domain),
                sensor_data=sensor_data,
                quality_requirement=FusionQualityLevel.OPERATIONAL_GRADE,
                classification_level=ClassificationLevel.CONFIDENTIAL
            )
            
            # Process fusion
            result = await self.fusion_system.process_intelligence_fusion(fusion_request)
            
            self.metrics.fusion_operations += 1
            
            return {
                "fusion_result": result.fusion_result,
                "confidence": result.confidence,
                "worker_id": self.worker_id
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    # Background tasks
    async def _metrics_collection_task(self):
        """Collect and update metrics"""
        
        while self.running:
            try:
                # Update system metrics
                self.metrics.update_system_metrics()
                
                # Update queue depth
                self.metrics.current_queue_depth = self.message_queue.qsize()
                
                if METRICS_AVAILABLE:
                    self.queue_depth_gauge.labels(
                        worker_id=self.worker_id,
                        worker_type=self.config.worker_type.value
                    ).set(self.metrics.current_queue_depth)
                
                await asyncio.sleep(30.0)  # Update every 30 seconds
                
            except Exception as e:
                log.error(f"Metrics collection failed: {e}")
                await asyncio.sleep(60.0)
    
    async def _health_monitoring_task(self):
        """Monitor worker health"""
        
        while self.running:
            try:
                # Check worker health
                if self.metrics.cpu_usage > 90.0:
                    log.warning(f"Worker {self.worker_id} CPU usage high: {self.metrics.cpu_usage:.1f}%")
                
                if self.metrics.memory_usage > 90.0:
                    log.warning(f"Worker {self.worker_id} memory usage high: {self.metrics.memory_usage:.1f}%")
                
                if self.metrics.current_queue_depth > 5000:
                    log.warning(f"Worker {self.worker_id} queue depth high: {self.metrics.current_queue_depth}")
                
                await asyncio.sleep(60.0)  # Check every minute
                
            except Exception as e:
                log.error(f"Health monitoring failed: {e}")
                await asyncio.sleep(120.0)
    
    async def _neural_mesh_sync_task(self):
        """Sync worker state to neural mesh"""
        
        while self.running:
            try:
                if self.neural_mesh:
                    # Sync worker metrics
                    await self.neural_mesh.store(
                        f"worker_metrics:{self.worker_id}",
                        json.dumps({
                            "messages_processed": self.metrics.messages_processed,
                            "messages_failed": self.metrics.messages_failed,
                            "active_jobs": self.metrics.active_jobs,
                            "cpu_usage": self.metrics.cpu_usage,
                            "memory_usage": self.metrics.memory_usage
                        }),
                        context={"type": "worker_metrics", "worker_id": self.worker_id},
                        metadata={"timestamp": time.time()}
                    )
                    
                    if METRICS_AVAILABLE:
                        self.neural_mesh_syncs_counter.labels(
                            worker_id=self.worker_id,
                            worker_type=self.config.worker_type.value
                        ).inc()
                
                await asyncio.sleep(300.0)  # Sync every 5 minutes
                
            except Exception as e:
                log.error(f"Neural mesh sync failed: {e}")
                await asyncio.sleep(600.0)
    
    def get_worker_status(self) -> Dict[str, Any]:
        """Get comprehensive worker status"""
        
        return {
            "worker_id": self.worker_id,
            "worker_type": self.config.worker_type.value,
            "processing_mode": self.config.processing_mode.value,
            "running": self.running,
            "metrics": {
                "messages_processed": self.metrics.messages_processed,
                "messages_failed": self.metrics.messages_failed,
                "success_rate": (
                    self.metrics.messages_processed / 
                    max(1, self.metrics.messages_processed + self.metrics.messages_failed)
                ),
                "active_jobs": self.metrics.active_jobs,
                "queue_depth": self.metrics.current_queue_depth,
                "cpu_usage": self.metrics.cpu_usage,
                "memory_usage": self.metrics.memory_usage,
                "neural_mesh_syncs": self.metrics.neural_mesh_syncs,
                "fusion_operations": self.metrics.fusion_operations
            },
            "configuration": {
                "max_concurrent_jobs": self.config.max_concurrent_jobs,
                "batch_size": self.config.batch_size,
                "fetch_timeout": self.config.fetch_timeout
            },
            "integrations": {
                "neural_mesh_available": self.neural_mesh is not None,
                "fusion_system_available": self.fusion_system is not None,
                "jetstream_available": self.jetstream is not None,
                "temporal_available": self.temporal_client is not None
            }
        }

# Factory for creating unified workers
async def create_unified_worker_system(
    worker_type: WorkerType = WorkerType.GENERAL,
    processing_mode: ProcessingMode = ProcessingMode.BATCH_PROCESSING,
    neural_mesh: Optional[Any] = None,
    fusion_system: Optional[ProductionFusionSystem] = None
) -> UnifiedWorker:
    """Create unified worker system"""
    
    config = WorkerConfiguration(
        worker_type=worker_type,
        processing_mode=processing_mode,
        max_concurrent_jobs=int(os.getenv("MAX_CONCURRENT_JOBS", "100")),
        batch_size=int(os.getenv("BATCH_SIZE", "50")),
        fetch_timeout=float(os.getenv("FETCH_TIMEOUT", "5.0")),
        metrics_port=int(os.getenv("METRICS_PORT", "8080"))
    )
    
    worker = UnifiedWorker(config, neural_mesh, fusion_system)
    
    if await worker.initialize():
        log.info(f"Unified worker system created: {worker_type.value}")
        return worker
    else:
        raise RuntimeError("Failed to initialize unified worker system")

# Backwards compatibility exports
__all__ = [
    "UnifiedWorker",
    "WorkerType",
    "ProcessingMode",
    "WorkerConfiguration",
    "create_unified_worker_system"
]
