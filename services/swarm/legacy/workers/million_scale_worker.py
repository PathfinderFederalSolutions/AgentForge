"""
Million-Scale NATS Worker with Enhanced Backpressure Management
Optimized for processing millions of agent coordination messages
"""
import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import psutil

# Enhanced JetStream and backpressure imports
import sys
sys.path.append('/Users/baileymahoney/AgentForge')

from services.swarm.enhanced_jetstream import get_enhanced_jetstream, EnhancedJetStream
from services.swarm.backpressure_manager import (
    get_backpressure_manager, 
    BackpressureManager, 
    BackpressureMetrics,
    BackpressureStrategy
)

# Metrics imports (graceful degradation)
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = lambda *args, **kwargs: None

log = logging.getLogger("million-scale-worker")

# Configuration
WORKER_ID = os.getenv("WORKER_ID", f"worker-{os.getpid()}")
WORKER_TYPE = os.getenv("WORKER_TYPE", "general")  # general, neural_mesh, lifecycle, etc.
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
FETCH_TIMEOUT = float(os.getenv("FETCH_TIMEOUT", "5.0"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8080"))

# Stream and consumer configuration based on worker type
WORKER_CONFIGS = {
    "general": {
        "stream": "agentforge_jobs_million",
        "consumer": f"job_processor_{WORKER_ID}",
        "subjects": ["jobs.>"],
        "description": "General job processing worker"
    },
    "neural_mesh": {
        "stream": "agentforge_neural_mesh",
        "consumer": f"mesh_worker_{WORKER_ID}",
        "subjects": ["mesh.sync.>", "mesh.crdt.>"],
        "description": "Neural mesh synchronization worker"
    },
    "lifecycle": {
        "stream": "agentforge_agent_lifecycle",
        "consumer": f"lifecycle_worker_{WORKER_ID}",
        "subjects": ["agent.spawn.>", "agent.terminate.>"],
        "description": "Agent lifecycle management worker"
    },
    "results": {
        "stream": "agentforge_results_million",
        "consumer": f"results_worker_{WORKER_ID}",
        "subjects": ["results.>"],
        "description": "Results processing worker"
    }
}

@dataclass
class WorkerMetrics:
    """Worker performance metrics"""
    messages_processed: int = 0
    messages_failed: int = 0
    total_processing_time: float = 0.0
    current_queue_depth: int = 0
    active_jobs: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_update: float = 0.0

class MillionScaleWorker:
    """High-performance worker optimized for million-scale message processing"""
    
    def __init__(self, worker_type: str = "general"):
        self.worker_type = worker_type
        self.worker_id = WORKER_ID
        self.config = WORKER_CONFIGS.get(worker_type, WORKER_CONFIGS["general"])
        
        # Core components
        self.jetstream: Optional[EnhancedJetStream] = None
        self.backpressure_manager: Optional[BackpressureManager] = None
        
        # State
        self.running = False
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.metrics = WorkerMetrics()
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        
        # Prometheus metrics
        if METRICS_AVAILABLE:
            self.messages_processed_counter = Counter(
                'worker_messages_processed_total',
                'Total messages processed',
                ['worker_id', 'worker_type', 'status']
            )
            self.processing_latency_histogram = Histogram(
                'worker_processing_latency_seconds',
                'Message processing latency',
                ['worker_id', 'worker_type']
            )
            self.active_jobs_gauge = Gauge(
                'worker_active_jobs',
                'Currently active jobs',
                ['worker_id', 'worker_type']
            )
            self.queue_depth_gauge = Gauge(
                'worker_queue_depth',
                'Current queue depth',
                ['worker_id', 'worker_type']
            )
            self.backpressure_level_gauge = Gauge(
                'worker_backpressure_level',
                'Current backpressure level',
                ['worker_id', 'worker_type']
            )
    
    async def initialize(self):
        """Initialize worker components"""
        log.info(f"Initializing million-scale worker {self.worker_id} (type: {self.worker_type})")
        
        # Initialize enhanced JetStream
        self.jetstream = await get_enhanced_jetstream()
        
        # Initialize backpressure manager
        self.backpressure_manager = get_backpressure_manager()
        
        # Register backpressure callbacks
        self.backpressure_manager.register_action_callback(
            BackpressureStrategy.RATE_LIMIT,
            self._handle_rate_limit_action
        )
        self.backpressure_manager.register_action_callback(
            BackpressureStrategy.BATCH_REDUCE,
            self._handle_batch_reduce_action
        )
        
        # Create consumer for this worker
        await self._create_consumer()
        
        # Start metrics server
        if METRICS_AVAILABLE:
            start_http_server(METRICS_PORT)
            log.info(f"Metrics server started on port {METRICS_PORT}")
        
        log.info(f"Worker {self.worker_id} initialized successfully")
    
    async def _create_consumer(self):
        """Create optimized consumer for this worker"""
        consumer_name = self.config["consumer"]
        stream_name = self.config["stream"]
        filter_subject = self.config["subjects"][0] if self.config["subjects"] else ">"
        
        # Calculate optimal consumer settings based on worker type
        if self.worker_type == "neural_mesh":
            max_ack_pending = 25000
            ack_wait = 600  # 10 minutes for mesh operations
        elif self.worker_type == "lifecycle":
            max_ack_pending = 10000
            ack_wait = 300  # 5 minutes for lifecycle operations
        else:
            max_ack_pending = 50000
            ack_wait = 300  # 5 minutes default
        
        await self.jetstream.create_million_scale_consumer(
            stream_name=stream_name,
            consumer_name=consumer_name,
            filter_subject=filter_subject,
            max_ack_pending=max_ack_pending,
            ack_wait=ack_wait
        )
        
        log.info(f"Created consumer {consumer_name} for stream {stream_name}")
    
    async def start(self):
        """Start the worker main loop"""
        if self.running:
            log.warning("Worker is already running")
            return
        
        self.running = True
        log.info(f"Starting million-scale worker {self.worker_id}")
        
        # Start background tasks
        background_tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._main_processing_loop())
        ]
        
        try:
            await asyncio.gather(*background_tasks)
        except asyncio.CancelledError:
            log.info("Worker tasks cancelled")
        except Exception as e:
            log.error(f"Worker error: {e}")
        finally:
            self.running = False
            await self._cleanup()
    
    async def _main_processing_loop(self):
        """Main message processing loop with backpressure management"""
        async with self.jetstream.connection_pool.get_connection() as nc:
            js = nc.jetstream()
            
            # Subscribe to consumer
            consumer_name = self.config["consumer"]
            stream_name = self.config["stream"]
            
            sub = await js.pull_subscribe(
                subject=self.config["subjects"][0],
                durable=consumer_name,
                stream=stream_name
            )
            
            log.info(f"Subscribed to {stream_name} with consumer {consumer_name}")
            
            while self.running:
                try:
                    # Get optimal batch size from backpressure manager
                    current_batch = self.backpressure_manager.get_optimal_batch_size(
                        queue_depth=self.metrics.current_queue_depth
                    )
                    
                    # Fetch messages
                    try:
                        messages = await sub.fetch(batch=current_batch, timeout=FETCH_TIMEOUT)
                    except asyncio.TimeoutError:
                        # No messages available, continue
                        continue
                    
                    if not messages:
                        continue
                    
                    # Update queue depth metric
                    self.metrics.current_queue_depth = len(messages)
                    
                    # Check if we should process messages (backpressure check)
                    should_process = await self.backpressure_manager.should_process_message()
                    if not should_process:
                        # Acknowledge messages without processing (load shedding)
                        for msg in messages:
                            await msg.ack()
                        continue
                    
                    # Process messages in parallel
                    await self._process_message_batch(messages)
                    
                except Exception as e:
                    log.error(f"Error in processing loop: {e}")
                    await asyncio.sleep(1.0)  # Brief pause on error
    
    async def _process_message_batch(self, messages: List):
        """Process a batch of messages with concurrency control"""
        batch_start_time = time.time()
        
        # Create processing tasks
        tasks = []
        for msg in messages:
            # Acquire semaphore for concurrency control
            await self.semaphore.acquire()
            task = asyncio.create_task(self._process_single_message(msg))
            tasks.append(task)
            
            # Track active job
            job_id = self._extract_job_id(msg)
            if job_id:
                self.active_jobs[job_id] = task
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update metrics
        batch_processing_time = time.time() - batch_start_time
        success_count = sum(1 for result in results if not isinstance(result, Exception))
        
        # Record batch results for adaptive batching
        self.backpressure_manager.batch_controller.record_batch_result(
            batch_size=len(messages),
            latency=batch_processing_time,
            success_count=success_count
        )
        
        # Update worker metrics
        self.metrics.messages_processed += success_count
        self.metrics.messages_failed += len(messages) - success_count
        self.metrics.total_processing_time += batch_processing_time
        
        if METRICS_AVAILABLE:
            self.processing_latency_histogram.labels(
                worker_id=self.worker_id,
                worker_type=self.worker_type
            ).observe(batch_processing_time / len(messages))
    
    async def _process_single_message(self, msg):
        """Process a single message"""
        job_id = None
        try:
            # Parse message
            data = json.loads(msg.data.decode())
            job_id = self._extract_job_id(msg)
            
            # Update active jobs count
            self.metrics.active_jobs = len(self.active_jobs)
            if METRICS_AVAILABLE:
                self.active_jobs_gauge.labels(
                    worker_id=self.worker_id,
                    worker_type=self.worker_type
                ).set(self.metrics.active_jobs)
            
            # Process based on worker type
            if self.worker_type == "neural_mesh":
                await self._process_neural_mesh_message(data)
            elif self.worker_type == "lifecycle":
                await self._process_lifecycle_message(data)
            elif self.worker_type == "results":
                await self._process_results_message(data)
            else:
                await self._process_general_message(data)
            
            # Acknowledge successful processing
            await msg.ack()
            
            if METRICS_AVAILABLE:
                self.messages_processed_counter.labels(
                    worker_id=self.worker_id,
                    worker_type=self.worker_type,
                    status="success"
                ).inc()
            
        except Exception as e:
            log.error(f"Error processing message {job_id}: {e}")
            
            # Handle failed message (could implement retry logic here)
            await msg.ack()  # For now, acknowledge to prevent redelivery
            
            if METRICS_AVAILABLE:
                self.messages_processed_counter.labels(
                    worker_id=self.worker_id,
                    worker_type=self.worker_type,
                    status="error"
                ).inc()
        
        finally:
            # Clean up active job tracking
            if job_id and job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            # Release semaphore
            self.semaphore.release()
    
    def _extract_job_id(self, msg) -> Optional[str]:
        """Extract job ID from message"""
        try:
            data = json.loads(msg.data.decode())
            return (
                data.get("job_id") or
                data.get("id") or
                data.get("request_id") or
                data.get("invocation_id")
            )
        except:
            return None
    
    async def _process_neural_mesh_message(self, data: Dict[str, Any]):
        """Process neural mesh synchronization message"""
        operation_type = data.get("operation", "sync")
        
        if operation_type == "sync":
            # Handle mesh synchronization
            await asyncio.sleep(0.01)  # Simulate processing
        elif operation_type == "crdt_update":
            # Handle CRDT update
            await asyncio.sleep(0.005)  # Simulate processing
        elif operation_type == "broadcast":
            # Handle broadcast message
            await asyncio.sleep(0.002)  # Simulate processing
        
        log.debug(f"Processed neural mesh {operation_type} operation")
    
    async def _process_lifecycle_message(self, data: Dict[str, Any]):
        """Process agent lifecycle message"""
        action = data.get("action", "unknown")
        agent_id = data.get("agent_id")
        
        if action == "spawn":
            # Handle agent spawning
            await asyncio.sleep(0.02)  # Simulate processing
        elif action == "terminate":
            # Handle agent termination
            await asyncio.sleep(0.01)  # Simulate processing
        elif action == "health_check":
            # Handle health check
            await asyncio.sleep(0.001)  # Simulate processing
        
        log.debug(f"Processed lifecycle {action} for agent {agent_id}")
    
    async def _process_results_message(self, data: Dict[str, Any]):
        """Process results message"""
        result_type = data.get("type", "unknown")
        
        if result_type == "success":
            # Handle successful result
            await asyncio.sleep(0.005)  # Simulate processing
        elif result_type == "error":
            # Handle error result
            await asyncio.sleep(0.01)  # Simulate processing
        
        log.debug(f"Processed result of type {result_type}")
    
    async def _process_general_message(self, data: Dict[str, Any]):
        """Process general job message"""
        job_type = data.get("type", "unknown")
        
        # Simulate different processing times based on job type
        if job_type == "compute_heavy":
            await asyncio.sleep(0.1)
        elif job_type == "io_bound":
            await asyncio.sleep(0.05)
        else:
            await asyncio.sleep(0.01)
        
        log.debug(f"Processed general job of type {job_type}")
    
    async def _metrics_collection_loop(self):
        """Collect and update system metrics"""
        while self.running:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                
                # Update metrics
                self.metrics.cpu_usage = cpu_percent
                self.metrics.memory_usage = memory_info.percent
                self.metrics.last_update = time.time()
                
                # Create backpressure metrics
                bp_metrics = BackpressureMetrics(
                    queue_depth=self.metrics.current_queue_depth,
                    avg_latency=self.metrics.total_processing_time / max(1, self.metrics.messages_processed),
                    cpu_usage=cpu_percent,
                    memory_usage=memory_info.percent,
                    error_rate=self.metrics.messages_failed / max(1, time.time() - self.metrics.last_update + 1),
                    active_agents=self.metrics.active_jobs,
                    message_rate=self.metrics.messages_processed / max(1, time.time() - self.metrics.last_update + 1)
                )
                
                # Update backpressure manager
                self.backpressure_manager.update_metrics(bp_metrics)
                
                # Update Prometheus metrics
                if METRICS_AVAILABLE:
                    self.backpressure_level_gauge.labels(
                        worker_id=self.worker_id,
                        worker_type=self.worker_type
                    ).set(self.backpressure_manager.current_level.value)
                    
                    self.queue_depth_gauge.labels(
                        worker_id=self.worker_id,
                        worker_type=self.worker_type
                    ).set(self.metrics.current_queue_depth)
                
                await asyncio.sleep(5.0)  # Collect metrics every 5 seconds
                
            except Exception as e:
                log.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5.0)
    
    async def _health_check_loop(self):
        """Periodic health checks"""
        while self.running:
            try:
                # Check JetStream health
                health = await self.jetstream.health_check()
                
                if health["status"] != "healthy":
                    log.warning(f"JetStream health check failed: {health}")
                
                # Check backpressure status
                bp_status = self.backpressure_manager.get_status()
                if bp_status["level"] in ["HIGH", "CRITICAL"]:
                    log.warning(f"High backpressure detected: {bp_status}")
                
                await asyncio.sleep(30.0)  # Health check every 30 seconds
                
            except Exception as e:
                log.error(f"Error in health check: {e}")
                await asyncio.sleep(30.0)
    
    def _handle_rate_limit_action(self, action):
        """Handle rate limiting backpressure action"""
        log.info(f"Applying rate limiting: {action.parameters}")
        # Rate limiting is handled by the backpressure manager
    
    def _handle_batch_reduce_action(self, action):
        """Handle batch reduction backpressure action"""
        log.info(f"Reducing batch size: {action.parameters}")
        # Batch reduction is handled by the adaptive batch controller
    
    async def _cleanup(self):
        """Clean up resources"""
        log.info(f"Cleaning up worker {self.worker_id}")
        
        # Cancel active jobs
        for job_id, task in self.active_jobs.items():
            if not task.done():
                task.cancel()
        
        if self.active_jobs:
            await asyncio.gather(*self.active_jobs.values(), return_exceptions=True)
        
        # Close JetStream connection
        if self.jetstream:
            await self.jetstream.close()
        
        log.info(f"Worker {self.worker_id} cleanup completed")
    
    async def stop(self):
        """Stop the worker"""
        log.info(f"Stopping worker {self.worker_id}")
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive worker status"""
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "running": self.running,
            "metrics": {
                "messages_processed": self.metrics.messages_processed,
                "messages_failed": self.metrics.messages_failed,
                "active_jobs": self.metrics.active_jobs,
                "queue_depth": self.metrics.current_queue_depth,
                "cpu_usage": self.metrics.cpu_usage,
                "memory_usage": self.metrics.memory_usage,
                "avg_processing_time": (
                    self.metrics.total_processing_time / max(1, self.metrics.messages_processed)
                ),
            },
            "backpressure": self.backpressure_manager.get_status() if self.backpressure_manager else {},
            "jetstream_health": "unknown"  # Would need async call to get actual health
        }

async def main():
    """Main entry point"""
    worker_type = os.getenv("WORKER_TYPE", "general")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    worker = MillionScaleWorker(worker_type=worker_type)
    
    try:
        await worker.initialize()
        await worker.start()
    except KeyboardInterrupt:
        log.info("Received interrupt signal")
    except Exception as e:
        log.error(f"Worker failed: {e}")
    finally:
        await worker.stop()

if __name__ == "__main__":
    asyncio.run(main())

