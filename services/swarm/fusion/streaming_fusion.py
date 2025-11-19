"""
Streaming Fusion Algorithms for Real-Time Intelligence Processing
High-performance distributed coordination mechanisms for massive-scale operations
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable, AsyncIterator, Union
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import queue
import weakref
import gc

# Optional high-performance imports
try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

log = logging.getLogger("streaming-fusion")

class StreamProcessingMode(Enum):
    """Streaming processing modes"""
    REAL_TIME = "real_time"
    BATCH_STREAMING = "batch_streaming"
    MICRO_BATCH = "micro_batch"
    CONTINUOUS = "continuous"

class CoordinationStrategy(Enum):
    """Distributed coordination strategies"""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    GOSSIP = "gossip"

class FusionPriority(Enum):
    """Fusion task priorities"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class StreamingDataPoint:
    """Individual data point in streaming fusion"""
    sensor_id: str
    timestamp: float
    value: Union[float, np.ndarray]
    quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_deadline: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if data point has expired"""
        if self.processing_deadline is None:
            return False
        return time.time() > self.processing_deadline
    
    def age_seconds(self) -> float:
        """Get age of data point in seconds"""
        return time.time() - self.timestamp

@dataclass
class FusionTask:
    """Fusion task for distributed processing"""
    task_id: str
    priority: FusionPriority
    data_points: List[StreamingDataPoint]
    fusion_algorithm: str
    parameters: Dict[str, Any]
    deadline: Optional[float] = None
    assigned_node: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if task has expired"""
        if self.deadline is None:
            return False
        return time.time() > self.deadline

@dataclass
class ProcessingNode:
    """Distributed processing node"""
    node_id: str
    endpoint: str
    capabilities: List[str]
    current_load: float = 0.0
    max_capacity: int = 100
    last_heartbeat: float = field(default_factory=time.time)
    performance_score: float = 1.0
    
    def is_alive(self, timeout: float = 30.0) -> bool:
        """Check if node is alive"""
        return (time.time() - self.last_heartbeat) < timeout
    
    def available_capacity(self) -> float:
        """Get available processing capacity"""
        return max(0.0, self.max_capacity - self.current_load)

class StreamingBuffer:
    """High-performance streaming buffer with automatic memory management"""
    
    def __init__(self, max_size: int = 10000, max_age_seconds: float = 300.0):
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.buffer: deque = deque(maxlen=max_size)
        self.sensor_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        
        # Performance tracking
        self.total_points_processed = 0
        self.buffer_overflows = 0
        self.expired_points_removed = 0
        
        # Start cleanup task
        self._cleanup_task = None
        self._running = True
    
    def add_data_point(self, data_point: StreamingDataPoint) -> bool:
        """Add data point to buffer"""
        try:
            with self._lock:
                # Check if expired
                if data_point.is_expired():
                    self.expired_points_removed += 1
                    return False
                
                # Add to main buffer
                if len(self.buffer) >= self.max_size:
                    self.buffer_overflows += 1
                
                self.buffer.append(data_point)
                
                # Add to sensor-specific buffer
                self.sensor_buffers[data_point.sensor_id].append(data_point)
                
                self.total_points_processed += 1
                return True
                
        except Exception as e:
            log.error(f"Failed to add data point: {e}")
            return False
    
    def get_recent_data(self, 
                       sensor_ids: Optional[List[str]] = None,
                       max_age_seconds: Optional[float] = None,
                       max_count: Optional[int] = None) -> List[StreamingDataPoint]:
        """Get recent data points"""
        try:
            with self._lock:
                current_time = time.time()
                max_age = max_age_seconds or self.max_age_seconds
                
                # Filter data points
                filtered_points = []
                
                if sensor_ids is None:
                    # Get from main buffer
                    source_buffer = self.buffer
                else:
                    # Get from sensor-specific buffers
                    source_buffer = []
                    for sensor_id in sensor_ids:
                        if sensor_id in self.sensor_buffers:
                            source_buffer.extend(self.sensor_buffers[sensor_id])
                
                for point in reversed(source_buffer):
                    # Check age
                    if (current_time - point.timestamp) > max_age:
                        continue
                    
                    # Check if expired
                    if point.is_expired():
                        continue
                    
                    filtered_points.append(point)
                    
                    # Check count limit
                    if max_count and len(filtered_points) >= max_count:
                        break
                
                return list(reversed(filtered_points))  # Restore chronological order
                
        except Exception as e:
            log.error(f"Failed to get recent data: {e}")
            return []
    
    def cleanup_expired_data(self):
        """Remove expired data points"""
        try:
            with self._lock:
                current_time = time.time()
                removed_count = 0
                
                # Clean main buffer
                while self.buffer and (current_time - self.buffer[0].timestamp) > self.max_age_seconds:
                    self.buffer.popleft()
                    removed_count += 1
                
                # Clean sensor buffers
                for sensor_id in list(self.sensor_buffers.keys()):
                    sensor_buffer = self.sensor_buffers[sensor_id]
                    while sensor_buffer and (current_time - sensor_buffer[0].timestamp) > self.max_age_seconds:
                        sensor_buffer.popleft()
                        removed_count += 1
                    
                    # Remove empty buffers
                    if not sensor_buffer:
                        del self.sensor_buffers[sensor_id]
                
                if removed_count > 0:
                    log.debug(f"Cleaned up {removed_count} expired data points")
                    
        except Exception as e:
            log.error(f"Data cleanup failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            return {
                "total_points_processed": self.total_points_processed,
                "current_buffer_size": len(self.buffer),
                "sensor_buffer_count": len(self.sensor_buffers),
                "buffer_overflows": self.buffer_overflows,
                "expired_points_removed": self.expired_points_removed,
                "max_size": self.max_size,
                "max_age_seconds": self.max_age_seconds
            }

class StreamingFusionProcessor:
    """High-performance streaming fusion processor"""
    
    def __init__(self, 
                 processing_mode: StreamProcessingMode = StreamProcessingMode.REAL_TIME,
                 batch_size: int = 100,
                 batch_timeout_ms: int = 100,
                 max_workers: int = None):
        
        self.processing_mode = processing_mode
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        # Threading setup
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count() or 1)
        
        # Streaming components
        self.streaming_buffer = StreamingBuffer()
        self.fusion_algorithms: Dict[str, Callable] = {}
        self.processing_queue = asyncio.Queue(maxsize=10000)
        self.result_callbacks: Dict[str, Callable] = {}
        
        # Performance tracking
        self.processing_stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "average_latency_ms": 0.0,
            "throughput_per_second": 0.0,
            "last_reset": time.time()
        }
        
        # Control flags
        self._running = False
        self._processing_tasks = []
        
        log.info(f"Streaming fusion processor initialized: mode={processing_mode.value}, "
                f"batch_size={batch_size}, workers={self.max_workers}")
    
    def register_fusion_algorithm(self, name: str, algorithm: Callable):
        """Register fusion algorithm"""
        self.fusion_algorithms[name] = algorithm
        log.info(f"Registered fusion algorithm: {name}")
    
    def register_result_callback(self, name: str, callback: Callable):
        """Register result callback"""
        self.result_callbacks[name] = callback
        log.info(f"Registered result callback: {name}")
    
    async def start_processing(self):
        """Start streaming processing"""
        if self._running:
            log.warning("Processor already running")
            return
        
        self._running = True
        
        # Start processing tasks based on mode
        if self.processing_mode == StreamProcessingMode.REAL_TIME:
            self._processing_tasks.append(asyncio.create_task(self._real_time_processor()))
        elif self.processing_mode == StreamProcessingMode.BATCH_STREAMING:
            self._processing_tasks.append(asyncio.create_task(self._batch_streaming_processor()))
        elif self.processing_mode == StreamProcessingMode.MICRO_BATCH:
            self._processing_tasks.append(asyncio.create_task(self._micro_batch_processor()))
        elif self.processing_mode == StreamProcessingMode.CONTINUOUS:
            self._processing_tasks.append(asyncio.create_task(self._continuous_processor()))
        
        # Start cleanup task
        self._processing_tasks.append(asyncio.create_task(self._cleanup_task()))
        
        log.info("Streaming processing started")
    
    async def stop_processing(self):
        """Stop streaming processing"""
        self._running = False
        
        # Cancel processing tasks
        for task in self._processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        log.info("Streaming processing stopped")
    
    async def add_data_stream(self, data_stream: AsyncIterator[StreamingDataPoint]):
        """Add data stream for processing"""
        try:
            async for data_point in data_stream:
                # Add to buffer
                if self.streaming_buffer.add_data_point(data_point):
                    # Create fusion task if needed
                    await self._maybe_create_fusion_task(data_point)
                
        except Exception as e:
            log.error(f"Data stream processing failed: {e}")
    
    async def _maybe_create_fusion_task(self, data_point: StreamingDataPoint):
        """Create fusion task if conditions are met"""
        try:
            # Get related data points
            related_points = self.streaming_buffer.get_recent_data(
                max_age_seconds=1.0,  # 1 second window
                max_count=100
            )
            
            if len(related_points) >= 2:  # Need at least 2 points for fusion
                # Determine fusion algorithm
                fusion_algorithm = self._select_fusion_algorithm(related_points)
                
                # Create fusion task
                task = FusionTask(
                    task_id=f"fusion_{int(time.time() * 1000)}_{len(related_points)}",
                    priority=FusionPriority.NORMAL,
                    data_points=related_points,
                    fusion_algorithm=fusion_algorithm,
                    parameters={},
                    deadline=time.time() + 5.0  # 5 second deadline
                )
                
                # Add to processing queue
                try:
                    await self.processing_queue.put(task)
                except asyncio.QueueFull:
                    log.warning("Processing queue full, dropping task")
                    
        except Exception as e:
            log.error(f"Fusion task creation failed: {e}")
    
    def _select_fusion_algorithm(self, data_points: List[StreamingDataPoint]) -> str:
        """Select appropriate fusion algorithm"""
        # Simple selection logic - can be enhanced
        sensor_count = len(set(point.sensor_id for point in data_points))
        
        if sensor_count >= 3:
            return "advanced_bayesian"
        elif sensor_count == 2:
            return "bayesian_fusion"
        else:
            return "simple_average"
    
    async def _real_time_processor(self):
        """Real-time processing loop"""
        while self._running:
            try:
                # Get task with timeout
                task = await asyncio.wait_for(self.processing_queue.get(), timeout=0.1)
                
                # Process immediately
                await self._process_fusion_task(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Real-time processing error: {e}")
                await asyncio.sleep(0.01)
    
    async def _batch_streaming_processor(self):
        """Batch streaming processing loop"""
        batch = []
        last_batch_time = time.time()
        
        while self._running:
            try:
                # Try to get task with timeout
                try:
                    task = await asyncio.wait_for(self.processing_queue.get(), timeout=0.01)
                    batch.append(task)
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if conditions met
                current_time = time.time()
                batch_timeout = (current_time - last_batch_time) * 1000 > self.batch_timeout_ms
                
                if len(batch) >= self.batch_size or (batch and batch_timeout):
                    await self._process_fusion_batch(batch)
                    batch.clear()
                    last_batch_time = current_time
                
                await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                
            except Exception as e:
                log.error(f"Batch streaming processing error: {e}")
                await asyncio.sleep(0.01)
    
    async def _micro_batch_processor(self):
        """Micro-batch processing loop"""
        micro_batch_size = min(10, self.batch_size)
        
        while self._running:
            try:
                batch = []
                
                # Collect micro-batch
                for _ in range(micro_batch_size):
                    try:
                        task = await asyncio.wait_for(self.processing_queue.get(), timeout=0.005)
                        batch.append(task)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    await self._process_fusion_batch(batch)
                
                await asyncio.sleep(0.001)
                
            except Exception as e:
                log.error(f"Micro-batch processing error: {e}")
                await asyncio.sleep(0.01)
    
    async def _continuous_processor(self):
        """Continuous processing loop"""
        while self._running:
            try:
                # Process all available tasks
                tasks_to_process = []
                
                # Drain queue up to batch size
                for _ in range(self.batch_size):
                    try:
                        task = self.processing_queue.get_nowait()
                        tasks_to_process.append(task)
                    except asyncio.QueueEmpty:
                        break
                
                if tasks_to_process:
                    # Process in parallel
                    await asyncio.gather(
                        *[self._process_fusion_task(task) for task in tasks_to_process],
                        return_exceptions=True
                    )
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                log.error(f"Continuous processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_fusion_task(self, task: FusionTask):
        """Process individual fusion task"""
        start_time = time.time()
        
        try:
            # Check if task expired
            if task.is_expired():
                log.debug(f"Task {task.task_id} expired, skipping")
                return
            
            # Get fusion algorithm
            if task.fusion_algorithm not in self.fusion_algorithms:
                log.warning(f"Unknown fusion algorithm: {task.fusion_algorithm}")
                return
            
            algorithm = self.fusion_algorithms[task.fusion_algorithm]
            
            # Prepare data for fusion
            fusion_data = self._prepare_fusion_data(task.data_points)
            
            # Execute fusion in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                algorithm,
                fusion_data,
                task.parameters
            )
            
            # Process result
            await self._handle_fusion_result(task, result, start_time)
            
            # Update statistics
            self.processing_stats["tasks_processed"] += 1
            
        except Exception as e:
            log.error(f"Fusion task processing failed: {e}")
            self.processing_stats["tasks_failed"] += 1
    
    async def _process_fusion_batch(self, batch: List[FusionTask]):
        """Process batch of fusion tasks"""
        if not batch:
            return
        
        try:
            # Process batch in parallel
            await asyncio.gather(
                *[self._process_fusion_task(task) for task in batch],
                return_exceptions=True
            )
            
            log.debug(f"Processed batch of {len(batch)} tasks")
            
        except Exception as e:
            log.error(f"Batch processing failed: {e}")
    
    def _prepare_fusion_data(self, data_points: List[StreamingDataPoint]) -> Dict[str, Any]:
        """Prepare data for fusion algorithm"""
        # Group by sensor
        sensor_data = defaultdict(list)
        
        for point in data_points:
            sensor_data[point.sensor_id].append({
                "timestamp": point.timestamp,
                "value": point.value,
                "quality": point.quality,
                "metadata": point.metadata
            })
        
        return {
            "sensor_data": dict(sensor_data),
            "data_points": len(data_points),
            "time_span": max(p.timestamp for p in data_points) - min(p.timestamp for p in data_points),
            "sensors": list(sensor_data.keys())
        }
    
    async def _handle_fusion_result(self, task: FusionTask, result: Any, start_time: float):
        """Handle fusion result"""
        try:
            processing_time = (time.time() - start_time) * 1000
            
            # Create result object
            fusion_result = {
                "task_id": task.task_id,
                "result": result,
                "processing_time_ms": processing_time,
                "data_points_fused": len(task.data_points),
                "sensors_involved": len(set(p.sensor_id for p in task.data_points)),
                "algorithm_used": task.fusion_algorithm,
                "timestamp": time.time()
            }
            
            # Call result callbacks
            for callback_name, callback in self.result_callbacks.items():
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(fusion_result)
                    else:
                        callback(fusion_result)
                except Exception as e:
                    log.error(f"Result callback {callback_name} failed: {e}")
            
            # Update latency statistics
            current_avg = self.processing_stats["average_latency_ms"]
            task_count = self.processing_stats["tasks_processed"]
            
            if task_count > 0:
                self.processing_stats["average_latency_ms"] = (
                    (current_avg * (task_count - 1) + processing_time) / task_count
                )
            
        except Exception as e:
            log.error(f"Result handling failed: {e}")
    
    async def _cleanup_task(self):
        """Periodic cleanup task"""
        while self._running:
            try:
                # Clean up expired data
                self.streaming_buffer.cleanup_expired_data()
                
                # Update throughput statistics
                current_time = time.time()
                time_elapsed = current_time - self.processing_stats["last_reset"]
                
                if time_elapsed >= 60.0:  # Update every minute
                    tasks_processed = self.processing_stats["tasks_processed"]
                    self.processing_stats["throughput_per_second"] = tasks_processed / time_elapsed
                    
                    # Reset counters
                    self.processing_stats["tasks_processed"] = 0
                    self.processing_stats["tasks_failed"] = 0
                    self.processing_stats["last_reset"] = current_time
                
                # Force garbage collection periodically
                if int(current_time) % 30 == 0:  # Every 30 seconds
                    gc.collect()
                
                await asyncio.sleep(5.0)  # Run every 5 seconds
                
            except Exception as e:
                log.error(f"Cleanup task failed: {e}")
                await asyncio.sleep(10.0)
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        buffer_stats = self.streaming_buffer.get_statistics()
        
        return {
            "processing_stats": self.processing_stats.copy(),
            "buffer_stats": buffer_stats,
            "queue_size": self.processing_queue.qsize(),
            "processing_mode": self.processing_mode.value,
            "max_workers": self.max_workers,
            "algorithms_registered": list(self.fusion_algorithms.keys()),
            "callbacks_registered": list(self.result_callbacks.keys())
        }

class DistributedCoordinator:
    """Distributed coordination for massive-scale fusion operations"""
    
    def __init__(self, 
                 node_id: str,
                 coordination_strategy: CoordinationStrategy = CoordinationStrategy.HIERARCHICAL):
        
        self.node_id = node_id
        self.coordination_strategy = coordination_strategy
        
        # Node management
        self.processing_nodes: Dict[str, ProcessingNode] = {}
        self.local_processor = StreamingFusionProcessor()
        
        # Task distribution
        self.task_queue = asyncio.Queue(maxsize=50000)
        self.task_assignments: Dict[str, str] = {}  # task_id -> node_id
        self.node_loads: Dict[str, float] = defaultdict(float)
        
        # Communication setup
        self.zmq_context = None
        self.communication_endpoints: Dict[str, str] = {}
        
        if ZMQ_AVAILABLE:
            self.zmq_context = zmq.asyncio.Context()
        
        # Coordination state
        self.leader_node: Optional[str] = None
        self.is_leader = False
        self.heartbeat_interval = 10.0  # seconds
        
        # Performance tracking
        self.coordination_stats = {
            "tasks_distributed": 0,
            "nodes_coordinated": 0,
            "load_balancing_decisions": 0,
            "failover_events": 0
        }
        
        log.info(f"Distributed coordinator initialized: node={node_id}, "
                f"strategy={coordination_strategy.value}")
    
    async def start_coordination(self):
        """Start distributed coordination"""
        # Start local processor
        await self.local_processor.start_processing()
        
        # Start coordination tasks
        coordination_tasks = [
            asyncio.create_task(self._heartbeat_task()),
            asyncio.create_task(self._task_distribution_task()),
            asyncio.create_task(self._load_balancing_task()),
            asyncio.create_task(self._failure_detection_task())
        ]
        
        # Strategy-specific initialization
        if self.coordination_strategy == CoordinationStrategy.HIERARCHICAL:
            coordination_tasks.append(asyncio.create_task(self._leader_election_task()))
        elif self.coordination_strategy == CoordinationStrategy.GOSSIP:
            coordination_tasks.append(asyncio.create_task(self._gossip_protocol_task()))
        
        log.info("Distributed coordination started")
        return coordination_tasks
    
    async def add_processing_node(self, node: ProcessingNode):
        """Add processing node to coordination"""
        self.processing_nodes[node.node_id] = node
        self.node_loads[node.node_id] = 0.0
        
        # Setup communication if ZMQ available
        if self.zmq_context and node.endpoint:
            self.communication_endpoints[node.node_id] = node.endpoint
        
        self.coordination_stats["nodes_coordinated"] = len(self.processing_nodes)
        log.info(f"Added processing node: {node.node_id}")
    
    async def submit_fusion_task(self, task: FusionTask) -> bool:
        """Submit fusion task for distributed processing"""
        try:
            # Select best node for task
            selected_node = await self._select_processing_node(task)
            
            if selected_node:
                # Assign task
                task.assigned_node = selected_node
                self.task_assignments[task.task_id] = selected_node
                
                # Update node load
                self.node_loads[selected_node] += 1.0
                
                # Send task to node
                success = await self._send_task_to_node(task, selected_node)
                
                if success:
                    self.coordination_stats["tasks_distributed"] += 1
                    return True
                else:
                    # Remove assignment on failure
                    del self.task_assignments[task.task_id]
                    self.node_loads[selected_node] -= 1.0
                    return False
            else:
                # No available nodes - process locally
                await self.local_processor.processing_queue.put(task)
                return True
                
        except Exception as e:
            log.error(f"Task submission failed: {e}")
            return False
    
    async def _select_processing_node(self, task: FusionTask) -> Optional[str]:
        """Select best processing node for task"""
        try:
            if not self.processing_nodes:
                return None
            
            # Filter alive nodes
            alive_nodes = [
                node for node in self.processing_nodes.values()
                if node.is_alive()
            ]
            
            if not alive_nodes:
                return None
            
            # Selection strategy based on coordination approach
            if self.coordination_strategy == CoordinationStrategy.CENTRALIZED:
                return self._centralized_node_selection(alive_nodes, task)
            elif self.coordination_strategy == CoordinationStrategy.DECENTRALIZED:
                return self._decentralized_node_selection(alive_nodes, task)
            elif self.coordination_strategy == CoordinationStrategy.HIERARCHICAL:
                return self._hierarchical_node_selection(alive_nodes, task)
            else:
                # Default to load-based selection
                return self._load_based_selection(alive_nodes)
                
        except Exception as e:
            log.error(f"Node selection failed: {e}")
            return None
    
    def _centralized_node_selection(self, nodes: List[ProcessingNode], task: FusionTask) -> str:
        """Centralized node selection"""
        # Select node with lowest load
        best_node = min(nodes, key=lambda n: self.node_loads[n.node_id])
        return best_node.node_id
    
    def _decentralized_node_selection(self, nodes: List[ProcessingNode], task: FusionTask) -> str:
        """Decentralized node selection using random choice with weights"""
        # Weight by available capacity
        weights = []
        node_ids = []
        
        for node in nodes:
            available_capacity = node.available_capacity()
            if available_capacity > 0:
                weights.append(available_capacity * node.performance_score)
                node_ids.append(node.node_id)
        
        if not weights:
            return nodes[0].node_id  # Fallback
        
        # Weighted random selection
        total_weight = sum(weights)
        import random
        r = random.uniform(0, total_weight)
        
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return node_ids[i]
        
        return node_ids[-1]  # Fallback
    
    def _hierarchical_node_selection(self, nodes: List[ProcessingNode], task: FusionTask) -> str:
        """Hierarchical node selection"""
        if self.is_leader:
            # Leader makes centralized decision
            return self._centralized_node_selection(nodes, task)
        else:
            # Non-leader uses local information
            return self._load_based_selection(nodes)
    
    def _load_based_selection(self, nodes: List[ProcessingNode]) -> str:
        """Simple load-based selection"""
        return min(nodes, key=lambda n: self.node_loads[n.node_id]).node_id
    
    async def _send_task_to_node(self, task: FusionTask, node_id: str) -> bool:
        """Send task to processing node"""
        try:
            if node_id == self.node_id:
                # Local processing
                await self.local_processor.processing_queue.put(task)
                return True
            
            elif self.zmq_context and node_id in self.communication_endpoints:
                # ZMQ communication
                return await self._send_task_via_zmq(task, node_id)
            
            else:
                # Fallback - process locally
                log.warning(f"Cannot send task to {node_id}, processing locally")
                await self.local_processor.processing_queue.put(task)
                return True
                
        except Exception as e:
            log.error(f"Task sending failed: {e}")
            return False
    
    async def _send_task_via_zmq(self, task: FusionTask, node_id: str) -> bool:
        """Send task via ZeroMQ"""
        try:
            endpoint = self.communication_endpoints[node_id]
            
            # Create socket
            socket = self.zmq_context.socket(zmq.PUSH)
            socket.connect(endpoint)
            
            # Serialize task
            task_data = {
                "task_id": task.task_id,
                "priority": task.priority.value,
                "fusion_algorithm": task.fusion_algorithm,
                "parameters": task.parameters,
                "deadline": task.deadline,
                "data_points": [
                    {
                        "sensor_id": dp.sensor_id,
                        "timestamp": dp.timestamp,
                        "value": dp.value.tolist() if isinstance(dp.value, np.ndarray) else dp.value,
                        "quality": dp.quality,
                        "metadata": dp.metadata
                    }
                    for dp in task.data_points
                ]
            }
            
            # Send task
            await socket.send_json(task_data)
            socket.close()
            
            return True
            
        except Exception as e:
            log.error(f"ZMQ task sending failed: {e}")
            return False
    
    async def _heartbeat_task(self):
        """Periodic heartbeat task"""
        while True:
            try:
                # Update own heartbeat
                if self.node_id in self.processing_nodes:
                    self.processing_nodes[self.node_id].last_heartbeat = time.time()
                
                # Send heartbeat to other nodes
                await self._broadcast_heartbeat()
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                log.error(f"Heartbeat task failed: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _broadcast_heartbeat(self):
        """Broadcast heartbeat to other nodes"""
        # Implementation would depend on communication mechanism
        log.debug("Broadcasting heartbeat")
    
    async def _task_distribution_task(self):
        """Task distribution management"""
        while True:
            try:
                # Check for completed tasks
                completed_tasks = []
                
                for task_id, node_id in list(self.task_assignments.items()):
                    # In production, would check actual task status
                    # For now, simulate task completion
                    if time.time() % 60 < 1:  # Simulate some tasks completing
                        completed_tasks.append(task_id)
                
                # Clean up completed tasks
                for task_id in completed_tasks:
                    if task_id in self.task_assignments:
                        node_id = self.task_assignments[task_id]
                        del self.task_assignments[task_id]
                        self.node_loads[node_id] = max(0.0, self.node_loads[node_id] - 1.0)
                
                await asyncio.sleep(5.0)
                
            except Exception as e:
                log.error(f"Task distribution management failed: {e}")
                await asyncio.sleep(10.0)
    
    async def _load_balancing_task(self):
        """Load balancing management"""
        while True:
            try:
                # Check for load imbalances
                if len(self.processing_nodes) > 1:
                    loads = list(self.node_loads.values())
                    avg_load = sum(loads) / len(loads)
                    max_load = max(loads)
                    
                    # Rebalance if significant imbalance
                    if max_load > avg_load * 1.5:
                        await self._rebalance_tasks()
                        self.coordination_stats["load_balancing_decisions"] += 1
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                log.error(f"Load balancing failed: {e}")
                await asyncio.sleep(60.0)
    
    async def _rebalance_tasks(self):
        """Rebalance tasks across nodes"""
        log.info("Performing load rebalancing")
        # Implementation would move tasks from overloaded to underloaded nodes
    
    async def _failure_detection_task(self):
        """Detect and handle node failures"""
        while True:
            try:
                failed_nodes = []
                
                for node_id, node in self.processing_nodes.items():
                    if not node.is_alive():
                        failed_nodes.append(node_id)
                
                # Handle failures
                for node_id in failed_nodes:
                    await self._handle_node_failure(node_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                log.error(f"Failure detection failed: {e}")
                await asyncio.sleep(self.heartbeat_interval * 2)
    
    async def _handle_node_failure(self, node_id: str):
        """Handle node failure"""
        log.warning(f"Node failure detected: {node_id}")
        
        # Reassign tasks from failed node
        failed_tasks = [
            task_id for task_id, assigned_node in self.task_assignments.items()
            if assigned_node == node_id
        ]
        
        for task_id in failed_tasks:
            # Remove assignment
            del self.task_assignments[task_id]
            self.node_loads[node_id] = 0.0
            
            # Would reassign task to another node in production
            log.info(f"Task {task_id} needs reassignment due to node failure")
        
        self.coordination_stats["failover_events"] += 1
    
    async def _leader_election_task(self):
        """Leader election for hierarchical coordination"""
        while True:
            try:
                if not self.leader_node or self.leader_node not in self.processing_nodes:
                    # Elect new leader (simple: lowest node_id)
                    alive_nodes = [
                        node_id for node_id, node in self.processing_nodes.items()
                        if node.is_alive()
                    ]
                    
                    if alive_nodes:
                        self.leader_node = min(alive_nodes)
                        self.is_leader = (self.leader_node == self.node_id)
                        
                        log.info(f"Leader elected: {self.leader_node}")
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                log.error(f"Leader election failed: {e}")
                await asyncio.sleep(60.0)
    
    async def _gossip_protocol_task(self):
        """Gossip protocol for decentralized coordination"""
        while True:
            try:
                # Exchange information with random subset of nodes
                # Implementation would depend on specific gossip protocol
                log.debug("Gossip protocol exchange")
                
                await asyncio.sleep(5.0)
                
            except Exception as e:
                log.error(f"Gossip protocol failed: {e}")
                await asyncio.sleep(10.0)
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination statistics"""
        return {
            "coordination_stats": self.coordination_stats.copy(),
            "node_count": len(self.processing_nodes),
            "active_tasks": len(self.task_assignments),
            "node_loads": dict(self.node_loads),
            "leader_node": self.leader_node,
            "is_leader": self.is_leader,
            "coordination_strategy": self.coordination_strategy.value,
            "processor_stats": self.local_processor.get_performance_statistics()
        }

# Utility functions for algorithm registration
def simple_average_fusion(fusion_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Simple averaging fusion algorithm"""
    sensor_data = fusion_data["sensor_data"]
    
    all_values = []
    for sensor_id, points in sensor_data.items():
        for point in points:
            if isinstance(point["value"], (int, float)):
                all_values.append(point["value"])
            elif isinstance(point["value"], list):
                all_values.extend(point["value"])
    
    if not all_values:
        return {"fused_value": 0.0, "confidence": 0.0}
    
    fused_value = sum(all_values) / len(all_values)
    confidence = min(1.0, len(all_values) / 10.0)  # Higher confidence with more data
    
    return {
        "fused_value": fused_value,
        "confidence": confidence,
        "algorithm": "simple_average",
        "data_points_used": len(all_values)
    }

async def create_high_performance_fusion_system(
    node_id: str,
    processing_mode: StreamProcessingMode = StreamProcessingMode.REAL_TIME,
    coordination_strategy: CoordinationStrategy = CoordinationStrategy.HIERARCHICAL
) -> Tuple[StreamingFusionProcessor, DistributedCoordinator]:
    """Create high-performance streaming fusion system"""
    
    # Use uvloop if available for better performance
    if UVLOOP_AVAILABLE:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    # Create processor
    processor = StreamingFusionProcessor(
        processing_mode=processing_mode,
        batch_size=200 if processing_mode == StreamProcessingMode.BATCH_STREAMING else 50,
        max_workers=min(64, mp.cpu_count() * 4)
    )
    
    # Register basic algorithms
    processor.register_fusion_algorithm("simple_average", simple_average_fusion)
    
    # Create coordinator
    coordinator = DistributedCoordinator(
        node_id=node_id,
        coordination_strategy=coordination_strategy
    )
    
    # Start systems
    await processor.start_processing()
    await coordinator.start_coordination()
    
    log.info(f"High-performance fusion system created for node {node_id}")
    
    return processor, coordinator
