#!/usr/bin/env python3
"""
Memory Synchronization Protocol for Neural Mesh
Event-driven updates, conflict resolution, and consistency management
"""

import asyncio
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from collections import defaultdict

# Message bus imports
try:
    import nats
    from nats.aio.client import Client as NATS
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False

log = logging.getLogger("memory-sync-protocol")

class SyncEventType(Enum):
    """Types of synchronization events"""
    MEMORY_CREATE = "memory_create"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    MEMORY_MERGE = "memory_merge"
    BATCH_SYNC = "batch_sync"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"

class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies"""
    LAST_WRITER_WINS = "last_writer_wins"
    VECTOR_CLOCK = "vector_clock"
    CRDT_MERGE = "crdt_merge"
    HUMAN_INTERVENTION = "human_intervention"
    AGENT_CONSENSUS = "agent_consensus"

@dataclass
class VectorClock:
    """Vector clock for distributed synchronization"""
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, agent_id: str):
        """Increment clock for agent"""
        self.clocks[agent_id] = self.clocks.get(agent_id, 0) + 1
    
    def update(self, other_clock: 'VectorClock'):
        """Update with another vector clock"""
        for agent_id, clock_value in other_clock.clocks.items():
            self.clocks[agent_id] = max(self.clocks.get(agent_id, 0), clock_value)
    
    def compare(self, other_clock: 'VectorClock') -> str:
        """Compare with another vector clock"""
        all_agents = set(self.clocks.keys()) | set(other_clock.clocks.keys())
        
        self_greater = False
        other_greater = False
        
        for agent_id in all_agents:
            self_value = self.clocks.get(agent_id, 0)
            other_value = other_clock.clocks.get(agent_id, 0)
            
            if self_value > other_value:
                self_greater = True
            elif self_value < other_value:
                other_greater = True
        
        if self_greater and not other_greater:
            return "greater"
        elif other_greater and not self_greater:
            return "less"
        elif not self_greater and not other_greater:
            return "equal"
        else:
            return "concurrent"

@dataclass
class SyncEvent:
    """Memory synchronization event"""
    event_id: str
    event_type: SyncEventType
    memory_id: str
    agent_id: str
    data: Dict[str, Any]
    vector_clock: VectorClock
    timestamp: float = field(default_factory=time.time)
    partition_key: str = ""
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class ConflictRecord:
    """Record of memory conflict"""
    conflict_id: str
    memory_id: str
    conflicting_agents: List[str]
    conflict_type: str
    resolution_strategy: ConflictResolutionStrategy
    original_versions: Dict[str, Any]
    resolved_version: Optional[Dict[str, Any]] = None
    resolved_at: Optional[float] = None
    resolution_confidence: float = 0.0

@dataclass
class BatchSyncOperation:
    """Batch synchronization operation"""
    batch_id: str
    operations: List[SyncEvent]
    target_agents: List[str]
    batch_size: int
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    success_count: int = 0
    failure_count: int = 0

class MemorySynchronizationProtocol:
    """Advanced memory synchronization with conflict resolution"""
    
    def __init__(self):
        # Message bus clients
        self.nats_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
        self.rabbitmq_connection = None
        
        # Synchronization state
        self.vector_clocks: Dict[str, VectorClock] = defaultdict(VectorClock)
        self.pending_events: Dict[str, SyncEvent] = {}
        self.conflict_records: Dict[str, ConflictRecord] = {}
        self.batch_operations: Dict[str, BatchSyncOperation] = {}
        
        # Configuration
        self.batch_size = 100
        self.batch_timeout = 5.0  # seconds
        self.conflict_resolution_strategy = ConflictResolutionStrategy.VECTOR_CLOCK
        
        # Event queues
        self.sync_event_queue = asyncio.Queue()
        self.conflict_resolution_queue = asyncio.Queue()
        self.batch_sync_queue = asyncio.Queue()
        
        # Initialize (defer async initialization)
        self._initialized = False
    
    async def _initialize_async(self):
        """Initialize synchronization protocol"""
        if self._initialized:
            return
            
        try:
            await self._initialize_message_bus()
            await self._start_sync_workers()
            
            self._initialized = True
            log.info("✅ Memory synchronization protocol initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize sync protocol: {e}")
    
    async def ensure_initialized(self):
        """Ensure the system is initialized"""
        if not self._initialized:
            await self._initialize_async()
    
    async def _initialize_message_bus(self):
        """Initialize message bus for event-driven updates"""
        
        # Try NATS first (preferred for AgentForge)
        if NATS_AVAILABLE:
            try:
                nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
                self.nats_client = await nats.connect(nats_url)
                
                # Subscribe to memory sync events
                await self.nats_client.subscribe(
                    "neural_mesh.memory.sync.*",
                    cb=self._handle_nats_sync_event
                )
                
                log.info("✅ NATS message bus initialized for memory sync")
                return
                
            except Exception as e:
                log.error(f"Failed to initialize NATS: {e}")
        
        # Try Kafka as fallback
        if KAFKA_AVAILABLE:
            try:
                kafka_servers = os.getenv("KAFKA_SERVERS", "localhost:9092").split(",")
                
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=kafka_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None
                )
                
                self.kafka_consumer = KafkaConsumer(
                    'neural_mesh_memory_sync',
                    bootstrap_servers=kafka_servers,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    group_id='neural_mesh_sync_group'
                )
                
                # Start Kafka consumer task
                asyncio.create_task(self._kafka_consumer_worker())
                
                log.info("✅ Kafka message bus initialized for memory sync")
                return
                
            except Exception as e:
                log.error(f"Failed to initialize Kafka: {e}")
        
        # Try RabbitMQ as final fallback
        if RABBITMQ_AVAILABLE:
            try:
                rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://localhost:5672")
                self.rabbitmq_connection = await pika.SelectConnection(
                    pika.URLParameters(rabbitmq_url)
                )
                
                log.info("✅ RabbitMQ message bus initialized for memory sync")
                return
                
            except Exception as e:
                log.error(f"Failed to initialize RabbitMQ: {e}")
        
        log.warning("No message bus available, using local event processing")
    
    async def _start_sync_workers(self):
        """Start synchronization worker tasks"""
        
        # Event processing worker
        asyncio.create_task(self._sync_event_worker())
        
        # Conflict resolution worker
        asyncio.create_task(self._conflict_resolution_worker())
        
        # Batch synchronization worker
        asyncio.create_task(self._batch_sync_worker())
        
        # Failure handling worker
        asyncio.create_task(self._failure_handling_worker())
        
        log.info("✅ Synchronization workers started")
    
    async def publish_memory_event(
        self,
        event_type: SyncEventType,
        memory_id: str,
        agent_id: str,
        data: Dict[str, Any],
        target_agents: List[str] = None,
        consistency_level: str = "eventual"
    ) -> str:
        """Publish memory synchronization event"""
        
        try:
            # Increment vector clock
            self.vector_clocks[agent_id].increment(agent_id)
            
            # Create sync event
            event = SyncEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                memory_id=memory_id,
                agent_id=agent_id,
                data=data,
                vector_clock=self.vector_clocks[agent_id],
                partition_key=self._get_partition_key(memory_id, agent_id)
            )
            
            # Publish via message bus
            if self.nats_client:
                await self._publish_nats_event(event, target_agents)
            elif self.kafka_producer:
                await self._publish_kafka_event(event, target_agents)
            elif self.rabbitmq_connection:
                await self._publish_rabbitmq_event(event, target_agents)
            else:
                # Local processing
                await self.sync_event_queue.put(event)
            
            log.debug(f"Published sync event {event.event_id}")
            return event.event_id
            
        except Exception as e:
            log.error(f"Error publishing memory event: {e}")
            raise
    
    async def _publish_nats_event(self, event: SyncEvent, target_agents: List[str] = None):
        """Publish event via NATS"""
        
        event_data = asdict(event)
        event_data["vector_clock"] = asdict(event.vector_clock)
        
        if target_agents:
            # Send to specific agents
            for agent_id in target_agents:
                subject = f"neural_mesh.memory.sync.{agent_id}"
                await self.nats_client.publish(subject, json.dumps(event_data).encode())
        else:
            # Broadcast to all agents
            subject = f"neural_mesh.memory.sync.broadcast"
            await self.nats_client.publish(subject, json.dumps(event_data).encode())
    
    async def _handle_nats_sync_event(self, msg):
        """Handle incoming NATS sync event"""
        
        try:
            event_data = json.loads(msg.data.decode())
            
            # Reconstruct event
            event = SyncEvent(
                event_id=event_data["event_id"],
                event_type=SyncEventType(event_data["event_type"]),
                memory_id=event_data["memory_id"],
                agent_id=event_data["agent_id"],
                data=event_data["data"],
                vector_clock=VectorClock(**event_data["vector_clock"]),
                timestamp=event_data["timestamp"],
                partition_key=event_data["partition_key"]
            )
            
            # Queue for processing
            await self.sync_event_queue.put(event)
            
        except Exception as e:
            log.error(f"Error handling NATS sync event: {e}")
    
    async def _sync_event_worker(self):
        """Worker for processing synchronization events"""
        
        while True:
            try:
                # Get event from queue
                event = await self.sync_event_queue.get()
                
                # Update vector clock
                self.vector_clocks[event.agent_id].update(event.vector_clock)
                
                # Check for conflicts
                conflict = await self._detect_memory_conflict(event)
                
                if conflict:
                    # Queue for conflict resolution
                    await self.conflict_resolution_queue.put((event, conflict))
                else:
                    # Apply event directly
                    await self._apply_sync_event(event)
                
                # Mark as done
                self.sync_event_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in sync event worker: {e}")
                await asyncio.sleep(1)
    
    async def _conflict_resolution_worker(self):
        """Worker for resolving memory conflicts"""
        
        while True:
            try:
                # Get conflict from queue
                event, conflict = await self.conflict_resolution_queue.get()
                
                # Resolve conflict based on strategy
                resolved_event = await self._resolve_memory_conflict(event, conflict)
                
                if resolved_event:
                    # Apply resolved event
                    await self._apply_sync_event(resolved_event)
                    
                    # Record conflict resolution
                    await self._record_conflict_resolution(event, conflict, resolved_event)
                
                # Mark as done
                self.conflict_resolution_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in conflict resolution worker: {e}")
                await asyncio.sleep(1)
    
    async def _batch_sync_worker(self):
        """Worker for batch synchronization operations"""
        
        batch_buffer = []
        last_batch_time = time.time()
        
        while True:
            try:
                # Check if we should process batch
                current_time = time.time()
                should_process_batch = (
                    len(batch_buffer) >= self.batch_size or
                    (batch_buffer and current_time - last_batch_time > self.batch_timeout)
                )
                
                if should_process_batch:
                    # Process batch
                    if batch_buffer:
                        await self._process_batch_sync(batch_buffer)
                        batch_buffer.clear()
                        last_batch_time = current_time
                
                # Try to get new batch operation
                try:
                    batch_op = await asyncio.wait_for(
                        self.batch_sync_queue.get(),
                        timeout=1.0
                    )
                    batch_buffer.append(batch_op)
                    self.batch_sync_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No new operations, continue
                    pass
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in batch sync worker: {e}")
                await asyncio.sleep(1)
    
    async def _failure_handling_worker(self):
        """Worker for handling synchronization failures"""
        
        while True:
            try:
                # Check for failed operations
                current_time = time.time()
                failed_events = []
                
                for event_id, event in self.pending_events.items():
                    if (current_time - event.timestamp > 30 and  # 30 seconds timeout
                        event.retry_count < event.max_retries):
                        failed_events.append(event)
                
                # Retry failed events
                for event in failed_events:
                    event.retry_count += 1
                    await self._retry_sync_event(event)
                
                # Clean up old pending events
                expired_events = [
                    event_id for event_id, event in self.pending_events.items()
                    if current_time - event.timestamp > 300  # 5 minutes
                ]
                
                for event_id in expired_events:
                    del self.pending_events[event_id]
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in failure handling worker: {e}")
                await asyncio.sleep(5)
    
    async def _detect_memory_conflict(self, event: SyncEvent) -> Optional[ConflictRecord]:
        """Detect memory conflicts using vector clocks"""
        
        try:
            # Get current memory state
            from services.neural_mesh.core.distributed_memory_store import distributed_memory_store
            current_memory = await distributed_memory_store.retrieve_memory(event.memory_id)
            
            if not current_memory:
                # No conflict if memory doesn't exist
                return None
            
            # Check vector clock for conflicts
            current_clock = self.vector_clocks.get(current_memory.agent_id, VectorClock())
            event_clock = event.vector_clock
            
            comparison = current_clock.compare(event_clock)
            
            if comparison == "concurrent":
                # Concurrent updates detected - conflict!
                conflict = ConflictRecord(
                    conflict_id=str(uuid.uuid4()),
                    memory_id=event.memory_id,
                    conflicting_agents=[current_memory.agent_id, event.agent_id],
                    conflict_type="concurrent_update",
                    resolution_strategy=self.conflict_resolution_strategy,
                    original_versions={
                        current_memory.agent_id: asdict(current_memory),
                        event.agent_id: event.data
                    }
                )
                
                self.conflict_records[conflict.conflict_id] = conflict
                
                log.warning(f"Memory conflict detected: {conflict.conflict_id}")
                return conflict
            
            return None
            
        except Exception as e:
            log.error(f"Error detecting conflict: {e}")
            return None
    
    async def _resolve_memory_conflict(
        self,
        event: SyncEvent,
        conflict: ConflictRecord
    ) -> Optional[SyncEvent]:
        """Resolve memory conflict using configured strategy"""
        
        try:
            if conflict.resolution_strategy == ConflictResolutionStrategy.LAST_WRITER_WINS:
                # Simple last writer wins
                resolved_event = event  # Use the incoming event
                conflict.resolution_confidence = 0.7
                
            elif conflict.resolution_strategy == ConflictResolutionStrategy.VECTOR_CLOCK:
                # Use vector clock ordering
                resolved_event = await self._resolve_with_vector_clock(event, conflict)
                conflict.resolution_confidence = 0.9
                
            elif conflict.resolution_strategy == ConflictResolutionStrategy.CRDT_MERGE:
                # CRDT-style merge
                resolved_event = await self._resolve_with_crdt_merge(event, conflict)
                conflict.resolution_confidence = 0.8
                
            elif conflict.resolution_strategy == ConflictResolutionStrategy.AGENT_CONSENSUS:
                # Agent consensus resolution
                resolved_event = await self._resolve_with_agent_consensus(event, conflict)
                conflict.resolution_confidence = 0.95
                
            else:
                # Default to last writer wins
                resolved_event = event
                conflict.resolution_confidence = 0.5
            
            if resolved_event:
                conflict.resolved_version = resolved_event.data
                conflict.resolved_at = time.time()
                
                log.info(f"Resolved conflict {conflict.conflict_id} with strategy {conflict.resolution_strategy.value}")
                return resolved_event
            
            return None
            
        except Exception as e:
            log.error(f"Error resolving conflict: {e}")
            return None
    
    async def _resolve_with_vector_clock(
        self,
        event: SyncEvent,
        conflict: ConflictRecord
    ) -> Optional[SyncEvent]:
        """Resolve conflict using vector clock ordering"""
        
        # Get all conflicting versions
        versions = []
        for agent_id, version_data in conflict.original_versions.items():
            clock = self.vector_clocks.get(agent_id, VectorClock())
            versions.append((agent_id, version_data, clock))
        
        # Add incoming event
        versions.append((event.agent_id, event.data, event.vector_clock))
        
        # Find the version with the most recent vector clock
        latest_version = max(
            versions,
            key=lambda v: sum(v[2].clocks.values())
        )
        
        # Create resolved event
        resolved_event = SyncEvent(
            event_id=str(uuid.uuid4()),
            event_type=SyncEventType.MEMORY_UPDATE,
            memory_id=event.memory_id,
            agent_id=latest_version[0],
            data=latest_version[1],
            vector_clock=latest_version[2]
        )
        
        return resolved_event
    
    async def _resolve_with_crdt_merge(
        self,
        event: SyncEvent,
        conflict: ConflictRecord
    ) -> Optional[SyncEvent]:
        """Resolve conflict using CRDT-style merge"""
        
        try:
            # Implement CRDT merge logic
            merged_data = {}
            
            # Collect all versions
            all_versions = list(conflict.original_versions.values())
            all_versions.append(event.data)
            
            # Merge strategy: union for lists, max for numbers, latest for strings
            all_keys = set()
            for version in all_versions:
                if isinstance(version, dict):
                    all_keys.update(version.keys())
            
            for key in all_keys:
                values = []
                for version in all_versions:
                    if isinstance(version, dict) and key in version:
                        values.append(version[key])
                
                if not values:
                    continue
                
                # Merge based on value type
                if all(isinstance(v, list) for v in values):
                    # Union of all lists
                    merged_data[key] = list(set().union(*values))
                elif all(isinstance(v, (int, float)) for v in values):
                    # Maximum value
                    merged_data[key] = max(values)
                elif all(isinstance(v, str) for v in values):
                    # Latest non-empty string
                    non_empty = [v for v in values if v.strip()]
                    merged_data[key] = non_empty[-1] if non_empty else values[-1]
                else:
                    # Default to latest value
                    merged_data[key] = values[-1]
            
            # Create merged event
            merged_event = SyncEvent(
                event_id=str(uuid.uuid4()),
                event_type=SyncEventType.MEMORY_MERGE,
                memory_id=event.memory_id,
                agent_id="system_merger",
                data=merged_data,
                vector_clock=self._merge_vector_clocks([v.vector_clock for _, _, v in 
                    [(a, d, self.vector_clocks.get(a, VectorClock())) 
                     for a, d in conflict.original_versions.items()]] + [event.vector_clock])
            )
            
            return merged_event
            
        except Exception as e:
            log.error(f"Error in CRDT merge: {e}")
            return None
    
    async def _resolve_with_agent_consensus(
        self,
        event: SyncEvent,
        conflict: ConflictRecord
    ) -> Optional[SyncEvent]:
        """Resolve conflict using agent consensus"""
        
        try:
            # Get all conflicting agents
            conflicting_agents = conflict.conflicting_agents
            
            # Request consensus from agents
            consensus_request = {
                "conflict_id": conflict.conflict_id,
                "memory_id": conflict.memory_id,
                "versions": conflict.original_versions,
                "new_version": event.data
            }
            
            # Send consensus request via message bus
            if self.nats_client:
                for agent_id in conflicting_agents:
                    subject = f"neural_mesh.consensus.request.{agent_id}"
                    await self.nats_client.publish(
                        subject,
                        json.dumps(consensus_request).encode()
                    )
            
            # Wait for consensus responses (simplified - would implement proper voting)
            await asyncio.sleep(5)  # Give agents time to respond
            
            # For now, default to CRDT merge
            return await self._resolve_with_crdt_merge(event, conflict)
            
        except Exception as e:
            log.error(f"Error in agent consensus: {e}")
            return None
    
    def _merge_vector_clocks(self, clocks: List[VectorClock]) -> VectorClock:
        """Merge multiple vector clocks"""
        
        merged = VectorClock()
        
        for clock in clocks:
            merged.update(clock)
        
        return merged
    
    async def _apply_sync_event(self, event: SyncEvent):
        """Apply synchronization event to memory store"""
        
        try:
            from services.neural_mesh.core.distributed_memory_store import distributed_memory_store
            
            if event.event_type == SyncEventType.MEMORY_CREATE:
                # Create new memory entry
                await distributed_memory_store.store_memory(
                    agent_id=event.agent_id,
                    memory_type=event.data["memory_type"],
                    memory_tier=event.data["memory_tier"],
                    content=event.data["content"],
                    metadata=event.data.get("metadata", {})
                )
                
            elif event.event_type == SyncEventType.MEMORY_UPDATE:
                # Update existing memory
                await distributed_memory_store.update_memory(
                    memory_id=event.memory_id,
                    agent_id=event.agent_id,
                    updates=event.data["updates"]
                )
                
            elif event.event_type == SyncEventType.MEMORY_DELETE:
                # Delete memory (mark as deleted)
                await self._mark_memory_deleted(event.memory_id, event.agent_id)
                
            elif event.event_type == SyncEventType.MEMORY_MERGE:
                # Apply merged memory
                await self._apply_merged_memory(event)
            
            # Remove from pending events
            if event.event_id in self.pending_events:
                del self.pending_events[event.event_id]
            
            log.debug(f"Applied sync event {event.event_id}")
            
        except Exception as e:
            log.error(f"Error applying sync event: {e}")
            # Add to pending for retry
            self.pending_events[event.event_id] = event
    
    async def _process_batch_sync(self, batch_operations: List[BatchSyncOperation]):
        """Process batch synchronization operations"""
        
        try:
            for batch_op in batch_operations:
                success_count = 0
                failure_count = 0
                
                # Process all operations in batch
                for operation in batch_op.operations:
                    try:
                        await self._apply_sync_event(operation)
                        success_count += 1
                    except Exception as e:
                        log.error(f"Batch operation failed: {e}")
                        failure_count += 1
                
                # Update batch status
                batch_op.success_count = success_count
                batch_op.failure_count = failure_count
                batch_op.completed_at = time.time()
                
                log.info(f"Processed batch {batch_op.batch_id}: {success_count} success, {failure_count} failures")
                
        except Exception as e:
            log.error(f"Error processing batch sync: {e}")
    
    async def create_batch_sync(
        self,
        operations: List[SyncEvent],
        target_agents: List[str]
    ) -> str:
        """Create batch synchronization operation"""
        
        batch_id = str(uuid.uuid4())
        
        batch_op = BatchSyncOperation(
            batch_id=batch_id,
            operations=operations,
            target_agents=target_agents,
            batch_size=len(operations)
        )
        
        # Queue for batch processing
        await self.batch_sync_queue.put(batch_op)
        
        log.info(f"Created batch sync operation {batch_id} with {len(operations)} operations")
        return batch_id
    
    async def _record_conflict_resolution(
        self,
        original_event: SyncEvent,
        conflict: ConflictRecord,
        resolved_event: SyncEvent
    ):
        """Record conflict resolution for audit trail"""
        
        try:
            # Store in neural mesh for learning
            from services.neural_mesh.core.enhanced_memory import EnhancedNeuralMesh
            
            neural_mesh = EnhancedNeuralMesh()
            await neural_mesh.store_knowledge(
                agent_id="sync_protocol",
                knowledge_type="conflict_resolution",
                data={
                    "conflict_id": conflict.conflict_id,
                    "memory_id": conflict.memory_id,
                    "conflicting_agents": conflict.conflicting_agents,
                    "resolution_strategy": conflict.resolution_strategy.value,
                    "resolution_confidence": conflict.resolution_confidence,
                    "original_event": asdict(original_event),
                    "resolved_event": asdict(resolved_event),
                    "timestamp": time.time()
                },
                memory_tier="L4"  # Long-term storage for audit
            )
            
        except Exception as e:
            log.error(f"Error recording conflict resolution: {e}")
    
    def _get_partition_key(self, memory_id: str, agent_id: str) -> str:
        """Get partition key for memory entry"""
        
        # Use agent_id as primary partition key
        return f"agent:{agent_id}"
    
    async def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        
        return {
            "timestamp": time.time(),
            "vector_clocks": len(self.vector_clocks),
            "pending_events": len(self.pending_events),
            "conflict_records": len(self.conflict_records),
            "batch_operations": len(self.batch_operations),
            "message_bus": {
                "nats_connected": self.nats_client is not None,
                "kafka_connected": self.kafka_producer is not None,
                "rabbitmq_connected": self.rabbitmq_connection is not None
            },
            "sync_queues": {
                "sync_events": self.sync_event_queue.qsize(),
                "conflict_resolution": self.conflict_resolution_queue.qsize(),
                "batch_sync": self.batch_sync_queue.qsize()
            }
        }
    
    async def force_memory_sync(
        self,
        agent_id: str,
        target_agents: List[str] = None
    ) -> Dict[str, Any]:
        """Force synchronization of all memories for an agent"""
        
        try:
            # Get all memories for agent
            from services.neural_mesh.core.distributed_memory_store import distributed_memory_store
            memories = await distributed_memory_store.search_memories(agent_id=agent_id, limit=1000)
            
            # Create sync events for all memories
            sync_events = []
            for memory in memories:
                event = SyncEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=SyncEventType.MEMORY_UPDATE,
                    memory_id=memory.memory_id,
                    agent_id=agent_id,
                    data=asdict(memory),
                    vector_clock=self.vector_clocks[agent_id]
                )
                sync_events.append(event)
            
            # Create batch sync
            batch_id = await self.create_batch_sync(sync_events, target_agents or [])
            
            return {
                "batch_id": batch_id,
                "memories_synced": len(memories),
                "target_agents": target_agents or [],
                "success": True
            }
            
        except Exception as e:
            log.error(f"Error forcing memory sync: {e}")
            return {"success": False, "error": str(e)}

# Global instance
memory_sync_protocol = MemorySynchronizationProtocol()
