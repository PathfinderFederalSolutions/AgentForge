"""
Enhanced JetStream Infrastructure - Million-Scale Agent Coordination
Provides production-ready NATS JetStream configuration with:
- Circuit breakers for fault tolerance
- Advanced connection pooling
- Million-scale stream configuration
- Comprehensive backpressure handling
- Performance monitoring and alerting
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from contextlib import asynccontextmanager

import nats
from nats.aio.client import Client as NATS
from nats.js import JetStreamContext
from nats.js.api import (
    StreamConfig,
    ConsumerConfig,
    RetentionPolicy,
    StorageType,
    DeliverPolicy,
    AckPolicy,
    DiscardPolicy,
    ReplayPolicy,
)
from nats.js.errors import NotFoundError, BadRequestError

# Metrics imports (graceful degradation)
try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = lambda *args, **kwargs: None

log = logging.getLogger("enhanced-jetstream")

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 3
    timeout: float = 10.0

@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration"""
    min_connections: int = 5
    max_connections: int = 100
    max_idle_time: float = 300.0  # 5 minutes
    connection_timeout: float = 10.0
    max_reconnect_attempts: int = -1  # Unlimited
    reconnect_time_wait: float = 2.0

@dataclass
class StreamScaleConfig:
    """Million-scale stream configuration"""
    max_msgs_per_stream: int = 10_000_000  # 10M messages per stream
    max_bytes_per_stream: int = 100 * 1024 * 1024 * 1024  # 100GB per stream
    max_msg_size: int = 64 * 1024 * 1024  # 64MB max message size
    num_replicas: int = 3  # High availability
    max_consumers: int = 1000  # Support 1K consumers per stream
    duplicate_window: int = 120_000_000_000  # 2 minutes in nanoseconds

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.next_attempt_time = 0.0
        
        # Metrics
        if METRICS_AVAILABLE:
            self.state_gauge = Gauge(f'circuit_breaker_state_{name}', 'Circuit breaker state (0=closed, 1=open, 2=half_open)')
            self.failure_counter = Counter(f'circuit_breaker_failures_{name}_total', 'Circuit breaker failures')
            self.success_counter = Counter(f'circuit_breaker_successes_{name}_total', 'Circuit breaker successes')
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        if self.state == CircuitState.OPEN:
            if time.time() < self.next_attempt_time:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
            else:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                log.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful execution"""
        if METRICS_AVAILABLE:
            self.success_counter.inc()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                log.info(f"Circuit breaker {self.name} recovered to CLOSED")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
        
        self._update_metrics()
    
    async def _on_failure(self):
        """Handle failed execution"""
        if METRICS_AVAILABLE:
            self.failure_counter.inc()
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
            log.warning(f"Circuit breaker {self.name} opened due to {self.failure_count} failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
            log.warning(f"Circuit breaker {self.name} reopened during recovery test")
        
        self._update_metrics()
    
    def _update_metrics(self):
        """Update Prometheus metrics"""
        if METRICS_AVAILABLE:
            state_value = {
                CircuitState.CLOSED: 0,
                CircuitState.OPEN: 1,
                CircuitState.HALF_OPEN: 2
            }[self.state]
            self.state_gauge.set(state_value)

class ConnectionPool:
    """Advanced connection pool for NATS"""
    
    def __init__(self, config: ConnectionPoolConfig, servers: List[str]):
        self.config = config
        self.servers = servers
        self.connections: List[NATS] = []
        self.available_connections: asyncio.Queue = asyncio.Queue()
        self.connection_count = 0
        self.lock = asyncio.Lock()
        self.circuit_breaker = CircuitBreaker("nats_connection", CircuitBreakerConfig())
        
        # Metrics
        if METRICS_AVAILABLE:
            self.active_connections = Gauge('nats_pool_active_connections', 'Active NATS connections')
            self.available_connections_gauge = Gauge('nats_pool_available_connections', 'Available NATS connections')
            self.connection_requests = Counter('nats_pool_connection_requests_total', 'Connection requests')
            self.connection_errors = Counter('nats_pool_connection_errors_total', 'Connection errors')
    
    async def initialize(self):
        """Initialize connection pool with minimum connections"""
        async with self.lock:
            for _ in range(self.config.min_connections):
                try:
                    conn = await self._create_connection()
                    self.connections.append(conn)
                    await self.available_connections.put(conn)
                    self.connection_count += 1
                except Exception as e:
                    log.error(f"Failed to initialize connection: {e}")
        
        log.info(f"Connection pool initialized with {self.connection_count} connections")
    
    async def _create_connection(self) -> NATS:
        """Create a new NATS connection"""
        nc = nats.NATS()
        await nc.connect(
            servers=self.servers,
            connect_timeout=self.config.connection_timeout,
            allow_reconnect=True,
            max_reconnect_attempts=self.config.max_reconnect_attempts,
            reconnect_time_wait=self.config.reconnect_time_wait,
            ping_interval=20,
            max_outstanding_pings=5,
            drain_timeout=10,
        )
        return nc
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        if METRICS_AVAILABLE:
            self.connection_requests.inc()
        
        conn = None
        try:
            # Try to get available connection
            try:
                conn = await asyncio.wait_for(
                    self.available_connections.get(),
                    timeout=self.config.connection_timeout
                )
            except asyncio.TimeoutError:
                # Create new connection if under max limit
                async with self.lock:
                    if self.connection_count < self.config.max_connections:
                        conn = await self._create_connection()
                        self.connections.append(conn)
                        self.connection_count += 1
                    else:
                        raise Exception("Connection pool exhausted")
            
            # Verify connection health
            if not conn.is_connected:
                await conn.connect(servers=self.servers)
            
            if METRICS_AVAILABLE:
                self.active_connections.inc()
            
            yield conn
            
        except Exception as e:
            if METRICS_AVAILABLE:
                self.connection_errors.inc()
            raise e
        finally:
            if conn:
                # Return connection to pool
                await self.available_connections.put(conn)
                if METRICS_AVAILABLE:
                    self.active_connections.dec()
    
    async def close_all(self):
        """Close all connections in the pool"""
        async with self.lock:
            for conn in self.connections:
                try:
                    if not conn.is_closed:
                        await conn.drain()
                except Exception as e:
                    log.error(f"Error closing connection: {e}")
            self.connections.clear()
            self.connection_count = 0

class EnhancedJetStream:
    """Enhanced JetStream client with million-scale capabilities"""
    
    def __init__(self, servers: Optional[List[str]] = None):
        self.servers = servers or [os.getenv("NATS_URL", "nats://localhost:4222")]
        
        # Configuration
        self.pool_config = ConnectionPoolConfig()
        self.scale_config = StreamScaleConfig()
        
        # Components
        self.connection_pool = ConnectionPool(self.pool_config, self.servers)
        self.circuit_breaker = CircuitBreaker("jetstream", CircuitBreakerConfig())
        
        # State
        self.initialized = False
        
        # Metrics
        if METRICS_AVAILABLE:
            self.publish_latency = Histogram('jetstream_publish_latency_seconds', 'JetStream publish latency', ['stream'])
            self.publish_errors = Counter('jetstream_publish_errors_total', 'JetStream publish errors', ['stream', 'error_type'])
            self.stream_messages = Gauge('jetstream_stream_messages', 'Messages in stream', ['stream'])
            self.consumer_pending = Gauge('jetstream_consumer_pending', 'Pending messages for consumer', ['stream', 'consumer'])
    
    async def initialize(self):
        """Initialize the enhanced JetStream system"""
        if self.initialized:
            return
        
        await self.connection_pool.initialize()
        await self.ensure_million_scale_streams()
        self.initialized = True
        log.info("Enhanced JetStream system initialized")
    
    async def ensure_million_scale_streams(self):
        """Create million-scale optimized streams"""
        streams_config = {
            "swarm_jobs_million": {
                "subjects": ["swarm.jobs.>"],
                "retention": RetentionPolicy.WORK_QUEUE,
                "description": "Million-scale job queue with work-queue retention"
            },
            "swarm_results_million": {
                "subjects": ["swarm.results.>"],
                "retention": RetentionPolicy.LIMITS,
                "description": "Million-scale results stream with limits retention"
            },
            "neural_mesh_million": {
                "subjects": ["mesh.ops.>", "mesh.sync.>", "mesh.broadcast.>"],
                "retention": RetentionPolicy.LIMITS,
                "description": "Million-scale neural mesh operations"
            },
            "agent_lifecycle_million": {
                "subjects": ["agent.spawn.>", "agent.terminate.>", "agent.health.>"],
                "retention": RetentionPolicy.LIMITS,
                "description": "Million-scale agent lifecycle management"
            }
        }
        
        async with self.connection_pool.get_connection() as nc:
            js = nc.jetstream()
            
            for stream_name, config in streams_config.items():
                await self._ensure_stream(js, stream_name, config)
    
    async def _ensure_stream(self, js: JetStreamContext, name: str, config: Dict[str, Any]):
        """Ensure a million-scale stream exists"""
        try:
            await js.stream_info(name)
            log.debug(f"Stream {name} already exists")
            return
        except NotFoundError:
            pass
        
        # Create million-scale optimized stream
        stream_config = StreamConfig(
            name=name,
            subjects=config["subjects"],
            retention=config["retention"],
            storage=StorageType.File,
            max_consumers=self.scale_config.max_consumers,
            max_msgs=self.scale_config.max_msgs_per_stream,
            max_bytes=self.scale_config.max_bytes_per_stream,
            max_msg_size=self.scale_config.max_msg_size,
            duplicate_window=self.scale_config.duplicate_window,
            num_replicas=self.scale_config.num_replicas,
            discard=DiscardPolicy.OLD,
            allow_direct=True,
            allow_rollup_hdrs=True,
        )
        
        try:
            await js.add_stream(stream_config)
            log.info(f"Created million-scale stream: {name}")
        except BadRequestError as e:
            log.error(f"Failed to create stream {name}: {e}")
            raise
    
    async def publish_with_circuit_breaker(self, subject: str, data: Any, headers: Optional[Dict[str, str]] = None) -> bool:
        """Publish message through circuit breaker"""
        async def _publish():
            async with self.connection_pool.get_connection() as nc:
                js = nc.jetstream()
                
                # Prepare message
                if isinstance(data, dict):
                    payload = json.dumps(data).encode('utf-8')
                elif isinstance(data, str):
                    payload = data.encode('utf-8')
                else:
                    payload = str(data).encode('utf-8')
                
                # Add idempotency headers
                if headers is None:
                    headers = {}
                
                if isinstance(data, dict):
                    msg_id = (
                        data.get("job_id") or 
                        data.get("invocation_id") or 
                        data.get("id") or 
                        data.get("request_id")
                    )
                    if msg_id:
                        headers["Nats-Msg-Id"] = str(msg_id)
                
                # Publish with metrics
                start_time = time.time()
                try:
                    await js.publish(subject, payload, headers=headers)
                    
                    if METRICS_AVAILABLE:
                        stream_name = subject.split('.')[0]
                        self.publish_latency.labels(stream=stream_name).observe(time.time() - start_time)
                    
                    return True
                except Exception as e:
                    if METRICS_AVAILABLE:
                        stream_name = subject.split('.')[0]
                        error_type = type(e).__name__
                        self.publish_errors.labels(stream=stream_name, error_type=error_type).inc()
                    raise
        
        return await self.circuit_breaker.call(_publish)
    
    async def create_million_scale_consumer(self, stream_name: str, consumer_name: str, filter_subject: str, 
                                          max_ack_pending: int = 10000, ack_wait: int = 300) -> str:
        """Create a consumer optimized for million-scale processing"""
        consumer_config = ConsumerConfig(
            durable_name=consumer_name,
            deliver_policy=DeliverPolicy.ALL,
            ack_policy=AckPolicy.EXPLICIT,
            replay_policy=ReplayPolicy.INSTANT,
            filter_subject=filter_subject,
            max_ack_pending=max_ack_pending,
            ack_wait=ack_wait * 1_000_000_000,  # Convert to nanoseconds
            max_deliver=3,  # Retry failed messages 3 times
            backoff=[1_000_000_000, 5_000_000_000, 10_000_000_000],  # 1s, 5s, 10s backoff
        )
        
        async with self.connection_pool.get_connection() as nc:
            js = nc.jetstream()
            try:
                await js.add_consumer(stream_name, consumer_config)
                log.info(f"Created million-scale consumer {consumer_name} for stream {stream_name}")
                return consumer_name
            except Exception as e:
                log.error(f"Failed to create consumer {consumer_name}: {e}")
                raise
    
    async def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get comprehensive stream information"""
        async with self.connection_pool.get_connection() as nc:
            js = nc.jetstream()
            try:
                info = await js.stream_info(stream_name)
                return {
                    "name": info.config.name,
                    "subjects": info.config.subjects,
                    "messages": info.state.messages,
                    "bytes": info.state.bytes,
                    "consumers": info.state.consumer_count,
                    "first_seq": info.state.first_seq,
                    "last_seq": info.state.last_seq,
                    "last_time": info.state.last_time,
                }
            except NotFoundError:
                return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "connection_pool": {
                "active_connections": self.connection_pool.connection_count,
                "available_connections": self.connection_pool.available_connections.qsize(),
            },
            "circuit_breaker": {
                "state": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count,
            },
            "streams": {}
        }
        
        # Check stream health
        stream_names = ["swarm_jobs_million", "swarm_results_million", "neural_mesh_million", "agent_lifecycle_million"]
        for stream_name in stream_names:
            try:
                info = await self.get_stream_info(stream_name)
                health["streams"][stream_name] = info
            except Exception as e:
                health["streams"][stream_name] = {"error": str(e)}
                health["status"] = "degraded"
        
        return health
    
    async def close(self):
        """Clean shutdown"""
        await self.connection_pool.close_all()
        log.info("Enhanced JetStream system closed")

# Global instance
enhanced_jetstream: Optional[EnhancedJetStream] = None

async def get_enhanced_jetstream() -> EnhancedJetStream:
    """Get or create the global enhanced JetStream instance"""
    global enhanced_jetstream
    if enhanced_jetstream is None:
        enhanced_jetstream = EnhancedJetStream()
        await enhanced_jetstream.initialize()
    return enhanced_jetstream

async def publish_million_scale(subject: str, data: Any, headers: Optional[Dict[str, str]] = None) -> bool:
    """Publish message through million-scale infrastructure"""
    js = await get_enhanced_jetstream()
    return await js.publish_with_circuit_breaker(subject, data, headers)

async def create_million_scale_consumer(stream_name: str, consumer_name: str, filter_subject: str) -> str:
    """Create million-scale consumer"""
    js = await get_enhanced_jetstream()
    return await js.create_million_scale_consumer(stream_name, consumer_name, filter_subject)
