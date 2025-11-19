"""
Redis Cluster Manager - Production-Grade Redis Cluster Implementation
Implements proper sharding, failover, and cluster management for million-agent deployments
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import hashlib
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
from collections import defaultdict
import random

# Redis imports
try:
    import redis.asyncio as redis
    from redis.asyncio.cluster import RedisCluster
    from redis.exceptions import (
        RedisClusterException, 
        RedisError, 
        ConnectionError, 
        TimeoutError,
        ClusterDownError,
        MovedError,
        AskError
    )
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    RedisCluster = None
    RedisClusterException = None
    RedisError = None
    REDIS_AVAILABLE = False

# Metrics imports
try:
    from prometheus_client import Counter, Histogram, Gauge, Enum as PrometheusEnum
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = PrometheusEnum = lambda *args, **kwargs: None

log = logging.getLogger("redis-cluster-manager")

class ClusterNodeState(Enum):
    """Redis cluster node states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"

class ShardingStrategy(Enum):
    """Data sharding strategies"""
    CONSISTENT_HASH = "consistent_hash"
    RANGE_BASED = "range_based"
    HASH_SLOT = "hash_slot"
    CUSTOM = "custom"

@dataclass
class ClusterNode:
    """Redis cluster node information"""
    node_id: str
    host: str
    port: int
    role: str  # master, slave
    master_id: Optional[str] = None
    slots: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) slot ranges
    state: ClusterNodeState = ClusterNodeState.HEALTHY
    last_seen: float = field(default_factory=time.time)
    connection_count: int = 0
    error_count: int = 0
    
    @property
    def address(self) -> str:
        """Get node address"""
        return f"{self.host}:{self.port}"
    
    def owns_slot(self, slot: int) -> bool:
        """Check if node owns a specific slot"""
        for start, end in self.slots:
            if start <= slot <= end:
                return True
        return False
    
    def slot_count(self) -> int:
        """Get number of slots owned by this node"""
        return sum(end - start + 1 for start, end in self.slots)

@dataclass
class ClusterConfig:
    """Redis cluster configuration"""
    nodes: List[str]  # host:port addresses
    password: Optional[str] = None
    max_connections: int = 100
    connection_timeout: float = 5.0
    socket_timeout: float = 5.0
    retry_attempts: int = 3
    retry_delay: float = 0.1
    health_check_interval: float = 30.0
    failover_timeout: float = 60.0
    cluster_require_full_coverage: bool = False
    readonly_mode: bool = False

class RedisClusterManager:
    """Production-grade Redis Cluster manager with advanced features"""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        
        # Cluster state
        self.cluster_nodes: Dict[str, ClusterNode] = {}
        self.slot_map: Dict[int, str] = {}  # slot -> node_id mapping
        self.master_nodes: Set[str] = set()
        self.replica_nodes: Dict[str, str] = {}  # replica_id -> master_id
        
        # Connection management
        self.cluster_client: Optional[RedisCluster] = None
        self.node_clients: Dict[str, redis.Redis] = {}
        self.connection_pool_stats = defaultdict(int)
        
        # Health monitoring
        self.node_health: Dict[str, Dict[str, Any]] = {}
        self.cluster_health_score = 1.0
        self.last_health_check = 0.0
        
        # Failover management
        self.failover_in_progress: Set[str] = set()
        self.failover_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.operation_stats = defaultdict(lambda: {"count": 0, "errors": 0, "total_time": 0})
        self.slot_distribution_stats = defaultdict(int)
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Metrics
        if METRICS_AVAILABLE:
            self.cluster_nodes_gauge = Gauge(
                'redis_cluster_nodes_total',
                'Total number of cluster nodes',
                ['state']
            )
            self.cluster_operations = Counter(
                'redis_cluster_operations_total',
                'Cluster operations',
                ['operation', 'status']
            )
            self.cluster_latency = Histogram(
                'redis_cluster_operation_latency_seconds',
                'Cluster operation latency',
                ['operation']
            )
            self.cluster_health_gauge = Gauge(
                'redis_cluster_health_score',
                'Cluster health score (0-1)'
            )
            self.slot_distribution_gauge = Gauge(
                'redis_cluster_slot_distribution',
                'Slot distribution across nodes',
                ['node_id']
            )
    
    async def initialize(self):
        """Initialize Redis cluster manager"""
        log.info("Initializing Redis cluster manager")
        
        if not REDIS_AVAILABLE:
            raise ImportError("redis library required for cluster management")
        
        # Initialize cluster client
        await self._initialize_cluster_client()
        
        # Discover cluster topology
        await self._discover_cluster_topology()
        
        # Initialize individual node clients
        await self._initialize_node_clients()
        
        # Start background tasks
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._topology_monitor()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._failover_detector())
        ]
        
        log.info(f"Redis cluster manager initialized with {len(self.cluster_nodes)} nodes")
    
    async def shutdown(self):
        """Shutdown Redis cluster manager"""
        log.info("Shutting down Redis cluster manager")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        # Close connections
        if self.cluster_client:
            await self.cluster_client.close()
        
        for client in self.node_clients.values():
            await client.close()
        
        log.info("Redis cluster manager shutdown complete")
    
    async def _initialize_cluster_client(self):
        """Initialize Redis cluster client"""
        try:
            startup_nodes = []
            for node_addr in self.config.nodes:
                host, port = node_addr.split(':')
                startup_nodes.append({"host": host, "port": int(port)})
            
            self.cluster_client = RedisCluster(
                startup_nodes=startup_nodes,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.connection_timeout,
                retry_on_timeout=True,
                retry_on_error=[ConnectionError, TimeoutError],
                decode_responses=False,
                skip_full_coverage_check=not self.config.cluster_require_full_coverage,
                readonly_mode=self.config.readonly_mode
            )
            
            # Test connection
            await self.cluster_client.ping()
            log.info("Redis cluster client initialized successfully")
            
        except Exception as e:
            log.error(f"Failed to initialize Redis cluster client: {e}")
            raise
    
    async def _discover_cluster_topology(self):
        """Discover and map cluster topology"""
        try:
            # Get cluster nodes information
            cluster_info = await self.cluster_client.cluster_nodes()
            
            with self.lock:
                self.cluster_nodes.clear()
                self.slot_map.clear()
                self.master_nodes.clear()
                self.replica_nodes.clear()
                
                for line in cluster_info.split('\n'):
                    if not line.strip():
                        continue
                    
                    parts = line.split()
                    if len(parts) < 8:
                        continue
                    
                    node_id = parts[0]
                    host_port = parts[1].split('@')[0]  # Remove cluster bus port
                    host, port = host_port.split(':')
                    flags = parts[2].split(',')
                    master_id = parts[3] if parts[3] != '-' else None
                    
                    # Determine role
                    if 'master' in flags:
                        role = 'master'
                        self.master_nodes.add(node_id)
                    else:
                        role = 'slave'
                        if master_id:
                            self.replica_nodes[node_id] = master_id
                    
                    # Parse slot ranges
                    slots = []
                    for slot_info in parts[8:]:
                        if '-' in slot_info:
                            start, end = map(int, slot_info.split('-'))
                            slots.append((start, end))
                            
                            # Map slots to node
                            for slot in range(start, end + 1):
                                self.slot_map[slot] = node_id
                        elif slot_info.isdigit():
                            slot = int(slot_info)
                            slots.append((slot, slot))
                            self.slot_map[slot] = node_id
                    
                    # Create node object
                    node = ClusterNode(
                        node_id=node_id,
                        host=host,
                        port=int(port),
                        role=role,
                        master_id=master_id,
                        slots=slots
                    )
                    
                    self.cluster_nodes[node_id] = node
                    
                    # Update slot distribution stats
                    self.slot_distribution_stats[node_id] = node.slot_count()
            
            log.info(f"Discovered cluster topology: {len(self.master_nodes)} masters, "
                    f"{len(self.replica_nodes)} replicas, {len(self.slot_map)} slots mapped")
            
            # Update metrics
            if METRICS_AVAILABLE:
                for state in ClusterNodeState:
                    count = sum(1 for node in self.cluster_nodes.values() if node.state == state)
                    self.cluster_nodes_gauge.labels(state=state.value).set(count)
                
                for node_id, slot_count in self.slot_distribution_stats.items():
                    self.slot_distribution_gauge.labels(node_id=node_id).set(slot_count)
            
        except Exception as e:
            log.error(f"Failed to discover cluster topology: {e}")
            raise
    
    async def _initialize_node_clients(self):
        """Initialize individual node clients for direct communication"""
        try:
            for node_id, node in self.cluster_nodes.items():
                try:
                    client = redis.Redis(
                        host=node.host,
                        port=node.port,
                        password=self.config.password,
                        socket_timeout=self.config.socket_timeout,
                        socket_connect_timeout=self.config.connection_timeout,
                        decode_responses=False
                    )
                    
                    # Test connection
                    await client.ping()
                    self.node_clients[node_id] = client
                    
                    log.debug(f"Initialized client for node {node_id} ({node.address})")
                    
                except Exception as e:
                    log.warning(f"Failed to initialize client for node {node_id}: {e}")
                    node.state = ClusterNodeState.FAILED
            
            log.info(f"Initialized {len(self.node_clients)} node clients")
            
        except Exception as e:
            log.error(f"Failed to initialize node clients: {e}")
    
    async def _health_monitor(self):
        """Background health monitoring"""
        while self.is_running:
            try:
                await self._check_cluster_health()
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                log.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def _check_cluster_health(self):
        """Check health of all cluster nodes"""
        current_time = time.time()
        healthy_nodes = 0
        total_nodes = len(self.cluster_nodes)
        
        for node_id, node in self.cluster_nodes.items():
            try:
                client = self.node_clients.get(node_id)
                if not client:
                    node.state = ClusterNodeState.FAILED
                    continue
                
                # Ping node
                start_time = time.time()
                await asyncio.wait_for(client.ping(), timeout=5.0)
                ping_time = time.time() - start_time
                
                # Get node info
                info = await client.info()
                
                # Update health status
                previous_state = node.state
                
                if ping_time > 1.0:  # Slow response
                    node.state = ClusterNodeState.DEGRADED
                else:
                    node.state = ClusterNodeState.HEALTHY
                    healthy_nodes += 1
                
                # Store health metrics
                self.node_health[node_id] = {
                    "ping_time": ping_time,
                    "memory_used": info.get("used_memory", 0),
                    "memory_peak": info.get("used_memory_peak", 0),
                    "connected_clients": info.get("connected_clients", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "last_check": current_time
                }
                
                node.last_seen = current_time
                node.error_count = 0
                
                # Log state changes
                if previous_state != node.state and previous_state == ClusterNodeState.FAILED:
                    log.info(f"Node {node_id} ({node.address}) recovered from failure")
                
            except asyncio.TimeoutError:
                log.warning(f"Node {node_id} ({node.address}) health check timeout")
                node.state = ClusterNodeState.DEGRADED
                node.error_count += 1
                
            except Exception as e:
                log.error(f"Node {node_id} ({node.address}) health check failed: {e}")
                node.state = ClusterNodeState.FAILED
                node.error_count += 1
        
        # Calculate cluster health score
        if total_nodes > 0:
            self.cluster_health_score = healthy_nodes / total_nodes
        else:
            self.cluster_health_score = 0.0
        
        self.last_health_check = current_time
        
        # Update metrics
        if METRICS_AVAILABLE:
            self.cluster_health_gauge.set(self.cluster_health_score)
            
            for state in ClusterNodeState:
                count = sum(1 for node in self.cluster_nodes.values() if node.state == state)
                self.cluster_nodes_gauge.labels(state=state.value).set(count)
        
        # Log health summary
        if self.cluster_health_score < 0.8:
            log.warning(f"Cluster health degraded: {self.cluster_health_score:.2f} "
                       f"({healthy_nodes}/{total_nodes} nodes healthy)")
        else:
            log.debug(f"Cluster health: {self.cluster_health_score:.2f}")
    
    async def _topology_monitor(self):
        """Monitor cluster topology changes"""
        while self.is_running:
            try:
                # Check for topology changes every 5 minutes
                await asyncio.sleep(300)
                
                # Rediscover topology
                old_node_count = len(self.cluster_nodes)
                await self._discover_cluster_topology()
                new_node_count = len(self.cluster_nodes)
                
                if old_node_count != new_node_count:
                    log.info(f"Cluster topology changed: {old_node_count} -> {new_node_count} nodes")
                    
                    # Reinitialize node clients for new nodes
                    await self._initialize_node_clients()
                
            except Exception as e:
                log.error(f"Topology monitoring error: {e}")
                await asyncio.sleep(600)  # Retry after 10 minutes
    
    async def _performance_monitor(self):
        """Monitor cluster performance metrics"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Collect performance metrics
                total_ops = sum(stats["count"] for stats in self.operation_stats.values())
                total_errors = sum(stats["errors"] for stats in self.operation_stats.values())
                
                if total_ops > 0:
                    error_rate = total_errors / total_ops
                    
                    if error_rate > 0.05:  # 5% error rate threshold
                        log.warning(f"High cluster error rate: {error_rate:.2%}")
                
                # Log performance summary
                log.debug(f"Cluster performance: {total_ops} ops, {error_rate:.2%} errors")
                
            except Exception as e:
                log.error(f"Performance monitoring error: {e}")
    
    async def _failover_detector(self):
        """Detect and handle node failures"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                failed_masters = []
                
                for node_id, node in self.cluster_nodes.items():
                    if (node.role == 'master' and 
                        node.state == ClusterNodeState.FAILED and
                        node_id not in self.failover_in_progress):
                        
                        failed_masters.append(node_id)
                
                # Handle failed masters
                for master_id in failed_masters:
                    await self._handle_master_failure(master_id)
                
            except Exception as e:
                log.error(f"Failover detection error: {e}")
    
    async def _handle_master_failure(self, master_id: str):
        """Handle master node failure"""
        try:
            log.warning(f"Handling failure of master node {master_id}")
            
            self.failover_in_progress.add(master_id)
            failover_start = time.time()
            
            # Find replicas of the failed master
            replicas = [
                node_id for node_id, replica_master_id in self.replica_nodes.items()
                if replica_master_id == master_id
            ]
            
            if not replicas:
                log.error(f"No replicas found for failed master {master_id}")
                return
            
            # Select best replica for promotion
            best_replica = await self._select_failover_replica(replicas)
            if not best_replica:
                log.error(f"No suitable replica found for master {master_id}")
                return
            
            # Attempt failover
            success = await self._promote_replica_to_master(best_replica, master_id)
            
            failover_duration = time.time() - failover_start
            
            # Record failover event
            failover_event = {
                "failed_master": master_id,
                "promoted_replica": best_replica if success else None,
                "success": success,
                "duration": failover_duration,
                "timestamp": time.time()
            }
            
            self.failover_history.append(failover_event)
            
            # Keep only recent failover history
            if len(self.failover_history) > 100:
                self.failover_history = self.failover_history[-100:]
            
            if success:
                log.info(f"Successfully promoted replica {best_replica} to master "
                        f"for failed master {master_id} in {failover_duration:.2f}s")
            else:
                log.error(f"Failed to promote replica for master {master_id}")
            
        except Exception as e:
            log.error(f"Master failover handling error: {e}")
        finally:
            self.failover_in_progress.discard(master_id)
    
    async def _select_failover_replica(self, replicas: List[str]) -> Optional[str]:
        """Select the best replica for failover"""
        best_replica = None
        best_score = -1
        
        for replica_id in replicas:
            try:
                replica_node = self.cluster_nodes.get(replica_id)
                if not replica_node or replica_node.state == ClusterNodeState.FAILED:
                    continue
                
                client = self.node_clients.get(replica_id)
                if not client:
                    continue
                
                # Get replica info
                info = await client.info('replication')
                
                # Calculate score based on:
                # - Replication lag
                # - Node health
                # - Connection stability
                repl_offset = info.get('slave_repl_offset', 0)
                health_score = 1.0 if replica_node.state == ClusterNodeState.HEALTHY else 0.5
                error_penalty = min(0.5, replica_node.error_count * 0.1)
                
                score = health_score - error_penalty + (repl_offset / 1000000)  # Normalize offset
                
                if score > best_score:
                    best_score = score
                    best_replica = replica_id
                
            except Exception as e:
                log.debug(f"Error evaluating replica {replica_id}: {e}")
                continue
        
        return best_replica
    
    async def _promote_replica_to_master(self, replica_id: str, failed_master_id: str) -> bool:
        """Promote replica to master"""
        try:
            client = self.node_clients.get(replica_id)
            if not client:
                return False
            
            # Execute cluster failover
            await client.cluster('failover')
            
            # Wait for promotion to complete
            for _ in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                
                # Check if promotion succeeded
                info = await client.info('replication')
                if info.get('role') == 'master':
                    # Update local topology
                    replica_node = self.cluster_nodes[replica_id]
                    replica_node.role = 'master'
                    
                    # Move to master set
                    self.master_nodes.add(replica_id)
                    self.replica_nodes.pop(replica_id, None)
                    
                    # Update slot mapping
                    failed_node = self.cluster_nodes.get(failed_master_id)
                    if failed_node:
                        for slot_start, slot_end in failed_node.slots:
                            for slot in range(slot_start, slot_end + 1):
                                self.slot_map[slot] = replica_id
                        
                        # Transfer slots to promoted replica
                        replica_node.slots = failed_node.slots.copy()
                    
                    return True
            
            log.error(f"Replica {replica_id} promotion timeout")
            return False
            
        except Exception as e:
            log.error(f"Failed to promote replica {replica_id}: {e}")
            return False
    
    # Public API methods
    
    async def execute_command(self, command: str, *args, **kwargs) -> Any:
        """Execute Redis command with cluster awareness"""
        operation_start = time.time()
        
        try:
            result = await self.cluster_client.execute_command(command, *args, **kwargs)
            
            # Track successful operation
            self.operation_stats[command]["count"] += 1
            self.operation_stats[command]["total_time"] += time.time() - operation_start
            
            if METRICS_AVAILABLE:
                self.cluster_operations.labels(
                    operation=command,
                    status="success"
                ).inc()
                
                self.cluster_latency.labels(operation=command).observe(
                    time.time() - operation_start
                )
            
            return result
            
        except (MovedError, AskError) as e:
            # Handle cluster redirections
            log.debug(f"Cluster redirection for {command}: {e}")
            
            # Refresh topology and retry
            await self._discover_cluster_topology()
            
            try:
                result = await self.cluster_client.execute_command(command, *args, **kwargs)
                self.operation_stats[command]["count"] += 1
                return result
            except Exception as retry_error:
                self.operation_stats[command]["errors"] += 1
                raise retry_error
            
        except Exception as e:
            # Track failed operation
            self.operation_stats[command]["errors"] += 1
            
            if METRICS_AVAILABLE:
                self.cluster_operations.labels(
                    operation=command,
                    status="error"
                ).inc()
            
            log.error(f"Cluster command {command} failed: {e}")
            raise
    
    async def get_key_location(self, key: str) -> Optional[str]:
        """Get the node ID that owns a specific key"""
        try:
            # Calculate hash slot for key
            slot = self._calculate_key_slot(key)
            return self.slot_map.get(slot)
            
        except Exception as e:
            log.error(f"Failed to get key location for {key}: {e}")
            return None
    
    def _calculate_key_slot(self, key: str) -> int:
        """Calculate Redis cluster hash slot for key"""
        # Extract hashtag if present
        start = key.find('{')
        if start != -1:
            end = key.find('}', start + 1)
            if end != -1 and end > start + 1:
                key = key[start + 1:end]
        
        # Calculate CRC16 hash
        crc = 0
        for byte in key.encode('utf-8'):
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        
        return crc % 16384  # Redis uses 16384 slots
    
    async def get_node_client(self, node_id: str) -> Optional[redis.Redis]:
        """Get direct client for specific node"""
        return self.node_clients.get(node_id)
    
    async def execute_on_node(self, node_id: str, command: str, *args, **kwargs) -> Any:
        """Execute command on specific node"""
        client = self.node_clients.get(node_id)
        if not client:
            raise ValueError(f"No client available for node {node_id}")
        
        try:
            return await client.execute_command(command, *args, **kwargs)
        except Exception as e:
            log.error(f"Command {command} failed on node {node_id}: {e}")
            raise
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get comprehensive cluster information"""
        try:
            cluster_info = await self.cluster_client.cluster_info()
            cluster_nodes = await self.cluster_client.cluster_nodes()
            
            # Parse cluster info
            info_dict = {}
            for line in cluster_info.split('\r\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info_dict[key] = value
            
            # Add our enhanced information
            enhanced_info = {
                "basic_info": info_dict,
                "node_count": len(self.cluster_nodes),
                "master_count": len(self.master_nodes),
                "replica_count": len(self.replica_nodes),
                "health_score": self.cluster_health_score,
                "last_health_check": self.last_health_check,
                "slot_coverage": len(self.slot_map),
                "failover_history": len(self.failover_history),
                "nodes": {
                    node_id: {
                        "address": node.address,
                        "role": node.role,
                        "state": node.state.value,
                        "slot_count": node.slot_count(),
                        "last_seen": node.last_seen,
                        "error_count": node.error_count
                    }
                    for node_id, node in self.cluster_nodes.items()
                }
            }
            
            return enhanced_info
            
        except Exception as e:
            log.error(f"Failed to get cluster info: {e}")
            return {"error": str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get cluster performance statistics"""
        total_ops = sum(stats["count"] for stats in self.operation_stats.values())
        total_errors = sum(stats["errors"] for stats in self.operation_stats.values())
        total_time = sum(stats["total_time"] for stats in self.operation_stats.values())
        
        error_rate = total_errors / total_ops if total_ops > 0 else 0
        avg_latency = total_time / total_ops if total_ops > 0 else 0
        
        return {
            "total_operations": total_ops,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "average_latency": avg_latency,
            "operations_per_second": total_ops / max(1, time.time() - self.last_health_check),
            "operation_breakdown": {
                op: {
                    "count": stats["count"],
                    "errors": stats["errors"],
                    "error_rate": stats["errors"] / max(1, stats["count"]),
                    "avg_latency": stats["total_time"] / max(1, stats["count"])
                }
                for op, stats in self.operation_stats.items()
            },
            "node_health": self.node_health,
            "failover_events": len(self.failover_history)
        }
    
    async def rebalance_cluster(self) -> bool:
        """Rebalance cluster slots for optimal distribution"""
        try:
            log.info("Starting cluster rebalancing")
            
            # This would implement cluster rebalancing logic
            # For now, just log the current distribution
            
            total_slots = 16384
            master_count = len(self.master_nodes)
            
            if master_count == 0:
                log.error("No master nodes available for rebalancing")
                return False
            
            target_slots_per_master = total_slots // master_count
            
            log.info(f"Target slots per master: {target_slots_per_master}")
            
            for node_id in self.master_nodes:
                node = self.cluster_nodes[node_id]
                current_slots = node.slot_count()
                
                log.info(f"Master {node_id}: {current_slots} slots "
                        f"(target: {target_slots_per_master})")
            
            # In production, this would implement actual slot migration
            log.info("Cluster rebalancing completed (simulation)")
            return True
            
        except Exception as e:
            log.error(f"Cluster rebalancing failed: {e}")
            return False

# Factory function for easy initialization
async def create_redis_cluster_manager(nodes: List[str], **kwargs) -> RedisClusterManager:
    """Factory function to create and initialize Redis cluster manager"""
    config = ClusterConfig(nodes=nodes, **kwargs)
    manager = RedisClusterManager(config)
    await manager.initialize()
    return manager
