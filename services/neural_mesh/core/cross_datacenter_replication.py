#!/usr/bin/env python3
"""
Cross-Datacenter Replication for Neural Mesh
Multi-region memory synchronization and disaster recovery
"""

import asyncio
import json
import time
import logging
import hashlib
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import aiohttp

log = logging.getLogger("cross-datacenter-replication")

class ReplicationMode(Enum):
    """Replication modes"""
    ACTIVE_PASSIVE = "active_passive"    # Master-slave replication
    ACTIVE_ACTIVE = "active_active"      # Multi-master replication
    MESH = "mesh"                        # Full mesh replication
    HIERARCHICAL = "hierarchical"       # Hierarchical replication

class ReplicationStatus(Enum):
    """Replication status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    OFFLINE = "offline"

@dataclass
class ReplicationNode:
    """Replication node configuration"""
    node_id: str
    endpoint: str
    region: str
    datacenter: str
    priority: int = 1
    mode: ReplicationMode = ReplicationMode.ACTIVE_PASSIVE
    status: ReplicationStatus = ReplicationStatus.OFFLINE
    last_sync: float = 0.0
    sync_lag: float = 0.0
    error_count: int = 0
    total_syncs: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReplicationOperation:
    """Replication operation record"""
    operation_id: str
    source_node: str
    target_nodes: List[str]
    operation_type: str
    data_size: int
    started_at: float
    completed_at: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0

class CrossDatacenterReplication:
    """Cross-datacenter replication system"""
    
    def __init__(self):
        self.local_node_id = os.getenv("NEURAL_MESH_NODE_ID", f"node_{uuid.uuid4().hex[:8]}")
        self.local_region = os.getenv("NEURAL_MESH_REGION", "us-east-1")
        self.local_datacenter = os.getenv("NEURAL_MESH_DATACENTER", "dc1")
        
        # Replication configuration
        self.replication_nodes: Dict[str, ReplicationNode] = {}
        self.replication_operations: List[ReplicationOperation] = []
        self.replication_mode = ReplicationMode.ACTIVE_PASSIVE
        
        # Synchronization state
        self.sync_queue = asyncio.Queue()
        self.conflict_resolution_queue = asyncio.Queue()
        
        # Performance metrics
        self.replication_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_sync_time": 0.0,
            "data_transferred": 0,
            "last_full_sync": 0.0
        }
        
        # Initialize (defer async initialization)
        self._initialized = False
    
    async def _initialize_async(self):
        """Initialize replication system"""
        if self._initialized:
            return
            
        try:
            await self._load_replication_configuration()
            await self._start_replication_workers()
            await self._perform_initial_sync()
            
            self._initialized = True
            log.info(f"✅ Cross-datacenter replication initialized for node {self.local_node_id}")
            
        except Exception as e:
            log.error(f"Failed to initialize replication system: {e}")
    
    async def ensure_initialized(self):
        """Ensure the system is initialized"""
        if not self._initialized:
            await self._initialize_async()
    
    async def _load_replication_configuration(self):
        """Load replication configuration"""
        
        try:
            # Load from environment or configuration file
            replication_config = os.getenv("NEURAL_MESH_REPLICATION_CONFIG")
            
            if replication_config:
                config = json.loads(replication_config)
                
                # Set replication mode
                self.replication_mode = ReplicationMode(
                    config.get("mode", ReplicationMode.ACTIVE_PASSIVE.value)
                )
                
                # Configure replication nodes
                for node_config in config.get("nodes", []):
                    if node_config["node_id"] != self.local_node_id:  # Don't replicate to self
                        node = ReplicationNode(
                            node_id=node_config["node_id"],
                            endpoint=node_config["endpoint"],
                            region=node_config.get("region", "unknown"),
                            datacenter=node_config.get("datacenter", "unknown"),
                            priority=node_config.get("priority", 1),
                            mode=ReplicationMode(node_config.get("mode", self.replication_mode.value))
                        )
                        
                        self.replication_nodes[node.node_id] = node
                
                log.info(f"Configured replication to {len(self.replication_nodes)} nodes")
            
        except Exception as e:
            log.error(f"Error loading replication configuration: {e}")
    
    async def _start_replication_workers(self):
        """Start replication worker tasks"""
        
        # Replication sync worker
        asyncio.create_task(self._replication_sync_worker())
        
        # Health monitoring worker
        asyncio.create_task(self._replication_health_monitor())
        
        # Conflict resolution worker
        asyncio.create_task(self._replication_conflict_resolver())
        
        # Performance monitoring worker
        asyncio.create_task(self._replication_performance_monitor())
        
        log.info("✅ Replication workers started")
    
    async def replicate_memory_operation(
        self,
        operation_type: str,
        memory_data: Dict[str, Any],
        target_nodes: List[str] = None,
        priority: int = 1
    ) -> str:
        """Replicate memory operation to other datacenters"""
        
        try:
            operation_id = str(uuid.uuid4())
            
            # Determine target nodes
            if target_nodes is None:
                target_nodes = [
                    node_id for node_id, node in self.replication_nodes.items()
                    if node.status in [ReplicationStatus.HEALTHY, ReplicationStatus.DEGRADED]
                ]
            
            # Create replication operation
            operation = ReplicationOperation(
                operation_id=operation_id,
                source_node=self.local_node_id,
                target_nodes=target_nodes,
                operation_type=operation_type,
                data_size=len(json.dumps(memory_data)),
                started_at=time.time()
            )
            
            # Queue for replication
            await self.sync_queue.put((operation, memory_data, priority))
            
            self.replication_operations.append(operation)
            
            log.debug(f"Queued replication operation {operation_id} to {len(target_nodes)} nodes")
            return operation_id
            
        except Exception as e:
            log.error(f"Error queuing replication operation: {e}")
            return ""
    
    async def _replication_sync_worker(self):
        """Worker for processing replication operations"""
        
        while True:
            try:
                # Get operation from queue
                operation, memory_data, priority = await self.sync_queue.get()
                
                # Execute replication
                await self._execute_replication_operation(operation, memory_data)
                
                # Mark as done
                self.sync_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in replication sync worker: {e}")
                await asyncio.sleep(1)
    
    async def _execute_replication_operation(
        self,
        operation: ReplicationOperation,
        memory_data: Dict[str, Any]
    ):
        """Execute replication operation to target nodes"""
        
        successful_nodes = 0
        failed_nodes = 0
        
        # Replicate to each target node
        for node_id in operation.target_nodes:
            try:
                node = self.replication_nodes.get(node_id)
                if not node:
                    continue
                
                # Send replication request
                success = await self._send_replication_request(node, operation, memory_data)
                
                if success:
                    successful_nodes += 1
                    node.last_sync = time.time()
                    node.total_syncs += 1
                    node.error_count = max(0, node.error_count - 1)  # Decrease error count on success
                else:
                    failed_nodes += 1
                    node.error_count += 1
                    
                    # Update node status based on error count
                    if node.error_count > 5:
                        node.status = ReplicationStatus.FAILED
                    elif node.error_count > 2:
                        node.status = ReplicationStatus.DEGRADED
                
            except Exception as e:
                log.error(f"Error replicating to node {node_id}: {e}")
                failed_nodes += 1
        
        # Update operation status
        operation.completed_at = time.time()
        operation.success = successful_nodes > 0
        
        if failed_nodes > 0:
            operation.error_message = f"Failed to replicate to {failed_nodes} nodes"
        
        # Update metrics
        self.replication_metrics["total_operations"] += 1
        if operation.success:
            self.replication_metrics["successful_operations"] += 1
        else:
            self.replication_metrics["failed_operations"] += 1
        
        # Update average sync time
        sync_time = operation.completed_at - operation.started_at
        total_ops = self.replication_metrics["total_operations"]
        current_avg = self.replication_metrics["average_sync_time"]
        self.replication_metrics["average_sync_time"] = (
            (current_avg * (total_ops - 1) + sync_time) / total_ops
        )
        
        log.debug(f"Replication operation {operation.operation_id} completed: {successful_nodes} success, {failed_nodes} failed")
    
    async def _send_replication_request(
        self,
        node: ReplicationNode,
        operation: ReplicationOperation,
        memory_data: Dict[str, Any]
    ) -> bool:
        """Send replication request to target node"""
        
        try:
            # Prepare replication payload
            payload = {
                "operation_id": operation.operation_id,
                "source_node": self.local_node_id,
                "operation_type": operation.operation_type,
                "memory_data": memory_data,
                "timestamp": time.time(),
                "checksum": hashlib.sha256(json.dumps(memory_data, sort_keys=True).encode()).hexdigest()
            }
            
            # Send HTTP request to target node
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{node.endpoint}/neural-mesh/replicate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update sync lag
                        node.sync_lag = time.time() - operation.started_at
                        
                        return result.get("success", False)
                    else:
                        log.error(f"Replication request failed: HTTP {response.status}")
                        return False
            
        except Exception as e:
            log.error(f"Error sending replication request to {node.node_id}: {e}")
            return False
    
    async def _replication_health_monitor(self):
        """Monitor replication node health"""
        
        while True:
            try:
                for node_id, node in self.replication_nodes.items():
                    # Check node health
                    health_status = await self._check_node_health(node)
                    
                    # Update node status
                    if health_status:
                        if node.status == ReplicationStatus.FAILED:
                            node.status = ReplicationStatus.RECOVERING
                        elif node.status == ReplicationStatus.RECOVERING:
                            node.status = ReplicationStatus.HEALTHY
                        elif node.status == ReplicationStatus.OFFLINE:
                            node.status = ReplicationStatus.HEALTHY
                    else:
                        if node.status == ReplicationStatus.HEALTHY:
                            node.status = ReplicationStatus.DEGRADED
                        elif node.status == ReplicationStatus.DEGRADED:
                            node.status = ReplicationStatus.FAILED
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in replication health monitor: {e}")
                await asyncio.sleep(30)
    
    async def _check_node_health(self, node: ReplicationNode) -> bool:
        """Check health of replication node"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{node.endpoint}/neural-mesh/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        health_data = await response.json()
                        return health_data.get("status") == "healthy"
                    else:
                        return False
                        
        except Exception as e:
            log.debug(f"Health check failed for node {node.node_id}: {e}")
            return False
    
    async def _perform_initial_sync(self):
        """Perform initial synchronization with replication nodes"""
        
        try:
            if not self.replication_nodes:
                return
            
            # Get local memory state
            from services.neural_mesh.core.distributed_memory_store import distributed_memory_store
            
            memory_stats = await distributed_memory_store.get_memory_statistics()
            
            # Send sync request to all nodes
            for node_id, node in self.replication_nodes.items():
                try:
                    sync_payload = {
                        "sync_type": "initial",
                        "source_node": self.local_node_id,
                        "memory_stats": memory_stats,
                        "timestamp": time.time()
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{node.endpoint}/neural-mesh/sync",
                            json=sync_payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as response:
                            
                            if response.status == 200:
                                sync_result = await response.json()
                                node.status = ReplicationStatus.HEALTHY
                                node.last_sync = time.time()
                                
                                log.info(f"Initial sync completed with node {node_id}")
                            else:
                                node.status = ReplicationStatus.FAILED
                                log.error(f"Initial sync failed with node {node_id}: HTTP {response.status}")
                
                except Exception as e:
                    log.error(f"Error in initial sync with node {node_id}: {e}")
                    node.status = ReplicationStatus.FAILED
            
        except Exception as e:
            log.error(f"Error in initial sync: {e}")
    
    async def _perform_incremental_sync(self, node_id: str):
        """Perform incremental sync with specific node"""
        
        try:
            node = self.replication_nodes.get(node_id)
            if not node:
                return
            
            # Get changes since last sync
            last_sync_time = node.last_sync
            
            # Query for recent memory changes
            from services.neural_mesh.core.distributed_memory_store import distributed_memory_store
            
            # This would get memories modified since last sync
            # For now, we'll simulate with a basic approach
            
            incremental_data = {
                "sync_type": "incremental",
                "source_node": self.local_node_id,
                "since_timestamp": last_sync_time,
                "changes": [],  # Would contain actual changes
                "timestamp": time.time()
            }
            
            # Send incremental sync
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{node.endpoint}/neural-mesh/sync",
                    json=incremental_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        sync_result = await response.json()
                        node.last_sync = time.time()
                        node.sync_lag = time.time() - incremental_data["timestamp"]
                        
                        log.debug(f"Incremental sync completed with node {node_id}")
                        return True
                    else:
                        log.error(f"Incremental sync failed with node {node_id}: HTTP {response.status}")
                        return False
            
        except Exception as e:
            log.error(f"Error in incremental sync with node {node_id}: {e}")
            return False
    
    async def _replication_performance_monitor(self):
        """Monitor replication performance"""
        
        while True:
            try:
                # Calculate performance metrics
                current_time = time.time()
                
                # Calculate sync lag for each node
                total_lag = 0.0
                healthy_nodes = 0
                
                for node in self.replication_nodes.values():
                    if node.status == ReplicationStatus.HEALTHY:
                        healthy_nodes += 1
                        total_lag += node.sync_lag
                
                # Update metrics
                if healthy_nodes > 0:
                    avg_sync_lag = total_lag / healthy_nodes
                    
                    # Store performance metrics
                    performance_data = {
                        "timestamp": current_time,
                        "healthy_nodes": healthy_nodes,
                        "total_nodes": len(self.replication_nodes),
                        "average_sync_lag": avg_sync_lag,
                        "replication_health": healthy_nodes / max(len(self.replication_nodes), 1),
                        "total_operations": self.replication_metrics["total_operations"],
                        "success_rate": (
                            self.replication_metrics["successful_operations"] /
                            max(self.replication_metrics["total_operations"], 1)
                        )
                    }
                    
                    # Store in local memory for monitoring
                    from services.neural_mesh.core.distributed_memory_store import distributed_memory_store
                    
                    await distributed_memory_store.store_memory(
                        agent_id="replication_system",
                        memory_type="performance_metrics",
                        memory_tier="L2",
                        content=performance_data,
                        metadata={"metric_type": "replication_performance"}
                    )
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in replication performance monitor: {e}")
                await asyncio.sleep(60)
    
    async def handle_replication_request(
        self,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle incoming replication request from another node"""
        
        try:
            operation_id = request_data.get("operation_id")
            source_node = request_data.get("source_node")
            operation_type = request_data.get("operation_type")
            memory_data = request_data.get("memory_data")
            checksum = request_data.get("checksum")
            
            # Validate checksum
            calculated_checksum = hashlib.sha256(
                json.dumps(memory_data, sort_keys=True).encode()
            ).hexdigest()
            
            if checksum != calculated_checksum:
                return {
                    "success": False,
                    "error": "Checksum validation failed",
                    "operation_id": operation_id
                }
            
            # Apply replication operation
            from services.neural_mesh.core.distributed_memory_store import distributed_memory_store
            
            if operation_type == "store_memory":
                await distributed_memory_store.store_memory(
                    agent_id=memory_data["agent_id"],
                    memory_type=memory_data["memory_type"],
                    memory_tier=memory_data["memory_tier"],
                    content=memory_data["content"],
                    metadata=memory_data.get("metadata", {})
                )
            
            elif operation_type == "update_memory":
                await distributed_memory_store.update_memory(
                    memory_id=memory_data["memory_id"],
                    agent_id=memory_data["agent_id"],
                    updates=memory_data["updates"]
                )
            
            # Record successful replication
            log.debug(f"Applied replication operation {operation_id} from node {source_node}")
            
            return {
                "success": True,
                "operation_id": operation_id,
                "applied_at": time.time(),
                "node_id": self.local_node_id
            }
            
        except Exception as e:
            log.error(f"Error handling replication request: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation_id": request_data.get("operation_id")
            }
    
    async def get_replication_status(self) -> Dict[str, Any]:
        """Get comprehensive replication status"""
        
        try:
            node_statuses = {}
            for node_id, node in self.replication_nodes.items():
                node_statuses[node_id] = {
                    "status": node.status.value,
                    "region": node.region,
                    "datacenter": node.datacenter,
                    "last_sync": node.last_sync,
                    "sync_lag": node.sync_lag,
                    "error_count": node.error_count,
                    "total_syncs": node.total_syncs,
                    "success_rate": (
                        (node.total_syncs - node.error_count) / max(node.total_syncs, 1)
                    )
                }
            
            # Calculate overall replication health
            healthy_nodes = len([n for n in self.replication_nodes.values() if n.status == ReplicationStatus.HEALTHY])
            total_nodes = len(self.replication_nodes)
            replication_health = healthy_nodes / max(total_nodes, 1)
            
            return {
                "local_node_id": self.local_node_id,
                "local_region": self.local_region,
                "local_datacenter": self.local_datacenter,
                "replication_mode": self.replication_mode.value,
                "total_nodes": total_nodes,
                "healthy_nodes": healthy_nodes,
                "replication_health": replication_health,
                "node_statuses": node_statuses,
                "performance_metrics": self.replication_metrics,
                "last_updated": time.time()
            }
            
        except Exception as e:
            log.error(f"Error getting replication status: {e}")
            return {"error": str(e)}
    
    async def trigger_disaster_recovery(
        self,
        failed_region: str,
        recovery_strategy: str = "automatic"
    ) -> Dict[str, Any]:
        """Trigger disaster recovery procedures"""
        
        try:
            recovery_id = str(uuid.uuid4())
            
            log.critical(f"Initiating disaster recovery for region {failed_region}")
            
            # Identify healthy nodes for recovery
            healthy_nodes = [
                node for node in self.replication_nodes.values()
                if node.status == ReplicationStatus.HEALTHY and node.region != failed_region
            ]
            
            if not healthy_nodes:
                return {
                    "recovery_id": recovery_id,
                    "success": False,
                    "error": "No healthy nodes available for recovery"
                }
            
            # Select recovery node (highest priority)
            recovery_node = max(healthy_nodes, key=lambda n: n.priority)
            
            # Initiate recovery process
            recovery_request = {
                "recovery_id": recovery_id,
                "failed_region": failed_region,
                "recovery_strategy": recovery_strategy,
                "recovery_node": recovery_node.node_id,
                "timestamp": time.time()
            }
            
            # Send recovery request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{recovery_node.endpoint}/neural-mesh/disaster-recovery",
                    json=recovery_request,
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes for recovery
                ) as response:
                    
                    if response.status == 200:
                        recovery_result = await response.json()
                        
                        log.info(f"Disaster recovery initiated successfully: {recovery_id}")
                        
                        return {
                            "recovery_id": recovery_id,
                            "success": True,
                            "recovery_node": recovery_node.node_id,
                            "recovery_result": recovery_result
                        }
                    else:
                        return {
                            "recovery_id": recovery_id,
                            "success": False,
                            "error": f"Recovery request failed: HTTP {response.status}"
                        }
            
        except Exception as e:
            log.error(f"Error in disaster recovery: {e}")
            return {
                "recovery_id": str(uuid.uuid4()),
                "success": False,
                "error": str(e)
            }

# Global instance
cross_dc_replication = CrossDatacenterReplication()
