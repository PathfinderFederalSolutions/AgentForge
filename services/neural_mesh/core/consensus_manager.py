"""
Distributed Consensus Manager - Ensures Data Consistency Across Memory Tiers
Implements Raft consensus algorithm and eventual consistency mechanisms
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import random
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

# Import base classes
from .memory_types import MemoryItem, Knowledge

# Optional imports
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

# Metrics imports
try:
    from prometheus_client import Counter, Histogram, Gauge, Enum as PrometheusEnum
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = PrometheusEnum = lambda *args, **kwargs: None

log = logging.getLogger("consensus-manager")

class NodeState(Enum):
    """Raft node states"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

class ConsensusOperation(Enum):
    """Types of consensus operations"""
    STORE = "store"
    UPDATE = "update"
    DELETE = "delete"
    PROPAGATE_KNOWLEDGE = "propagate_knowledge"
    TIER_SYNC = "tier_sync"

@dataclass
class LogEntry:
    """Raft log entry"""
    term: int
    index: int
    operation: ConsensusOperation
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    committed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "term": self.term,
            "index": self.index,
            "operation": self.operation.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "committed": self.committed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LogEntry:
        """Deserialize from dictionary"""
        return cls(
            term=data["term"],
            index=data["index"],
            operation=ConsensusOperation(data["operation"]),
            data=data["data"],
            timestamp=data.get("timestamp", time.time()),
            committed=data.get("committed", False)
        )

@dataclass
class VoteRequest:
    """Raft vote request"""
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int

@dataclass
class VoteResponse:
    """Raft vote response"""
    term: int
    vote_granted: bool

@dataclass
class AppendEntriesRequest:
    """Raft append entries request"""
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry]
    leader_commit: int

@dataclass
class AppendEntriesResponse:
    """Raft append entries response"""
    term: int
    success: bool
    match_index: int = 0

class ConsensusNode:
    """Individual node in the consensus cluster"""
    
    def __init__(self, node_id: str, cluster_nodes: List[str], redis_url: Optional[str] = None):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.redis_url = redis_url
        
        # Raft state
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Timing
        self.last_heartbeat = time.time()
        self.election_timeout = random.uniform(150, 300) / 1000  # 150-300ms
        self.heartbeat_interval = 50 / 1000  # 50ms
        
        # Networking
        self.redis_client = None
        self.http_session = None
        
        # Threading
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
        
        # Callbacks
        self.on_leader_elected: Optional[Callable[[str], None]] = None
        self.on_log_committed: Optional[Callable[[LogEntry], None]] = None
        
        # Metrics
        if METRICS_AVAILABLE:
            self.node_state_metric = PrometheusEnum(
                'consensus_node_state',
                'Current state of consensus node',
                ['node_id'],
                states=[s.value for s in NodeState]
            )
            self.consensus_operations = Counter(
                'consensus_operations_total',
                'Consensus operations',
                ['node_id', 'operation', 'status']
            )
            self.election_counter = Counter(
                'consensus_elections_total',
                'Number of elections',
                ['node_id', 'result']
            )
            self.log_size = Gauge(
                'consensus_log_size',
                'Size of consensus log',
                ['node_id']
            )
    
    async def initialize(self):
        """Initialize consensus node"""
        log.info(f"Initializing consensus node {self.node_id}")
        
        # Initialize Redis connection
        if self.redis_url and REDIS_AVAILABLE:
            self.redis_client = redis.from_url(self.redis_url)
            try:
                await self.redis_client.ping()
                log.info("Redis connection established for consensus")
            except Exception as e:
                log.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Initialize HTTP session
        if AIOHTTP_AVAILABLE:
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5.0)
            )
        
        # Load persistent state
        await self._load_persistent_state()
        
        # Start background tasks
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._election_timer()),
            asyncio.create_task(self._heartbeat_timer()),
            asyncio.create_task(self._log_replication_loop()),
            asyncio.create_task(self._state_machine_loop())
        ]
        
        log.info(f"Consensus node {self.node_id} initialized")
    
    async def shutdown(self):
        """Shutdown consensus node"""
        log.info(f"Shutting down consensus node {self.node_id}")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        # Close connections
        if self.http_session:
            await self.http_session.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        log.info(f"Consensus node {self.node_id} shutdown complete")
    
    async def _load_persistent_state(self):
        """Load persistent state from storage"""
        if not self.redis_client:
            return
        
        try:
            # Load current term and voted_for
            term_data = await self.redis_client.get(f"consensus:{self.node_id}:term")
            if term_data:
                self.current_term = int(term_data)
            
            vote_data = await self.redis_client.get(f"consensus:{self.node_id}:voted_for")
            if vote_data:
                self.voted_for = vote_data.decode()
            
            # Load log entries
            log_data = await self.redis_client.get(f"consensus:{self.node_id}:log")
            if log_data:
                log_entries = json.loads(log_data)
                self.log = [LogEntry.from_dict(entry) for entry in log_entries]
            
            log.info(f"Loaded persistent state: term={self.current_term}, log_size={len(self.log)}")
            
        except Exception as e:
            log.error(f"Failed to load persistent state: {e}")
    
    async def _save_persistent_state(self):
        """Save persistent state to storage"""
        if not self.redis_client:
            return
        
        try:
            # Save current term and voted_for
            await self.redis_client.set(f"consensus:{self.node_id}:term", self.current_term)
            
            if self.voted_for:
                await self.redis_client.set(f"consensus:{self.node_id}:voted_for", self.voted_for)
            else:
                await self.redis_client.delete(f"consensus:{self.node_id}:voted_for")
            
            # Save log entries (keep last 1000 for performance)
            log_entries = [entry.to_dict() for entry in self.log[-1000:]]
            await self.redis_client.set(
                f"consensus:{self.node_id}:log", 
                json.dumps(log_entries)
            )
            
        except Exception as e:
            log.error(f"Failed to save persistent state: {e}")
    
    async def _election_timer(self):
        """Election timeout timer"""
        while self.is_running:
            try:
                await asyncio.sleep(0.01)  # Check every 10ms
                
                if self.state == NodeState.LEADER:
                    continue
                
                # Check if election timeout has expired
                if time.time() - self.last_heartbeat > self.election_timeout:
                    await self._start_election()
                    
            except Exception as e:
                log.error(f"Election timer error: {e}")
    
    async def _heartbeat_timer(self):
        """Heartbeat timer for leader"""
        while self.is_running:
            try:
                if self.state == NodeState.LEADER:
                    await self._send_heartbeats()
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                log.error(f"Heartbeat timer error: {e}")
    
    async def _start_election(self):
        """Start leader election"""
        with self.lock:
            log.info(f"Node {self.node_id} starting election for term {self.current_term + 1}")
            
            # Become candidate
            self.state = NodeState.CANDIDATE
            self.current_term += 1
            self.voted_for = self.node_id
            self.last_heartbeat = time.time()
            
            # Reset election timeout
            self.election_timeout = random.uniform(150, 300) / 1000
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.node_state_metric.labels(node_id=self.node_id).state(self.state.value)
        
        # Save persistent state
        await self._save_persistent_state()
        
        # Request votes from other nodes
        votes_received = 1  # Vote for self
        vote_tasks = []
        
        for node in self.cluster_nodes:
            if node != self.node_id:
                task = asyncio.create_task(self._request_vote(node))
                vote_tasks.append(task)
        
        if vote_tasks:
            vote_results = await asyncio.gather(*vote_tasks, return_exceptions=True)
            
            for result in vote_results:
                if isinstance(result, VoteResponse) and result.vote_granted:
                    votes_received += 1
                elif isinstance(result, VoteResponse) and result.term > self.current_term:
                    # Found higher term, become follower
                    await self._become_follower(result.term)
                    return
        
        # Check if won election
        majority = len(self.cluster_nodes) // 2 + 1
        
        with self.lock:
            if self.state == NodeState.CANDIDATE and votes_received >= majority:
                await self._become_leader()
            else:
                # Election failed, become follower
                self.state = NodeState.FOLLOWER
                
                if METRICS_AVAILABLE:
                    self.election_counter.labels(
                        node_id=self.node_id, 
                        result="failed"
                    ).inc()
    
    async def _request_vote(self, node_id: str) -> Optional[VoteResponse]:
        """Request vote from a node"""
        try:
            last_log_index = len(self.log) - 1 if self.log else 0
            last_log_term = self.log[-1].term if self.log else 0
            
            request = VoteRequest(
                term=self.current_term,
                candidate_id=self.node_id,
                last_log_index=last_log_index,
                last_log_term=last_log_term
            )
            
            # Send vote request (simplified - would use actual networking)
            if self.http_session:
                async with self.http_session.post(
                    f"http://{node_id}/consensus/vote",
                    json=request.__dict__
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return VoteResponse(**data)
            
            return None
            
        except Exception as e:
            log.debug(f"Vote request to {node_id} failed: {e}")
            return None
    
    async def handle_vote_request(self, request: VoteRequest) -> VoteResponse:
        """Handle incoming vote request"""
        with self.lock:
            vote_granted = False
            
            # Reply false if term < currentTerm
            if request.term < self.current_term:
                return VoteResponse(term=self.current_term, vote_granted=False)
            
            # If term > currentTerm, become follower
            if request.term > self.current_term:
                await self._become_follower(request.term)
            
            # Grant vote if haven't voted or voted for this candidate
            if (self.voted_for is None or self.voted_for == request.candidate_id):
                # Check if candidate's log is at least as up-to-date
                last_log_index = len(self.log) - 1 if self.log else 0
                last_log_term = self.log[-1].term if self.log else 0
                
                if (request.last_log_term > last_log_term or
                    (request.last_log_term == last_log_term and 
                     request.last_log_index >= last_log_index)):
                    
                    vote_granted = True
                    self.voted_for = request.candidate_id
                    self.last_heartbeat = time.time()
            
            response = VoteResponse(term=self.current_term, vote_granted=vote_granted)
            
            # Save persistent state if voted
            if vote_granted:
                asyncio.create_task(self._save_persistent_state())
            
            return response
    
    async def _become_leader(self):
        """Become leader"""
        with self.lock:
            log.info(f"Node {self.node_id} became leader for term {self.current_term}")
            
            self.state = NodeState.LEADER
            
            # Initialize leader state
            for node in self.cluster_nodes:
                if node != self.node_id:
                    self.next_index[node] = len(self.log)
                    self.match_index[node] = 0
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.node_state_metric.labels(node_id=self.node_id).state(self.state.value)
                self.election_counter.labels(
                    node_id=self.node_id, 
                    result="won"
                ).inc()
            
            # Notify callback
            if self.on_leader_elected:
                self.on_leader_elected(self.node_id)
        
        # Send initial heartbeats
        await self._send_heartbeats()
    
    async def _become_follower(self, term: int):
        """Become follower"""
        with self.lock:
            if term > self.current_term:
                self.current_term = term
                self.voted_for = None
            
            self.state = NodeState.FOLLOWER
            self.last_heartbeat = time.time()
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.node_state_metric.labels(node_id=self.node_id).state(self.state.value)
        
        # Save persistent state
        await self._save_persistent_state()
    
    async def _send_heartbeats(self):
        """Send heartbeats to all followers"""
        if self.state != NodeState.LEADER:
            return
        
        heartbeat_tasks = []
        for node in self.cluster_nodes:
            if node != self.node_id:
                task = asyncio.create_task(self._send_append_entries(node))
                heartbeat_tasks.append(task)
        
        if heartbeat_tasks:
            await asyncio.gather(*heartbeat_tasks, return_exceptions=True)
    
    async def _send_append_entries(self, node_id: str) -> Optional[AppendEntriesResponse]:
        """Send append entries to a node"""
        try:
            with self.lock:
                prev_log_index = self.next_index.get(node_id, 0) - 1
                prev_log_term = 0
                
                if prev_log_index >= 0 and prev_log_index < len(self.log):
                    prev_log_term = self.log[prev_log_index].term
                
                # Get entries to send
                entries = []
                start_index = self.next_index.get(node_id, 0)
                if start_index < len(self.log):
                    entries = self.log[start_index:]
                
                request = AppendEntriesRequest(
                    term=self.current_term,
                    leader_id=self.node_id,
                    prev_log_index=prev_log_index,
                    prev_log_term=prev_log_term,
                    entries=entries,
                    leader_commit=self.commit_index
                )
            
            # Send append entries (simplified - would use actual networking)
            if self.http_session:
                async with self.http_session.post(
                    f"http://{node_id}/consensus/append_entries",
                    json={
                        **request.__dict__,
                        "entries": [entry.to_dict() for entry in request.entries]
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_obj = AppendEntriesResponse(**data)
                        await self._handle_append_entries_response(node_id, response_obj)
                        return response_obj
            
            return None
            
        except Exception as e:
            log.debug(f"Append entries to {node_id} failed: {e}")
            return None
    
    async def handle_append_entries(self, request: AppendEntriesRequest) -> AppendEntriesResponse:
        """Handle incoming append entries request"""
        with self.lock:
            # Reply false if term < currentTerm
            if request.term < self.current_term:
                return AppendEntriesResponse(term=self.current_term, success=False)
            
            # Become follower if term >= currentTerm
            if request.term >= self.current_term:
                asyncio.create_task(self._become_follower(request.term))
            
            self.last_heartbeat = time.time()
            
            # Reply false if log doesn't contain entry at prevLogIndex
            if (request.prev_log_index >= 0 and
                (request.prev_log_index >= len(self.log) or
                 self.log[request.prev_log_index].term != request.prev_log_term)):
                return AppendEntriesResponse(term=self.current_term, success=False)
            
            # Delete conflicting entries and append new ones
            if request.entries:
                # Find insertion point
                insert_index = request.prev_log_index + 1
                
                # Remove conflicting entries
                if insert_index < len(self.log):
                    self.log = self.log[:insert_index]
                
                # Append new entries
                self.log.extend(request.entries)
                
                # Save persistent state
                asyncio.create_task(self._save_persistent_state())
            
            # Update commit index
            if request.leader_commit > self.commit_index:
                self.commit_index = min(request.leader_commit, len(self.log) - 1)
            
            match_index = len(self.log) - 1
            return AppendEntriesResponse(
                term=self.current_term, 
                success=True, 
                match_index=match_index
            )
    
    async def _handle_append_entries_response(self, node_id: str, response: AppendEntriesResponse):
        """Handle append entries response"""
        with self.lock:
            if response.term > self.current_term:
                # Found higher term, become follower
                asyncio.create_task(self._become_follower(response.term))
                return
            
            if self.state != NodeState.LEADER:
                return
            
            if response.success:
                # Update next_index and match_index
                self.next_index[node_id] = response.match_index + 1
                self.match_index[node_id] = response.match_index
                
                # Update commit index
                await self._update_commit_index()
            else:
                # Decrement next_index and retry
                self.next_index[node_id] = max(0, self.next_index[node_id] - 1)
    
    async def _update_commit_index(self):
        """Update commit index based on majority replication"""
        if self.state != NodeState.LEADER:
            return
        
        # Find highest index replicated on majority
        for n in range(len(self.log) - 1, self.commit_index, -1):
            if self.log[n].term == self.current_term:
                # Count replicas
                replicas = 1  # Leader
                for node_id, match_index in self.match_index.items():
                    if match_index >= n:
                        replicas += 1
                
                # Check if majority
                if replicas > len(self.cluster_nodes) // 2:
                    self.commit_index = n
                    break
    
    async def _log_replication_loop(self):
        """Background log replication for leader"""
        while self.is_running:
            try:
                if self.state == NodeState.LEADER:
                    # Send append entries to followers that need updates
                    tasks = []
                    for node_id in self.cluster_nodes:
                        if (node_id != self.node_id and 
                            self.next_index.get(node_id, 0) < len(self.log)):
                            task = asyncio.create_task(self._send_append_entries(node_id))
                            tasks.append(task)
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(0.01)  # 10ms
                
            except Exception as e:
                log.error(f"Log replication loop error: {e}")
    
    async def _state_machine_loop(self):
        """Apply committed entries to state machine"""
        while self.is_running:
            try:
                if self.last_applied < self.commit_index:
                    # Apply next entry
                    self.last_applied += 1
                    entry = self.log[self.last_applied]
                    
                    # Mark as committed
                    entry.committed = True
                    
                    # Notify callback
                    if self.on_log_committed:
                        self.on_log_committed(entry)
                    
                    # Update metrics
                    if METRICS_AVAILABLE:
                        self.consensus_operations.labels(
                            node_id=self.node_id,
                            operation=entry.operation.value,
                            status="committed"
                        ).inc()
                
                await asyncio.sleep(0.001)  # 1ms
                
            except Exception as e:
                log.error(f"State machine loop error: {e}")
    
    async def append_entry(self, operation: ConsensusOperation, data: Dict[str, Any]) -> bool:
        """Append entry to log (leader only)"""
        if self.state != NodeState.LEADER:
            return False
        
        with self.lock:
            entry = LogEntry(
                term=self.current_term,
                index=len(self.log),
                operation=operation,
                data=data
            )
            
            self.log.append(entry)
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.log_size.labels(node_id=self.node_id).set(len(self.log))
                self.consensus_operations.labels(
                    node_id=self.node_id,
                    operation=operation.value,
                    status="appended"
                ).inc()
        
        # Save persistent state
        await self._save_persistent_state()
        
        return True
    
    def get_leader(self) -> Optional[str]:
        """Get current leader ID"""
        # In a real implementation, this would track the current leader
        # For now, return self if leader, None otherwise
        return self.node_id if self.state == NodeState.LEADER else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consensus node statistics"""
        with self.lock:
            return {
                "node_id": self.node_id,
                "state": self.state.value,
                "current_term": self.current_term,
                "voted_for": self.voted_for,
                "log_size": len(self.log),
                "commit_index": self.commit_index,
                "last_applied": self.last_applied,
                "cluster_size": len(self.cluster_nodes),
                "is_leader": self.state == NodeState.LEADER,
                "next_index": dict(self.next_index),
                "match_index": dict(self.match_index)
            }

class ConsensusManager:
    """Manages consensus across memory tiers"""
    
    def __init__(self, node_id: str, cluster_nodes: List[str], redis_url: Optional[str] = None):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.redis_url = redis_url
        
        # Consensus node
        self.consensus_node = ConsensusNode(node_id, cluster_nodes, redis_url)
        
        # Operation queues
        self.pending_operations = asyncio.Queue()
        self.operation_callbacks = {}
        
        # Tier synchronization state
        self.tier_sync_state = {}
        self.conflict_resolution_handlers = {}
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
        
        # Set up callbacks
        self.consensus_node.on_leader_elected = self._on_leader_elected
        self.consensus_node.on_log_committed = self._on_log_committed
        
        # Metrics
        if METRICS_AVAILABLE:
            self.consensus_manager_operations = Counter(
                'consensus_manager_operations_total',
                'Consensus manager operations',
                ['operation', 'status']
            )
            self.pending_operations_gauge = Gauge(
                'consensus_manager_pending_operations',
                'Number of pending consensus operations'
            )
    
    async def initialize(self):
        """Initialize consensus manager"""
        log.info(f"Initializing consensus manager for node {self.node_id}")
        
        # Initialize consensus node
        await self.consensus_node.initialize()
        
        # Start background tasks
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._operation_processor()),
            asyncio.create_task(self._tier_sync_monitor())
        ]
        
        log.info("Consensus manager initialized")
    
    async def shutdown(self):
        """Shutdown consensus manager"""
        log.info("Shutting down consensus manager")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        # Shutdown consensus node
        await self.consensus_node.shutdown()
        
        log.info("Consensus manager shutdown complete")
    
    async def ensure_consistency(self, operation: ConsensusOperation, data: Dict[str, Any]) -> bool:
        """Ensure operation is applied consistently across cluster"""
        if not self.is_running:
            return False
        
        try:
            # Create operation ID
            operation_id = hashlib.sha256(
                f"{operation.value}:{json.dumps(data, sort_keys=True)}:{time.time()}".encode()
            ).hexdigest()[:12]
            
            # Add to pending operations
            operation_future = asyncio.Future()
            self.operation_callbacks[operation_id] = operation_future
            
            await self.pending_operations.put({
                "id": operation_id,
                "operation": operation,
                "data": data,
                "timestamp": time.time()
            })
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.pending_operations_gauge.set(self.pending_operations.qsize())
            
            # Wait for consensus
            try:
                result = await asyncio.wait_for(operation_future, timeout=10.0)
                
                if METRICS_AVAILABLE:
                    self.consensus_manager_operations.labels(
                        operation=operation.value,
                        status="success"
                    ).inc()
                
                return result
                
            except asyncio.TimeoutError:
                log.warning(f"Consensus timeout for operation {operation_id}")
                
                if METRICS_AVAILABLE:
                    self.consensus_manager_operations.labels(
                        operation=operation.value,
                        status="timeout"
                    ).inc()
                
                return False
            finally:
                # Clean up callback
                if operation_id in self.operation_callbacks:
                    del self.operation_callbacks[operation_id]
        
        except Exception as e:
            log.error(f"Consensus error for operation {operation.value}: {e}")
            
            if METRICS_AVAILABLE:
                self.consensus_manager_operations.labels(
                    operation=operation.value,
                    status="error"
                ).inc()
            
            return False
    
    async def _operation_processor(self):
        """Process pending operations"""
        while self.is_running:
            try:
                # Get pending operation
                operation_data = await asyncio.wait_for(
                    self.pending_operations.get(),
                    timeout=1.0
                )
                
                # Check if we're the leader
                if self.consensus_node.state == NodeState.LEADER:
                    # Append to log
                    success = await self.consensus_node.append_entry(
                        operation_data["operation"],
                        {
                            "operation_id": operation_data["id"],
                            **operation_data["data"]
                        }
                    )
                    
                    if not success:
                        # Failed to append, reject operation
                        operation_id = operation_data["id"]
                        if operation_id in self.operation_callbacks:
                            self.operation_callbacks[operation_id].set_result(False)
                else:
                    # Not leader, forward to leader or retry
                    leader = self.consensus_node.get_leader()
                    if leader:
                        # Forward to leader (simplified)
                        log.debug(f"Forwarding operation to leader {leader}")
                        # In real implementation, would forward via network
                    
                    # For now, reject non-leader operations
                    operation_id = operation_data["id"]
                    if operation_id in self.operation_callbacks:
                        self.operation_callbacks[operation_id].set_result(False)
                
                # Update metrics
                if METRICS_AVAILABLE:
                    self.pending_operations_gauge.set(self.pending_operations.qsize())
                
            except asyncio.TimeoutError:
                continue  # No operations, continue loop
            except Exception as e:
                log.error(f"Operation processor error: {e}")
    
    def _on_leader_elected(self, leader_id: str):
        """Callback when leader is elected"""
        log.info(f"New leader elected: {leader_id}")
        
        if leader_id == self.node_id:
            # We became leader, process pending operations
            log.info("This node is now the leader")
    
    def _on_log_committed(self, entry: LogEntry):
        """Callback when log entry is committed"""
        try:
            operation_id = entry.data.get("operation_id")
            
            # Apply operation to state machine
            success = self._apply_operation(entry)
            
            # Notify waiting operation
            if operation_id and operation_id in self.operation_callbacks:
                self.operation_callbacks[operation_id].set_result(success)
            
            log.debug(f"Applied committed operation: {entry.operation.value}")
            
        except Exception as e:
            log.error(f"Error applying committed operation: {e}")
    
    def _apply_operation(self, entry: LogEntry) -> bool:
        """Apply operation to local state machine"""
        try:
            operation = entry.operation
            data = entry.data
            
            if operation == ConsensusOperation.STORE:
                return self._apply_store_operation(data)
            elif operation == ConsensusOperation.UPDATE:
                return self._apply_update_operation(data)
            elif operation == ConsensusOperation.DELETE:
                return self._apply_delete_operation(data)
            elif operation == ConsensusOperation.PROPAGATE_KNOWLEDGE:
                return self._apply_knowledge_propagation(data)
            elif operation == ConsensusOperation.TIER_SYNC:
                return self._apply_tier_sync(data)
            else:
                log.warning(f"Unknown operation type: {operation}")
                return False
        
        except Exception as e:
            log.error(f"Error applying operation {entry.operation.value}: {e}")
            return False
    
    def _apply_store_operation(self, data: Dict[str, Any]) -> bool:
        """Apply store operation"""
        # This would integrate with the actual memory layers
        log.debug(f"Applied store operation: {data.get('key')}")
        return True
    
    def _apply_update_operation(self, data: Dict[str, Any]) -> bool:
        """Apply update operation"""
        log.debug(f"Applied update operation: {data.get('key')}")
        return True
    
    def _apply_delete_operation(self, data: Dict[str, Any]) -> bool:
        """Apply delete operation"""
        log.debug(f"Applied delete operation: {data.get('key')}")
        return True
    
    def _apply_knowledge_propagation(self, data: Dict[str, Any]) -> bool:
        """Apply knowledge propagation operation"""
        log.debug(f"Applied knowledge propagation: {data.get('knowledge_id')}")
        return True
    
    def _apply_tier_sync(self, data: Dict[str, Any]) -> bool:
        """Apply tier synchronization operation"""
        log.debug(f"Applied tier sync: {data.get('sync_id')}")
        return True
    
    async def _tier_sync_monitor(self):
        """Monitor and resolve tier synchronization issues"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check for tier inconsistencies
                inconsistencies = await self._detect_tier_inconsistencies()
                
                for inconsistency in inconsistencies:
                    await self._resolve_tier_inconsistency(inconsistency)
                
            except Exception as e:
                log.error(f"Tier sync monitor error: {e}")
    
    async def _detect_tier_inconsistencies(self) -> List[Dict[str, Any]]:
        """Detect inconsistencies between memory tiers"""
        inconsistencies = []
        
        # This would implement actual inconsistency detection
        # For now, return empty list
        
        return inconsistencies
    
    async def _resolve_tier_inconsistency(self, inconsistency: Dict[str, Any]):
        """Resolve tier inconsistency through consensus"""
        try:
            # Create tier sync operation
            sync_data = {
                "inconsistency_type": inconsistency.get("type"),
                "affected_tiers": inconsistency.get("tiers", []),
                "resolution_strategy": "last_write_wins",  # Could be configurable
                "data": inconsistency.get("data", {})
            }
            
            # Ensure consistency through consensus
            success = await self.ensure_consistency(
                ConsensusOperation.TIER_SYNC,
                sync_data
            )
            
            if success:
                log.info(f"Resolved tier inconsistency: {inconsistency.get('type')}")
            else:
                log.warning(f"Failed to resolve tier inconsistency: {inconsistency.get('type')}")
                
        except Exception as e:
            log.error(f"Error resolving tier inconsistency: {e}")
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        node_stats = self.consensus_node.get_stats()
        
        return {
            **node_stats,
            "pending_operations": self.pending_operations.qsize(),
            "operation_callbacks": len(self.operation_callbacks),
            "is_running": self.is_running
        }
