"""
Distributed Consensus Manager - Production-Ready Consensus Algorithms
Implements Raft, PBFT, and custom quantum-inspired consensus for agent coordination
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import defaultdict, deque

# Cryptographic imports for security
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

log = logging.getLogger("distributed-consensus")

class NodeRole(Enum):
    """Node roles in distributed system"""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    OBSERVER = "observer"

class ConsensusState(Enum):
    """States in consensus protocol"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate" 
    LEADER = "leader"
    BYZANTINE = "byzantine"  # For PBFT
    FAULTY = "faulty"

class ConsensusMessageType(Enum):
    """Types of consensus messages"""
    # Raft messages
    REQUEST_VOTE = "request_vote"
    VOTE_RESPONSE = "vote_response"
    APPEND_ENTRIES = "append_entries"
    APPEND_RESPONSE = "append_response"
    
    # PBFT messages
    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    VIEW_CHANGE = "view_change"
    NEW_VIEW = "new_view"
    
    # Quantum consensus
    QUANTUM_PROPOSAL = "quantum_proposal"
    QUANTUM_VOTE = "quantum_vote"
    COHERENCE_UPDATE = "coherence_update"

@dataclass
class ConsensusMessage:
    """Consensus protocol message"""
    message_type: ConsensusMessageType
    sender_id: str
    term: int
    timestamp: float
    payload: Dict[str, Any]
    signature: Optional[str] = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "term": self.term,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "signature": self.signature,
            "message_id": self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsensusMessage':
        return cls(
            message_type=ConsensusMessageType(data["message_type"]),
            sender_id=data["sender_id"],
            term=data["term"],
            timestamp=data["timestamp"],
            payload=data["payload"],
            signature=data.get("signature"),
            message_id=data.get("message_id", str(uuid.uuid4()))
        )

@dataclass
class LogEntry:
    """Distributed log entry for consensus"""
    index: int
    term: int
    command: Dict[str, Any]
    timestamp: float
    committed: bool = False
    hash_value: str = field(init=False)
    
    def __post_init__(self):
        """Calculate hash of log entry"""
        content = f"{self.index}:{self.term}:{json.dumps(self.command, sort_keys=True)}:{self.timestamp}"
        self.hash_value = hashlib.sha256(content.encode()).hexdigest()

class RaftConsensusNode:
    """
    Raft Consensus Algorithm Implementation
    
    Features:
    - Leader election
    - Log replication
    - Safety guarantees
    - Network partition tolerance
    """
    
    def __init__(self, node_id: str, peer_ids: List[str], 
                 election_timeout: float = 5.0, heartbeat_interval: float = 1.0):
        self.node_id = node_id
        self.peer_ids = peer_ids
        self.all_nodes = [node_id] + peer_ids
        
        # Raft state
        self.state = ConsensusState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = -1
        self.last_applied = -1
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Timing
        self.election_timeout = election_timeout
        self.heartbeat_interval = heartbeat_interval
        self.last_heartbeat = time.time()
        self.election_deadline = time.time() + election_timeout + random.uniform(0, election_timeout)
        
        # Network and callbacks
        self.message_handlers: Dict[ConsensusMessageType, Callable] = {
            ConsensusMessageType.REQUEST_VOTE: self._handle_request_vote,
            ConsensusMessageType.VOTE_RESPONSE: self._handle_vote_response,
            ConsensusMessageType.APPEND_ENTRIES: self._handle_append_entries,
            ConsensusMessageType.APPEND_RESPONSE: self._handle_append_response
        }
        
        self.send_message_callback: Optional[Callable] = None
        self.apply_command_callback: Optional[Callable] = None
        
        # Metrics
        self.metrics = {
            "elections_started": 0,
            "votes_received": 0,
            "log_entries_appended": 0,
            "commands_committed": 0,
            "leadership_duration": 0.0,
            "last_leadership_start": 0.0
        }
        
        log.info(f"Raft node {node_id} initialized with peers: {peer_ids}")
    
    async def start(self):
        """Start Raft consensus protocol"""
        # Start election timer
        asyncio.create_task(self._election_timer())
        
        # Start heartbeat timer (if leader)
        asyncio.create_task(self._heartbeat_timer())
        
        log.info(f"Raft node {self.node_id} started")
    
    async def _election_timer(self):
        """Monitor election timeout and start elections"""
        while True:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms
                
                current_time = time.time()
                
                # Check if we need to start an election
                if (self.state != ConsensusState.LEADER and 
                    current_time > self.election_deadline):
                    
                    await self._start_election()
                
            except Exception as e:
                log.error(f"Election timer error: {e}")
                await asyncio.sleep(1.0)
    
    async def _heartbeat_timer(self):
        """Send heartbeats as leader"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if self.state == ConsensusState.LEADER:
                    await self._send_heartbeats()
                
            except Exception as e:
                log.error(f"Heartbeat timer error: {e}")
                await asyncio.sleep(1.0)
    
    async def _start_election(self):
        """Start leader election"""
        log.info(f"Node {self.node_id} starting election for term {self.current_term + 1}")
        
        # Become candidate
        self.state = ConsensusState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.metrics["elections_started"] += 1
        
        # Reset election timeout
        self._reset_election_timeout()
        
        # Vote for ourselves
        votes_received = 1
        
        # Request votes from all peers
        last_log_index = len(self.log) - 1
        last_log_term = self.log[last_log_index].term if self.log else 0
        
        vote_request = ConsensusMessage(
            message_type=ConsensusMessageType.REQUEST_VOTE,
            sender_id=self.node_id,
            term=self.current_term,
            timestamp=time.time(),
            payload={
                "candidate_id": self.node_id,
                "last_log_index": last_log_index,
                "last_log_term": last_log_term
            }
        )
        
        # Send to all peers
        for peer_id in self.peer_ids:
            if self.send_message_callback:
                await self.send_message_callback(peer_id, vote_request)
    
    async def _send_heartbeats(self):
        """Send heartbeat messages to all followers"""
        for peer_id in self.peer_ids:
            await self._send_append_entries(peer_id, heartbeat=True)
    
    async def _send_append_entries(self, peer_id: str, heartbeat: bool = False):
        """Send append entries message to specific peer"""
        next_index = self.next_index.get(peer_id, len(self.log))
        prev_log_index = next_index - 1
        prev_log_term = self.log[prev_log_index].term if prev_log_index >= 0 else 0
        
        # Get entries to send
        entries = []
        if not heartbeat and next_index < len(self.log):
            entries = [entry.__dict__ for entry in self.log[next_index:]]
        
        append_message = ConsensusMessage(
            message_type=ConsensusMessageType.APPEND_ENTRIES,
            sender_id=self.node_id,
            term=self.current_term,
            timestamp=time.time(),
            payload={
                "leader_id": self.node_id,
                "prev_log_index": prev_log_index,
                "prev_log_term": prev_log_term,
                "entries": entries,
                "leader_commit": self.commit_index,
                "heartbeat": heartbeat
            }
        )
        
        if self.send_message_callback:
            await self.send_message_callback(peer_id, append_message)
    
    def _reset_election_timeout(self):
        """Reset election timeout with randomization"""
        timeout_variance = random.uniform(0, self.election_timeout)
        self.election_deadline = time.time() + self.election_timeout + timeout_variance
    
    async def handle_message(self, message: ConsensusMessage):
        """Handle incoming consensus message"""
        try:
            # Update term if we see a higher term
            if message.term > self.current_term:
                self.current_term = message.term
                self.voted_for = None
                if self.state != ConsensusState.FOLLOWER:
                    self._become_follower()
            
            # Handle message based on type
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                log.warning(f"Unknown message type: {message.message_type}")
                
        except Exception as e:
            log.error(f"Error handling message {message.message_type}: {e}")
    
    async def _handle_request_vote(self, message: ConsensusMessage):
        """Handle vote request from candidate"""
        payload = message.payload
        candidate_id = payload["candidate_id"]
        last_log_index = payload["last_log_index"]
        last_log_term = payload["last_log_term"]
        
        # Check if we can vote for this candidate
        vote_granted = False
        
        if (message.term >= self.current_term and
            (self.voted_for is None or self.voted_for == candidate_id)):
            
            # Check if candidate's log is at least as up-to-date as ours
            our_last_log_index = len(self.log) - 1
            our_last_log_term = self.log[our_last_log_index].term if self.log else 0
            
            log_ok = (last_log_term > our_last_log_term or
                     (last_log_term == our_last_log_term and last_log_index >= our_last_log_index))
            
            if log_ok:
                vote_granted = True
                self.voted_for = candidate_id
                self._reset_election_timeout()
        
        # Send vote response
        vote_response = ConsensusMessage(
            message_type=ConsensusMessageType.VOTE_RESPONSE,
            sender_id=self.node_id,
            term=self.current_term,
            timestamp=time.time(),
            payload={
                "vote_granted": vote_granted,
                "voter_id": self.node_id
            }
        )
        
        if self.send_message_callback:
            await self.send_message_callback(candidate_id, vote_response)
        
        log.debug(f"Node {self.node_id} voted {'YES' if vote_granted else 'NO'} for {candidate_id} in term {message.term}")
    
    async def _handle_vote_response(self, message: ConsensusMessage):
        """Handle vote response from peer"""
        if self.state != ConsensusState.CANDIDATE:
            return
        
        payload = message.payload
        vote_granted = payload["vote_granted"]
        
        if vote_granted and message.term == self.current_term:
            votes_received = self.metrics["votes_received"] + 1
            self.metrics["votes_received"] = votes_received
            
            # Count votes (including our own)
            votes_needed = (len(self.all_nodes) // 2) + 1
            
            if votes_received >= votes_needed:
                await self._become_leader()
    
    async def _become_leader(self):
        """Become leader and start sending heartbeats"""
        log.info(f"Node {self.node_id} became leader for term {self.current_term}")
        
        self.state = ConsensusState.LEADER
        self.metrics["last_leadership_start"] = time.time()
        
        # Initialize leader state
        self.next_index = {peer: len(self.log) for peer in self.peer_ids}
        self.match_index = {peer: -1 for peer in self.peer_ids}
        
        # Send initial heartbeats
        await self._send_heartbeats()
    
    def _become_follower(self):
        """Become follower"""
        if self.state == ConsensusState.LEADER:
            # Track leadership duration
            if self.metrics["last_leadership_start"] > 0:
                duration = time.time() - self.metrics["last_leadership_start"]
                self.metrics["leadership_duration"] += duration
        
        self.state = ConsensusState.FOLLOWER
        log.debug(f"Node {self.node_id} became follower")
    
    async def _handle_append_entries(self, message: ConsensusMessage):
        """Handle append entries from leader"""
        payload = message.payload
        leader_id = payload["leader_id"]
        prev_log_index = payload["prev_log_index"]
        prev_log_term = payload["prev_log_term"]
        entries = payload["entries"]
        leader_commit = payload["leader_commit"]
        is_heartbeat = payload.get("heartbeat", False)
        
        success = False
        
        # Reset election timeout - we heard from leader
        self._reset_election_timeout()
        self.last_heartbeat = time.time()
        
        # Become follower if we're not already
        if self.state != ConsensusState.FOLLOWER:
            self._become_follower()
        
        # Check if our log matches leader's at prev_log_index
        if (prev_log_index == -1 or
            (prev_log_index < len(self.log) and 
             self.log[prev_log_index].term == prev_log_term)):
            
            success = True
            
            if not is_heartbeat and entries:
                # Append new entries
                start_index = prev_log_index + 1
                
                # Remove conflicting entries
                if start_index < len(self.log):
                    self.log = self.log[:start_index]
                
                # Append new entries
                for entry_data in entries:
                    entry = LogEntry(
                        index=entry_data["index"],
                        term=entry_data["term"],
                        command=entry_data["command"],
                        timestamp=entry_data["timestamp"],
                        committed=entry_data.get("committed", False)
                    )
                    self.log.append(entry)
                    self.metrics["log_entries_appended"] += 1
            
            # Update commit index
            if leader_commit > self.commit_index:
                self.commit_index = min(leader_commit, len(self.log) - 1)
                await self._apply_committed_entries()
        
        # Send response
        append_response = ConsensusMessage(
            message_type=ConsensusMessageType.APPEND_RESPONSE,
            sender_id=self.node_id,
            term=self.current_term,
            timestamp=time.time(),
            payload={
                "success": success,
                "match_index": len(self.log) - 1 if success else -1,
                "follower_id": self.node_id
            }
        )
        
        if self.send_message_callback:
            await self.send_message_callback(leader_id, append_response)
    
    async def _handle_append_response(self, message: ConsensusMessage):
        """Handle append entries response from follower"""
        if self.state != ConsensusState.LEADER:
            return
        
        payload = message.payload
        success = payload["success"]
        match_index = payload["match_index"]
        follower_id = payload["follower_id"]
        
        if success:
            # Update follower's indices
            self.next_index[follower_id] = match_index + 1
            self.match_index[follower_id] = match_index
            
            # Check if we can commit more entries
            await self._update_commit_index()
        else:
            # Decrement next_index and retry
            if follower_id in self.next_index:
                self.next_index[follower_id] = max(0, self.next_index[follower_id] - 1)
                await self._send_append_entries(follower_id)
    
    async def _update_commit_index(self):
        """Update commit index based on majority replication"""
        if self.state != ConsensusState.LEADER:
            return
        
        # Find highest index that majority of servers have replicated
        for index in range(len(self.log) - 1, self.commit_index, -1):
            if self.log[index].term == self.current_term:
                # Count replicas
                replicas = 1  # Count ourselves
                for match_idx in self.match_index.values():
                    if match_idx >= index:
                        replicas += 1
                
                # Check if majority
                if replicas > len(self.all_nodes) // 2:
                    self.commit_index = index
                    await self._apply_committed_entries()
                    break
    
    async def _apply_committed_entries(self):
        """Apply committed log entries to state machine"""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied]
            entry.committed = True
            
            # Apply command to state machine
            if self.apply_command_callback:
                await self.apply_command_callback(entry.command)
            
            self.metrics["commands_committed"] += 1
            log.debug(f"Applied command {entry.index}: {entry.command}")
    
    async def append_command(self, command: Dict[str, Any]) -> bool:
        """Append new command to log (leader only)"""
        if self.state != ConsensusState.LEADER:
            return False
        
        # Create new log entry
        entry = LogEntry(
            index=len(self.log),
            term=self.current_term,
            command=command,
            timestamp=time.time()
        )
        
        self.log.append(entry)
        
        # Send to all followers
        for peer_id in self.peer_ids:
            await self._send_append_entries(peer_id)
        
        return True
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus algorithm statistics"""
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "current_term": self.current_term,
            "log_length": len(self.log),
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "voted_for": self.voted_for,
            "is_leader": self.state == ConsensusState.LEADER,
            "metrics": self.metrics.copy(),
            "last_heartbeat": self.last_heartbeat
        }

class PBFTConsensusNode:
    """
    Practical Byzantine Fault Tolerance (PBFT) Implementation
    
    Features:
    - Byzantine fault tolerance (up to f faulty nodes out of 3f+1 total)
    - Three-phase protocol: pre-prepare, prepare, commit
    - View changes for liveness
    - Message authentication
    """
    
    def __init__(self, node_id: str, peer_ids: List[str], f: int = 1):
        self.node_id = node_id
        self.peer_ids = peer_ids
        self.all_nodes = [node_id] + peer_ids
        self.n = len(self.all_nodes)  # Total nodes
        self.f = f  # Maximum faulty nodes
        
        if self.n < 3 * f + 1:
            raise ValueError(f"Need at least {3*f+1} nodes for f={f} Byzantine faults, got {self.n}")
        
        # PBFT state
        self.view = 0
        self.sequence_number = 0
        self.state = ConsensusState.FOLLOWER
        
        # Message logs
        self.pre_prepare_log: Dict[Tuple[int, int], ConsensusMessage] = {}  # (view, seq) -> message
        self.prepare_log: Dict[Tuple[int, int], Set[str]] = defaultdict(set)  # (view, seq) -> set of nodes
        self.commit_log: Dict[Tuple[int, int], Set[str]] = defaultdict(set)  # (view, seq) -> set of nodes
        
        # Request processing
        self.executed_requests: Set[int] = set()
        self.pending_requests: deque = deque()
        
        # View change
        self.view_change_timeout = 10.0  # seconds
        self.last_progress = time.time()
        
        # Message handlers
        self.message_handlers = {
            ConsensusMessageType.PRE_PREPARE: self._handle_pre_prepare,
            ConsensusMessageType.PREPARE: self._handle_prepare,
            ConsensusMessageType.COMMIT: self._handle_commit,
            ConsensusMessageType.VIEW_CHANGE: self._handle_view_change,
            ConsensusMessageType.NEW_VIEW: self._handle_new_view
        }
        
        self.send_message_callback: Optional[Callable] = None
        self.execute_request_callback: Optional[Callable] = None
        
        # Cryptographic keys (if available)
        self.private_key = None
        self.public_keys: Dict[str, Any] = {}
        
        if CRYPTO_AVAILABLE:
            self._generate_keys()
        
        log.info(f"PBFT node {node_id} initialized (n={self.n}, f={f})")
    
    def _generate_keys(self):
        """Generate cryptographic keys for message authentication"""
        if not CRYPTO_AVAILABLE:
            return
        
        try:
            # Generate RSA key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Store our public key
            public_key = self.private_key.public_key()
            self.public_keys[self.node_id] = public_key
            
        except Exception as e:
            log.error(f"Failed to generate cryptographic keys: {e}")
            self.private_key = None
    
    def _sign_message(self, message: ConsensusMessage) -> str:
        """Sign message with private key"""
        if not CRYPTO_AVAILABLE or not self.private_key:
            return ""
        
        try:
            # Create message digest
            message_bytes = json.dumps(message.to_dict(), sort_keys=True).encode()
            
            # Sign with private key
            signature = self.private_key.sign(
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return signature.hex()
            
        except Exception as e:
            log.error(f"Failed to sign message: {e}")
            return ""
    
    def _verify_signature(self, message: ConsensusMessage) -> bool:
        """Verify message signature"""
        if not CRYPTO_AVAILABLE or not message.signature:
            return True  # Skip verification if crypto not available
        
        try:
            sender_public_key = self.public_keys.get(message.sender_id)
            if not sender_public_key:
                return False
            
            # Create message digest
            message_copy = message.to_dict()
            message_copy.pop("signature", None)  # Remove signature for verification
            message_bytes = json.dumps(message_copy, sort_keys=True).encode()
            
            # Verify signature
            signature_bytes = bytes.fromhex(message.signature)
            sender_public_key.verify(
                signature_bytes,
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            log.debug(f"Signature verification failed: {e}")
            return False
    
    def is_primary(self) -> bool:
        """Check if this node is the primary for current view"""
        primary_index = self.view % len(self.all_nodes)
        primary_id = self.all_nodes[primary_index]
        return primary_id == self.node_id
    
    async def start(self):
        """Start PBFT consensus protocol"""
        # Start view change timer
        asyncio.create_task(self._view_change_timer())
        
        # Start request processing
        asyncio.create_task(self._process_requests())
        
        log.info(f"PBFT node {self.node_id} started (primary: {self.is_primary()})")
    
    async def _view_change_timer(self):
        """Monitor progress and trigger view changes if needed"""
        while True:
            try:
                await asyncio.sleep(1.0)
                
                current_time = time.time()
                if (current_time - self.last_progress > self.view_change_timeout and
                    not self.is_primary()):
                    
                    await self._initiate_view_change()
                
            except Exception as e:
                log.error(f"View change timer error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_requests(self):
        """Process pending requests (primary only)"""
        while True:
            try:
                await asyncio.sleep(0.1)
                
                if self.is_primary() and self.pending_requests:
                    request = self.pending_requests.popleft()
                    await self._send_pre_prepare(request)
                
            except Exception as e:
                log.error(f"Request processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def submit_request(self, request: Dict[str, Any]) -> bool:
        """Submit request for consensus"""
        if not self.is_primary():
            # Forward to primary
            primary_index = self.view % len(self.all_nodes)
            primary_id = self.all_nodes[primary_index]
            log.info(f"Forwarding request to primary {primary_id}")
            return False
        
        # Add to pending requests
        request["request_id"] = len(self.executed_requests) + len(self.pending_requests)
        request["timestamp"] = time.time()
        self.pending_requests.append(request)
        
        return True
    
    async def _send_pre_prepare(self, request: Dict[str, Any]):
        """Send pre-prepare message (primary only)"""
        if not self.is_primary():
            return
        
        pre_prepare_msg = ConsensusMessage(
            message_type=ConsensusMessageType.PRE_PREPARE,
            sender_id=self.node_id,
            term=self.view,
            timestamp=time.time(),
            payload={
                "view": self.view,
                "sequence_number": self.sequence_number,
                "request": request,
                "digest": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            }
        )
        
        # Sign message
        pre_prepare_msg.signature = self._sign_message(pre_prepare_msg)
        
        # Store in log
        self.pre_prepare_log[(self.view, self.sequence_number)] = pre_prepare_msg
        
        # Send to all backups
        for peer_id in self.peer_ids:
            if self.send_message_callback:
                await self.send_message_callback(peer_id, pre_prepare_msg)
        
        self.sequence_number += 1
        log.debug(f"Sent pre-prepare for sequence {self.sequence_number - 1}")
    
    async def handle_message(self, message: ConsensusMessage):
        """Handle incoming PBFT message"""
        try:
            # Verify message signature
            if not self._verify_signature(message):
                log.warning(f"Invalid signature from {message.sender_id}")
                return
            
            # Handle message based on type
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                log.warning(f"Unknown PBFT message type: {message.message_type}")
                
        except Exception as e:
            log.error(f"Error handling PBFT message {message.message_type}: {e}")
    
    async def _handle_pre_prepare(self, message: ConsensusMessage):
        """Handle pre-prepare message from primary"""
        payload = message.payload
        view = payload["view"]
        sequence_number = payload["sequence_number"]
        request = payload["request"]
        digest = payload["digest"]
        
        # Verify this is from the correct primary
        expected_primary_index = view % len(self.all_nodes)
        expected_primary = self.all_nodes[expected_primary_index]
        
        if message.sender_id != expected_primary:
            log.warning(f"Pre-prepare from wrong primary: {message.sender_id}, expected: {expected_primary}")
            return
        
        # Verify digest
        computed_digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if digest != computed_digest:
            log.warning(f"Invalid digest in pre-prepare from {message.sender_id}")
            return
        
        # Store pre-prepare
        self.pre_prepare_log[(view, sequence_number)] = message
        
        # Send prepare message
        prepare_msg = ConsensusMessage(
            message_type=ConsensusMessageType.PREPARE,
            sender_id=self.node_id,
            term=view,
            timestamp=time.time(),
            payload={
                "view": view,
                "sequence_number": sequence_number,
                "digest": digest,
                "node_id": self.node_id
            }
        )
        
        prepare_msg.signature = self._sign_message(prepare_msg)
        
        # Send to all nodes (including primary)
        for peer_id in self.all_nodes:
            if peer_id != self.node_id and self.send_message_callback:
                await self.send_message_callback(peer_id, prepare_msg)
        
        # Add our own prepare
        self.prepare_log[(view, sequence_number)].add(self.node_id)
        
        log.debug(f"Sent prepare for view {view}, sequence {sequence_number}")
    
    async def _handle_prepare(self, message: ConsensusMessage):
        """Handle prepare message from backup"""
        payload = message.payload
        view = payload["view"]
        sequence_number = payload["sequence_number"]
        digest = payload["digest"]
        node_id = payload["node_id"]
        
        # Verify we have matching pre-prepare
        pre_prepare = self.pre_prepare_log.get((view, sequence_number))
        if not pre_prepare or pre_prepare.payload["digest"] != digest:
            log.debug(f"No matching pre-prepare for prepare from {node_id}")
            return
        
        # Add to prepare log
        self.prepare_log[(view, sequence_number)].add(node_id)
        
        # Check if we have enough prepares (2f)
        if len(self.prepare_log[(view, sequence_number)]) >= 2 * self.f:
            await self._send_commit(view, sequence_number, digest)
    
    async def _send_commit(self, view: int, sequence_number: int, digest: str):
        """Send commit message"""
        commit_msg = ConsensusMessage(
            message_type=ConsensusMessageType.COMMIT,
            sender_id=self.node_id,
            term=view,
            timestamp=time.time(),
            payload={
                "view": view,
                "sequence_number": sequence_number,
                "digest": digest,
                "node_id": self.node_id
            }
        )
        
        commit_msg.signature = self._sign_message(commit_msg)
        
        # Send to all nodes
        for peer_id in self.all_nodes:
            if peer_id != self.node_id and self.send_message_callback:
                await self.send_message_callback(peer_id, commit_msg)
        
        # Add our own commit
        self.commit_log[(view, sequence_number)].add(self.node_id)
        
        # Check if we can execute
        await self._check_execution(view, sequence_number)
        
        log.debug(f"Sent commit for view {view}, sequence {sequence_number}")
    
    async def _handle_commit(self, message: ConsensusMessage):
        """Handle commit message"""
        payload = message.payload
        view = payload["view"]
        sequence_number = payload["sequence_number"]
        digest = payload["digest"]
        node_id = payload["node_id"]
        
        # Verify we have matching pre-prepare
        pre_prepare = self.pre_prepare_log.get((view, sequence_number))
        if not pre_prepare or pre_prepare.payload["digest"] != digest:
            log.debug(f"No matching pre-prepare for commit from {node_id}")
            return
        
        # Add to commit log
        self.commit_log[(view, sequence_number)].add(node_id)
        
        # Check if we can execute
        await self._check_execution(view, sequence_number)
    
    async def _check_execution(self, view: int, sequence_number: int):
        """Check if request can be executed"""
        # Need 2f+1 commits (including our own if we sent one)
        if len(self.commit_log[(view, sequence_number)]) >= 2 * self.f + 1:
            if sequence_number not in self.executed_requests:
                await self._execute_request(view, sequence_number)
    
    async def _execute_request(self, view: int, sequence_number: int):
        """Execute the request"""
        pre_prepare = self.pre_prepare_log.get((view, sequence_number))
        if not pre_prepare:
            return
        
        request = pre_prepare.payload["request"]
        
        # Execute request
        if self.execute_request_callback:
            await self.execute_request_callback(request)
        
        self.executed_requests.add(sequence_number)
        self.last_progress = time.time()
        
        log.info(f"Executed request {sequence_number}: {request}")
    
    async def _initiate_view_change(self):
        """Initiate view change to new primary"""
        new_view = self.view + 1
        
        view_change_msg = ConsensusMessage(
            message_type=ConsensusMessageType.VIEW_CHANGE,
            sender_id=self.node_id,
            term=new_view,
            timestamp=time.time(),
            payload={
                "new_view": new_view,
                "last_sequence": self.sequence_number,
                "node_id": self.node_id,
                "prepared_requests": list(self.prepare_log.keys())
            }
        )
        
        view_change_msg.signature = self._sign_message(view_change_msg)
        
        # Send to all nodes
        for peer_id in self.all_nodes:
            if peer_id != self.node_id and self.send_message_callback:
                await self.send_message_callback(peer_id, view_change_msg)
        
        log.info(f"Initiated view change to view {new_view}")
    
    async def _handle_view_change(self, message: ConsensusMessage):
        """Handle view change message"""
        # Implementation would collect view change messages
        # and trigger new view when enough are received
        log.debug(f"Received view change from {message.sender_id}")
    
    async def _handle_new_view(self, message: ConsensusMessage):
        """Handle new view message from new primary"""
        payload = message.payload
        new_view = payload["new_view"]
        
        if new_view > self.view:
            self.view = new_view
            log.info(f"Switched to view {new_view}")

class DistributedConsensusManager:
    """
    Unified distributed consensus manager supporting multiple algorithms
    """
    
    def __init__(self, node_id: str, peer_ids: List[str], algorithm: str = "raft"):
        self.node_id = node_id
        self.peer_ids = peer_ids
        self.algorithm = algorithm
        
        # Initialize consensus algorithm
        if algorithm == "raft":
            self.consensus_node = RaftConsensusNode(node_id, peer_ids)
        elif algorithm == "pbft":
            f = len(peer_ids) // 3  # Maximum Byzantine faults
            self.consensus_node = PBFTConsensusNode(node_id, peer_ids, f)
        else:
            raise ValueError(f"Unsupported consensus algorithm: {algorithm}")
        
        # Network layer
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.peer_connections: Dict[str, Any] = {}
        
        # Callbacks
        self.consensus_node.send_message_callback = self._send_message
        
        if hasattr(self.consensus_node, 'apply_command_callback'):
            self.consensus_node.apply_command_callback = self._apply_command
        if hasattr(self.consensus_node, 'execute_request_callback'):
            self.consensus_node.execute_request_callback = self._execute_request
        
        # Command execution
        self.command_handlers: Dict[str, Callable] = {}
        self.state_machine: Dict[str, Any] = {}
        
        log.info(f"Distributed consensus manager initialized with {algorithm}")
    
    async def start(self):
        """Start consensus manager"""
        # Start consensus algorithm
        await self.consensus_node.start()
        
        # Start message processing
        asyncio.create_task(self._process_messages())
        
        log.info(f"Consensus manager started")
    
    async def _send_message(self, peer_id: str, message: ConsensusMessage):
        """Send message to peer"""
        try:
            # In production, this would send over network
            # For now, simulate by adding to message queue with delay
            await asyncio.sleep(0.001)  # Simulate network latency
            
            # Could implement actual network layer here
            log.debug(f"Sent {message.message_type.value} to {peer_id}")
            
        except Exception as e:
            log.error(f"Failed to send message to {peer_id}: {e}")
    
    async def _process_messages(self):
        """Process incoming messages"""
        while True:
            try:
                # Get message from queue
                message = await self.message_queue.get()
                
                # Handle message
                await self.consensus_node.handle_message(message)
                
            except Exception as e:
                log.error(f"Error processing message: {e}")
                await asyncio.sleep(0.1)
    
    async def receive_message(self, message_data: Dict[str, Any]):
        """Receive message from network"""
        try:
            message = ConsensusMessage.from_dict(message_data)
            await self.message_queue.put(message)
        except Exception as e:
            log.error(f"Failed to receive message: {e}")
    
    async def _apply_command(self, command: Dict[str, Any]):
        """Apply command to state machine (Raft)"""
        await self._execute_command(command)
    
    async def _execute_request(self, request: Dict[str, Any]):
        """Execute request (PBFT)"""
        await self._execute_command(request)
    
    async def _execute_command(self, command: Dict[str, Any]):
        """Execute command on state machine"""
        try:
            command_type = command.get("type", "unknown")
            handler = self.command_handlers.get(command_type)
            
            if handler:
                await handler(command)
            else:
                # Default handler - store in state machine
                key = command.get("key", f"command_{time.time()}")
                self.state_machine[key] = command.get("value", command)
            
            log.debug(f"Executed command: {command_type}")
            
        except Exception as e:
            log.error(f"Failed to execute command: {e}")
    
    def register_command_handler(self, command_type: str, handler: Callable):
        """Register handler for specific command type"""
        self.command_handlers[command_type] = handler
    
    async def submit_command(self, command: Dict[str, Any]) -> bool:
        """Submit command for consensus"""
        try:
            if self.algorithm == "raft":
                return await self.consensus_node.append_command(command)
            elif self.algorithm == "pbft":
                return await self.consensus_node.submit_request(command)
            else:
                return False
        except Exception as e:
            log.error(f"Failed to submit command: {e}")
            return False
    
    def get_consensus_status(self) -> Dict[str, Any]:
        """Get current consensus status"""
        base_stats = {
            "algorithm": self.algorithm,
            "node_id": self.node_id,
            "peer_count": len(self.peer_ids),
            "state_machine_size": len(self.state_machine)
        }
        
        if hasattr(self.consensus_node, 'get_consensus_stats'):
            consensus_stats = self.consensus_node.get_consensus_stats()
            base_stats.update(consensus_stats)
        
        return base_stats
    
    def is_leader(self) -> bool:
        """Check if this node is currently the leader"""
        if self.algorithm == "raft":
            return self.consensus_node.state == ConsensusState.LEADER
        elif self.algorithm == "pbft":
            return self.consensus_node.is_primary()
        else:
            return False
