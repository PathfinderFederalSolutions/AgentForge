from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
import time
import hashlib
import json

# Helpers to avoid lambda (pylint unnecessary-lambda)

def _uuid_hex() -> str:
    return uuid.uuid4().hex

def _now() -> float:
    return time.time()

# Core task specification coming from client
class TaskSpec(BaseModel):
    id: str = Field(default_factory=_uuid_hex)
    goal: str
    priority: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ToolInvocation(BaseModel):
    invocation_id: str = Field(default_factory=_uuid_hex)
    task_id: str
    tool: str
    args: Dict[str, Any]
    attempt: int = 1
    max_attempts: int = 3
    created_ts: float = Field(default_factory=_now)
    trace_id: Optional[str] = None
    parent_invocation_id: Optional[str] = None

class ToolResult(BaseModel):
    invocation_id: str
    task_id: str
    tool: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    attempt: int = 1
    latency_ms: float = 0.0
    trace_id: Optional[str] = None
    started_ts: float = Field(default_factory=_now)
    completed_ts: float = Field(default_factory=_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AggregatedResult(BaseModel):
    task_id: str
    goal: str
    results: List[ToolResult]
    final_artifact: Any = None
    started_ts: float = Field(default_factory=_now)
    completed_ts: float = Field(default_factory=_now)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    lineage: List[Dict[str, Any]] = Field(default_factory=list)

class BacklogMetric(BaseModel):
    subject: str
    pending: int
    ts: float = Field(default_factory=_now)

class DAGNode(BaseModel):
    node_id: str
    capability: str
    args: Dict[str, Any] = Field(default_factory=dict)

class DAGEdge(BaseModel):
    source: str
    target: str

class DAGSpec(BaseModel):
    goal: str
    seed: int
    nodes: List[DAGNode]
    edges: List[DAGEdge]
    created_ts: float = Field(default_factory=_now)
    hash: Optional[str] = None
    latency_budget_ms: Optional[int] = None

    def compute_hash(self) -> str:
        if self.hash:
            return self.hash
        # Canonical serialization (order by node_id, edges tuple order)
        payload = {
            "goal": self.goal,
            "seed": self.seed,
            "nodes": [{"id": n.node_id, "cap": n.capability, "args": n.args} for n in sorted(self.nodes, key=lambda x: x.node_id)],
            "edges": [(e.source, e.target) for e in sorted(self.edges, key=lambda x: (x.source, x.target))],
            "latency_budget_ms": self.latency_budget_ms,
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        self.hash = hashlib.sha256(raw).hexdigest()
        return self.hash

__all__ = [
    'TaskSpec','ToolInvocation','ToolResult','AggregatedResult','BacklogMetric', 'DAGNode', 'DAGEdge', 'DAGSpec'
]
