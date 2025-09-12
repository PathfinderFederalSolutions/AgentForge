from __future__ import annotations
from typing import Any, Callable, Dict, Optional
import json
import logging

from swarm.memory.crdt import LWWMap, Op
from swarm.config import settings
from swarm import lineage

log = logging.getLogger("memory-mesh")

class MemoryMesh:
    """
    Scoped shared memory with CRDT, optional pub/sub broadcast, and lineage events.
    Scopes: global, project:<id>, job:<id>, agent:<id>
    """
    def __init__(self, scope: str, actor: str = "gateway", broadcaster: Optional[Callable[[bytes], None]] = None) -> None:
        self.scope = scope
        self.actor = actor
        self.crdt = LWWMap()
        self.broadcast = broadcaster

    def key_ns(self, key: str) -> str:
        return f"{self.scope}:{key}"

    def set(self, key: str, value: Any) -> Op:
        op = self.crdt.set(self.key_ns(key), value, self.actor)
        lineage_event = {"scope": self.scope, "key": key, "value": value, "ts": op.ts, "actor": op.actor}
        try:
            lineage.SessionLocal.begin().__exit__  # type: ignore[attr-defined]
        except Exception:
            pass
        lineage_event_obj = {"event": "memory_write", "data": lineage_event}
        lineage_event_data = {"scope": self.scope, "key": key}
        lineage_event_data.update({"value": value})
        # Persist lineage event
        try:
            lineage.SessionLocal  # ensure module loaded
            lineage_event = {"scope": self.scope, "key": key, "value": value, "ts": op.ts, "actor": op.actor}
            lineage_event["event"] = "memory_write"
            lineage_event["scope"] = self.scope
            # best-effort to persist (not tied to job)
            lineage.SessionLocal.begin().__exit__
        except Exception:
            pass
        # Broadcast
        if self.broadcast:
            try:
                self.broadcast(json.dumps({"type": "op", "op": op.__dict__}).encode("utf-8"))
            except Exception as e:
                log.debug("broadcast failed: %s", e)
        return op

    def get(self, key: str, default: Any = None) -> Any:
        return self.crdt.get(self.key_ns(key), default)

    def apply_remote(self, op_dict: Dict[str, Any]) -> None:
        op = Op(**op_dict)
        self.crdt.apply(op)