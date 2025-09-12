from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

@dataclass(frozen=True)
class Op:
    key: str
    value: Any
    ts: int  # lamport-like monotonic timestamp
    actor: str

class LWWMap:
    """
    Simple Last-Write-Wins CRDT Map.
    Keeps per-key (value, ts, actor). Merge prefers greater ts; on tie, lexicographically higher actor.
    """
    def __init__(self) -> None:
        self._data: Dict[str, Tuple[Any, int, str]] = {}
        self._clock: int = 0

    def clock(self) -> int:
        self._clock += 1
        return self._clock

    def apply(self, op: Op) -> None:
        cur = self._data.get(op.key)
        if cur is None:
            self._data[op.key] = (op.value, op.ts, op.actor)
            return
        _, cts, cactor = cur
        if op.ts > cts or (op.ts == cts and op.actor > cactor):
            self._data[op.key] = (op.value, op.ts, op.actor)

    def set(self, key: str, value: Any, actor: str) -> Op:
        ts = self.clock()
        op = Op(key=key, value=value, ts=ts, actor=actor)
        self.apply(op)
        return op

    def get(self, key: str, default: Any = None) -> Any:
        cur = self._data.get(key)
        return cur[0] if cur else default

    def merge(self, other: "LWWMap") -> None:
        for k, (v, ts, actor) in other._data.items():
            self.apply(Op(k, v, ts, actor))

    def to_dict(self) -> Dict[str, Any]:
        return {k: v[0] for k, v in self._data.items()}