from __future__ import annotations
import asyncio
import json
import logging
import time
from typing import Any, Dict

from swarm.memory.crdt import LWWMap, Op
from swarm.nats_conn import get_js
from swarm.storage import store

log = logging.getLogger("mesh-dist")

class DistMemoryMesh:
    def __init__(self, scope: str, actor: str = "gateway", ttl_seconds: int = 7*24*3600) -> None:
        self.scope = scope
        self.actor = actor
        self.crdt = LWWMap()
        self.ttl = ttl_seconds
        self._wal: list[Dict[str, Any]] = []
        self._tombstones: set[str] = set()

    @property
    def subj(self) -> str:
        # mesh.ops.scope.<scope>
        safe = self.scope.replace(":", "_").replace("/", "_")
        return f"mesh.ops.scope.{safe}"

    def _now_ts(self) -> int:
        return int(time.time())

    async def publish_op(self, op: Op, tombstone: bool = False) -> None:
        _, js = await get_js()
        payload = {"type": "op", "scope": self.scope, "op": op.__dict__, "tombstone": tombstone, "ts": self._now_ts()}
        msg_id = f"{self.scope}:{op.key}:{op.ts}"
        await js.publish(self.subj, json.dumps(payload).encode("utf-8"), headers={"Nats-Msg-Id": msg_id})
        # Drain handled by connection manager

    def set(self, key: str, value: Any) -> Op:
        op = self.crdt.set(key, value, self.actor)
        self._wal.append({"op": op.__dict__, "ts": self._now_ts()})
        asyncio.create_task(self.publish_op(op))
        return op

    def delete(self, key: str) -> None:
        # tombstone pattern: set None with later GC
        op = self.crdt.set(key, None, self.actor)
        self._tombstones.add(key)
        self._wal.append({"op": op.__dict__, "tombstone": True, "ts": self._now_ts()})
        asyncio.create_task(self.publish_op(op, tombstone=True))

    def get(self, key: str, default: Any = None) -> Any:
        val = self.crdt.get(key, default)
        if key in self._tombstones:
            return default
        return val

    async def read_repair(self) -> None:
        # periodic snapshot publish and compaction
        snap = {"scope": self.scope, "state": self.crdt.to_dict(), "tombstones": list(self._tombstones), "ttl": self.ttl, "ts": self._now_ts()}
        # persist snapshot to MinIO/S3
        data = json.dumps(snap).encode("utf-8")
        from io import BytesIO
        store.save_file(BytesIO(data), filename=f"mesh-{self.scope}.json", content_type="application/json")
        # clear W(A)L if successful
        self._wal.clear()

    async def subscribe_and_apply(self) -> None:
        _, js = await get_js()
        sub = await js.subscribe(self.subj)
        async for msg in sub.messages:
            try:
                payload = json.loads(msg.data.decode("utf-8"))
                if payload.get("type") == "op":
                    op = Op(**payload["op"])
                    self.crdt.apply(op)
                    if payload.get("tombstone"):
                        self._tombstones.add(op.key)
                await msg.ack()
            except Exception as e:
                log.debug("mesh apply failed: %s", e)
                await msg.nak()

async def _snapshot_loop(mesh: "DistMemoryMesh", interval_sec: int = 60):
    while True:
        try:
            await mesh.read_repair()
        except Exception as e:
            log.debug("snapshot failed for %s: %s", mesh.scope, e)
        await asyncio.sleep(interval_sec)

def start_snapshot_task(mesh: "DistMemoryMesh", interval_sec: int = 60) -> asyncio.Task:
    """
    Launch periodic snapshot compaction for this mesh.
    """
    return asyncio.create_task(_snapshot_loop(mesh, interval_sec))

async def start_subscribe_task(mesh: "DistMemoryMesh") -> asyncio.Task:
    """
    Subscribe to NATS subject for this mesh scope and apply ops.
    """
    async def _runner():
        await mesh.subscribe_and_apply()
    return asyncio.create_task(_runner())