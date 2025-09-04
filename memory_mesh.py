from __future__ import annotations
from typing import Any, Dict, Optional
import os
import json
import time

try:
    import redis
except Exception:  # Redis optional
    redis = None  # type: ignore


class MemoryMesh:
    def __init__(self, ns: str = "default"):
        url = (
            os.getenv("REDIS_URL")
            or f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/{os.getenv('REDIS_DB', '0')}"
        )
        self.ns = ns
        self.stream = f"{ns}:events"
        self.kv_prefix = f"{ns}:kv:"
        self._kv: Dict[str, str] = {}
        self._events: list[Dict[str, Any]] = []
        self.r = None
        # Attempt to connect and ping; fall back to in-memory when unavailable
        try:
            if redis is not None:
                client = redis.Redis.from_url(url, decode_responses=True)
                try:
                    client.ping()
                    self.r = client
                except Exception:
                    self.r = None
        except Exception:
            self.r = None

    def get(self, key: str) -> Optional[str]:
        rkey = self.kv_prefix + key
        try:
            if self.r:
                return self.r.get(rkey)
        except Exception:
            pass
        return self._kv.get(rkey)

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        rkey = self.kv_prefix + key
        try:
            if self.r:
                if ttl:
                    self.r.set(rkey, value, ex=ttl)
                else:
                    self.r.set(rkey, value)
                return
        except Exception:
            pass
        self._kv[rkey] = value

    def publish(self, topic: str, payload: Dict[str, Any]) -> str:
        event = {
            "ts": str(time.time()),
            "topic": topic,
            "payload": json.dumps(payload),
        }
        try:
            if self.r:
                return self.r.xadd(self.stream, event)
        except Exception:
            pass
        # In-memory fallback; synthesize an ID
        self._events.append(event)
        return f"{self.stream}-{len(self._events)}"

    def subscribe(self, group: str, consumer: str, block_ms: int = 1000):
        if not self.r:
            # No-op generator when Redis is unavailable
            if False:
                yield {}
            return
        try:
            self.r.xgroup_create(self.stream, group, mkstream=True)
        except Exception:
            pass
        while True:
            try:
                res = self.r.xreadgroup(
                    group, consumer, {self.stream: ">"}, count=64, block=block_ms
                )
            except Exception:
                res = []
            if not res:
                yield from ()
            else:
                for _, msgs in res:
                    for msg_id, fields in msgs:
                        yield {"id": msg_id, "fields": fields}
                        try:
                            self.r.xack(self.stream, group, msg_id)
                        except Exception:
                            pass
