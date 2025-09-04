from __future__ import annotations
from typing import Any, Dict, Iterable, Optional
import os, json, time, redis

class MemoryMesh:
    def __init__(self, ns: str = "default"):
        url = os.getenv("REDIS_URL") or f"redis://{os.getenv('REDIS_HOST','localhost')}:{os.getenv('REDIS_PORT','6379')}/{os.getenv('REDIS_DB','0')}"
        self.ns = ns
        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.stream = f"{ns}:events"
        self.kv_prefix = f"{ns}:kv:"

    def get(self, key: str) -> Optional[str]:
        return self.r.get(self.kv_prefix + key)

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        self.r.set(self.kv_prefix + key, value, ex=ttl)

    def publish(self, topic: str, payload: Dict[str, Any]) -> str:
        event = {"ts": str(time.time()), "topic": topic, "payload": json.dumps(payload)}
        return self.r.xadd(self.stream, event)

    def subscribe(self, group: str, consumer: str, block_ms: int = 1000) -> Iterable[Dict[str, Any]]:
        try:
            self.r.xgroup_create(self.stream, group, mkstream=True)
        except redis.exceptions.ResponseError:
            pass
        while True:
            res = self.r.xreadgroup(group, consumer, {self.stream: ">"}, count=64, block=block_ms)
            if not res:
                yield from ()
            else:
                for _, msgs in res:
                    for msg_id, fields in msgs:
                        yield {"id": msg_id, "fields": fields}
                        self.r.xack(self.stream, group, msg_id)