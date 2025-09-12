from __future__ import annotations
from typing import Any, Dict, List
import os
from swarm.config import settings
from swarm.memory.mesh import MemoryMesh
try:
    from swarm.memory.mesh_dist import DistMemoryMesh
except Exception:
    DistMemoryMesh = None  # optional
from swarm.capabilities.registry import registry
try:
    from swarm.vector import service as vector_service
except Exception:
    vector_service = None

class Agent:
    def __init__(self, name: str, scope: str) -> None:
        self.name = name
        # Select distributed memory when enabled; fallback to local mesh
        mem_mode = os.getenv("MEMORY_MESH_MODE", "local").lower()
        if mem_mode == "dist" and DistMemoryMesh:
            self.memory = DistMemoryMesh(scope=scope, actor=name)
        else:
            self.memory = MemoryMesh(scope=scope, actor=name)

    def _maybe_vector_upsert(self, key: str, value: Any) -> None:
        if not vector_service:
            return
        try:
            content = value if isinstance(value, str) else str(value)
            vector_service.upsert(
                scope=self.memory.scope,
                key=f"{key}:{self.name}",
                content=content,
                meta={"agent": self.name, "capability_key": key},
                ttl_seconds=int(os.getenv("VECTOR_TTL_SECONDS", "604800")),  # 7d default
            )
        except Exception:
            # Non-fatal on vector persistence
            pass

    def run_step(self, capability: str, args: Dict[str, Any]) -> Any:
        cap = registry.get(capability)
        if not cap:
            return {"error": f"capability {capability} not found"}
        result = cap.func(**args)
        key = f"result:{capability}"
        self.memory.set(key, result)
        self._maybe_vector_upsert(key, result)
        return result

class AgentFactory:
    def create(self, name: str, scope: str) -> Agent:
        return Agent(name=name, scope=scope)