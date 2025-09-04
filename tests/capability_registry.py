from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Union
import importlib
import threading
import time


HandlerT = Union[str, Callable[..., Any]]  # "pkg.mod:ClassOrFunc" or callable


@dataclass
class Capability:
    name: str
    handler: HandlerT
    version: str = "1.0.0"
    provides: List[str] = field(default_factory=list)     # logical capabilities
    requires: List[str] = field(default_factory=list)     # dependencies
    aliases: List[str] = field(default_factory=list)      # alternate names
    tags: List[str] = field(default_factory=list)         # search hints
    qos: Dict[str, Any] = field(default_factory=dict)     # latency/throughput/etc
    cost: Dict[str, Any] = field(default_factory=dict)    # unit cost estimates
    resources: Dict[str, Any] = field(default_factory=dict)  # gpu/cpu/ram hints
    health: Dict[str, Any] = field(default_factory=lambda: {"status": "unknown", "ts": time.time()})

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # hide non-serializable handler
        d["handler"] = self._handler_repr()
        return d

    def _handler_repr(self) -> str:
        if isinstance(self.handler, str):
            return self.handler
        if callable(self.handler):
            return getattr(self.handler, "__name__", "callable")
        return type(self.handler).__name__


class CapabilityRegistry:
    def __init__(self) -> None:
        self._caps: Dict[str, Capability] = {}
        self._index: Dict[str, str] = {}  # alias -> name
        self._lock = threading.RLock()

    def register(self, cap: Capability) -> None:
        with self._lock:
            self._caps[cap.name] = cap
            self._index[cap.name] = cap.name
            for a in cap.aliases + cap.provides + cap.tags:
                self._index.setdefault(a, cap.name)

    def register_capability(
        self,
        name: str,
        handler: HandlerT,
        *,
        version: str = "1.0.0",
        provides: Optional[List[str]] = None,
        requires: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        qos: Optional[Dict[str, Any]] = None,
        cost: Optional[Dict[str, Any]] = None,
        resources: Optional[Dict[str, Any]] = None,
    ) -> None:
        cap = Capability(
            name=name,
            handler=handler,
            version=version,
            provides=provides or [],
            requires=requires or [],
            aliases=aliases or [],
            tags=tags or [],
            qos=qos or {},
            cost=cost or {},
            resources=resources or {},
        )
        self.register(cap)

    def resolve_capability(self, key: str) -> Optional[Capability]:
        with self._lock:
            name = self._index.get(key, key)
            return self._caps.get(name)

    def list(self) -> List[Capability]:
        with self._lock:
            return list(self._caps.values())

    def find_by_tags(self, *tags: str) -> List[Capability]:
        with self._lock:
            res = []
            for c in self._caps.values():
                if all(t in c.tags for t in tags):
                    res.append(c)
            return res

    def update_health(self, name_or_alias: str, status: str, **extra: Any) -> None:
        with self._lock:
            cap = self.resolve_capability(name_or_alias)
            if not cap:
                return
            cap.health.update({"status": status, "ts": time.time(), **extra})

    def _load_handler(self, cap: Capability):
        # String entrypoint: "pkg.mod:ClassOrFunc"
        if isinstance(cap.handler, str):
            mod, _, sym = cap.handler.partition(":")
            if not sym:
                m = importlib.import_module(mod)
                return getattr(m, "default", m)
            m = importlib.import_module(mod)
            return getattr(m, sym)
        return cap.handler

    def create_from_capability(self, key: str, **kwargs) -> Any:
        cap = self.resolve_capability(key)
        if not cap:
            return None
        handler = self._load_handler(cap)
        # Prefer conventional constructors
        if hasattr(handler, "from_spec") and callable(getattr(handler, "from_spec")):
            return handler.from_spec(**kwargs)
        if hasattr(handler, "__call__"):
            return handler(**kwargs)  # class or factory function
        return handler

# Global singleton
_global_registry: Optional[CapabilityRegistry] = None

def get_registry() -> CapabilityRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = CapabilityRegistry()
    return _global_registry

def capability(name: str, **meta):
    """Decorator to register a callable/class as a capability."""
    def _wrap(fn_or_cls):
        get_registry().register_capability(name=name, handler=fn_or_cls, **meta)
        return fn_or_cls
    return _wrap