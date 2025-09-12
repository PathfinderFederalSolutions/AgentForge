from __future__ import annotations
from typing import Any, Callable, Dict, Optional

class Capability:
    def __init__(self, name: str, func: Callable[..., Any], inputs: Dict[str, str], outputs: Dict[str, str], desc: str = "", meta: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.func = func
        self.inputs = inputs
        self.outputs = outputs
        self.desc = desc
        self.meta = meta or {}

class CapabilityRegistry:
    def __init__(self) -> None:
        self._caps: Dict[str, Capability] = {}

    def register(self, cap: Capability) -> None:
        self._caps[cap.name] = cap

    def decorator(self, name: str, inputs: Dict[str, str], outputs: Dict[str, str], desc: str = "", meta: Optional[Dict[str, Any]] = None):
        def _wrap(fn: Callable[..., Any]):
            self.register(Capability(name, fn, inputs, outputs, desc, meta))
            return fn
        return _wrap

    def get(self, name: str) -> Optional[Capability]:
        return self._caps.get(name)

    def list(self) -> Dict[str, Capability]:
        return dict(self._caps)

registry = CapabilityRegistry()