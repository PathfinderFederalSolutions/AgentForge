from typing import Callable, Dict, Any
from .models import SwarmJob, TaskSpec
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
    def register(self, name: str, kind: str, validator: Callable | None = None):
        self._tools[name] = {"kind": kind, "validator": validator}
    def list(self):
        return list(self._tools.keys())
    def allowed_for(self, job: SwarmJob):
        if not job.tools_allowed:
            return self.list()
        return [t for t in self.list() if t in job.tools_allowed]
registry = ToolRegistry()
# Baseline tools
for t in ["browser.fetch","browser.render","code.exec","file.ingest","rag.search","review.validate"]:
    registry.register(t, kind="remote")