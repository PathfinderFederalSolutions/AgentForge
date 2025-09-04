from __future__ import annotations
from typing import List, Dict
from agent_factory import AgentFactory
from memory_mesh import MemoryMesh

try:
    from capability_registry import get_registry
    _cap_registry = get_registry()
except Exception:
    _cap_registry = None

class TaskPlanner:
    def __init__(self, mesh: MemoryMesh, factory: AgentFactory):
        self.mesh = mesh
        self.factory = factory

    def decompose(self, goal: str) -> List[Dict]:
        parts = [s.strip() for s in goal.replace("\n", " ").split(".") if s.strip()]
        subtasks = []
        for i, p in enumerate(parts):
            needs = []
            pl = p.lower()
            if any(w in pl for w in ["ui", "screen", "click", "type", "app"]): needs.append("ui")
            if any(w in pl for w in ["api", "http", "backend", "auth"]): needs.append("api")
            if any(w in pl for w in ["db", "schema", "sql"]): needs.append("db")
            if any(w in pl for w in ["code", "build", "bug", "implement"]): needs.append("code")
            if any(w in pl for w in ["test", "verify", "unit"]): needs.append("test")
            if not needs: needs.append("general")
            subtasks.append({"id": f"st{i+1}", "desc": p, "needs": needs})
        return subtasks

    def ensure_skills(self, goal: str, subtasks: List[Dict]) -> None:
        for st in subtasks:
            for skill in st["needs"]:
                try:
                    self.factory.ensure_agent(goal, skill)
                except Exception:
                    pass

    def _normalize_agent_type(self, agent_type: str) -> str:
        if _cap_registry is None:
            return agent_type
        cap = _cap_registry.resolve_capability(agent_type)
        return cap.name if cap else agent_type

    def plan(self, goal: str, context=None):
        # Backward-compatible: use _plan_impl if present, else original flow
        if hasattr(self, "_plan_impl") and callable(getattr(self, "_plan_impl")):
            plan = self._plan_impl(goal, context)
        else:
            plan = self.decompose(goal)
            try:
                self.ensure_skills(goal, plan)
            except Exception:
                pass

        # Normalize any task.agent_type or dict["agent_type"] if present
        try:
            for t in plan or []:
                if hasattr(t, "agent_type"):
                    t.agent_type = self._normalize_agent_type(getattr(t, "agent_type"))
                elif isinstance(t, dict) and "agent_type" in t:
                    t["agent_type"] = self._normalize_agent_type(t["agent_type"])
        except Exception:
            pass
        return plan