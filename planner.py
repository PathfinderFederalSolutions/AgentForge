from __future__ import annotations
from typing import List, Dict
from agent_factory import AgentFactory
from memory_mesh import MemoryMesh

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

    def plan(self, goal: str) -> List[Dict]:
        subtasks = self.decompose(goal)
        self.ensure_skills(goal, subtasks)
        self.mesh.publish("plan.created", {"goal": goal, "subtasks": subtasks})
        return subtasks