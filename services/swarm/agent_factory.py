from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any
import time

try:
    from capability_registry import get_registry
    _cap_registry = get_registry()
except Exception:
    _cap_registry = None

@dataclass
class AgentSpec:
    name: str
    capabilities: List[str]
    tools: List[str] = field(default_factory=list)
    llm: str = "gpt-5"
    policy: Dict[str, Any] = field(default_factory=dict)

class AgentRegistry:
    def __init__(self):
        self.specs: Dict[str, AgentSpec] = {}
        self.builders: Dict[str, Callable[[AgentSpec], Any]] = {}

    def register_builder(self, llm: str, builder: Callable[[AgentSpec], Any]):
        self.builders[llm] = builder

    def add(self, spec: AgentSpec) -> None:
        self.specs[spec.name] = spec

    def build(self, name: str):
        spec = self.specs[name]
        builder = self.builders.get(spec.llm)
        if not builder:
            raise ValueError(f"No builder for llm={spec.llm}")
        return builder(spec)

class AgentFactory:
    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def synthesize(self, goal: str, missing_skill: str) -> AgentSpec:
        name = f"auto_{missing_skill}_{int(time.time())}"
        spec = AgentSpec(name=name, capabilities=[missing_skill], llm="gpt-5")
        self.registry.add(spec)
        return spec

    def evaluate(self, agent, eval_tasks: List[str]) -> float:
        return 0.8  # stub; replace with real evaluator

    def ensure_agent(self, goal: str, required_skill: str) -> str:
        spec = self.synthesize(goal, required_skill)
        agent = self.registry.build(spec.name)
        score = self.evaluate(agent, [f"Eval: {required_skill}"])
        if score < 0.7:
            del self.registry.specs[spec.name]
            raise RuntimeError(f"Agent for {required_skill} failed eval")
        return spec.name

    def create_agent(self, agent_type: str, **kwargs):
        # Try CapabilityRegistry first (if present), then fall back to legacy paths.
        if _cap_registry is not None:
            try:
                instance = _cap_registry.create_from_capability(agent_type, **kwargs)
                if instance is not None:
                    return instance
            except Exception:
                # Soft-fail to legacy behavior
                pass

        # return legacy construction by agent_type