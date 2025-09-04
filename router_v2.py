from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import random
from forge_types import Task

@dataclass
class Provider:
    key: str
    model: str
    capabilities: Set[str]
    rpm: int = 500
    cost_per_1k: float = 0.0
    uses: int = 0
    reward_sum: float = 0.0

    @property
    def avg_reward(self) -> float:
        return (self.reward_sum / self.uses) if self.uses else 0.0

class MoERouter:
    """Mixture-of-experts router with epsilon-greedy exploration."""
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.providers: Dict[str, Provider] = {}
        self.cap_index: Dict[str, List[str]] = {}

    def register(self, p: Provider) -> None:
        self.providers[p.key] = p
        for cap in p.capabilities:
            self.cap_index.setdefault(cap, []).append(p.key)

    def _infer_cap(self, description: str) -> str:
        d = (description or "").lower()
        if any(w in d for w in ["code", "implement", "bug", "build"]): return "code"
        if any(w in d for w in ["analyze", "critique", "review", "summarize"]): return "analysis"
        if any(w in d for w in ["web", "search", "browse"]): return "search"
        return "general"

    def route(self, task: Task | str, required_cap: Optional[str] = None) -> str:
        description = task if isinstance(task, str) else getattr(task, "description", "")
        cap = required_cap or self._infer_cap(description)
        candidates = self.cap_index.get(cap, list(self.providers.keys()))
        if not candidates:
            raise RuntimeError("No providers registered")
        if random.random() < self.epsilon:
            return random.choice(candidates)
        return max(candidates, key=lambda k: self.providers[k].avg_reward)

    def feedback(self, provider_key: str, reward: float) -> None:
        if provider_key in self.providers:
            p = self.providers[provider_key]
            p.uses += 1
            p.reward_sum += reward