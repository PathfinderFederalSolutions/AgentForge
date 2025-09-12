from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Any
import random, time
import math
from forge_types import Task
from swarm.observability.costs import wrap_llm_call
from swarm.observability.costs import set_observability_context  # type: ignore
try:
    from swarm.observability.otel import tag_span as _tag_span  # type: ignore
except Exception:  # pragma: no cover
    _tag_span = None  # type: ignore

# Policy loader (lazy import to avoid mandatory dependency during older tests)
try:
    from swarm.router_policy_loader import match_policy  # type: ignore
except Exception:  # pragma: no cover
    def match_policy(_desc: str) -> Optional[dict]:  # type: ignore
        return None

@dataclass
class Provider:
    key: str
    model: str
    capabilities: Set[str]
    rpm: int = 500
    cost_per_1k: float = 0.0
    uses: int = 0
    reward_sum: float = 0.0
    cost_spent: float = 0.0  # USD
    avg_latency_ms: float = 0.0

    @property
    def avg_reward(self) -> float:
        return (self.reward_sum / self.uses) if self.uses else 0.0

    def _estimate_cost(self, resp: Any) -> float:
        # Prefer token attributes if present
        pt = getattr(resp, 'prompt_tokens', None)
        ct = getattr(resp, 'completion_tokens', None)
        if isinstance(pt, (int, float)) and isinstance(ct, (int, float)):
            total_tokens = max(0, int(pt) + int(ct))
            return (total_tokens / 1000.0) * self.cost_per_1k
        # Fallback heuristic (assume 250 tokens)
        return (250 / 1000.0) * self.cost_per_1k

    def _base_call(self, *args, **kwargs):
        start = time.perf_counter()
        resp = self.client.invoke(*args, **kwargs)
        lat_ms = (time.perf_counter() - start) * 1000.0
        # Exponential moving average latency (alpha=0.2)
        if self.avg_latency_ms <= 0:
            self.avg_latency_ms = lat_ms
        else:
            self.avg_latency_ms = 0.2 * lat_ms + 0.8 * self.avg_latency_ms
        try:
            c = self._estimate_cost(resp)
            if not math.isnan(c) and c >= 0:
                self.cost_spent += c
        except Exception:
            pass
        return resp

    def call(self, *args, **kwargs):  # public
        return self._base_call(*args, **kwargs)

    def instrument(self):
        self.call = wrap_llm_call(self.key, self.model, self.call)  # type: ignore[attr-defined]

class MoERouter:
    """Mixture-of-experts router with epsilon-greedy exploration + policy enforcement.

    Policies (JSON via ROUTER_POLICY_PATH or default config/router_policies.json) support:
      - task_regex: regex applied to task description
      - max_cost_usd: cumulative cost ceiling per provider for matching tasks
      - max_latency_ms: (soft) average latency ceiling (skips providers exceeding)
      - sensitive_local_only: when true restrict to providers considered 'local'
        A provider is treated as local if cost_per_1k == 0 or key/model contains one of: ['mock','local']
      - allowed_providers: optional explicit allowlist when sensitive_local_only true
    """
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.providers: Dict[str, Provider] = {}
        self.cap_index: Dict[str, List[str]] = {}

    def register(self, p: Provider) -> None:
        self.providers[p.key] = p
        for cap in p.capabilities:
            self.cap_index.setdefault(cap, []).append(p.key)

    def _infer_cap(self, description: str) -> str:
        d = (description or '').lower()
        if any(w in d for w in ['code', 'implement', 'bug', 'build']): return 'code'
        if any(w in d for w in ['analyze', 'critique', 'review', 'summarize']): return 'analysis'
        if any(w in d for w in ['web', 'search', 'browse']): return 'search'
        return 'general'

    def _is_local(self, p: Provider) -> bool:
        tag = f"{p.key}:{p.model}".lower()
        if p.cost_per_1k == 0:
            return True
        return any(x in tag for x in ('mock','local'))

    def _apply_policy_filters(self, description: str, candidates: List[str]) -> List[str]:
        pol = match_policy(description)
        if not pol:
            return candidates
        allow = set(pol.get('allowed_providers', []) or [])
        max_cost = pol.get('max_cost_usd')
        max_lat = pol.get('max_latency_ms')
        sens = pol.get('sensitive_local_only')
        out: List[str] = []
        for k in candidates:
            p = self.providers[k]
            if sens and not (k in allow or self._is_local(p)):
                continue
            if isinstance(max_cost, (int, float)) and max_cost >= 0 and p.cost_spent >= max_cost:
                continue
            if isinstance(max_lat, (int, float)) and max_lat > 0 and p.avg_latency_ms > 0 and p.avg_latency_ms > max_lat:
                continue
            out.append(k)
        return out or candidates

    def route(self, task: Task | str, required_cap: Optional[str] = None) -> str:
        description = task if isinstance(task, str) else getattr(task, 'description', '')
        cap = required_cap or self._infer_cap(description)
        candidates = self.cap_index.get(cap, list(self.providers.keys()))
        if not candidates:
            raise RuntimeError('No providers registered')
        candidates = self._apply_policy_filters(description, candidates)
        if random.random() < self.epsilon:
            choice = random.choice(candidates)
        else:
            choice = max(candidates, key=lambda k: self.providers[k].avg_reward)
        if _tag_span:
            try:
                # Best effort tagging; ignore signature mismatch silently
                _tag_span()
            except Exception:
                pass
        return choice

    def invoke(self, provider_key: str, task: Task | str, mission: str = 'default', task_id: str = 'unknown', *args, **kwargs):
        set_observability_context(mission_id=mission, task_id=task_id)
        if _tag_span:
            try: _tag_span()
            except Exception: pass
        p = self.providers[provider_key]
        return p.call(*args, **kwargs)

    def feedback(self, provider_key: str, reward: float) -> None:
        if provider_key in self.providers:
            p = self.providers[provider_key]
            p.uses += 1
            p.reward_sum += reward