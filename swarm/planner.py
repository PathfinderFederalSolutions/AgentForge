from __future__ import annotations
from typing import Any, Dict, List, Tuple
from swarm.capabilities.registry import registry, Capability
from swarm.factory import Agent
from swarm.consensus.patterns import aggregate
try:
    from swarm.vector import service as vector_service
except Exception:
    vector_service = None

import os as _os
if _os.getenv("VECTOR_ENABLED", "0") == "0":
    vector_service = None

try:
    from swarm.learning.feedback import get_boost
except Exception:
    def get_boost(name: str) -> float:
        return 0.0

from swarm.protocol.messages import DAGSpec, DAGNode, DAGEdge
import random

class PlanStep:
    def __init__(self, capability: str, args: Dict[str, Any]) -> None:
        self.capability = capability
        self.args = args

class Planner:
    def score_cap(self, cap: Capability, goal_l: str, budget_ms: int = 1000) -> float:
        s = 0.0
        meta = cap.meta or {}
        if any(m in goal_l for m in (cap.name, *(meta.get("modalities", [])))):
            s += 1.0
        lat = meta.get("latency_ms", 100)
        s += max(0.0, 0.5 - (lat / max(1.0, budget_ms)))
        s += meta.get("trust", 0.5) * 0.5
        # Learned boost based on prior approvals/failures
        s += 0.5 * get_boost(cap.name)
        return s

    def make_plan(self, goal: str, budget_ms: int = 1000) -> List[PlanStep]:
        goal_l = goal.lower()
        caps = registry.list()
        scored: List[Tuple[float, Capability]] = [(self.score_cap(c, goal_l, budget_ms), c) for c in caps.values()]
        scored.sort(key=lambda x: x[0], reverse=True)
        steps = [PlanStep(c.name, {}) for s, c in scored[:3]]
        return steps

    def make_dag(self, goal: str, seed: int, budget_ms: int = 1000, max_steps: int = 5) -> DAGSpec:
        """
        Deterministically generate a DAGSpec from goal & seed.
        Steps:
        1. Score capabilities
        2. Use seeded RNG to pick subset deterministically
        3. Create linear or shallow-fork DAG pattern deterministically from RNG decisions
        """
        rng = random.Random(seed)
        goal_l = goal.lower()
        caps = registry.list()
        scored: List[Tuple[float, Capability]] = [(self.score_cap(c, goal_l, budget_ms), c) for c in caps.values()]
        scored.sort(key=lambda x: (-(x[0]), x[1].name))
        # Deterministic subset selection: take top K then maybe branch
        k = min(max_steps, max(1, int(rng.random() * min(4, len(scored))) + 1))
        subset = [c for _, c in scored[:k]]
        nodes: List[DAGNode] = []
        edges: List[DAGEdge] = []
        for idx, cap in enumerate(subset):
            nid = f"n{idx}"
            nodes.append(DAGNode(node_id=nid, capability=cap.name, args={}))
            if idx > 0:
                # Decide deterministic parent pattern: linear chain or star from n0
                if k > 2 and rng.random() < 0.3:
                    edges.append(DAGEdge(source="n0", target=nid))
                else:
                    edges.append(DAGEdge(source=f"n{idx-1}", target=nid))
        dag = DAGSpec(goal=goal, seed=seed, nodes=nodes, edges=edges, latency_budget_ms=budget_ms)
        dag.compute_hash()
        return dag

class Executor:
    def __init__(self, scope: str) -> None:
        self.scope = scope

    def run(self, plan: List[PlanStep], agents: int = 3) -> Dict[str, Any]:
        # Encourage agent self-spawn: scale workers to plan size*2, capped
        size = max(agents, min(agents * 2, len(plan) * 2))
        workers = [Agent(name=f"agent-{i}", scope=self.scope) for i in range(size)]
        results = []
        for step in plan:
            # Memory-assisted args enrichment using vector results (optional)
            if vector_service:
                try:
                    hits = vector_service.search(scope=self.scope, query=step.capability, top_k=3)
                    _ = hits  # avoid unused var if not consumed
                except Exception:
                    pass
            # Provide safe defaults for known capabilities regardless of vector availability
            try:
                if step.capability == "bayesian_fusion":
                    step.args.setdefault("eo", [1, 2, 3, 4, 5])
                    step.args.setdefault("ir", [2, 3, 4, 5, 6])
                elif step.capability == "conformal_validate":
                    step.args.setdefault("residuals", [0.1, -0.2, 0.05, 0.0])
                    step.args.setdefault("alpha", 0.1)
            except Exception:
                # Do not let arg population errors break execution
                pass
            out = [w.run_step(step.capability, step.args) for w in workers]
            results.append(aggregate(out))
        return {"steps": [s.capability for s in plan], "results": results}