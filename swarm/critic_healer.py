from __future__ import annotations
import random
from typing import Any, Dict, List, Tuple

from swarm.enforcement import enforcer
from swarm.bkg import store as bkg
from swarm.canary.router import CanaryRouter

# Lazy import to avoid heavy startup costs
_build_orchestrator = None

def _get_build_orchestrator():
    global _build_orchestrator
    if _build_orchestrator is None:
        from orchestrator import build_orchestrator
        _build_orchestrator = build_orchestrator
    return _build_orchestrator

class CriticHealer:
    def __init__(self, max_rounds: int = 3, canary_fraction: float = 0.2) -> None:
        self.max_rounds = max_rounds
        self.router = CanaryRouter(canary_fraction=canary_fraction)

    def _run(self, goal: str, agents: int) -> Tuple[List[dict], Dict[str, Any]]:
        build_orchestrator = _get_build_orchestrator()
        orch = build_orchestrator(num_agents=agents)
        results = orch.run_goal_sync(goal)
        decision = enforcer.post(goal=goal, results=results)
        return list(results), decision

    def _mutate(self, goal: str, round_idx: int) -> str:
        # Simple mutation strategy: append hints
        hints = [
            "Increase strictness and validate invariants.",
            "Prioritize robustness over latency.",
            "Use alternative parsing strategy.",
            "Retry failed substeps with backoff.",
            "Use consensus aggregation."
        ]
        return goal + "\n\nHealer Hint: " + hints[round_idx % len(hints)]

    def run(self, goal: str, total_agents: int = 6) -> Dict[str, Any]:
        # Consult best-known-good but do not short-circuit; continue to evaluate
        b = bkg.get(goal)
        base_n, canary_n = self.router.route(total_agents)
        # Initial base run
        base_results, base_decision = self._run(goal, base_n)
        best_decision = base_decision
        best_results = base_results
        best_source = "base"

        # If BKG exists and is approved, prefer it if our base is not approved
        if b and b.get("decision", {}).get("approved", False) and not base_decision.get("approved", False):
            best_decision = b["decision"]
            best_results = b["results"]
            best_source = "base"  # treat as base-equivalent for routing semantics

        if not best_decision.get("approved", False):
            # Canary trials with mutations
            for r in range(self.max_rounds):
                mutated = self._mutate(goal, r)
                canary_results, canary_decision = self._run(mutated, canary_n)
                choose, chosen = self._choose(best_decision, canary_decision)
                if choose == "variant":
                    best_decision, best_results, best_source = canary_decision, canary_results, "canary"
                    if best_decision.get("approved", False):
                        break  # promote immediately
                # else keep current best
        if best_decision.get("approved", False):
            bkg.update(goal, best_decision, best_results)
        return {"source": best_source, "decision": best_decision, "results": best_results}

    def _choose(self, base_decision: Dict[str, Any], canary_decision: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        from swarm.canary.router import choose_better
        which, chosen = choose_better(base_decision, canary_decision)
        return which, chosen