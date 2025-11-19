from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, Callable
import time, statistics
import os
try:  # metrics
    from ..api.metrics import Histogram, Counter, CANARY_EVAL_LATENCY_MS, CANARY_EVAL_LATENCY_BUDGET_VIOLATIONS  # type: ignore
except Exception:  # pragma: no cover
    Histogram = Counter = None  # type: ignore
    CANARY_EVAL_LATENCY_MS = None  # type: ignore
    CANARY_EVAL_LATENCY_BUDGET_VIOLATIONS = None  # type: ignore
try:  # BKG persistence
    from ..bkg import store as bkg_store  # type: ignore
except Exception:  # pragma: no cover
    bkg_store = None  # type: ignore

def choose_better(base: Dict[str, Any], variant: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    bm = base.get("metrics", {})
    vm = variant.get("metrics", {})
    # Prefer lower error_rate, then higher completeness, then approved
    be = bm.get("error_rate", 1.0); ve = vm.get("error_rate", 1.0)
    bc = bm.get("completeness", 0.0); vc = vm.get("completeness", 0.0)
    if ve < be:
        return "variant", variant
    if ve == be and vc > bc:
        return "variant", variant
    if ve == be and vc == bc and variant.get("approved", False) and not base.get("approved", False):
        return "variant", variant
    return "base", base

# Promotion phases
CANARY_PHASES = ("idle","canary","promote","rollback")

DEFAULT_REGRESSION_THRESHOLDS = {
    "error_rate_delta": 0.02,   # absolute increase allowed
    "latency_p95_increase": 0.25,  # 25% increase
    "completeness_drop": 0.03,
}

class CanaryState:
    def __init__(self) -> None:
        self.phase: str = "idle"
        self.started_at: float = 0.0
        self.base_metrics: Dict[str, list] = {"latency": [], "error": [], "completeness": []}
        self.canary_metrics: Dict[str, list] = {"latency": [], "error": [], "completeness": []}
        self.target_fraction: float = 0.1
        self.current_fraction: float = 0.0
        self.best_variant: Optional[Dict[str, Any]] = None
        self.goal: str = "router_policy"

    def reset(self):
        self.__init__()

state = CanaryState()

if Histogram and Counter:
    try:
        CANARY_TRAFFIC_FRACTION = Histogram(
            "canary_traffic_fraction","Assigned traffic fraction to canary",["phase"],buckets=(0.01,0.05,0.1,0.2,0.3,0.5,1.0))  # type: ignore
        CANARY_REGRESSIONS = Counter("canary_regressions_total","Detected canary regressions",["reason"])  # type: ignore
        CANARY_PROMOTIONS = Counter("canary_promotions_total","Successful promotions",[])  # type: ignore
        CANARY_ROLLBACKS = Counter("canary_rollbacks_total","Rollback events",[])  # type: ignore
    except Exception:  # pragma: no cover
        CANARY_TRAFFIC_FRACTION = CANARY_REGRESSIONS = CANARY_PROMOTIONS = CANARY_ROLLBACKS = None  # type: ignore
else:
    CANARY_TRAFFIC_FRACTION = CANARY_REGRESSIONS = CANARY_PROMOTIONS = CANARY_ROLLBACKS = None  # type: ignore


def start_canary(target_fraction: float = 0.1, goal: str = "router_policy") -> None:
    state.reset()
    state.phase = "canary"
    state.started_at = time.time()
    state.target_fraction = max(0.01, min(target_fraction, 0.9))
    state.goal = goal
    if CANARY_TRAFFIC_FRACTION:
        try: CANARY_TRAFFIC_FRACTION.labels(state.phase).observe(state.current_fraction)
        except Exception: pass


def record_observation(is_canary: bool, latency_ms: float, error: bool, completeness: float) -> None:
    bucket = state.canary_metrics if is_canary else state.base_metrics
    bucket["latency"].append(latency_ms)
    bucket["error"].append(1.0 if error else 0.0)
    bucket["completeness"].append(completeness)


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    idx = int(0.95 * (len(vs) - 1))
    return vs[idx]


def _stable_enough(window: list[float], min_samples: int = 30) -> bool:
    return len(window) >= min_samples


def evaluate(thresholds: Optional[Dict[str, float]] = None) -> Tuple[bool, Dict[str, float]]:
    thr = {**DEFAULT_REGRESSION_THRESHOLDS, **(thresholds or {})}
    def mean_safe(key: str) -> float:
        vals = state.base_metrics[key]
        can = state.canary_metrics[key]
        m_base = statistics.fmean(vals) if _stable_enough(vals) else 0.0
        m_can = statistics.fmean(can) if _stable_enough(can) else 0.0
        return m_base, m_can
    base_err, can_err = mean_safe("error")
    base_c, can_c = mean_safe("completeness")
    base_p95 = p95(state.base_metrics["latency"]) if _stable_enough(state.base_metrics["latency"]) else 0.0
    can_p95 = p95(state.canary_metrics["latency"]) if _stable_enough(state.canary_metrics["latency"]) else 0.0
    deltas = {
        "error_rate_delta": max(0.0, can_err - base_err),
        "latency_p95_increase": ( (can_p95 - base_p95) / base_p95 ) if base_p95 > 1 else 0.0,
        "completeness_drop": max(0.0, base_c - can_c),
    }
    regression = any(deltas[k] > thr[k] for k in deltas)
    return (not regression, deltas)


def _persist_promotion(deltas: Dict[str,float]):
    if bkg_store:
        try:
            bkg_store.update(state.goal, {"phase": "promote","deltas": deltas}, results=["canary_policy"])
        except Exception:
            pass


def _record_regressions(deltas: Dict[str,float]):
    if CANARY_REGRESSIONS:
        try:
            for k,v in deltas.items():
                if v > 0: CANARY_REGRESSIONS.labels(k).inc()
        except Exception:
            pass


def _restore_bkg():
    if not bkg_store:
        return
    try:
        record = bkg_store.get(state.goal)
        if not record:
            return
        # Example structure: decision contains previous policy metadata, results list of identifiers
        decision = record.get("decision", {})
        # For router policies we might store a serialized policy variant; reapply by writing file/env
        policy_blob = decision.get("policy_json")
        policy_path = os.getenv("ROUTER_POLICY_PATH", "config/router_policies.json")
        if policy_blob:
            try:
                with open(policy_path, "w", encoding="utf-8") as f:
                    f.write(policy_blob)
            except Exception:
                pass
    except Exception:
        pass


def maybe_progress(thresholds: Optional[Dict[str,float]] = None, persist_best: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if state.phase != "canary":
        return {"phase": state.phase}
    healthy, deltas = evaluate(thresholds)
    if healthy:
        state.current_fraction = min(state.target_fraction, state.current_fraction + 0.05)
        if state.current_fraction >= state.target_fraction and _stable_enough(state.canary_metrics["error"], 50):
            state.phase = "promote"
            if persist_best:
                _persist_promotion(deltas)
            if CANARY_PROMOTIONS:
                try: CANARY_PROMOTIONS.inc()
                except Exception: pass
    else:
        state.phase = "rollback"
        _record_regressions(deltas)
        _restore_bkg()
        if CANARY_ROLLBACKS:
            try: CANARY_ROLLBACKS.inc()
            except Exception: pass
    if CANARY_TRAFFIC_FRACTION:
        try: CANARY_TRAFFIC_FRACTION.labels(state.phase).observe(state.current_fraction)
        except Exception: pass
    # Evaluation latency metric + budget
    eval_lat_ms = (time.perf_counter() - t0) * 1000.0
    if CANARY_EVAL_LATENCY_MS:
        try: CANARY_EVAL_LATENCY_MS.observe(eval_lat_ms)
        except Exception: pass
    try:
        budget_ms = float(os.getenv("CANARY_EVAL_LATENCY_BUDGET_MS", "0"))
    except Exception:
        budget_ms = 0.0
    if budget_ms > 0 and eval_lat_ms > budget_ms and CANARY_EVAL_LATENCY_BUDGET_VIOLATIONS:
        try: CANARY_EVAL_LATENCY_BUDGET_VIOLATIONS.labels("hard_cap").inc()
        except Exception: pass
    return {"phase": state.phase, "fraction": state.current_fraction, "deltas": deltas, "eval_latency_ms": eval_lat_ms}


class CanaryRouter:
    def __init__(self, canary_fraction: float = 0.1, selector: Optional[Callable[..., Tuple[int,int]]] = None) -> None:
        self.canary_fraction = canary_fraction
        self.selector = selector or self._default_selector

    def _default_selector(self, total_agents: int, fraction: float) -> Tuple[int,int]:
        canary = max(1, int(total_agents * fraction))
        base = max(1, total_agents - canary)
        return base, canary

    def route(self, total_agents: int) -> Tuple[int, int]:
        # If active promotion adjust using dynamic state fraction else static fraction
        fraction = state.current_fraction if state.phase in ("canary","promote") else self.canary_fraction
        return self.selector(total_agents, fraction)