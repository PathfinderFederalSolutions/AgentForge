"""
Central SLA / KPI configuration.

All services (planner, router, workers, API) must import from here only.
No hard‑coded numeric thresholds elsewhere.

Supports environment overrides so ops can tune without code edits.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default

@dataclass(frozen=True)
class LatencySLO:
    name: str
    p95_ms: int
    max_cap_ms: int
    description: str

@dataclass(frozen=True)
class BacklogSLO:
    name: str
    warning_threshold: int
    critical_threshold: int
    sustained_minutes_for_warning: int
    sustained_minutes_for_critical: int
    description: str

@dataclass(frozen=True)
class DrainSLO:
    name: str
    p95_seconds: int
    hard_cap_seconds: int
    description: str

@dataclass(frozen=True)
class ErrorBudgetPolicy:
    slo_name: str
    period_days: int
    target_availability: float  # e.g. 0.995
    fast_window: str            # Prometheus range (e.g. 5m)
    slow_window: str            # Prometheus range (e.g. 1h)

@dataclass(frozen=True)
class TaskLatencyBudget:
    name: str
    p50_ms: int
    p90_ms: int
    p95_ms: int
    p99_ms: int
    hard_cap_ms: int
    description: str

# --- Definitions (with env overrides) ---

LATENCY_SLOS: Dict[str, LatencySLO] = {
    "job_dispatch": LatencySLO(
        name="job_dispatch",
        p95_ms=_env_int("SLO_JOB_DISPATCH_P95_MS", 600_000),          # 600s
        max_cap_ms=_env_int("SLO_JOB_DISPATCH_CAP_MS", 1_200_000),    # 1200s
        description="Time from job enqueue to first worker start."
    ),
}

BACKLOG_SLOS: Dict[str, BacklogSLO] = {
    "jetstream_consumer": BacklogSLO(
        name="jetstream_consumer",
        warning_threshold=_env_int("SLO_BACKLOG_WARN", 3000),
        critical_threshold=_env_int("SLO_BACKLOG_CRIT", 6000),
        sustained_minutes_for_warning=_env_int("SLO_BACKLOG_WARN_MINUTES", 10),
        sustained_minutes_for_critical=_env_int("SLO_BACKLOG_CRIT_MINUTES", 10),
        description="Pending messages across critical consumers."
    )
}

DRAIN_SLOS: Dict[str, DrainSLO] = {
    "backlog_drain": DrainSLO(
        name="backlog_drain",
        p95_seconds=_env_int("SLO_DRAIN_P95_SECONDS", 600),
        hard_cap_seconds=_env_int("SLO_DRAIN_CAP_SECONDS", 1_200),
        description="Time to drain backlog after scaling workers back up."
    )
}

ERROR_BUDGET_POLICIES: List[ErrorBudgetPolicy] = [
    ErrorBudgetPolicy(
        slo_name="backlog_drain",
        period_days=_env_int("ERR_BUDGET_PERIOD_DAYS", 30),
        target_availability=_env_float("ERR_BUDGET_TARGET", 0.995),
        fast_window=os.getenv("ERR_BUDGET_FAST_WINDOW", "5m"),
        slow_window=os.getenv("ERR_BUDGET_SLOW_WINDOW", "1h"),
    )
]

# Task type latency budgets (higher cardinality per task avoided; this is per task class)
# Environment variable naming convention:
#   TASK_<UPPER_NAME>_P50_MS, TASK_<UPPER_NAME>_P90_MS, etc.
# A generic "default" budget is always present.
_TASK_DEFAULT_PREFIX = "TASK_DEFAULT_"
TASK_LATENCY_BUDGETS: Dict[str, TaskLatencyBudget] = {
    "default": TaskLatencyBudget(
        name="default",
        p50_ms=_env_int(f"{_TASK_DEFAULT_PREFIX}P50_MS", 1_000),
        p90_ms=_env_int(f"{_TASK_DEFAULT_PREFIX}P90_MS", 5_000),
        p95_ms=_env_int(f"{_TASK_DEFAULT_PREFIX}P95_MS", 10_000),
        p99_ms=_env_int(f"{_TASK_DEFAULT_PREFIX}P99_MS", 30_000),
        hard_cap_ms=_env_int(f"{_TASK_DEFAULT_PREFIX}CAP_MS", 120_000),
        description="Generic task end-to-end latency expectations"
    )
}

# Prometheus metric names (single source)
METRIC_NAMES = {
    "jetstream_backlog": "sum(nats_jetstream_consumer_num_pending)",
    "backlog_drain_histogram": "backlog_drain_seconds_bucket",
    "connections": "nats_varz_connections",
    "slo_violation_counter": "slo_violation_events_total",
}

def get_latency_slo(name: str) -> LatencySLO:
    return LATENCY_SLOS[name]

def get_backlog_slo(name: str) -> BacklogSLO:
    return BACKLOG_SLOS[name]

def get_drain_slo(name: str) -> DrainSLO:
    return DRAIN_SLOS[name]

def get_task_budget(task_type: str) -> TaskLatencyBudget:
    """Return latency budget for a logical task type (case-insensitive).
    Falls back to 'default' if not defined.
    """
    key = (task_type or "default").lower()
    return TASK_LATENCY_BUDGETS.get(key, TASK_LATENCY_BUDGETS["default"])

def list_error_budget_policies() -> List[ErrorBudgetPolicy]:
    return ERROR_BUDGET_POLICIES[:]

def export_summary() -> Dict[str, Dict]:
    return {
        "latency_slos": {k: vars(v) for k, v in LATENCY_SLOS.items()},
        "backlog_slos": {k: vars(v) for k, v in BACKLOG_SLOS.items()},
        "drain_slos": {k: vars(v) for k, v in DRAIN_SLOS.items()},
        "error_budget_policies": [vars(p) for p in ERROR_BUDGET_POLICIES],
        "metric_names": METRIC_NAMES,
        "task_latency_budgets": {k: vars(v) for k, v in TASK_LATENCY_BUDGETS.items()},
    }

if __name__ == "__main__":
    import json
    print(json.dumps(export_summary(), indent=2))