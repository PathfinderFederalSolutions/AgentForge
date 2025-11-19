# filepath: swarm/observability/task_latency.py
from __future__ import annotations
"""Task latency budgeting metrics & helpers.

Records end-to-end task (job) latency and evaluates against configured latency budgets
(from sla_kpi_config.get_task_budget). A single generic "default" task type is used
unless additional classification is later introduced.

Exports:
  record_task_completion(latency_seconds, task_type, mission_id, task_id=None)

Metrics:
  task_latency_seconds (Histogram) labels: mission_id, task_type, budget_name
  task_latency_budget_violations_total (Counter) labels: mission_id, budget_name, threshold

Violation thresholds emitted when latency exceeds configured p99 or hard_cap boundaries.
(p95/p90 not surfaced to limit cardinality & noise.)
"""
from prometheus_client import Histogram, Counter, Gauge
from typing import Optional

# Hist buckets (seconds) wide enough for long running tasks
_LATENCY_BUCKETS = (
    0.05, 0.1, 0.25, 0.5,
    1, 2, 5, 10,
    20, 30, 45, 60,
    90, 120, 180, 300,
    600, 1200, 1800
)

TASK_LATENCY_SECONDS = Histogram(
    "task_latency_seconds",
    "End-to-end task latency (job start -> completion)",
    ["mission_id", "task_type", "budget_name"],
    buckets=_LATENCY_BUCKETS,
)

TASK_LATENCY_VIOLATIONS = Counter(
    "task_latency_budget_violations_total",
    "Task latency budget violations",
    ["mission_id", "budget_name", "threshold"],
)

dag_budget_violation = Counter(
    'swarm_dag_latency_budget_violations_total',
    'Count of DAG executions exceeding latency budget'
)

# New: total DAG executions & unique DAG hash occurrences (cardinality should be moderate due to hashing)
dag_executions_total = Counter(
    'swarm_dag_executions_total',
    'Total DAG executions observed'
)

dag_last_hash = Gauge(
    'swarm_dag_last_hash',
    'Dummy gauge set to 1 with label dag_hash for last seen DAG (allows scraping of current hash)',
    ['dag_hash']
)

# Optional: count occurrences per hash (helps frequency analysis)
dag_hash_occurrences = Counter(
    'swarm_dag_hash_occurrences_total',
    'Count of times a deterministic DAG hash has been executed',
    ['dag_hash']
)

def record_task_completion(latency_seconds: float, task_type: str, mission_id: str, budget_name: str, p99_ms: int, hard_cap_ms: int):
    """Observe latency & emit violation counters if thresholds exceeded."""
    try:
        TASK_LATENCY_SECONDS.labels(mission_id, task_type, budget_name).observe(latency_seconds)
        latency_ms = latency_seconds * 1000.0
        if latency_ms > p99_ms:
            TASK_LATENCY_VIOLATIONS.labels(mission_id, budget_name, "p99").inc()
        if latency_ms > hard_cap_ms:
            TASK_LATENCY_VIOLATIONS.labels(mission_id, budget_name, "hard_cap").inc()
    except Exception:
        # Metrics best-effort; swallow any registry issues
        pass

def record_dag_completion(latency_ms: float, budget_ms: Optional[int]):
    try:
        dag_executions_total.inc()
        if budget_ms and latency_ms > budget_ms:
            dag_budget_violation.inc()
    except Exception:
        pass

# Helper to tag a DAG hash metric-wise
def record_dag_hash(dag_hash: str):
    try:
        dag_last_hash.labels(dag_hash).set(1)
        dag_hash_occurrences.labels(dag_hash).inc()
    except Exception:
        pass

__all__ = [
    "record_task_completion",
    "TASK_LATENCY_SECONDS",
    "TASK_LATENCY_VIOLATIONS",
    'record_dag_completion', 'dag_budget_violation', 'dag_executions_total',
    'record_dag_hash', 'dag_last_hash', 'dag_hash_occurrences'
]
