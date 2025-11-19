# Latency Budgets Overview

Components Covered:
- Fusion pipeline (FUSION_LATENCY_BUDGET_MS)
- Canary evaluation loop (CANARY_EVAL_LATENCY_BUDGET_MS)
- DAG planning/execution (existing task & DAG latency metrics)

Environment Variables:
- FUSION_LATENCY_BUDGET_MS: e.g., 150 (ms)
- CANARY_EVAL_LATENCY_BUDGET_MS: e.g., 250 (ms)
- DAG_LATENCY_BUDGET_MS: (TBD; existing swarm_dag_latency_budget_violations_total used when provided in DAGSpec)

Metrics & Alerts:
- fusion_latency_budget_violations_total -> FusionLatencyBudgetBreached
- canary_eval_latency_budget_violations_total -> CanaryEvalLatencyBudgetBreached
- swarm_dag_latency_budget_violations_total -> (add alert in future ruleset)

Operational Response:
1. Confirm rising latency correlates with resource saturation (CPU, memory) or external dependency slowness.
2. Capture p95/p99 deltas before scaling. If increase >25% vs baseline sustain >10m -> scale or optimize.
3. If budget violations coincide with regressions (canary) consider pausing traffic ramp.

Tuning Guidance:
- Set initial budget at p95 + 20% headroom from baseline load test.
- Recalibrate quarterly or after substantial code changes.

## UI Streaming Budgets (SSE/WS)

Targets:
- Tactical map stream p95 end-to-end latency: < 200 ms under 100 concurrent clients
- Heartbeat interval: 10 s (SSE and WS)

Measurement:
- Measure server processing + network + client render enqueue
- Use synthetic emit endpoint or generator to push 5 rps for 60 s; record p50/p95/p99

Knobs:
- TACTICAL_STREAM_MAX_CLIENTS, TACTICAL_STREAM_MAX_PER_IP
- TACTICAL_REQUIRE_CLIENT_CERT, TACTICAL_REQUIRE_BEARER + TACTICAL_BEARER_TOKEN

Runbook Last Updated: 2025-09-11
