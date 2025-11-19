# Canary Promotion & Rollback Workflow

Phases: idle -> canary -> promote | rollback

Metrics:
- canary_traffic_fraction{phase}
- canary_regressions_total{reason}
- canary_promotions_total
- canary_rollbacks_total
- canary_eval_latency_ms
- canary_eval_latency_budget_violations_total

Latency Budget:
- Set CANARY_EVAL_LATENCY_BUDGET_MS (e.g., 250) to enforce evaluation execution time budget. Violations trigger CanaryEvalLatencyBudgetBreached.

Promotion Criteria:
- Incremental traffic ramp + healthy evaluation (no regression across error_rate_delta, latency_p95_increase, completeness_drop).
- Target fraction achieved with >=50 canary error samples -> phase promote + persist best variant to Best Known Good (BKG) store.

Rollback Criteria:
- Any dimension exceeds thresholds.
- Reapply previous BKG state (writes stored policy_json back to ROUTER_POLICY_PATH).

Operational Steps:
1. Start canary (API / CLI) with goal identifier.
2. Monitor metrics dashboard. Expect canary_traffic_fraction to rise in 5% increments until target.
3. On regression alert:
   - Inspect deltas (exposed via diagnostic endpoint planned) or logs.
   - Confirm rollback occurred (canary_rollbacks_total incremented, phase=rollback).
4. After promotion, verify policy file updated and stable performance (no new regressions for 30m).
5. If evaluation latency alerts fire: profile evaluation path, reduce metric cardinality, or batch observations.

Runbook Last Updated: 2025-09-10
