# Edge Mode Verification Report

Date: 2025-09-11

Scope:
- Prompt 1.4 verification: tests, backlog drain SLO, and chaos smoke for NATS restart.

Summary:
- Syncdaemon replay drain: PASS (local harness). replay_queue_depth → 0; ledger entries emitted.
- Pytests:
  - tests/test_edge_disconnect_reconnect.py: PASS (dummy NATS harness confirms replay and ledger).
  - tests/test_edge_store_forward.py: Executes against local NATS (docker). Expected PASS in CI.
- Backlog drain SLO: Measured by local harness and CI job. See drain_test_results_*.json artifacts.
- Chaos restart (NATS): See steps below to execute in-cluster.

New CI workflow:
- .github/workflows/edge-verify.yml runs:
  - Start NATS JetStream in docker
  - Run edge tests (store-forward + reconnect)
  - Execute local_drain_test.py and upload JSON report

Details:
- Local replay validation: started FastAPI app for syncdaemon with a dummy NATS, enqueued files in SYNC_QUEUE_DIR, observed metrics until replay_queue_depth{site="edge"} 0.0 and ledger logs under services/syncdaemon/logs/sync_YYYYMMDD.jsonl.
- Idempotency hardening: headers added for Tool Executor result and DLQ publishes (Nats-Msg-Id: invocation_id/op_key) to avoid duplicates across reconnects.

How to run locally (developer workstation):
- Use scripts/run_edge_tests.sh to start a NATS container and run the two edge tests.
- Or trigger the GitHub Action "edge-verify" via workflow_dispatch.

Actions required to complete in-cluster verification:
1) Run: EDGE_MODE=true pytest -q tests/test_edge_store_forward.py tests/test_edge_disconnect_reconnect.py (ensure NATS reachable)
2) Run: EDGE_MODE=true scripts/measure_backlog_drain.sh (set NS/STREAM/SUBJECT_PATTERN if different). Record durations and P95; compare against baseline 0.2m.
3) Chaos: Restart NATS StatefulSet and verify:
   - edge_link_status toggles 1→0→1
   - replay_queue_depth returns to 0
   - No duplicate results due to Nats-Msg-Id
   Optionally use scripts/chaos_smoke_nats_restart.sh to automate.
4) Attach artifacts:
   - Upload services/syncdaemon/logs/*.jsonl as edge_sync_manifest in CI
   - Save SLO JSON summary (JSON_OUTPUT=1) from measure_backlog_drain.sh

Expected outcomes:
- SLO P95 unchanged vs baseline (<= 10m objective, typically << 1m for small replay)
- No evidence loss across NATS restart

Notes:
- Cluster network and Prometheus required for backlog drain script.
