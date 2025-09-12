# AgentForge MVP

Quickstart
- Use the bundled venv: source source/bin/activate
- Copy .env.example to .env and fill in API keys as needed. All *_API_KEY are optional; a MockLLM is used when absent.
- To avoid heavy embedding downloads in tests, default is a lightweight HashEmbedder. To use sentence-transformers, set EMBEDDINGS_BACKEND=st.
- To enable Prometheus metrics server during local runs, set PROMETHEUS_ENABLE=1 (default off). Configure port via PROMETHEUS_PORT.

Run
- python main.py

Key env vars
- AF_GOAL: default goal string when not passed on CLI
- AF_AGENTS: number of agents in the swarm (default 2)
- AF_FORCE_MOCK: when "1", force mock model usage for all calls (no network)
- AF_SKIP_HEALTHCHECK: when "1" (default), skip provider health checks at call time
- REDIS_URL (or REDIS_HOST/REDIS_PORT/REDIS_DB): optional; if unavailable, streaming gracefully disables and system continues locally
- AF_ENABLE_PINECONE: opt-in Pinecone usage; disabled by default to avoid startup hangs; requires PINECONE_API_KEY
- EMBEDDINGS_BACKEND: "hash" (default) or "st" (sentence-transformers)
- AF_ENFORCE_STRICT: when "1", SLA/KPI violations will raise
- AF_APPROVAL_ENABLE: when "1", enable approval/HITL gating
- AF_APPROVAL_REQUIRE_STRICT: when "1", block on non-approved escalations
- AF_HITL_AUTOAPPROVE: when "1" (default), auto-approve escalations in CI/dev

SLA/KPI and HITL
- SLA/KPI policies live in sla_kpi_config.py and are enforced pre/post each subtask via orchestrator_enforcer.py (soft by default).
- Approval/HITL gating is lightweight and non-blocking by default; escalations publish events onto the memory mesh and can be strictly enforced via env.

## Backlog Drain SLO (Results & Tools)
We track JetStream consumer lag for both tool execution and results sink workers and define an SLO for backlog drain.

Targets (example baseline, adjust per environment):
- P95 time to drain a burst of 1,000 tool invocations: < 120s
- P95 time to persist 1,000 ToolResult messages (results sink): < 60s
- Steady-state lag (tool executor + results sink) < 50 messages for > 99% of 5s samples

Metrics exposed (Prometheus):
- tool executor: `tool_queue_backlog_gauge{stream="TOOLS",consumer="tool-exec"}`
- results sink: `results_backlog_gauge{stream="SWARM_RESULTS",consumer="results-<env>"}`
- persistence counters: `results_persisted_total{backend=...}`, `results_persist_fail_total{...}`

Procedure to Measure SLO Locally
1. Start NATS (with JetStream) and the following workers: tool executor, results sink.
2. Generate a synthetic burst (e.g., 1k messages) onto subject `tools.invocations.echo` using a short script.
3. Scrape Prometheus every 5s (or curl metrics endpoints) capturing backlog gauges.
4. Record timestamps when backlog first exceeds threshold (>= burst size) and when it returns to <= baseline (e.g., <= 5 messages).
5. Compute drain duration = t_end - t_start. Repeat for N runs (>= 20) and take P95.
6. For results sink, publish synthetic `ToolResult` messages directly to `swarm.results.<env>` (optionally disable DB to exercise JSONL path). Measure identical drain timing with `results_backlog_gauge`.

KEDA Scaling
- KEDA ScaledObjects (`keda-nats-worker.yaml`, `keda-results-sink.yaml`) use JetStream lag to scale replicas.
- Tune `lagThreshold` so that scaling triggers before SLO breach (e.g., choose threshold = (expected RPS * target drain window)/replica_capacity * safety_factor).

Operational Runbook
- Alert if: P95 drain > SLO target over rolling 1h OR backlog gauge > 5 * lagThreshold for > 5m.
- First actions: (a) check worker pod CPU/memory throttling, (b) verify NATS latency / store performance, (c) increase replicas or concurrency env vars.
- If persistence failures increase (results_persist_fail_total), inspect DB health; auto-fallback to JSONL ensures durability but may defer relational queries.

Synthetic Test Script Sketch
```
for i in $(seq 1 1000); do
  nats pub tools.invocations.echo '{"task_id":"t'$i'","tool":"echo","args":{"i":'$i'}}'
done
```
(Use official nats CLI; ensure TOOLS stream exists.)

Calculating P95
- Collect N drain durations into a list D.
- Sort D; pick element at index ceil(0.95 * N) - 1.

Document updates here should be kept in sync with any changes to worker metric names.

## Drift Monitoring (PSI & KL)
The platform now includes lightweight covariate/label drift monitoring utilities (`swarm.observability.drift`).

Capabilities:
- Baseline initialization on first observation per feature (histogram snapshot)
- Population Stability Index (PSI) and KL Divergence computed on subsequent batches
- Prometheus metrics:
  - `drift_psi{feature="<name>"}` latest PSI value
  - `drift_kl_divergence{feature="<name>"}` latest KL value
  - `drift_alerts_total{feature,metric,threshold}` count of threshold exceedances
- Defaults: PSI threshold 0.2, KL threshold 0.5 (tunable at call-site)

Usage example:
```python
from swarm.observability import drift
values = collect_feature_batch()
res = drift.evaluate_drift("embedding_norm", values)
if res["alert"]:
    # trigger mitigation / notify
    pass
```

Roadmap:
- Persist baselines to disk / object storage
- Adaptive binning & mixed data-type support
- Label drift (prediction vs actual) dedicated helper
- Prometheus alert rules file with multi-window evaluation

Tests
- pip install -r requirements.txt (inside venv)
- pytest -q

Services
- Redis (optional) for swarm streams and memory KV: set REDIS_URL or REDIS_HOST/REDIS_PORT. If not reachable, features degrade gracefully.
- Pinecone (optional) for semantic index: set PINECONE_API_KEY and ensure region setup. Disabled unless AF_ENABLE_PINECONE=1.

Security
- Do not commit real secrets. Rotate any leaked keys. .env is ignored by git.

## Evidence Bundle API
Provides a reproducible decision & lineage package for any job.

Endpoint
- GET /v1/evidence/{job_id}

Response Fields
- job_id: Job identifier
- goal: Original goal text (with any embedded artifact context)
- dag_hash: Deterministic hash of the planned DAG (if known)
- decision: Final approval/decision object (includes metrics & citations if present)
- results_summary: Compact summary of results used for approval (count + metrics)
- artifacts: Array of artifact metadata (id, filename, sha256, size, backend, uri, content_type, created_at)
- events: Ordered lineage events (id, event_type, data, created_at)
- citations: Decision evidence references (if provided by reviewers/fusion)
- confidence: Confidence score extracted from decision.metrics.confidence or decision.confidence
- reproducibility: { dag_path: path to persisted DAG JSON if present, dag_path_exists: boolean }

Notes
- Returns 404 if job_id unknown.
- DAG JSON is persisted via lineage.persist_dag; absence simply marks dag_path_exists=false.
- Designed for downstream XAI, audit export, and approval snapshot diffing.

Testing
- See tests/test_approval.py::test_evidence_bundle_endpoint and tests/test_evidence_bundle.py for coverage.

Planned Enhancements
- Optional pagination for large event sets
- ETag / cache-control headers
- Redaction hooks for sensitive artifact metadata
- Streaming variant for very large bundles
