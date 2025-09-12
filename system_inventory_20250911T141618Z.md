# System Inventory Snapshot

- Timestamp (UTC): 20250911T141618Z
- Hostname: Baileys-MacBook-Pro-2.local
- User: baileymahoney

## Summary for GPT-5
This bundle describes the AgentForge system: code, services, messaging, observability, deployment profiles, and artifacts.

- Workloads: FastAPI services (API, Orchestrator), NATS JetStream workers (tool executor, results sink), HITL.
- Messaging: NATS JetStream subjects swarm.jobs.<env>, swarm.results.<env>, hitl, DLQs; durable consumers.
- Observability: Prometheus /metrics, ServiceMonitors, OTEL traces, canary SLOs, drift monitors (PSI/KL), ROC/EER, executor/worker metrics.
- Storage: JSONL fallbacks, lineage DAG artifacts under var/artifacts, optional DB/pgvector.
- K8s: Profiles for SCIF/GovCloud/SaaS via Kustomize; KEDA scaling, ServiceMonitors, PrometheusRules, OTEL Collector.
- Supply chain: Syft SPDX SBOMs (source/image), Cosign keyless attest workflow in GitHub Actions.
- Evidence: /v1/evidence/{job_id} returns bundle; lineage DAG persisted with dag_hash.
- Drift: PSI, KL metrics (DRIFT_PSI, DRIFT_KL) with DRIFT_ALERTS; ROC EER metric (fusion_roc_eer).
- Throughput: GPU-aware AdaptiveBatchController for NATS workers; metrics worker_adaptive_batch_size, worker_gpu_avg_mem_mb, worker_queue_depth, worker_batch_latency_seconds, worker_jobs_processed_total.

## Git
Commit: 0d5b8a6ea742679fd742147dd79205a0e8d1f92c
Branch: main
Dirty: dirty

## Code & Dependencies
- Python dependencies captured
- GitHub Actions workflows copied

## Docker
- Docker runtime captured

## Kubernetes (agentforge-staging)
- Namespace resources enumerated

## Kustomize Profiles
- Kustomize builds captured for profiles and README copied

## NATS / JetStream
- JetStream status collected
- Stream & consumer listings captured

## Prometheus
- Prometheus rule files copied from repo
- Prometheus metadata & SLO queries captured

## Exporter Image Security

## Grafana
- Grafana skipped (vars not set)

## Host Resources
- Host resources snapshot captured

## Application Artifacts & Metrics
- Lineage artifacts listed (var/artifacts)
