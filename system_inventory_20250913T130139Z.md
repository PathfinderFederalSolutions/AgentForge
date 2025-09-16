# System Inventory Snapshot

- Timestamp (UTC): 20250913T130139Z
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
- CI/CD: .github/workflows/ci.yml, supply-chain.yml, SBOMs (Syft), Trivy image scan, Cosign sign for syncdaemon, comms-gateway, ar-context, route-engine, engagement, cds-bridge.
- Orchestration: PhaseRunner in orchestrator/app/main.py, plans/master_orchestration.yaml, worker_protocol.py, results_sink.py.
- Engagement: services/engagement/app/main.py, tests/test_engagement_dual_control.py, ui/tactical-dashboard/src/views/Engagement.tsx.
- HITL: services/hitl/app.py, dual approval endpoints, evidence DAG logging.
- CDS Bridge: services/cds-bridge/app/main.py, k8s/profiles/govcloud/cds-bridge.yaml, k8s/profiles/govcloud/network-policies.yaml, tests/test_cds_hash_verification.py.
- Evidence: var/artifacts/engagement, var/artifacts/phase_runs, services/cds-bridge/logs/transfers.
- Metrics: engagement_time_to_decision_seconds, cds_transfer_success_total, worker_adaptive_batch_size, worker_gpu_avg_mem_mb, worker_queue_depth, worker_batch_latency_seconds, worker_jobs_processed_total, fusion_roc_eer, DRIFT_PSI, DRIFT_KL, DRIFT_ALERTS.
- Security: Cosign signing, SBOMs, Trivy scan, supply chain attestation.
- UI: Tactical dashboard, engagement queue, evidence preview, ROE snapshot, WebAuthn/YubiKey approval.
- K8s: Kustomize profiles, KEDA scaling, ServiceMonitors, PrometheusRules, OTEL Collector, network policies.
- Storage: JSONL, lineage DAG, pgvector, /v1/evidence/{job_id}.
- Messaging: NATS JetStream, subjects, durable consumers, DLQs.
- Observability: Prometheus, OTEL, SLOs, drift monitors, ROC/EER, executor/worker metrics.
- Throughput: AdaptiveBatchController, metrics, GPU-aware scaling.

## Git
Commit: 262c435a37a0b9b2777000d6cf60c25c05f82a92
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
