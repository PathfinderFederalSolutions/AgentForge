# Runbook: NATS Prometheus Exporter

Alert Name:
- NATSExporterDown
- NATSJetStreamMetricsMissing

Symptoms:
- Exporter target down or jetstream_* recording rules show gaps.

Immediate Steps:
1. kubectl -n agentforge-staging get pods -l app=nats-prom-exporter
2. Describe pod for restart/crashloop: kubectl -n agentforge-staging describe pod <pod>
3. Check logs: kubectl -n agentforge-staging logs <pod>
4. Confirm Service endpoint: kubectl -n agentforge-staging get endpoints nats-prometheus-exporter
5. Verify NetworkPolicies allow Prometheus ingress (monitoring namespace) and NATS egress 8222.

Network:
- Test in-cluster: busybox wget http://nats-prometheus-exporter.agentforge-staging.svc.cluster.local:7777/metrics
- If connection refused: check container listening / liveness probe failures.

Common Causes:
- Image pull issue (local tag missing after node restart) -> rebuild & kind load image.
- NATS endpoint unreachable (NetworkPolicy regression) -> adjust allow-exporter-egress.
- OOMKilled -> increase memory limit slightly (observe usage first).

Restoration:
- Rollout restart: kubectl -n agentforge-staging rollout restart deploy/nats-prometheus-exporter
- If image updated: ensure digest pinned in manifest.

Post-Incident:
- Add gap annotation in monitoring timeline.
