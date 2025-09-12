# System Context (GPT-5 Input)

Purpose: Snapshot of current platform (NATS JetStream + Prometheus + SLO + Exporter + K8s namespace agentforge-staging) for future architecture & scaling plan.

Core Components:
- NATS + JetStream (streams/consumers for jobs & results)
- Prometheus (SLO rules: backlog, drain p95)
- Grafana (SLO + burn-rate panels planned/partial)
- Rebuilt prometheus-nats-exporter (SBOM + vuln scan)
- Kubernetes namespace: agentforge-staging
- Inventory script: generate_system_inventory.sh

Current SLO Performance:
- Drain P95 ≈ 12s (Target < 600s) PASS
- Capped latency << 1200s PASS

Key Metrics / Rules:
- jetstream_backlog
- slo_backlog_drain_p95
- (Planned) multi-window burn rate
- Backlog Warning/Critical thresholds

Security:
- Exporter SBOM + vulnerability scan done
- Need pipeline-wide image scans & attestations

Top Gaps / Next Opportunities:
1. Automate daily inventory snapshot (cron / Git artifact)
2. Implement multi-window (fast/slow) burn-rate alerts
3. Add error budget & depletion dashboard
4. Productionize Prometheus Adapter + HPA (backlog_current)
5. Chaos/resiliency tests (exporter / NATS restart scenarios)
6. Add supply chain provenance (SLSA baseline)

What To Produce (GPT-5 Request):
1. Two-sprint architecture & scaling roadmap
2. Refined SLO & error budget policy
3. Top 10 risk register + mitigations
4. Observability & security hardening plan
5. Prioritized backlog (Must / Should / Could) with rough effort (S/M/L)

