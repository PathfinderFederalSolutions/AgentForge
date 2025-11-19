# NATS JetStream Backlog Drain SLO

Objective:
P95 message backlog drain time (from arrival to backlog cleared) <= 10 minutes with a hard cap of 20 minutes.

Indicators & Derived Metrics:
- jetstream_backlog: current pending messages.
- jetstream_backlog_drain_rate: msgs/sec backlog is shrinking (positive = draining).
- jetstream_backlog_drain_estimate_minutes: (backlog / drain_rate)/60 (guarded to avoid divide-by-zero).
- jetstream_backlog_drain_estimate_minutes_p95_6h: 95th percentile drain estimate over rolling 6h.
- jetstream_backlog_slo_violation: 1 when drain_estimate_minutes > 20.
- jetstream_backlog_slo_violation_ratio_24h: fraction of last 24h in violation.

Targets:
- P95 drain estimate (6h lookback) < 10.
- Violation ratio (24h) < 0.01 (<= 1% of day over 20m estimate).

Error Budget:
- 24h * 1% = 14.4 minutes permissible time where estimate exceeds 20m.

Alert Coupling:
- SustainedNATSBacklogCritical + NATSBacklogDrainSLOAtRisk serve early warning.

Operational Use:
1. Track p95 and violation ratio on Grafana SLO panel.
2. During incident: compute remaining error budget = (0.01 * 1440) - (violation_ratio_24h * 1440) minutes.
3. If remaining budget < 5 minutes escalate capacity planning.

Data Quality Notes:
- Estimate accuracy depends on monotonic decrease periods; bursts can cause noisy derivative. Consider smoothing with rate() if deriv() unstable.
- For higher fidelity future improvement: emit processing latency histogram at worker side.

Improvements Roadmap:
- Replace estimate with direct end-to-end latency metric histogram.
- Add burn-rate multi-window alert (e.g., 5m & 1h) once latency metric available.
