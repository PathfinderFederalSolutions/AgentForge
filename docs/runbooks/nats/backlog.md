# Runbook: NATS JetStream Backlog (jetstream_backlog)

Alert Names:
- SustainedNATSBacklogWarning
- SustainedNATSBacklogCritical
- NATSBacklogDrainSLOAtRisk

SLO Context:
- Objective: Drain backlog so that P95 end-to-end processing latency < 10m (hard cap 20m)
- Backlog >= 4000 + drain estimate > 20m => SLO at risk

Primary Metrics:
- jetstream_backlog
- jetstream_backlog_drain_rate (msgs/sec, positive means draining)
- jetstream_backlog_drain_estimate_minutes
- jetstream_ack_pending
- jetstream_redelivery_rate

Immediate Triage:
1. Confirm exporter target is up: up{service="nats-prometheus-exporter"} == 1
2. Check backlog trend last 30m. Is drain rate positive and sufficient? (backlog / drain_rate)
3. Verify consumers running: kubectl -n agentforge-staging get deploy nats-worker
4. Inspect consumer info: nats con info swarm_jobs worker-staging --json
5. Look for elevated ack pending or redeliveries indicating slow/failing processing.

Diagnostics:
- Scale status: kubectl -n agentforge-staging get hpa,deployment nats-worker -o wide
- Pod logs (errors/timeouts): kubectl -n agentforge-staging logs deploy/nats-worker --tail=200
- Network latency / DNS issues: kubectl -n agentforge-staging exec <worker-pod> -- apk add --no-cache drill && drill nats
- Storage / stream limits: nats str info swarm_jobs --json

Common Causes & Actions:
- Insufficient worker replicas: Unpause autoscaling (remove autoscaling.keda.sh/paused) or manually scale.
- CPU / memory throttling: Check pod metrics; raise limits if sustained >90% utilization.
- Hot keys / large messages: Optimize message size or parallelism.
- Redeliveries high: See redeliveries runbook.

Mitigations:
- Temporary manual scale: kubectl -n agentforge-staging scale deploy/nats-worker --replicas=<N>
- Increase max ack pending in consumer if legitimate burst: adjust consumer config (max-ack-pending)
- If backlog huge and processing permanently slow, consider draining with a one-off batch worker deployment.

Post-Resolution:
- Record peak backlog, drain rate, time to recover.
- Update capacity plan if sustained load > provisioned capacity.

References:
- Consumer design: KEDA ScaledObject nats-jetstream trigger
- Metrics: Prometheus recording rules jetstream_*

## Edge Mode (Store-and-Forward) Verification

Purpose: Validate that Edge Mode persists evidence locally during link loss, replays without duplication on reconnect, and that backlog drain SLO remains within objective.

Prereqs:
- Edge overlay applied (k8s/profiles/edge) with NATS JetStream PV at /lib/nats and syncdaemon running.
- Prometheus scraping NATS and syncdaemon ServiceMonitors.
- ENV vars for Edge set in ConfigMap (EDGE_MODE=true, SITE, etc.).

Steps:
1) Functional tests (local or CI):
   - Run pytest for: tests/test_edge_store_forward.py tests/test_edge_disconnect_reconnect.py
   - Expectations:
     - Persistence + replay without duplication and edge_link_status toggles 1→0→1.
     - Sync Daemon replay_queue_depth drains to 0 after files in /agentforge/replay are processed; evidence ledger written to services/syncdaemon/logs/*.jsonl.

2) Metrics checks:
   - Confirm Prom targets up for NATS and syncdaemon.
   - Look for metrics:
     - edge_link_status{site=...} from orchestrator.
     - replay_queue_depth{site=...} and edge_bandwidth_bps{site=...} from syncdaemon.
   - During disconnect, replay_queue_depth may increase; on reconnect, it should monotonically decrease to 0.

3) Chaos smoke (NATS restart):
   - Restart NATS StatefulSet (1 pod) and observe:
     - No evidence loss: files queued under /agentforge/replay continue to replay post-reconnect.
     - No duplicate publishes: idempotency via Nats-Msg-Id (content SHA256) dedupes.
     - EDGE gauge flips 1→0→1, and replay_queue_depth recovers to 0.

4) Backlog drain SLO measurement:
   - Use scripts/measure_backlog_drain.sh with EDGE_MODE=true environment.
   - Ensures stream/consumer then publishes a controlled backlog and measures drain until 0 via Prometheus metric jetstream_backlog.
   - Acceptance: P95 <= 10m, no run > 20m. Record durations array and computed P95.

Evidence:
- Attach the syncdaemon ledger artifact (services/syncdaemon/logs/*.jsonl) as edge_sync_manifest in CI artifacts.
- Save a short report including durations and P95 along with timestamp and commit SHA.

Troubleshooting:
- If replay_queue_depth stalls > 0:
  - Check NATS availability and subject configuration (SYNC_SUBJECT).
  - Inspect syncdaemon logs (Deployment logs) for publish errors; verify NATS_URL.
  - Confirm filesystem permissions for /agentforge/replay and logs directories in Pod.
- If duplicates observed:
  - Verify Nats-Msg-Id header present at all publishers (orchestrator, syncdaemon); dedupe window should be active.
- If SLO breach:
  - See diagnostics and mitigations above; consider increasing worker replicas or consumer max-ack-pending temporarily.
