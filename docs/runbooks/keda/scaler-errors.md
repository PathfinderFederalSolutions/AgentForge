# Runbook: KEDA Scaler Errors

Alert Name:
- KEDAScalerErrors

Meaning:
KEDA metrics adapter is reporting scaler errors for one or more triggers.

Triage:
1. Logs:
   - kubectl -n keda logs deploy/keda-operator --tail=200
   - kubectl -n keda logs deploy/keda-metrics-apiserver --tail=200
2. Check ScaledObject status conditions: kubectl -n agentforge-staging describe scaledobject nats-worker-scaledobject
3. Validate NATS monitoring endpoint reachable from keda namespace (network-policies test).
4. Confirm NATS consumer/stream exist and names match trigger metadata.

Common Causes:
- Stream/consumer missing or renamed.
- NetworkPolicy blocking 8222.
- Authentication / TLS mismatch (future if enabled).

Resolution:
- Fix underlying resource naming or network issue.
- Reapply ScaledObject to refresh status: kubectl -n agentforge-staging apply -f k8s/staging/keda-nats-worker.yaml

Follow Up:
- Add regression test to verification script.

---
## Observed behaviors during chaos runs
- During NetworkChaos (partition API->NATS), external metrics may momentarily report no-data. KEDA retains last known value; scale decisions may pause. Once connectivity resumes, metrics recover without manual intervention.
- PodChaos (nats restart) caused a transient spike in jetstream_backlog; HPA reacted within 1-2 reconciliation periods.
- Replay storm (induced latency/loss) increases redeliveries; ensure maxAckPending >= backlog concurrency to avoid starvation.

## Remediation steps
- If ScaledObject shows Error status after chaos:
  - kubectl -n agentforge-staging describe scaledobject nats-worker-scaledobject | sed -n 's/.*Status.*//p'
  - Restart metrics server: kubectl -n keda rollout restart deploy/keda-metrics-apiserver
- Validate JSZ reachability from keda namespace using busybox wget against http://nats.agentforge-staging.svc.cluster.local:8222/jsz
- For persistent replays, bump consumer AckWait and maxAckPending; verify with: nats consumer info swarm_jobs worker-staging
