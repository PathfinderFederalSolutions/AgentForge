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
