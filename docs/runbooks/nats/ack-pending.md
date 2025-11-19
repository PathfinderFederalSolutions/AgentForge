# Runbook: NATS JetStream Ack Pending (jetstream_ack_pending)

Alert Name:
- NATSConsumerAckPendingHigh

Definition:
Ack pending represents messages delivered to the consumer but not yet acknowledged.

Why It Matters:
- Sustained high ack pending suggests slow processing, risk of redeliveries and backlog growth.

Key Metrics:
- jetstream_ack_pending
- jetstream_backlog
- jetstream_redelivery_rate

Triage Steps:
1. Confirm consumer replicas: kubectl -n agentforge-staging get deploy nats-worker -o wide
2. Inspect worker logs for timeouts/errors.
3. Check CPU / memory saturation of worker pods.
4. Confirm max_ack_pending value (nats con info swarm_jobs worker-staging --json).
5. Verify no external dependency latency (DB, network).

Diagnostics:
- Redeliveries trending up? Investigate failures.
- Message size / processing time increased recently?

Actions:
- Scale workers (unpause KEDA or manual scale).
- Optimize processing path (reduce synchronous I/O, batch operations).
- Increase max ack pending only after validating consumer stability.

Post Mortem:
- Capture peak ack pending, backlog correlation, recovery time.
