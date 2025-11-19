# Runbook: NATS JetStream Redeliveries (jetstream_redelivery_rate)

Alert Name:
- NATSConsumerRedeliveriesRateHigh

Meaning:
High redelivery rate indicates messages are being reprocessed due to failures or timeouts.

Impacts:
- Wasted capacity, increased latency, potential backlog expansion.

Key Checks:
1. Worker logs: errors, panics, timeouts.
2. Ack pending also high? Coupled issue.
3. Consumer configuration: ack wait too short causing premature redeliveries?
4. External dependencies failing (DB/network)?
5. Message poison? Specific message causing repeated failures (use nats con next to inspect messages if safe).

Metrics:
- jetstream_redelivery_rate
- jetstream_ack_pending
- jetstream_backlog

Mitigations:
- Increase ack wait if processing legitimately long.
- Add retry backoff / DLQ logic to avoid hot-loop redeliveries.
- Patch failing code path.

After Recovery:
- Document root cause, update test coverage.
