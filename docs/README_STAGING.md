Staging verification and end-to-end helper

Run the provided end-to-end script to apply kustomize, ensure NetworkPolicies, create a JetStream backlog Job, and poll KEDA metrics.

Usage (zsh):
  chmod +x ./scripts/run_staging_end_to_end.sh
  NS=agentforge-staging BACKLOG_COUNT=120 ./scripts/run_staging_end_to_end.sh

Troubleshooting:
- If the backlog Job fails with image pull errors, edit k8s/staging/nats-backlog-job.yaml to use an available nats-box tag.
- If Job pods cannot egress to NATS, ensure the NetworkPolicy `allow-backlog-to-nats` exists and the backlog pod has label `app=nats-backlog`.
- If KEDA external metrics remain zero, check NATS JSZ consumer info (Num Pending) and KEDA operator logs.
- For HPA CPU metrics, install metrics-server in the cluster.

# Staging Observability & Alerts

## Exporter Image (Local Build Workaround)
Upstream GHCR pulls returned 403 (anonymous forbidden). A local image build path is provided:
  docker build -t prometheus-nats-exporter:local -f build/prometheus-nats-exporter/Dockerfile .
If using kind:
  kind load docker-image prometheus-nats-exporter:local --name <kindCluster>
Then deploy (kustomize uses image prometheus-nats-exporter:local). After validation, optionally retag/pin an upstream digest and update the Deployment.

## Exporter and Prometheus
- Ensure exporter Pod Ready and Service endpoints present
- Prometheus target Up: up{namespace="agentforge-staging",service="nats-prometheus-exporter"} == 1
- Metrics present: nats_jetstream_consumer_num_pending, num_ack_pending, num_redelivered

Exporter args (trimmed for perf): -varz -jsz=streams,consumers,accounts (connz/routez/subz removed; re-add if needed)

Prometheus config notes:
- kube-prometheus-stack values configured to select ServiceMonitors/Rules by label release=kube-prom across namespaces.
- Ensure Prometheus is installed with scripts/install_monitoring.sh so CRDs exist and ServiceMonitors are picked up.

Exporter validation:
- kubectl -n agentforge-staging get deploy,svc,endpoints nats-prometheus-exporter
- Confirm Prometheus target Up: up{namespace="agentforge-staging",service="nats-prometheus-exporter"}
- In-cluster wget http://nats-prometheus-exporter.agentforge-staging.svc.cluster.local:7777/metrics and look for nats_jetstream_consumer_num_pending|ack_pending|redelivered

## Recording Rules
Added for simpler queries and SLOs:
- jetstream_backlog = sum(nats_jetstream_consumer_num_pending{stream="swarm_jobs",consumer="worker-staging"})
- jetstream_ack_pending = sum(nats_jetstream_consumer_num_ack_pending{...})
- jetstream_redelivery_rate = sum(rate(nats_jetstream_consumer_num_redelivered{...}[5m]))
- jetstream_backlog_drain_rate = -1 * deriv(jetstream_backlog[10m]) (negative means backlog is decreasing)

## Alerts
- SustainedNATSBacklogWarning: backlog >= 3000 for 10m (runbook: /nats/backlog)
- SustainedNATSBacklogCritical: backlog >= 4000 for 10m (runbook: /nats/backlog)
- NATSConsumerAckPendingHigh: ack pending > 1000 for 10m (/nats/ack-pending)
- NATSConsumerRedeliveriesRateHigh: redelivery rate > 1/s (5m avg) for 10m (/nats/redeliveries)
- NATSExporterDown: exporter not scraped for 5m (/nats/exporter)
- NATSJetStreamMetricsMissing: no backlog metric for 10m (/nats/metrics-missing)

## Threshold Rationale
- Warning backlog 3000 chosen to warn before 10m drain SLO risk; critical at 4000 indicates potential breach of 10m objective and approaching 20m hard cap assuming ~6 rps * replicas.
- Ack pending > 1000 sustained implies consumer processing delay that could lead to backlog growth.
- Redelivery rate >1/s sustained suggests systemic failures/timeouts.

## Grafana
- Auto-imported dashboard ConfigMap grafana-tool-executor-dashboard (label grafana_dashboard=1)
- Panels include backlog, drain rate, ack pending, redelivery rate.

## NetworkPolicies
- Consolidated exporter policies: single ingress (Prometheus + kubelet CIDR 10.0.0.0/16) and single egress (4222, 8222, DNS)
- kubelet probe ingress tightened: node CIDR 10.0.0.0/16 only (adjust if control-plane outside this range)
- exporter DNS egress to kube-dns; exporter -> NATS 4222 & 8222 allowed

Security TODO:
- Pin exporter image by digest after validation
- Evaluate if additional JSZ flags required; keep minimal
- Confirm control-plane/node CIDRs; refine ipBlock(s) if needed

## Test
- scripts/install_monitoring.sh
- kubectl apply -k k8s/staging
- scripts/verify_staging.sh with GENERATE_BACKLOG=1 to simulate backlog

---

# Edge Mode

Set EDGE_MODE=1 to enable file-backed JetStream limits per subject and longer consumer ack_wait to tolerate intermittent connectivity. The orchestrator exports edge_link_status{site="<SITE>"} where 1 means connected and 0 means disconnected.

Env vars:
- EDGE_MODE: 1 to enable edge behaviors
- EDGE_JS_MAX_MSGS_PER_SUBJECT: per-subject cap (default 10000)
- EDGE_JS_MAX_BYTES: stream byte cap (default 1GiB)
- EDGE_ACK_WAIT_S: consumer ack wait seconds (default 120)
- NATS_MAX_RECONNECT_WAIT: max reconnect backoff seconds (default 15)
- SITE: label for edge_link_status metric (defaults to ENV)

Behavior changes:
- Streams created with file storage; when EDGE_MODE=1, apply max_msgs_per_subject and max_bytes to prevent disk exhaustion.
- Publishers include Nats-Msg-Id for idempotent replay across disconnects.
- Consumers use durable pull subscriptions with increased ack_wait and max_ack_pending.
- Orchestrator NATS client reconnects with exponential backoff + jitter and updates edge_link_status gauge on disconnect/reconnect.

---

# Tactical Streams (SSE / WebSocket)

Endpoints exposed by the API service:
- GET /events/stream: Server-Sent Events; emits sanitized GeoJSON Feature markers with evidence_link (HMAC-signed).
- WS /events/ws: WebSocket; emits marker and heartbeat events with same payload shape.

Environment:
- EVIDENCE_SIGNING_SECRET: HMAC key used to sign evidence links (default dev-secret).
- EVIDENCE_SIGNING_TTL: Token lifetime seconds (default 300).
- TACTICAL_STREAM_MAX_CLIENTS: Max concurrent clients globally (default 50).
- TACTICAL_STREAM_MAX_PER_IP: Max concurrent clients per IP (default 0 = disabled).
- TACTICAL_SSE_HEARTBEAT_SECS / TACTICAL_WS_HEARTBEAT_SECS: Heartbeat cadence (seconds).
- TACTICAL_REQUIRE_CLIENT_CERT: If set to 1/true, requires a verified client certificate; ingress must forward one of headers ssl-client-verify|x-ssl-client-verify|x-client-verify=SUCCESS.

Security:
- Titles/descriptions are sanitized (printable-only) and length-limited (256/1024).
- Unknown fields are dropped from marker/zone schemas.
- Basic connection caps enforced; per-IP cap optional.

Metrics:
- tactical_alerts_published_total{channel="sse|ws"}
