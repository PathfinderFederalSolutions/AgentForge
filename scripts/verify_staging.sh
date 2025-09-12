#!/usr/bin/env bash
set -uo pipefail

NS=${NS:-agentforge-staging}
COLOR=${COLOR:-1}
BACKLOG_COUNT=${BACKLOG_COUNT:-100}
PROM_NAMESPACE=${PROM_NAMESPACE:-monitoring}
PROM_SERVICE=${PROM_SERVICE:-kube-prometheus-stack-prometheus}
PROM_PORT=${PROM_PORT:-9090}
CHECK_SLO=${CHECK_SLO:-1}
_ok() { if [[ "$COLOR" == "1" ]]; then echo -e "\033[32m✔\033[0m $1"; else echo "OK: $1"; fi }
_warn() { if [[ "$COLOR" == "1" ]]; then echo -e "\033[33m!\033[0m $1"; else echo "WARN: $1"; fi }
_err() { if [[ "$COLOR" == "1" ]]; then echo -e "\033[31m✘\033[0m $1"; else echo "ERR: $1"; fi }
_run() { echo "> $*"; eval "$*"; }

# Ensure kubectl is configured (prefer a lightweight API call over version)
if kubectl --request-timeout=5s get ns >/dev/null 2>&1 || kubectl --request-timeout=5s cluster-info >/dev/null 2>&1; then
  _ok "kubectl reachable"
else
  _err "kubectl not configured or cluster unreachable"
  exit 1
fi

# 1) External metrics APIService status
if kubectl get apiservice v1beta1.external.metrics.k8s.io >/dev/null 2>&1; then
  kubectl get apiservice v1beta1.external.metrics.k8s.io -o wide
  _ok "external.metrics.k8s.io APIService present"
else
  _err "external.metrics.k8s.io APIService missing"
fi

# 2) KEDA and ScaledObject
if kubectl -n keda get deploy keda-operator >/dev/null 2>&1; then
  kubectl -n keda get deploy,svc,pods -o wide | sed -n '1,200p'
  _ok "KEDA namespace healthy"
else
  _warn "KEDA namespace or operator not found"
fi

if kubectl -n "$NS" get scaledobject keda-nats-worker 2>/dev/null | grep -q .; then
  kubectl -n "$NS" describe scaledobject keda-nats-worker | sed -n '1,160p'
else
  # fall back to name used in repo
  if kubectl -n "$NS" get scaledobject nats-worker-scaledobject >/dev/null 2>&1; then
    kubectl -n "$NS" describe scaledobject nats-worker-scaledobject | sed -n '1,160p'
    _ok "ScaledObject present"
  else
    _err "ScaledObject not found"
  fi
fi

# Quick /jsz scrape from KEDA namespace to validate NetworkPolicies
_warn "Scraping NATS /jsz from keda namespace (ensures KEDA can reach monitoring 8222)"
JSZ_KEDA="jsz-check-$(date +%s)"
kubectl -n keda run "$JSZ_KEDA" --restart=Never --image=busybox:1.36 --env NATS_HOST="nats.$NS.svc.cluster.local" -- \
  sh -lc 'wget -qO- "http://$NATS_HOST:8222/jsz?streams=true&consumers=true&compact=true" | head -n 20' || true
sleep 2
kubectl -n keda logs "$JSZ_KEDA" --tail=-1 2>/dev/null || true
kubectl -n keda delete pod "$JSZ_KEDA" --ignore-not-found >/dev/null 2>&1 || true

# 3) HPA view
kubectl -n "$NS" get hpa -o wide || _warn "HPA not found"

# Also attempt to fetch the KEDA-created HPA name explicitly
if kubectl -n "$NS" get hpa keda-hpa-nats-worker-scaledobject >/dev/null 2>&1; then
  kubectl -n "$NS" describe hpa keda-hpa-nats-worker-scaledobject | sed -n '1,200p'
fi

# 4) External metrics raw endpoint for JetStream backlog
# Hardcoded URL-encoded value for 'scaledobject.keda.sh/name=nats-worker-scaledobject'
ENC_SO="scaledobject.keda.sh%2Fname%3Dnats-worker-scaledobject"
URL="/apis/external.metrics.k8s.io/v1beta1/namespaces/$NS/s0-nats-jetstream-swarm_jobs?labelSelector=$ENC_SO"
if kubectl get --raw "$URL" >/dev/null 2>&1; then
  kubectl get --raw "$URL" | sed -n '1,120p'
  _ok "External metric endpoint responsive"
else
  _warn "External metric endpoint not reachable or empty"
fi

# 4b) Dump JSZ for stream/consumer via HTTP from staging namespace (note: NATS ingress 8222 is locked to keda) 
_warn "Dumping NATS JSZ via HTTP from agentforge-staging"
JSZ_STAGING="nats-jsz-$(date +%s)"
kubectl -n "$NS" run "$JSZ_STAGING" --restart=Never --image=busybox:1.36 --labels "app=nats-backlog" --env NATS_HOST="nats.$NS.svc.cluster.local" -- \
  sh -lc 'wget -qO- "http://$NATS_HOST:8222/jsz?streams=true&consumers=true&compact=true" | sed -n "1,200p"' || true
sleep 2
kubectl -n "$NS" logs "$JSZ_STAGING" --tail=-1 2>/dev/null || true
kubectl -n "$NS" delete pod "$JSZ_STAGING" --ignore-not-found >/dev/null 2>&1 || true

# 5) Verify Services + ServiceMonitors
kubectl -n "$NS" get svc orchestrator tool-executor nats || true
kubectl -n "$NS" get servicemonitor -o name || true

# 6) NetworkPolicies
kubectl -n "$NS" get netpol -o name | sed -n '1,200p' || true

# 7) RBAC quick can-i checks
kubectl -n "$NS" auth can-i get secrets --as=system:serviceaccount:$NS:orchestrator-sa
kubectl -n "$NS" auth can-i get secrets --as=system:serviceaccount:$NS:tool-executor-sa

# 8) Optional smoke backlog to watch HPA react (guarded by env FLAG)
if [[ "${GENERATE_BACKLOG:-0}" == "1" ]]; then
  _warn "Ensuring stream+PULL durable consumer exist, then generating ${BACKLOG_COUNT} msgs on swarm.jobs.staging via nats-box"
  kubectl -n "$NS" run nats-box --restart=Never --image=natsio/nats-box:latest --labels "app=nats-backlog" --env SVC="nats.$NS.svc.cluster.local:4222" --env BACKLOG_COUNT="$BACKLOG_COUNT" -- \
    sh -lc '
      set -euo pipefail
      : "${SVC:?}"
      : "${BACKLOG_COUNT:?}"
      # Ensure stream and PULL durable consumer aligned with scaler (worker-staging)
      nats --server "$SVC" str add swarm_jobs --subjects "swarm.jobs.*" --retention workqueue --storage file --discard old --dupe-window 1m || true
      nats --server "$SVC" con add swarm_jobs worker-staging --pull --deliver all --ack explicit --max-deliver -1 --max-ack-pending 2048 --filter "swarm.jobs.staging" || true
      # Publish backlog
      for i in $(seq 1 "$BACKLOG_COUNT"); do nats --server "$SVC" pub "swarm.jobs.staging" "{\"job_id\":\"test-$i\",\"goal\":\"echo $i\"}" >/dev/null; done
      echo "Published $BACKLOG_COUNT messages"
      echo "--- Consumer info (JSON) ---"
      nats --server "$SVC" con info swarm_jobs worker-staging --json || true
    '
  # Capture logs and clean up
  sleep 2
  LOGS=$(kubectl -n "$NS" logs nats-box --tail=-1 2>/dev/null || true)
  echo "$LOGS" | sed -n '1,200p'
  kubectl -n "$NS" delete pod nats-box --ignore-not-found >/dev/null 2>&1 || true
  _ok "Backlog ensured and published"

  # Poll external metric a few times to observe backlog value
  for t in 1 2 3 4 5 6 7 8; do
    sleep 5
    echo "Polling external metric (attempt $t)"
    kubectl get --raw "$URL" 2>/dev/null | jq -r '.items[0].value // "no-data"' || true
    kubectl -n "$NS" get hpa -o wide || true
  done
fi

# 9) Describe HPA and recent events
if kubectl -n "$NS" get hpa keda-hpa-nats-worker-scaledobject >/dev/null 2>&1; then
  kubectl -n "$NS" describe hpa keda-hpa-nats-worker-scaledobject | sed -n '1,200p'
fi
kubectl -n "$NS" get events --sort-by=.lastTimestamp | tail -n 50 || true

# 10) KEDA logs (operator and metrics-apiserver)
for app in keda-operator keda-metrics-apiserver; do
  POD=$(kubectl -n keda get pods -l app=$app -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [[ -n "$POD" ]]; then
    echo "--- logs: $app ($POD)"; kubectl -n keda logs "$POD" --tail=200 || true
  fi
done

# Exporter pod/service checks
kubectl -n "$NS" get deploy nats-prometheus-exporter svc/nats-prometheus-exporter || true
kubectl -n "$NS" get endpoints nats-prometheus-exporter -o wide || true

# In-cluster metrics curl to confirm metric names
_warn "Fetching exporter /metrics from inside cluster"
CURL_POD="curl-exp-$(date +%s)"
kubectl -n "$NS" run "$CURL_POD" --restart=Never --image=busybox:1.36 --labels "app=nats-backlog" --env EXP_HOST="nats-prometheus-exporter.$NS.svc.cluster.local:7777" -- \
  sh -lc 'wget -qO- "http://$EXP_HOST/metrics" | grep -E "^nats_jetstream_consumer_num_(pending|ack_pending|redelivered)" | head -n 20 || true'
sleep 2
kubectl -n "$NS" logs "$CURL_POD" --tail=-1 2>/dev/null || true
kubectl -n "$NS" delete pod "$CURL_POD" --ignore-not-found >/dev/null 2>&1 || true

# Prometheus target status hint
_warn "If Prometheus is installed, check target: serviceMonitor/$NS/nats-prometheus-exporter/0"

if [[ "$CHECK_SLO" == "1" ]]; then
  _warn "Attempting SLO metric queries (drain estimate / violation ratios)"
  # Try an in-cluster curl via a temporary pod if Prometheus accessible
  PROMQL_BACKLOG_EST='jetstream_backlog_drain_estimate_minutes'
  PROMQL_P95='jetstream_backlog_drain_estimate_minutes_p95_6h'
  PROMQL_VIOL_1H='jetstream_backlog_slo_violation_ratio_1h'
  PROMQL_VIOL_6H='jetstream_backlog_slo_violation_ratio_6h'
  PROMQL_VIOL_24H='jetstream_backlog_slo_violation_ratio_24h'
  Q() { echo -n "${1}" | sed 's/\"/%22/g' | sed "s/'/%27/g" | sed 's/ /%20/g'; }
  CURL_POD_SLO="curl-slo-$(date +%s)"
  kubectl -n "$NS" run "$CURL_POD_SLO" --restart=Never --image=busybox:1.36 --env PROM="${PROM_SERVICE}.${PROM_NAMESPACE}.svc.cluster.local:${PROM_PORT}" -- \
    sh -lc 'for q in '"$PROMQL_BACKLOG_EST"' '"$PROMQL_P95"' '"$PROMQL_VIOL_1H"' '"$PROMQL_VIOL_6H"' '"$PROMQL_VIOL_24H"'; do echo "PROMQL: $q"; wget -qO- "http://$PROM/api/v1/query?query=$(echo $q | sed s/\"/%22/g)"; echo; done' 2>/dev/null || true
  sleep 2
  kubectl -n "$NS" logs "$CURL_POD_SLO" --tail=-1 2>/dev/null | sed -n '1,220p' || true
  kubectl -n "$NS" delete pod "$CURL_POD_SLO" --ignore-not-found >/dev/null 2>&1 || true
fi

_ok "Verification completed"
