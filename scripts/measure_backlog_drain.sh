#!/usr/bin/env bash
# Measure NATS JetStream backlog drain time and compare against SLO.
# SLO: P95 drain time <= 10m, hard cap 20m for any single run.
# Creates backlog, then allows workers (or KEDA) to drain and records durations.
# Requires: kubectl, jq, (optional) bc. Cluster NetworkPolicy must allow pods with label app=nats-backlog to reach NATS 4222.
set -euo pipefail

# Configurable parameters via env vars
NS="${NS:-agentforge-staging}"            # namespace containing NATS + workers
STREAM="${STREAM:-swarm_jobs}"            # JetStream stream name
SUBJECT_PATTERN="${SUBJECT_PATTERN:-swarm.jobs.staging}" # subject used to publish backlog
CONSUMER="${CONSUMER:-worker-staging}"    # Durable consumer name (pull)
WORKER_DEPLOY="${WORKER_DEPLOY:-nats-worker}" # Worker deployment name
BACKLOG_COUNT="${BACKLOG_COUNT:-3000}"    # Messages per run to publish
RUNS="${RUNS:-3}"                         # Number of measurement iterations
MAX_CAP_MINUTES="${MAX_CAP_MINUTES:-20}"  # Hard cap
OBJ_P95_MINUTES="${OBJ_P95_MINUTES:-10}"  # Objective for P95
PROM_NAMESPACE="${PROM_NAMESPACE:-monitoring}"
PROM_SERVICE="${PROM_SERVICE:-kube-prometheus-stack-prometheus}"
PROM_PORT="${PROM_PORT:-9090}"
SLEEP_POLL="${SLEEP_POLL:-5}"             # seconds between backlog polls
PAUSE_KEDA="${PAUSE_KEDA:-0}"             # 1 to pause autoscaling & manually scale workers
WORKER_REPLICAS="${WORKER_REPLICAS:-1}"   # Used if PAUSE_KEDA=1
PUBLISH_PARALLELISM="${PUBLISH_PARALLELISM:-50}" # concurrent publish shells (best effort)

log() { printf '%s %s\n' "[$(date +%H:%M:%S)]" "$*"; }
err() { log "ERROR: $*" >&2; }
ok() { log "OK: $*"; }

require_bin() { command -v "$1" >/dev/null 2>&1 || { err "Missing required binary: $1"; exit 1; }; }
for b in kubectl jq; do require_bin "$b"; done

get_backlog_prom() {
  local q='jetstream_backlog{stream="'"$STREAM"'"}'
  kubectl -n "$NS" run prom-q-$$ --restart=Never --image=busybox:1.36 --env Q="$q" --env PROM="${PROM_SERVICE}.${PROM_NAMESPACE}.svc.cluster.local:${PROM_PORT}" --labels app=nats-backlog -- \
    sh -lc 'wget -qO- "http://$PROM/api/v1/query?query=$(echo $Q | sed s/"/%22/g)"' 2>/dev/null || true
  sleep 1
  local pod
  pod=$(kubectl -n "$NS" get pods -l run=prom-q-$$ -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  local raw="" val=""
  if [[ -n "$pod" ]]; then
    raw=$(kubectl -n "$NS" logs "$pod" 2>/dev/null || true)
    kubectl -n "$NS" delete pod "$pod" --ignore-not-found >/dev/null 2>&1 || true
  fi
  val=$(echo "$raw" | jq -r 'try .data.result[0].value[1] catch "NaN"' 2>/dev/null || echo NaN)
  echo "$val"
}

ensure_stream_consumer() {
  log "Ensuring stream and durable consumer exist"
  kubectl -n "$NS" run js-admin-$$ --restart=Never --image=natsio/nats-box:latest --labels app=nats-backlog --env SVC="nats.${NS}.svc.cluster.local:4222" -- \
    sh -lc 'set -e; nats --server "$SVC" str add '"$STREAM"' --subjects '"$SUBJECT_PATTERN"' --retention workqueue --storage file --discard old --dupe-window 30s || true; \
              nats --server "$SVC" con add '"$STREAM"' '"$CONSUMER"' --pull --filter '"$SUBJECT_PATTERN"' --deliver all --ack explicit --max-deliver -1 --max-ack-pending 5000 || true'
  sleep 2 || true
  local pod; pod=$(kubectl -n "$NS" get pods -l run=js-admin-$$ -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  [[ -n "$pod" ]] && kubectl -n "$NS" logs "$pod" || true
  kubectl -n "$NS" delete pod -l run=js-admin-$$ --ignore-not-found >/dev/null 2>&1 || true
  ok "Stream & consumer ensured"
}

publish_backlog() {
  local count="$1"; log "Publishing $count messages (parallel=$PUBLISH_PARALLELISM)"
  kubectl -n "$NS" run pub-$$ --restart=Never --image=natsio/nats-box:latest --labels app=nats-backlog --env SVC="nats.${NS}.svc.cluster.local:4222" --env COUNT="$count" --env SUBJECT="$SUBJECT_PATTERN" --env PAR="$PUBLISH_PARALLELISM" -- \
    sh -lc 'set -e; c=${COUNT}; s=${SUBJECT}; svc=$SVC; par=${PAR}; seq 1 "$c" | xargs -P "$par" -I{} sh -c "nats --server $svc pub $s '{\"job_id\":\"bench-{}\",\"goal\":\"noop\"}' >/dev/null"; echo "Published $c"'
  sleep 2
  local pod; pod=$(kubectl -n "$NS" get pods -l run=pub-$$ -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  [[ -n "$pod" ]] && kubectl -n "$NS" logs "$pod" | sed -n '1,10p' || true
  kubectl -n "$NS" delete pod -l run=pub-$$ --ignore-not-found >/dev/null 2>&1 || true
}

scale_workers_for_test() {
  if [[ "$PAUSE_KEDA" == "1" ]]; then
    log "Pausing KEDA (remove annotations after test) & scaling workers to ${WORKER_REPLICAS}"
    kubectl -n "$NS" annotate scaledobject nats-worker-scaledobject autoscaling.keda.sh/paused-replicas="${WORKER_REPLICAS}" --overwrite || true
    kubectl -n "$NS" annotate scaledobject nats-worker-scaledobject keda.sh/paused="true" --overwrite || true
    kubectl -n "$NS" scale deploy/"$WORKER_DEPLOY" --replicas="$WORKER_REPLICAS" || true
  else
    log "Allowing KEDA to manage scaling (no manual pause)"
  fi
}

unpause_keda_if_needed() {
  if [[ "$PAUSE_KEDA" == "1" ]]; then
    log "Unpausing KEDA"
    kubectl -n "$NS" annotate scaledobject nats-worker-scaledobject autoscaling.keda.sh/paused-replicas- || true
    kubectl -n "$NS" annotate scaledobject nats-worker-scaledobject keda.sh/paused- || true
  fi
}

wait_for_workers_ready() {
  log "Waiting for worker pods to become Ready"
  for i in {1..30}; do
    local ready total
    ready=$(kubectl -n "$NS" get deploy "$WORKER_DEPLOY" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
    total=$(kubectl -n "$NS" get deploy "$WORKER_DEPLOY" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)
    [[ -z "$ready" ]] && ready=0
    if [[ "$ready" == "$total" && "$total" != "0" ]]; then ok "Workers ready ($ready)"; return 0; fi
    sleep 2
  done
  err "Workers not fully ready (continuing anyway)"
}

measure_single_run() {
  local run_id="$1"
  log "=== Run $run_id/$RUNS ==="
  publish_backlog "$BACKLOG_COUNT"
  # Confirm backlog via Prom metric (may lag); fallback to consumer info if NaN
  local start_backlog
  start_backlog=$(get_backlog_prom || echo NaN)
  log "Start backlog (prom metric) = $start_backlog"
  local start_ts; start_ts=$(date +%s)
  local last_backlog="$start_backlog" now_backlog elapsed drain_minutes
  for i in {1..240}; do # up to 240 * SLEEP_POLL seconds (~20m if 5s interval)
    now_backlog=$(get_backlog_prom || echo NaN)
    local now_ts; now_ts=$(date +%s)
    if [[ "$now_backlog" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      if (( ${now_backlog%.*} == 0 )); then
        elapsed=$(( now_ts - start_ts ))
        drain_minutes=$(awk 'BEGIN{printf "%.2f",('$elapsed'/60)}')
        log "Backlog drained to 0 in ${drain_minutes}m (elapsed ${elapsed}s)"
        echo "$drain_minutes"
        return 0
      fi
      last_backlog="$now_backlog"
    fi
    sleep "$SLEEP_POLL"
  done
  err "Run $run_id hit timeout without draining (cap ${MAX_CAP_MINUTES}m)"
  echo "$MAX_CAP_MINUTES" # treat as cap breach
  return 1
}

main() {
  ensure_stream_consumer
  scale_workers_for_test
  wait_for_workers_ready
  unpause_keda_if_needed # If using autoscaling we only paused momentarily to set initial state
  local durations=()
  for r in $(seq 1 "$RUNS"); do
    local d
    d=$(measure_single_run "$r" || true)
    durations+=("$d")
    log "Sleeping 10s between runs"; sleep 10
  done
  printf '\nDurations (minutes): %s\n' "${durations[*]}"
  # Compute P95 (simple sort and index)
  local sorted
  IFS=$'\n' sorted=($(printf '%s\n' "${durations[@]}" | sort -n))
  unset IFS
  local n=${#sorted[@]}
  if (( n == 0 )); then err "No durations collected"; exit 2; fi
  local idx=$(( (95 * n + 99) / 100 - 1 )) # ceil(0.95*n)-1
  (( idx < 0 )) && idx=0
  (( idx >= n )) && idx=$(( n - 1 ))
  local p95=${sorted[$idx]}
  log "Computed P95 drain time = ${p95}m (n=$n, idx=$idx)"
  local status=0
  awk -v p95="$p95" -v obj="$OBJ_P95_MINUTES" 'BEGIN{ if (p95+0 <= obj+0) exit 0; else exit 1 }' || status=1
  if (( status == 0 )); then ok "P95 meets objective (<= ${OBJ_P95_MINUTES}m)"; else err "P95 exceeds objective (>${OBJ_P95_MINUTES}m)"; fi
  # Check any individual run > cap
  for d in "${durations[@]}"; do
    awk -v d="$d" -v cap="$MAX_CAP_MINUTES" 'BEGIN{ if (d+0 <= cap+0) exit 0; else exit 1 }' || { err "Run exceeded hard cap ${MAX_CAP_MINUTES}m (d=${d})"; status=2; }
  done
  if (( status == 0 )); then ok "SLO PASS"; else err "SLO FAIL"; fi
  # Optional: emit JSON summary
  if [[ "${JSON_OUTPUT:-0}" == "1" ]]; then
    printf '{"runs":%d,"durations":["%s"],"p95":"%s","objective_minutes":%s,"cap_minutes":%s,"status":"%s"}\n' \
      "$RUNS" "$(IFS=,; echo "${durations[*]}")" "$p95" "$OBJ_P95_MINUTES" "$MAX_CAP_MINUTES" "$([[ $status == 0 ]] && echo pass || echo fail)"
  fi
  exit $status
}

main "$@"
