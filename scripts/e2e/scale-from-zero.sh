#!/usr/bin/env bash
set -euo pipefail
NS=agentforge-staging
FILE="k8s/staging/nats-backlog-job.yaml"

echo "[1/4] Scaling nats-worker to 0..."
kubectl -n "$NS" scale deploy/nats-worker --replicas=0
kubectl -n "$NS" rollout status deploy/nats-worker --timeout=300s || true

echo "[2/4] Publishing backlog..."
kubectl -n "$NS" delete job nats-backlog --ignore-not-found >/dev/null 2>&1 || true
kubectl -n "$NS" create -f "$FILE"
kubectl -n "$NS" wait --for=condition=complete --timeout=300s job/nats-backlog

echo "[3/4] External metric snapshot:"
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/$NS/s0-nats-jetstream-swarm_jobs?labelSelector=scaledobject.keda.sh%2Fname%3Dnats-worker-scaledobject" | jq -r 'try .items[0].value catch "no-data"'

echo "[4/4] Status:"
kubectl -n "$NS" get hpa,scaledobject
kubectl -n "$NS" get deploy nats-worker

# Chaos run orchestration (optional)
CHAOS="${CHAOS:-0}"               # 1 to run chaos experiments
CHAOS_LOG_EXPORT="${CHAOS_LOG_EXPORT:-0}" # 1 to also emit metrics via ephemeral pod logs (for Promtail/Loki)
CHAOS_DIR="var/artifacts/chaos"
TS="$(date +%Y%m%dT%H%M%S)"
RUN_DIR="${CHAOS_DIR}/${TS}"
EXPT_DIR="chaos/experiments"
NS="${NS:-agentforge-staging}"

mkdir -p "$RUN_DIR"

run_and_time() {
  local name="$1" file="$2"; shift 2
  echo "[chaos] Applying $name ($file)" | tee -a "$RUN_DIR/chaos.log"
  local start end dur
  start=$(date +%s)
  kubectl apply -f "$file" | tee -a "$RUN_DIR/chaos.log" || true
  # Wait for duration+grace by polling experiment status
  sleep 5
  # Capture initial metrics snapshot
  kubectl -n "$NS" get hpa,scaledobject >"$RUN_DIR/${name}_pre_scale.txt" 2>&1 || true
  # Best-effort wait based on embedded duration in yaml
  DUR=$(grep -E 'duration: *"?[0-9]+[smh]"?' "$file" | head -n1 | awk '{print $2}' | tr -d '"' || echo "60s")
  # Convert to seconds (supports s,m,h)
  UNIT=${DUR: -1}; NUM=${DUR::-1};
  case "$UNIT" in
    s) S=$NUM;; m) S=$((NUM*60));; h) S=$((NUM*3600));; *) S=60;;
  esac
  GRACE=15
  sleep $((S+GRACE))
  # Archive resource state
  kubectl -n "$NS" get pods -o wide >"$RUN_DIR/${name}_pods.txt" 2>&1 || true
  kubectl -n "$NS" describe deploy nats >"$RUN_DIR/${name}_nats_describe.txt" 2>&1 || true
  kubectl -n "$NS" get events --sort-by=.lastTimestamp -A | tail -n 200 >"$RUN_DIR/${name}_events.txt" 2>&1 || true
  # End timestamp
  end=$(date +%s)
  dur=$((end-start))
  # Per-experiment gauge for UI visibility
  echo "chaos_recovery_seconds{name=\"$name\"} $dur" | tee -a "$RUN_DIR/metrics.prom"
  echo "[chaos] ${name} elapsed ${dur}s" | tee -a "$RUN_DIR/chaos.log"
}

if [[ "$CHAOS" == "1" ]]; then
  echo "[chaos] Installing Chaos Mesh kustomization" | tee -a "$RUN_DIR/chaos.log"
  kubectl apply -k k8s/staging/chaos-mesh | tee -a "$RUN_DIR/chaos.log" || true
  echo "[chaos] Waiting for chaos-controller-manager" | tee -a "$RUN_DIR/chaos.log"
  kubectl -n chaos-mesh rollout status deploy/chaos-controller-manager --timeout=180s || true
  # Execute experiments sequentially
  for exp in nats_restart.yaml partition_api_nats.yaml replay_storm.yaml; do
    if [[ -f "$EXPT_DIR/$exp" ]]; then
      name="${exp%.yaml}"
      run_and_time "$name" "$EXPT_DIR/$exp"
    fi
  done
  # Build local Prometheus exposition and MTTR summary
  {
    echo "# HELP chaos_recovery_seconds Time to complete chaos experiment (s)"
    echo "# TYPE chaos_recovery_seconds gauge"
    cat "$RUN_DIR/metrics.prom"
  } > "$RUN_DIR/chaos_export.txt"

  # Optionally export via ephemeral pod logs for log-based scraping
  if [[ "$CHAOS_LOG_EXPORT" == "1" ]]; then
    echo "[chaos] Emitting metrics via ephemeral pod logs" | tee -a "$RUN_DIR/chaos.log"
    kubectl -n "$NS" run "chaos-metrics-$(date +%s)" --restart=Never --image=busybox:1.36 --env TS="$TS" --overrides='{"metadata":{"labels":{"app":"chaos-exporter"}}}' -- \
      sh -lc 'echo "# HELP chaos_recovery_seconds Time to complete chaos experiment (s)"; echo "# TYPE chaos_recovery_seconds gauge"; cat /dev/stdin' < "$RUN_DIR/metrics.prom" \
      | tee -a "$RUN_DIR/chaos_export.txt" || true
  fi

  # Generate MTTR summary table and stats + histogram
  awk 'match($0, /chaos_recovery_seconds\{name="([^"]+)"\} ([0-9]+)/, a) {printf "%s %s\n", a[1], a[2]}' "$RUN_DIR/metrics.prom" | sort -k2,2n > "$RUN_DIR/mttr_data.tmp" || true
  if [[ -s "$RUN_DIR/mttr_data.tmp" ]]; then
    mapfile -t VALS < <(awk '{print $2}' "$RUN_DIR/mttr_data.tmp")
    N=${#VALS[@]}
    MIN=${VALS[0]:-0}
    MAX=${VALS[$((N-1))]:-0}
    SUM=0
    for v in "${VALS[@]}"; do SUM=$((SUM+v)); done
    if [[ "$N" -gt 0 ]]; then MEAN=$(awk -v s="$SUM" -v n="$N" 'BEGIN{printf "%.2f", s/n}'); else MEAN=0; fi
    # p50
    if [[ "$N" -gt 0 ]]; then
      if (( N % 2 == 1 )); then P50=${VALS[$(((N+1)/2-1))]}; else P50=$(awk -v a="${VALS[$((N/2-1))]}" -v b="${VALS[$((N/2))]}" 'BEGIN{printf "%.2f", (a+b)/2}'); fi
    else P50=0; fi
    # nearest-rank percentiles
    if [[ "$N" -gt 0 ]]; then IDX=$(( (90*N + 99)/100 )); (( IDX<1 )) && IDX=1; (( IDX>N )) && IDX=$N; P90=${VALS[$((IDX-1))]}; else P90=0; fi
    if [[ "$N" -gt 0 ]]; then IDX=$(( (95*N + 99)/100 )); (( IDX<1 )) && IDX=1; (( IDX>N )) && IDX=$N; P95=${VALS[$((IDX-1))]}; else P95=0; fi
    if [[ "$N" -gt 0 ]]; then IDX=$(( (99*N + 99)/100 )); (( IDX<1 )) && IDX=1; (( IDX>N )) && IDX=$N; P99=${VALS[$((IDX-1))]}; else P99=0; fi

    # Build histogram exposition from observed durations
    BUCKETS=(30 60 90 120 180 300 600 900)
    : > "$RUN_DIR/histogram.prom"
    for b in "${BUCKETS[@]}"; do
      CNT=$(awk -v th="$b" '{if ($2<=th) c++} END{print c+0}' "$RUN_DIR/mttr_data.tmp")
      echo "chaos_recovery_duration_seconds_bucket{le=\"$b\"} $CNT" >> "$RUN_DIR/histogram.prom"
    done
    echo "chaos_recovery_duration_seconds_bucket{le=\"+Inf\"} $N" >> "$RUN_DIR/histogram.prom"
    echo "chaos_recovery_duration_seconds_sum $SUM" >> "$RUN_DIR/histogram.prom"
    echo "chaos_recovery_duration_seconds_count $N" >> "$RUN_DIR/histogram.prom"

    # Append histogram family to export
    {
      echo "# HELP chaos_recovery_duration_seconds Chaos recovery duration histogram (s)"
      echo "# TYPE chaos_recovery_duration_seconds histogram"
      cat "$RUN_DIR/histogram.prom"
    } >> "$RUN_DIR/chaos_export.txt"

    {
      echo "Chaos MTTR Summary (seconds)"
      echo "count=$N mean=$MEAN p50=$P50 p90=$P90 p95=$P95 p99=$P99 max=$MAX"
      echo "name,seconds"
      awk '{printf "%s,%s\n", $1, $2}' "$RUN_DIR/mttr_data.tmp"
    } > "$RUN_DIR/mttr_summary.txt"
  else
    echo "No chaos metrics collected" > "$RUN_DIR/mttr_summary.txt"
  fi
fi