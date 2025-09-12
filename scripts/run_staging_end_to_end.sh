#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-agentforge-staging}"
BACKLOG_COUNT="${BACKLOG_COUNT:-200}"
WAIT_FOR_JOB="${WAIT_FOR_JOB:-1}"
FINAL_PASS="${FINAL_PASS:-1}"   # run pause/resume validation by default
SO_NAME="${SO_NAME:-nats-worker-scaledobject}"

echo "Applying staging kustomization..."
kubectl apply -k k8s/staging

# Recreate HPA so KEDA rebuilds it fresh
if kubectl -n "$NS" get hpa keda-hpa-${SO_NAME} >/dev/null 2>&1; then
  echo "Deleting existing HPA keda-hpa-${SO_NAME} to let KEDA recreate with min=0"
  kubectl -n "$NS" delete hpa keda-hpa-${SO_NAME} || true
  sleep 3
fi

# Give KEDA a chance to reconcile ScaledObject/HPA after apply
sleep 3 || true

echo "Waiting for NATS rollout..."
kubectl -n "$NS" rollout status deploy/nats --timeout=180s

if [[ "$FINAL_PASS" == "1" ]]; then
  echo "Pausing KEDA scaling by setting autoscaling.keda.sh/paused-replicas=1 and keda.sh/paused=true on ScaledObject/${SO_NAME}"
  kubectl -n "$NS" annotate --overwrite scaledobject ${SO_NAME} autoscaling.keda.sh/paused-replicas="1" || true
  kubectl -n "$NS" annotate --overwrite scaledobject ${SO_NAME} keda.sh/paused="true" || true
  # Ensure Deployment stays at 0 replicas while paused so backlog can accumulate
  echo "Scaling nats-worker to 0 replicas while paused to accumulate backlog"
  kubectl -n "$NS" scale deploy/nats-worker --replicas=0 || true
fi

echo "Creating backlog Job (BACKLOG_COUNT=$BACKLOG_COUNT)"
JOB_JSON="$(kubectl -n "$NS" create -f k8s/staging/nats-backlog-job.yaml -o json)"
JOB_NAME="$(jq -r '.metadata.name' <<<"$JOB_JSON" || true)"
if [[ -z "${JOB_NAME:-}" || "${JOB_NAME}" == "null" ]]; then
  echo "Failed to capture job name, listing jobs..."
  kubectl -n "$NS" get jobs -l 'app=nats-backlog' -o name
fi

if [[ "${WAIT_FOR_JOB}" == "1" ]]; then
  echo "Waiting for backlog job to complete (300s)"
  if [[ -n "${JOB_NAME:-}" ]]; then
    kubectl -n "$NS" wait --for=condition=complete --timeout=300s "job/${JOB_NAME}" || true
  else
    kubectl -n "$NS" wait --for=condition=complete --timeout=300s job -l 'app=nats-backlog' || true
  fi
fi

# Show logs from the latest backlog pod
POD="$(kubectl -n "$NS" get pods -l 'app=nats-backlog' -o json \
  | jq -r '.items | sort_by(.metadata.creationTimestamp) | last(.[]).metadata.name')"
if [[ -n "${POD:-}" && "${POD}" != "null" ]]; then
  echo "--- Logs from ${POD} ---"
  kubectl -n "$NS" logs "$POD" --tail=200 || true
fi

echo "NATS Service Endpoints:"
kubectl -n "$NS" get endpoints nats -o wide || true

# Dump JSZ via short-lived pods (no --rm; capture logs; then delete)
echo "Dumping NATS JSZ via short-lived pod (agentforge-staging)"
JSZ_POD="nats-jsz-$(date +%s)"
kubectl -n "$NS" run "$JSZ_POD" --restart=Never --image=busybox:1.36 --env NATS_HOST="nats.$NS.svc.cluster.local" -- \
  sh -lc 'wget -qO- "http://$NATS_HOST:8222/jsz?streams=true&consumers=true&compact=true" | sed -n "1,120p"' || true
sleep 2
kubectl -n "$NS" logs "$JSZ_POD" --tail=-1 2>/dev/null || true
kubectl -n "$NS" delete pod "$JSZ_POD" --ignore-not-found >/dev/null 2>&1 || true

echo "Verifying KEDA namespace can scrape NATS /jsz"
KEDA_JSZ_POD="keda-jsz-$(date +%s)"
kubectl -n keda run "$KEDA_JSZ_POD" --restart=Never --image=busybox:1.36 --env NATS_HOST="nats.$NS.svc.cluster.local" -- \
  sh -lc 'wget -qO- "http://$NATS_HOST:8222/jsz?streams=true&consumers=true&compact=true" | head -n 20' || true
sleep 2
kubectl -n keda logs "$KEDA_JSZ_POD" --tail=-1 2>/dev/null || true
kubectl -n keda delete pod "$KEDA_JSZ_POD" --ignore-not-found >/dev/null 2>&1 || true

# Poll KEDA external metric endpoint a few times
ENC_SO="scaledobject.keda.sh%2Fname%3D${SO_NAME}"
URL="/apis/external.metrics.k8s.io/v1beta1/namespaces/$NS/s0-nats-jetstream-swarm_jobs?labelSelector=$ENC_SO"

echo "Polling KEDA external metric endpoint"
for i in {1..10}; do
  echo "Attempt $i"
  RAW="$(kubectl get --raw "$URL" 2>/dev/null || echo '{}')"
  echo "$RAW" | jq -r 'try .items[0].value catch "no-data"' || true
  kubectl -n "$NS" get hpa -o wide || true
  sleep 5
done

# New: verify external metric reaches > 0 while paused (backlog sustained)
echo "Verifying external metric > 0 while paused"
for i in {1..12}; do
  RAW="$(kubectl get --raw "$URL" 2>/dev/null || echo '{}')"
  VAL="$(echo "$RAW" | jq -r 'try .items[0].value catch "no-data"' || echo "no-data")"
  echo "metric=$VAL (paused check $i/12)"
  if [[ "$VAL" != "no-data" && "$VAL" != "0" ]]; then break; fi
  sleep 5
done

# New: Phase 2 only if FINAL_PASS=1
if [[ "$FINAL_PASS" == "1" ]]; then
  echo "Unpausing KEDA scaling by removing pause annotations"
  kubectl -n "$NS" annotate scaledobject ${SO_NAME} autoscaling.keda.sh/paused-replicas- || true
  kubectl -n "$NS" annotate scaledobject ${SO_NAME} keda.sh/paused- || true
  # Give the operator a moment to reconcile
  sleep 5 || true

  echo "Waiting for HPA to scale nats-worker above 0"
  for i in {1..24}; do
    DESIRED=$(kubectl -n "$NS" get hpa keda-hpa-${SO_NAME} -o jsonpath='{.status.desiredReplicas}' 2>/dev/null || echo "0")
    READY=$(kubectl -n "$NS" get deploy nats-worker -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    echo "HPA desired=${DESIRED:-0} ready=${READY:-0} (check $i/24)"
    if [[ "${READY:-0}" != "0" ]]; then break; fi
    sleep 5
  done
  kubectl -n "$NS" rollout status deploy/nats-worker --timeout=180s || true

  echo "Waiting for consumer backlog to drain to 0"
  SVC="nats.$NS.svc.cluster.local:4222"
  for i in {1..60}; do
    POD_CI="nats-ci-$RANDOM"
    kubectl -n "$NS" run "$POD_CI" --restart=Never --image=natsio/nats-box:latest --env SVC="$SVC" -- \
      sh -lc 'nats --server "$SVC" consumer info swarm_jobs worker-staging -j 2>/dev/null || true'
    sleep 2
    RAW=$(kubectl -n "$NS" logs "$POD_CI" 2>/dev/null || echo "")
    kubectl -n "$NS" delete pod "$POD_CI" --ignore-not-found >/dev/null 2>&1 || true
    PENDING=$(echo "$RAW" | sed -n 's/.*"num_pending"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | tail -n1)
    echo "pending: ${PENDING:-unknown} (drain attempt $i/60)"
    if [[ -n "${PENDING:-}" && "$PENDING" == "0" ]]; then break; fi
    sleep 3
  done

  echo "Verifying external metric drops to 0 and HPA scales down to 0"
  for i in {1..24}; do
    VAL_RAW="$(kubectl get --raw "$URL" 2>/dev/null || echo '{}')"
    VAL="$(echo "$VAL_RAW" | jq -r 'try .items[0].value catch "no-data"' || echo "no-data")"
    REPLICAS="$(kubectl -n "$NS" get deploy nats-worker -o jsonpath='{.status.replicas}' 2>/dev/null || echo "?")"
    echo "metric=$VAL replicas=${REPLICAS} (check $i/24)"
    if [[ "$VAL" == "0" && "${REPLICAS:-1}" == "0" ]]; then
      echo "OK: scale-to-zero validated"
      break
    fi
    sleep 10
  done
fi

echo "End-to-end script completed"