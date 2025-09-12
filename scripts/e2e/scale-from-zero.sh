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