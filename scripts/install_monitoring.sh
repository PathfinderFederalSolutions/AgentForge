#!/usr/bin/env bash
set -euo pipefail

NS_MON=${NS_MON:-monitoring}
RELEASE=${RELEASE:-kube-prom}
VALUES=${VALUES:-k8s/staging/monitoring-values.yaml}

# Create namespace if missing
kubectl get ns "$NS_MON" >/dev/null 2>&1 || kubectl create ns "$NS_MON"

# Add/Update repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo update >/dev/null

# Install or upgrade kube-prometheus-stack
helm upgrade --install "$RELEASE" prometheus-community/kube-prometheus-stack \
  --namespace "$NS_MON" -f "$VALUES" --wait

# After CRDs exist, apply ServiceMonitors in staging namespace
kubectl apply -f k8s/staging/servicemonitors.yaml

echo "Monitoring installed. Grafana and Prometheus are in namespace: $NS_MON"
