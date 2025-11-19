#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------
# Config (override via env or CLI flags)
# ---------------------------------------------
CLUSTER="${CLUSTER:-agentforge-demo}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Features (1=enable, 0=disable)
ENABLE_CLOUDWATCH_ADDON="${ENABLE_CLOUDWATCH_ADDON:-1}"   # Container Insights addon
ENABLE_FLUENT_BIT="${ENABLE_FLUENT_BIT:-1}"               # Fallback logs DaemonSet via Helm
ENABLE_KUBE_PROM_STACK="${ENABLE_KUBE_PROM_STACK:-1}"     # Prometheus + Grafana
ENABLE_GPU_OPERATOR="${ENABLE_GPU_OPERATOR:-1}"           # NVIDIA operator (safe now; activates when GPUs arrive)
ENABLE_DCGM_SCRAPE="${ENABLE_DCGM_SCRAPE:-1}"             # Prom scrape for DCGM exporter
ENABLE_ALB_CONTROLLER="${ENABLE_ALB_CONTROLLER:-0}"       # AWS Load Balancer Controller
EXPOSE_GRAFANA_INGRESS="${EXPOSE_GRAFANA_INGRESS:-0}"     # Internet-facing ALB for Grafana
ENABLE_DEMO_APP="${ENABLE_DEMO_APP:-1}"                   # Small demo app emitting logs + /metrics

# Grafana admin password (for kube-prometheus-stack)
GRAFANA_ADMIN_PASS="${GRAFANA_ADMIN_PASS:-changeme123}"

# ---------------------------------------------
# Helpers
# ---------------------------------------------
say() { echo -e "\033[1;36m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERR ]\033[0m $*"; }
need() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing dependency: $1"; exit 1; }
}

usage() {
  cat <<USAGE
Usage:
  CLUSTER=agentforge-demo AWS_REGION=us-east-1 \\
  ENABLE_ALB_CONTROLLER=1 EXPOSE_GRAFANA_INGRESS=1 \\
  ./observability.sh [install|verify|port-forward|destroy]

Defaults:
  ENABLE_CLOUDWATCH_ADDON=$ENABLE_CLOUDWATCH_ADDON
  ENABLE_FLUENT_BIT=$ENABLE_FLUENT_BIT
  ENABLE_KUBE_PROM_STACK=$ENABLE_KUBE_PROM_STACK
  ENABLE_GPU_OPERATOR=$ENABLE_GPU_OPERATOR
  ENABLE_DCGM_SCRAPE=$ENABLE_DCGM_SCRAPE
  ENABLE_ALB_CONTROLLER=$ENABLE_ALB_CONTROLLER
  EXPOSE_GRAFANA_INGRESS=$EXPOSE_GRAFANA_INGRESS
  ENABLE_DEMO_APP=$ENABLE_DEMO_APP

Targets:
  install       Install/upgrade everything enabled by flags
  verify        Print status & helpful next steps
  port-forward  Local Grafana (3000) and demo app (8080)
  destroy       Remove what this script installed
USAGE
}

# ---------------------------------------------
# Sanity checks
# ---------------------------------------------
prereqs() {
  need aws; need kubectl; need helm; need jq; need eksctl
  say "Setting kubeconfig for cluster=$CLUSTER region=$AWS_REGION"
  aws eks update-kubeconfig --name "$CLUSTER" --region "$AWS_REGION" >/dev/null
}

# ---------------------------------------------
# A) CloudWatch Container Insights + Fluent Bit
# ---------------------------------------------
install_cloudwatch_addon() {
  if [[ "$ENABLE_CLOUDWATCH_ADDON" != "1" ]]; then return 0; fi
  say "Installing/Updating CloudWatch Observability addon"
  if ! aws eks create-addon --cluster-name "$CLUSTER" --region "$AWS_REGION" \
      --addon-name amazon-cloudwatch-observability \
      --resolve-conflicts OVERWRITE >/dev/null 2>&1; then
    aws eks update-addon --cluster-name "$CLUSTER" --region "$AWS_REGION" \
      --addon-name amazon-cloudwatch-observability \
      --resolve-conflicts OVERWRITE >/dev/null || true
  fi
}

install_fluent_bit() {
  if [[ "$ENABLE_FLUENT_BIT" != "1" ]]; then return 0; fi
  say "Installing Fluent Bit (logs → CloudWatch)"
  kubectl create namespace amazon-cloudwatch 2>/dev/null || true

  # Create log group
  aws logs create-log-group --log-group-name "/eks/${CLUSTER}/workload" \
    --region "$AWS_REGION" >/dev/null 2>&1 || true

  # IAM policy
  cat >/tmp/fluentbit-policy.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect":"Allow","Action":["logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams","logs:CreateLogGroup"],"Resource":"*"}
  ]
}
JSON
  FB_POLICY_ARN=$(aws iam list-policies --query "Policies[?PolicyName=='EKSFluentBitLogsPolicy'].Arn" --output text)
  if [[ -z "$FB_POLICY_ARN" || "$FB_POLICY_ARN" == "None" ]]; then
    FB_POLICY_ARN=$(aws iam create-policy \
      --policy-name EKSFluentBitLogsPolicy \
      --policy-document file:///tmp/fluentbit-policy.json \
      --query 'Policy.Arn' --output text)
  fi

  eksctl utils associate-iam-oidc-provider --cluster "$CLUSTER" --region "$AWS_REGION" --approve >/dev/null

  eksctl create iamserviceaccount \
    --cluster "$CLUSTER" \
    --namespace amazon-cloudwatch \
    --name aws-for-fluent-bit \
    --attach-policy-arn "$FB_POLICY_ARN" \
    --approve \
    --region "$AWS_REGION" >/dev/null 2>&1 || true

  helm repo add eks https://aws.github.io/eks-charts >/dev/null
  helm repo update >/dev/null
  helm upgrade --install aws-for-fluent-bit eks/aws-for-fluent-bit \
    --namespace amazon-cloudwatch \
    --set cloudWatch.logGroupName="/eks/${CLUSTER}/workload" \
    --set cloudWatch.logStreamPrefix="app" \
    --set serviceAccount.create=false \
    --set serviceAccount.name=aws-for-fluent-bit >/dev/null
}

# ---------------------------------------------
# B) Prometheus + Grafana
# ---------------------------------------------
install_kube_prom_stack() {
  if [[ "$ENABLE_KUBE_PROM_STACK" != "1" ]]; then return 0; fi
  say "Installing kube-prometheus-stack (Prometheus + Grafana)"
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
  helm repo add grafana https://grafana.github.io/helm-charts >/dev/null
  helm repo update >/dev/null
  kubectl create namespace monitoring 2>/dev/null || true

  cat > /tmp/kps-values.yaml <<YAML
grafana:
  adminPassword: "${GRAFANA_ADMIN_PASS}"
  service:
    type: ClusterIP
  defaultDashboardsEnabled: true

prometheus:
  service:
    type: ClusterIP

nodeExporter:
  enabled: true
kubeStateMetrics:
  enabled: true
YAML

  helm upgrade --install kps prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    -f /tmp/kps-values.yaml >/dev/null
}

# ---------------------------------------------
# C) NVIDIA GPU Operator + DCGM scrape
# ---------------------------------------------
install_gpu_operator() {
  if [[ "$ENABLE_GPU_OPERATOR" != "1" ]]; then return 0; fi
  say "Installing NVIDIA GPU Operator (safe to install now; activates when GPUs appear)"
  helm repo add nvidia https://nvidia.github.io/gpu-operator >/dev/null
  helm repo update >/dev/null
  helm upgrade --install gpu-operator nvidia/gpu-operator \
    --namespace gpu-operator --create-namespace \
    --set driver.enabled=true \
    --set toolkit.enabled=true \
    --set devicePlugin.enabled=true >/dev/null
}

install_dcgm_servicemonitor() {
  if [[ "$ENABLE_DCGM_SCRAPE" != "1" ]]; then return 0; fi
  say "Creating Service & ServiceMonitor for DCGM exporter (GPU metrics)"
  cat > /tmp/dcgm-servicemonitor.yaml <<'YAML'
apiVersion: v1
kind: Service
metadata:
  name: dcgm-exporter
  namespace: gpu-operator
  labels:
    app.kubernetes.io/name: dcgm-exporter
spec:
  selector:
    app.kubernetes.io/name: dcgm-exporter
  ports:
    - name: metrics
      port: 9400
      targetPort: 9400
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
  namespace: monitoring
  labels:
    release: kps
spec:
  namespaceSelector:
    matchNames:
      - gpu-operator
  selector:
    matchLabels:
      app.kubernetes.io/name: dcgm-exporter
  endpoints:
    - port: metrics
      interval: 15s
YAML
  kubectl apply -f /tmp/dcgm-servicemonitor.yaml >/dev/null 2>&1 || true
}

# ---------------------------------------------
# D) Demo app
# ---------------------------------------------
install_demo_app() {
  if [[ "$ENABLE_DEMO_APP" != "1" ]]; then return 0; fi
  say "Deploying tiny demo app (FastAPI + /metrics)"
  cat > /tmp/demo-app.yaml <<'YAML'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: demo-hello
  labels: {app: demo-hello}
spec:
  replicas: 2
  selector: {matchLabels: {app: demo-hello}}
  template:
    metadata:
      labels: {app: demo-hello}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: web
        image: ghcr.io/tiangolo/uvicorn-gunicorn-fastapi:python3.11
        env:
        - {name: MODULE_NAME, value: "main"}
        - {name: VARIABLE_NAME, value: "app"}
        ports: [{containerPort: 8080}]
        command: ["bash","-lc"]
        args:
        - |
          cat > /app/main.py <<'PY'
          from fastapi import FastAPI
          from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
          app = FastAPI()
          c = Counter("hello_requests_total","Total hello requests")
          @app.get("/hello")
          def hello(): c.inc(); return {"ok": True}
          @app.get("/metrics")
          def metrics():
              return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}
          PY
          pip install -q prometheus_client fastapi uvicorn >/dev/null
          exec /start-reload.sh
---
apiVersion: v1
kind: Service
metadata:
  name: demo-hello
spec:
  selector: {app: demo-hello}
  ports: [{port: 80, targetPort: 8080}]
YAML
  kubectl apply -f /tmp/demo-app.yaml >/dev/null
  kubectl rollout status deploy/demo-hello >/dev/null
}

# ---------------------------------------------
# E) AWS Load Balancer Controller + Grafana Ingress
# ---------------------------------------------
install_alb_controller() {
  if [[ "$ENABLE_ALB_CONTROLLER" != "1" ]]; then return 0; fi
  say "Installing AWS Load Balancer Controller"
  eksctl utils associate-iam-oidc-provider --cluster "$CLUSTER" --region "$AWS_REGION" --approve >/dev/null

  curl -s -o /tmp/lbc-iam-policy.json \
    https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/main/docs/install/iam_policy.json

  LBC_POLICY_ARN=$(aws iam list-policies --query "Policies[?PolicyName=='AWSLoadBalancerControllerIAMPolicy'].Arn" --output text)
  if [[ -z "$LBC_POLICY_ARN" || "$LBC_POLICY_ARN" == "None" ]]; then
    LBC_POLICY_ARN=$(aws iam create-policy \
      --policy-name AWSLoadBalancerControllerIAMPolicy \
      --policy-document file:///tmp/lbc-iam-policy.json \
      --query 'Policy.Arn' --output text)
  fi

  eksctl create iamserviceaccount \
    --cluster "$CLUSTER" \
    --namespace kube-system \
    --name aws-load-balancer-controller \
    --attach-policy-arn "$LBC_POLICY_ARN" \
    --override-existing-serviceaccounts \
    --approve \
    --region "$AWS_REGION" >/dev/null 2>&1 || true

  helm repo add eks https://aws.github.io/eks-charts >/dev/null
  helm repo update >/dev/null
  helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
    -n kube-system \
    --set clusterName="$CLUSTER" \
    --set region="$AWS_REGION" \
    --set serviceAccount.create=false \
    --set serviceAccount.name=aws-load-balancer-controller >/dev/null
}

expose_grafana_ingress() {
  if [[ "$EXPOSE_GRAFANA_INGRESS" != "1" ]]; then return 0; fi
  say "Creating internet-facing ALB Ingress for Grafana"
  cat > /tmp/grafana-ing.yaml <<'YAML'
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: grafana
  namespace: monitoring
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
spec:
  rules:
    - http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: kps-grafana
                port:
                  number: 80
YAML
  kubectl apply -f /tmp/grafana-ing.yaml >/dev/null
  say "Waiting for ALB address..."
  for i in {1..30}; do
    ADDR=$(kubectl -n monitoring get ingress grafana -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)
    [[ -n "${ADDR:-}" ]] && break || sleep 10
  done
  [[ -n "${ADDR:-}" ]] && say "Grafana URL: http://${ADDR}" || warn "ALB not ready yet; check later: kubectl -n monitoring get ingress grafana"
}

# ---------------------------------------------
# VERIFY / PORT-FORWARD / DESTROY
# ---------------------------------------------
verify() {
  say "=== Core ==="
  kubectl get nodes -o wide || true
  say "=== Monitoring pods ==="
  kubectl -n monitoring get pods -o wide || true
  say "=== CloudWatch logs group ==="
  aws logs describe-log-groups --log-group-name-prefix "/eks/${CLUSTER}" --region "$AWS_REGION" || true
  say "=== GPU Operator pods ==="
  kubectl -n gpu-operator get pods -o wide || true
  say "=== GPU capacity (shows once nodes are GPU) ==="
  kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity."nvidia\.com/gpu" || true
  say "=== Grafana admin ==="
  echo "User: admin   Pass: ${GRAFANA_ADMIN_PASS}"
  say "Port-forward Grafana:   kubectl -n monitoring port-forward svc/kps-grafana 3000:80"
  say "Port-forward demo app:  kubectl port-forward svc/demo-hello 8080:80"
}

pf() {
  say "Port-forwarding Grafana → http://localhost:3000 (Ctrl+C to stop)"
  kubectl -n monitoring port-forward svc/kps-grafana 3000:80
}

destroy() {
  warn "Removing resources installed by this script"
  kubectl delete -f /tmp/demo-app.yaml 2>/dev/null || true
  kubectl -n monitoring delete ingress grafana 2>/dev/null || true
  helm -n kube-system uninstall aws-load-balancer-controller 2>/dev/null || true
  eksctl delete iamserviceaccount \
    --cluster "$CLUSTER" --region "$AWS_REGION" \
    --namespace kube-system --name aws-load-balancer-controller \
    --approve >/dev/null 2>&1 || true

  helm -n monitoring uninstall kps 2>/dev/null || true
  kubectl delete namespace monitoring 2>/dev/null || true

  helm -n amazon-cloudwatch uninstall aws-for-fluent-bit 2>/dev/null || true
  eksctl delete iamserviceaccount \
    --cluster "$CLUSTER" --region "$AWS_REGION" \
    --namespace amazon-cloudwatch --name aws-for-fluent-bit \
    --approve >/dev/null 2>&1 || true
  kubectl delete namespace amazon-cloudwatch 2>/dev/null || true

  helm -n gpu-operator uninstall gpu-operator 2>/dev/null || true
  kubectl delete namespace gpu-operator 2>/dev/null || true

  aws logs delete-log-group --log-group-name "/eks/${CLUSTER}/workload" --region "$AWS_REGION" >/dev/null 2>&1 || true

  warn "If you enabled the CloudWatch addon, you can remove it with:"
  echo "aws eks delete-addon --cluster-name \"$CLUSTER\" --region \"$AWS_REGION\" --addon-name amazon-cloudwatch-observability"
}

install_all() {
  prereqs
  install_cloudwatch_addon
  install_fluent_bit
  install_kube_prom_stack
  install_gpu_operator
  install_dcgm_servicemonitor
  install_alb_controller
  expose_grafana_ingress
  install_demo_app
  verify
}

# ---------------------------------------------
# Entry
# ---------------------------------------------
ACTION="${1:-install}"
case "$ACTION" in
  install)       install_all ;;
  verify)        prereqs; verify ;;
  port-forward)  prereqs; pf ;;
  destroy)       prereqs; destroy ;;
  *)             usage; exit 1 ;;
esac
