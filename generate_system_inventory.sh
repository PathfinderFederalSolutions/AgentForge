# filepath: ./generate_system_inventory.sh
#!/usr/bin/env bash
set -euo pipefail

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="inventory_${TS}"
ART_DIR="${OUT_DIR}/artifacts"
MD_OUT="system_inventory_${TS}.md"
JSON_OUT="system_inventory_${TS}.json"
BUNDLE_TGZ="system_inventory_${TS}.tar.gz"
HANDOFF_OUT="handoff_compiled_${TS}.md"

mkdir -p "${ART_DIR}"

echo "[*] Starting system inventory at ${TS}"

md() { printf "%s\n" "$*" >> "${MD_OUT}"; }

if command -v jq >/dev/null 2>&1; then
  JSON_TMP="$(mktemp)"; echo '{}' > "${JSON_TMP}"
  jq_add () {
    local path="$1" val="$2"
    if [[ -f "${val}" ]]; then
      if jq "${path} = (try (input | fromjson) catch (input))" "${JSON_TMP}" "${val}" > "${JSON_TMP}.new" 2>/dev/null; then
        :
      else
        jq "${path} = (input | @text)" "${JSON_TMP}" "${val}" > "${JSON_TMP}.new"
      fi
    else
      jq "${path} = \$v" --arg v "${val}" "${JSON_TMP}" > "${JSON_TMP}.new"
    fi
    mv "${JSON_TMP}.new" "${JSON_TMP}"
  }
else
  JSON_TMP=""
  jq_add () { :; }
  echo "[!] jq not found – JSON aggregation limited."
fi

md "# System Inventory Snapshot"
md ""
md "- Timestamp (UTC): ${TS}"
md "- Hostname: $(hostname || true)"
md "- User: $(whoami || true)"
md ""
md "## Summary for GPT-5"
md "This bundle describes the AgentForge system: code, services, messaging, observability, deployment profiles, and artifacts."
md ""
md "- Workloads: FastAPI services (API, Orchestrator), NATS JetStream workers (tool executor, results sink), HITL."
md "- Messaging: NATS JetStream subjects swarm.jobs.<env>, swarm.results.<env>, hitl, DLQs; durable consumers."
md "- Observability: Prometheus /metrics, ServiceMonitors, OTEL traces, canary SLOs, drift monitors (PSI/KL), ROC/EER, executor/worker metrics."
md "- Storage: JSONL fallbacks, lineage DAG artifacts under var/artifacts, optional DB/pgvector."
md "- K8s: Profiles for SCIF/GovCloud/SaaS via Kustomize; KEDA scaling, ServiceMonitors, PrometheusRules, OTEL Collector."
md "- Supply chain: Syft SPDX SBOMs (source/image), Cosign keyless attest workflow in GitHub Actions."
md "- Evidence: /v1/evidence/{job_id} returns bundle; lineage DAG persisted with dag_hash."
md "- Drift: PSI, KL metrics (DRIFT_PSI, DRIFT_KL) with DRIFT_ALERTS; ROC EER metric (fusion_roc_eer)."
md "- Throughput: GPU-aware AdaptiveBatchController for NATS workers; metrics worker_adaptive_batch_size, worker_gpu_avg_mem_mb, worker_queue_depth, worker_batch_latency_seconds, worker_jobs_processed_total."
md "- CI/CD: .github/workflows/ci.yml, supply-chain.yml, SBOMs (Syft), Trivy image scan, Cosign sign for syncdaemon, comms-gateway, ar-context, route-engine, engagement, cds-bridge."
md "- Orchestration: PhaseRunner in orchestrator/app/main.py, plans/master_orchestration.yaml, worker_protocol.py, results_sink.py."
md "- Engagement: services/engagement/app/main.py, tests/test_engagement_dual_control.py, ui/tactical-dashboard/src/views/Engagement.tsx."
md "- HITL: services/hitl/app.py, dual approval endpoints, evidence DAG logging."
md "- CDS Bridge: services/cds-bridge/app/main.py, k8s/profiles/govcloud/cds-bridge.yaml, k8s/profiles/govcloud/network-policies.yaml, tests/test_cds_hash_verification.py."
md "- Evidence: var/artifacts/engagement, var/artifacts/phase_runs, services/cds-bridge/logs/transfers."
md "- Metrics: engagement_time_to_decision_seconds, cds_transfer_success_total, worker_adaptive_batch_size, worker_gpu_avg_mem_mb, worker_queue_depth, worker_batch_latency_seconds, worker_jobs_processed_total, fusion_roc_eer, DRIFT_PSI, DRIFT_KL, DRIFT_ALERTS."
md "- Security: Cosign signing, SBOMs, Trivy scan, supply chain attestation."
md "- UI: Tactical dashboard, engagement queue, evidence preview, ROE snapshot, WebAuthn/YubiKey approval."
md "- K8s: Kustomize profiles, KEDA scaling, ServiceMonitors, PrometheusRules, OTEL Collector, network policies."
md "- Storage: JSONL, lineage DAG, pgvector, /v1/evidence/{job_id}."
md "- Messaging: NATS JetStream, subjects, durable consumers, DLQs."
md "- Observability: Prometheus, OTEL, SLOs, drift monitors, ROC/EER, executor/worker metrics."
md "- Throughput: AdaptiveBatchController, metrics, GPU-aware scaling."
md ""

#################### Git ####################
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  GIT_TAGS="$(git tag --points-at HEAD 2>/dev/null | paste -sd , - || true)"
  if git diff --quiet >/dev/null 2>&1; then GIT_DIRTY="clean"; else GIT_DIRTY="dirty"; fi

  # Safe remote extraction (name + URL unique)
  REMOTES_RAW="$(
    git remote -v 2>/dev/null | \
    while read -r name url rest; do
      [[ -n "${name:-}" && -n "${url:-}" ]] && printf '%s %s\n' "$name" "$url"
    done | sort -u
  )"
  REMOTES_JSON="["
  first=1
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    esc=$(printf '%s' "$line" | sed 's/"/\\"/g')
    if [[ $first -eq 1 ]]; then
      REMOTES_JSON="${REMOTES_JSON}\"${esc}\""
      first=0
    else
      REMOTES_JSON="${REMOTES_JSON},\"${esc}\""
    fi
  done <<< "${REMOTES_RAW}"
  REMOTES_JSON="${REMOTES_JSON}]"

  cat > "${ART_DIR}/git.json" <<EOF
{
  "commit": "${GIT_COMMIT}",
  "branch": "${GIT_BRANCH}",
  "tags": "${GIT_TAGS}",
  "dirty": "${GIT_DIRTY}",
  "remotes": ${REMOTES_JSON}
}
EOF
  jq_add '.git' "${ART_DIR}/git.json"
  md "## Git"
  md "Commit: ${GIT_COMMIT}"
  md "Branch: ${GIT_BRANCH}"
  md "Dirty: ${GIT_DIRTY}"
  md ""
fi

#################### Code & Dependencies ####################
md "## Code & Dependencies"
if command -v cloc >/dev/null 2>&1; then
  cloc --json . > "${ART_DIR}/cloc.json" 2>/dev/null || true
  jq_add '.codeStats' "${ART_DIR}/cloc.json"
  md "- LOC stats captured"
fi
# Source SBOM (SPDX) for entire repo (best for GPT-5 understanding of code surface)
if command -v syft >/dev/null 2>&1; then
  syft dir:. -o spdx-json > "${ART_DIR}/source_sbom.spdx.json" 2>/dev/null || true
  jq_add '.supplyChain.sourceSbom' "${ART_DIR}/source_sbom.spdx.json"
  md "- Source SBOM (SPDX JSON) generated"
fi
if [[ -f requirements.txt ]]; then
  (pip freeze || true) > "${ART_DIR}/python_freeze.txt" 2>/dev/null || true
  if command -v jq >/dev/null 2>&1; then
    jq -n --rawfile f "${ART_DIR}/python_freeze.txt" '{requirements: ($f|split("\n")|map(select(length>0)))}' > "${ART_DIR}/python_deps.json" || true
    jq_add '.dependencies.python' "${ART_DIR}/python_deps.json"
  fi
  md "- Python dependencies captured"
fi
if [[ -d ".github/workflows" ]]; then
  mkdir -p "${ART_DIR}/workflows"
  cp -R .github/workflows "${ART_DIR}/workflows" 2>/dev/null || true
  md "- GitHub Actions workflows copied"
fi
if [[ -f package.json ]]; then
  cp package.json "${ART_DIR}/package.json"
  jq_add '.dependencies.node.packageJson' "${ART_DIR}/package.json"
  md "- Node package.json captured"
fi
if [[ -f go.mod ]]; then
  (go list -m -json all || true) > "${ART_DIR}/gomod_modules.json" 2>/dev/null || true
  jq_add '.dependencies.go.modulesRaw' "${ART_DIR}/gomod_modules.json"
  md "- Go modules captured"
fi
md ""

#################### Docker ####################
md "## Docker"
if command -v docker >/dev/null 2>&1; then
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' > "${ART_DIR}/docker_ps.txt" || true
  docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}' > "${ART_DIR}/docker_images.txt" || true
  if command -v jq >/dev/null 2>&1; then
    jq -n --rawfile ps "${ART_DIR}/docker_ps.txt" --rawfile imgs "${ART_DIR}/docker_images.txt" \
      '{containers: ($ps|split("\n")), images: ($imgs|split("\n"))}' > "${ART_DIR}/docker.json" || true
    jq_add '.docker' "${ART_DIR}/docker.json"
  fi
  md "- Docker runtime captured"
fi
md ""

#################### Kubernetes ####################
NS="agentforge-staging"
md "## Kubernetes (${NS})"
if command -v kubectl >/dev/null 2>&1; then
  RES_LIST=(deployments statefulsets daemonsets jobs cronjobs pods services ingresses configmaps secrets persistentvolumeclaims horizontalpodautoscalers scaledobjects prometheusrules)
  for r in "${RES_LIST[@]}"; do
    kubectl -n "${NS}" get "${r}" -o wide > "${ART_DIR}/k8s_${r}.txt" 2>/dev/null || true
  done
  # KEDA resources (ScaledObjects) may be in CRD group keda.sh
  kubectl -n "${NS}" get scaledobjects.keda.sh -o wide > "${ART_DIR}/k8s_scaledobjects.txt" 2>/dev/null || true
  kubectl api-resources --verbs=list -o name > "${ART_DIR}/k8s_api_resources.txt" 2>/dev/null || true
  md "- Namespace resources enumerated"
fi
md ""

#################### Kustomize Profiles ####################
md "## Kustomize Profiles"
kustom_build () {
  local path="$1" out="$2"
  if command -v kustomize >/dev/null 2>&1; then
    kustomize build "$path" > "$out" 2>/dev/null || true
  elif command -v kubectl >/dev/null 2>&1; then
    kubectl kustomize "$path" > "$out" 2>/dev/null || true
  fi
}
for prof in scif govcloud saas; do
  if [[ -d "k8s/profiles/${prof}" ]]; then
    kustom_build "k8s/profiles/${prof}" "${ART_DIR}/kustomize_${prof}.yaml"
    if [[ -s "${ART_DIR}/kustomize_${prof}.yaml" ]]; then
      # Summarize kinds
      awk '/^kind:/{print $2}' "${ART_DIR}/kustomize_${prof}.yaml" | sort | uniq -c > "${ART_DIR}/kustomize_${prof}_kinds.txt" || true
    fi
  fi
done
# Copy profile README for differences reference
if [[ -f "k8s/profiles/README.md" ]]; then
  cp "k8s/profiles/README.md" "${ART_DIR}/k8s_profiles_README.md" || true
  md "- Kustomize builds captured for profiles and README copied"
else
  md "- Kustomize builds captured for available profiles (scif/govcloud/saas)"
fi
# If helper script exists, run it and capture output
if [[ -x "scripts/validate_kustomize.sh" ]]; then
  bash scripts/validate_kustomize.sh > "${ART_DIR}/validate_kustomize.log" 2>&1 || true
  md "- Kustomize validation log saved (scripts/validate_kustomize.sh)"
fi
md ""

#################### NATS / JetStream ####################
md "## NATS / JetStream"
NATS_MONITOR="${NATS_MONITOR:-http://localhost:8222}"
for ep in varz connz routez; do
  curl -fsSL "${NATS_MONITOR}/${ep}" -o "${ART_DIR}/nats_${ep}.json" 2>/dev/null || true
done
curl -fsSL "${NATS_MONITOR}/jsz?accounts=true&streams=true&consumers=true&config=true" -o "${ART_DIR}/nats_jsz.json" 2>/dev/null || true
[[ -s "${ART_DIR}/nats_jsz.json" ]] && jq_add '.nats.jsz' "${ART_DIR}/nats_jsz.json" && md "- JetStream status collected"
if command -v nats >/dev/null 2>&1; then
  nats stream ls -j > "${ART_DIR}/nats_streams.json" 2>/dev/null || true
  nats consumer ls "" -j > "${ART_DIR}/nats_consumers.json" 2>/dev/null || true
  md "- Stream & consumer listings captured"
fi
md ""

#################### Prometheus ####################
md "## Prometheus"
PROM="${PROM_URL:-http://localhost:9090}"
for p in rules targets alerts; do
  curl -fsSL "${PROM}/api/v1/${p}" -o "${ART_DIR}/prom_${p}.json" 2>/dev/null || true
  jq_add ".prometheus.${p}" "${ART_DIR}/prom_${p}.json"
done
# Include repo Prometheus rule files (alerts)
find k8s -maxdepth 3 -type f \( -name "*rules*.yml" -o -name "*rules*.yaml" -o -name "nats_rules.yml" \) -print -exec cp "{}" "${ART_DIR}/" \; 2>/dev/null || true
md "- Prometheus rule files copied from repo"
QUERIES=(
  "jetstream_backlog|sum(nats_jetstream_consumer_num_pending)"
  "slo_backlog_drain_p95|histogram_quantile(0.95, rate(backlog_drain_seconds_bucket[5m]))"
  "connections|sum(nats_varz_connections)"
  # New: Tool executor metrics
  "tool_exec_attempts|sum by (tool) (tool_executor_attempts_total)"
  "tool_exec_failures|sum by (tool,reason) (tool_executor_failures_total)"
  "tool_exec_retries|sum by (tool,reason) (executor_retry_total)"
  "tool_exec_replays|sum by (tool) (tool_executor_idempotent_replays_total)"
  "tool_exec_backlog_gauge|sum by (stream,consumer) (tool_queue_backlog_gauge)"
  # New: Worker adaptive batching
  "worker_batch_size|avg by (mission) (worker_adaptive_batch_size)"
  "worker_gpu_avg_mem_mb|avg by (mission) (worker_gpu_avg_mem_mb)"
  "worker_queue_depth|avg by (mission) (worker_queue_depth)"
  "worker_jobs_processed|sum by (mission) (worker_jobs_processed_total)"
  "worker_batch_p95|histogram_quantile(0.95, sum(rate(worker_batch_latency_seconds_bucket[5m])) by (le, mission))"
  # New: Drift and ROC/EER
  "drift_psi_avg|avg(drift_psi)"
  "drift_kl_avg|avg(drift_kl_divergence)"
  "drift_alerts|sum by (feature,metric) (drift_alerts_total)"
  "fusion_roc_eer_p50|histogram_quantile(0.5, sum(rate(fusion_roc_eer_bucket[15m])) by (le))"
)
for entry in "${QUERIES[@]}"; do
  name="${entry%%|*}"; query="${entry#*|}"
  enc="$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "${query}" 2>/dev/null || echo "")"
  [[ -n "${enc}" ]] && curl -fsSL "${PROM}/api/v1/query?query=${enc}" -o "${ART_DIR}/prom_q_${name}.json" 2>/dev/null || true
done
md "- Prometheus metadata & SLO queries captured"
md ""

#################### Exporter Image Security ####################
md "## Exporter Image Security"
EXPORTER_IMAGE="${EXPORTER_IMAGE:-ghcr.io/nats-io/prometheus-nats-exporter:latest}"
if command -v syft >/dev/null 2>&1; then
  syft "${EXPORTER_IMAGE}" -o json > "${ART_DIR}/exporter_sbom.json" 2>/dev/null || true
  md "- SBOM generated"
fi
if command -v trivy >/dev/null 2>&1; then
  trivy image --severity CRITICAL,HIGH --format json "${EXPORTER_IMAGE}" > "${ART_DIR}/exporter_vuln.json" 2>/dev/null || true
  md "- Vulnerability scan complete"
fi
md ""

#################### Grafana ####################
md "## Grafana"
if [[ -n "${GRAFANA_URL:-}" && -n "${GRAFANA_API_KEY:-}" ]]; then
  HDR="Authorization: Bearer ${GRAFANA_API_KEY}"
  curl -fsSL -H "${HDR}" "${GRAFANA_URL}/api/search?query=" > "${ART_DIR}/grafana_search.json" 2>/dev/null || true
  if command -v jq >/dev/null 2>&1; then
    for uid in $(jq -r '.[].uid' "${ART_DIR}/grafana_search.json" 2>/dev/null | head -n 15); do
      curl -fsSL -H "${HDR}" "${GRAFANA_URL}/api/dashboards/uid/${uid}" > "${ART_DIR}/grafana_dashboard_${uid}.json" 2>/dev/null || true
    done
  fi
  md "- Dashboards exported"
else
  md "- Grafana skipped (vars not set)"
fi
md ""

#################### Host Resources ####################
md "## Host Resources"
{
  echo "CPU:"; (command -v sysctl >/dev/null && sysctl -n machdep.cpu.brand_string) || true
  echo "Load:"; uptime || true
  echo "Memory:"; (vm_stat || free -h 2>/dev/null) | head -n 15
  echo "Disk:"; df -h
  echo "Open Ports (top 50 LISTEN/UDP):"; (netstat -an 2>/dev/null | grep -E 'LISTEN|UDP' | head -n 50) || true
} > "${ART_DIR}/host_resources.txt"
md "- Host resources snapshot captured"
md ""

#################### Application Artifacts & Metrics ####################
md "## Application Artifacts & Metrics"
# Lineage DAG artifacts
if [[ -d "var/artifacts" ]]; then
  find var/artifacts -type f -maxdepth 1 \( -name "*.dag.json" -o -name "*.json" \) -print > "${ART_DIR}/lineage_artifacts_list.txt" 2>/dev/null || true
  md "- Lineage artifacts listed (var/artifacts)"
fi
# Results sink JSONL samples
ls -1 var/swarm_results_*.jsonl 2>/dev/null | head -n 5 | while read -r f; do
  tail -n 50 "$f" > "${ART_DIR}/$(basename "$f").tail.txt" 2>/dev/null || true
done
if ls var/swarm_results_*.jsonl >/dev/null 2>&1; then
  md "- Results JSONL tails captured"
fi
# Try scraping local /metrics if API running
curl -fsS http://localhost:8000/metrics -o "${ART_DIR}/api_metrics.txt" 2>/dev/null || true
curl -fsS http://localhost:8001/metrics -o "${ART_DIR}/orchestrator_metrics.txt" 2>/dev/null || true
# Tool executor metrics (separate Prom server)
curl -fsS http://localhost:${TOOL_EXEC_METRICS_PORT:-9000}/metrics -o "${ART_DIR}/tool_executor_metrics.txt" 2>/dev/null || true
if [[ -s "${ART_DIR}/api_metrics.txt" ]]; then md "- /metrics snapshot (API) captured"; fi
if [[ -s "${ART_DIR}/orchestrator_metrics.txt" ]]; then md "- /metrics snapshot (Orchestrator) captured"; fi
if [[ -s "${ART_DIR}/tool_executor_metrics.txt" ]]; then md "- /metrics snapshot (Tool Executor) captured"; fi

# Build a metrics catalogue from code (best-effort)
if command -v rg >/dev/null 2>&1; then GREP=rg; elif command -v ag >/dev/null 2>&1; then GREP=ag; else GREP=grep; fi
$GREP -R --line-number -E "(Counter\(|Gauge\(|Histogram\()" swarm 2>/dev/null | sed 's#:.*Counter(# Counter(#; s#:.*Gauge(# Gauge(#; s#:.*Histogram(# Histogram(#;' | awk '{print $1}' | sort | uniq > "${ART_DIR}/metrics_catalogue.txt" || true
md "- Metrics catalogue generated from code"
md ""

#################### OTEL / Collector Configs ####################
md "## OTEL / Collector Configs"
find k8s -maxdepth 4 -type f -name "otel-collector*.yaml" -print -exec cp "{}" "${ART_DIR}/" \; 2>/dev/null || true
md "- OTEL collector configs copied (if present)"
md ""

#################### GPU Inventory ####################
md "## GPU Inventory"
{ (command -v nvidia-smi >/dev/null && nvidia-smi -q) || (command -v rocminfo >/dev/null && rocminfo) || echo "No GPU inventory tool found"; } > "${ART_DIR}/gpu_inventory.txt" 2>/dev/null || true
md "- GPU inventory attempted (nvidia-smi/rocminfo)"
md ""

#################### Capabilities & Decorators ####################
md "## Capabilities & Decorators"
# Grep for capability decorator usage for quick surface overview
grep -R --line-number -E "capability\(" . 2>/dev/null | grep -v "\.venv" | head -n 200 > "${ART_DIR}/capability_decorators.txt" || true
if [[ -s "${ART_DIR}/capability_decorators.txt" ]]; then
  md "- capability(...) decorator occurrences captured"
else
  md "- No capability decorators found or not statically referenced"
fi
md ""

#################### Static & Security Analysis ####################
md "## Static & Security Analysis"
# Flake8
if command -v flake8 >/dev/null 2>&1; then
  flake8 . > "${ART_DIR}/flake8.txt" 2>&1 || true
  md "- flake8 report saved"
fi
# Bandit (security)
if command -v bandit >/dev/null 2>&1; then
  bandit -q -r swarm -f txt > "${ART_DIR}/bandit_swarm.txt" 2>&1 || true
  md "- bandit security scan for swarm/ saved"
fi
# Safety (dependency vulns)
if command -v safety >/dev/null 2>&1; then
  safety check -r requirements.txt -f text --full-report > "${ART_DIR}/safety.txt" 2>&1 || true
  md "- safety dependency audit saved"
fi
# Shellcheck for scripts
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck scripts/*.sh generate_system_inventory.sh > "${ART_DIR}/shellcheck.txt" 2>&1 || true
  md "- shellcheck report saved"
fi
md ""

#################### Finalize ####################
if [[ -n "${JSON_TMP}" ]]; then
  cp "${JSON_TMP}" "${JSON_OUT}"
else
  echo '{}' > "${JSON_OUT}"
fi

md "## Artifacts Directory"
md "Raw outputs: ${ART_DIR}/"
md ""
md "## Usage"
md "Feed this markdown OR the JSON (${JSON_OUT}) to GPT-5 for planning."
md ""
md "## Packaging"
md "A single archive (${BUNDLE_TGZ}) with the markdown, JSON, and artifacts has been created for easy upload to GPT-5."

# Compose a handoff markdown if a preamble exists
if [[ -f "gpt5_context_preamble.md" ]]; then
  {
    echo "# AgentForge Handoff"
    echo
    echo "## Context Preamble"
    cat gpt5_context_preamble.md
    echo
    echo "## System Inventory"
    cat "${MD_OUT}"
    if [[ -f "${ART_DIR}/k8s_profiles_README.md" ]]; then
      echo
      echo "## K8s Profiles Guide"
      cat "${ART_DIR}/k8s_profiles_README.md"
    fi
  } > "${HANDOFF_OUT}"
fi

echo "[*] Inventory complete:"
echo " - ${MD_OUT}"
echo " - ${JSON_OUT}"
echo " - ${ART_DIR}/"
if command -v tar >/dev/null 2>&1; then
  tar -czf "${BUNDLE_TGZ}" "${MD_OUT}" "${JSON_OUT}" "${ART_DIR}" ${HANDOFF_OUT:+"${HANDOFF_OUT}"} 2>/dev/null || true
  echo " - ${BUNDLE_TGZ}"
fi