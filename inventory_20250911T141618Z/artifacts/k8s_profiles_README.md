# Deployment Profiles

This directory contains deployment profiles for different operational environments using Kustomize overlays.

Profiles:
- `staging` (existing baseline in `../staging`)
- `saas` (Commercial multi-tenant SaaS)
- `govcloud` (US GovCloud / GCC High hardened)
- `scif` (Air‑gapped / Sensitive Compartmented Information Facility)

Each profile customizes:
- Namespace & labeling
- NetworkPolicies & egress restrictions
- Image registry sources
- Security / compliance annotations
- Feature toggles via ConfigMap
- Secrets mounting strategy
- Optional GPU enablement & batch controller tuning

See individual profile README sections below.

---
## Common Base
All profiles start conceptually from the `k8s/staging` manifests (serving as a de facto base). For clarity and to avoid unintended drift, each profile today vendors the needed resource manifests with environment‑specific adjustments rather than using a multi-level Kustomize base chain. (Future: refactor into explicit `base/` + overlays.)

Shared components:
- NATS messaging plane
- Orchestrator & Tool Executor
- NATS Worker (adaptive batch)
- Temporal (temporalite for non‑prod; external Temporal for gov/scif via config)
- pgvector persistence
- OpenTelemetry Collector
- Monitoring resources (ServiceMonitors, exporter)

---
## SaaS Profile (`saas`)
Focus: multi‑tenant isolation & PII safeguards.
Key adjustments:
- Namespace: `agentforge-saas`
- ConfigMap enables PII redaction (`PII_REDACT=1`), multi‑tenant mode (`MULTI_TENANT=1`)
- NetworkPolicies allow egress only to required in-cluster services plus sanctioned outbound (HTTPS) for model APIs; restrict inbound by app.
- Adds label `deployment-tier=saas` for selection & cost allocation.
- ImagePullPolicy `IfNotPresent` (assumes internal registry mirror), can override via Kustomize.

---
## GovCloud Profile (`govcloud`)
Focus: compliance (FedRAMP / GCC High) & restricted registries.
Key adjustments:
- Namespace: `agentforge-gov`
- Image references use gov‑approved registry placeholder `registry.govcloud.example/agentforge`.
- Disables direct external egress except DNS + approved update proxy (placeholder host).
- Enforces FIPS annotations and sets `SECURITY_HARDENED=1` env toggle.
- Temporal endpoint externalized (`TEMPORAL_ADDRESS` not in-cluster temporalite) expecting managed service.

---
## SCIF Air‑Gapped Profile (`scif`)
Focus: zero external connectivity & offline model execution.
Key adjustments:
- Namespace: `agentforge-scif`
- Removes any outbound egress except DNS (optional) and intra‑namespace service traffic.
- Uses preloaded internal registry `registry.local/agentforge` images with digest pins (placeholders to be updated during air‑gap import pipeline).
- Disables features requiring outbound access (`VECTOR_SYNC=0`, `EXTERNAL_TOOLS=0`).
- Mounts persistent volume for artifact + model cache.

---
## Updating & Building
Validate a profile builds:
```
kustomize build k8s/profiles/saas | kubectl apply -f - --dry-run=client
kustomize build k8s/profiles/govcloud | head -n 50
kustomize build k8s/profiles/scif >/dev/null
```

---
## Future Enhancements
- Introduce explicit `base/` with patches via overlays to reduce duplication.
- Integrate policy-as-code (Kyverno / OPA Gatekeeper) constraints per profile.
- Automated conftest / kube-score validation in CI.
- Profile selection matrix tests.
