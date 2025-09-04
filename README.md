# AgentForge MVP

Quickstart
- Use the bundled venv: source source/bin/activate
- Copy .env.example to .env and fill in API keys as needed. All *_API_KEY are optional; a MockLLM is used when absent.
- To avoid heavy embedding downloads in tests, default is a lightweight HashEmbedder. To use sentence-transformers, set EMBEDDINGS_BACKEND=st.
- To enable Prometheus metrics server during local runs, set PROMETHEUS_ENABLE=1 (default off). Configure port via PROMETHEUS_PORT.

Run
- python main.py

Key env vars
- AF_GOAL: default goal string when not passed on CLI
- AF_AGENTS: number of agents in the swarm (default 2)
- AF_FORCE_MOCK: when "1", force mock model usage for all calls (no network)
- AF_SKIP_HEALTHCHECK: when "1" (default), skip provider health checks at call time
- REDIS_URL (or REDIS_HOST/REDIS_PORT/REDIS_DB): optional; if unavailable, streaming gracefully disables and system continues locally
- AF_ENABLE_PINECONE: opt-in Pinecone usage; disabled by default to avoid startup hangs; requires PINECONE_API_KEY
- EMBEDDINGS_BACKEND: "hash" (default) or "st" (sentence-transformers)
- AF_ENFORCE_STRICT: when "1", SLA/KPI violations will raise
- AF_APPROVAL_ENABLE: when "1", enable approval/HITL gating
- AF_APPROVAL_REQUIRE_STRICT: when "1", block on non-approved escalations
- AF_HITL_AUTOAPPROVE: when "1" (default), auto-approve escalations in CI/dev

SLA/KPI and HITL
- SLA/KPI policies live in sla_kpi_config.py and are enforced pre/post each subtask via orchestrator_enforcer.py (soft by default).
- Approval/HITL gating is lightweight and non-blocking by default; escalations publish events onto the memory mesh and can be strictly enforced via env.

Tests
- pip install -r requirements.txt (inside venv)
- pytest -q

Services
- Redis (optional) for swarm streams and memory KV: set REDIS_URL or REDIS_HOST/REDIS_PORT. If not reachable, features degrade gracefully.
- Pinecone (optional) for semantic index: set PINECONE_API_KEY and ensure region setup. Disabled unless AF_ENABLE_PINECONE=1.

Security
- Do not commit real secrets. Rotate any leaked keys. .env is ignored by git.
