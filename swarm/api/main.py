from __future__ import annotations
from typing import Any, Dict, Optional, List
import os
import asyncio
import time
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi import Request, Body
from fastapi.responses import StreamingResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from collections import defaultdict

# Centralized Prometheus metrics utilities (moved to top before class definitions)
from swarm.api.metrics import (
    metrics_router,
    instrument_app,
    TASK_SUBMIT,
    TASK_SUBMIT_ERRORS,
    TASK_SUBMIT_LATENCY,
    TACTICAL_ALERTS_PUBLISHED,
)  # type: ignore

from swarm.enforcement import enforcer
from swarm.config import settings
from swarm.storage import store
from swarm import lineage
from swarm.critic_healer import CriticHealer
from swarm.observability.task_latency import record_task_completion  # type: ignore
from sla_kpi_config import get_task_budget  # type: ignore

# Optional tracing provider init
try:
    from swarm.observability.otel import init_tracing as _init_tracing
except Exception:  # pragma: no cover
    _init_tracing = None  # type: ignore

# Optional protocol models
try:
    from swarm.protocol.messages import TaskSpec  # canonical task spec
except Exception:
    TaskSpec = None  # type: ignore

# Optional observability context
try:
    from swarm.observability.costs import set_observability_context as _set_obs_ctx  # type: ignore
except Exception:  # pragma: no cover
    _set_obs_ctx = None  # type: ignore

# Optional default observable context
try:
    from swarm.observability.otel import set_default_observable_context as _set_default_ctx  # type: ignore
except Exception:  # pragma: no cover
    _set_default_ctx = None  # type: ignore


# --- Request/Response Models -------------------------------------------------
class UploadResponse(BaseModel):
    artifact_id: str
    backend: str
    filename: str
    size: int
    sha256: str
    presigned_url: Optional[str] = None


class TaskSubmitResponse(BaseModel):
    task_id: str
    dispatch: Optional[str] = Field(None, description="Dispatcher used (e.g., nats|temporal)")


class JobRequest(BaseModel):
    goal: str
    agents: Optional[int] = Field(default=2, ge=1, le=2048)
    artifacts: Optional[List[Dict[str, Any]]] = None


class JobResponse(BaseModel):
    goal: str
    results: List[Dict[str, Any]]
    decision: Dict[str, Any]
    dag_hash: Optional[str] = None


class CHRequest(BaseModel):
    goal: str
    agents: Optional[int] = Field(default=6, ge=1, le=4096)
    artifacts: Optional[List[Dict[str, Any]]] = None


# Optional OpenTelemetry
try:
    from opentelemetry import trace  # type: ignore

    _tracer = trace.get_tracer("swarm.api")
    try:
        from opentelemetry.propagate import inject  # type: ignore
    except Exception:  # pragma: no cover
        inject = None  # type: ignore
except Exception:  # pragma: no cover
    _tracer = None
    inject = None  # type: ignore

# JetStream helpers
try:
    from swarm.jetstream import publish_job, ensure_streams
except Exception:
    publish_job = None
    ensure_streams = None

# Test helper export
try:
    import fastapi.testclient as _fastapi_testclient  # module import to avoid unused warnings

    TestClient = getattr(_fastapi_testclient, "TestClient", None)  # re-export for tests
    _ = TestClient  # reference to avoid linter flagging as unused
except Exception:  # pragma: no cover
    TestClient = None

try:
    from orchestrator import build_orchestrator
except Exception:
    build_orchestrator = None

try:
    from swarm.memory.mesh_dist import DistMemoryMesh, start_snapshot_task
except Exception:
    DistMemoryMesh = None
    start_snapshot_task = None

try:  # Added import for fused track retrieval
    from swarm.storage import load_fused_track  # type: ignore
except Exception:  # pragma: no cover
    load_fused_track = None  # type: ignore


# --- helpers ---------------------------------------------------------------

def _trace_carrier() -> Dict[str, str]:
    carrier: Dict[str, str] = {}
    if inject is not None:
        try:
            inject(carrier)  # type: ignore
        except Exception:
            return {}
    return carrier


async def _dispatch_nats(mission: str, payload: Dict[str, Any]) -> None:
    # Try JetStream helper first, then fallback to bare NATS publish
    try:
        if publish_job:
            await publish_job(mission, payload)
            return
    except Exception:
        pass  # fall through to bare NATS
    try:
        import importlib
        nats = importlib.import_module("nats")
        nc = await nats.connect(servers=[settings.nats_url])
        subj = f"swarm.jobs.{mission}"
        # Add idempotency header if possible
        msg_id = (
            payload.get("job_id")
            or payload.get("invocation_id")
            or payload.get("id")
            or payload.get("request_id")
        )
        headers = {"Nats-Msg-Id": str(msg_id)} if msg_id else None
        await nc.publish(subj, json.dumps(payload).encode("utf-8"), headers=headers)
        await nc.drain()
    except Exception as e:
        raise RuntimeError("nats_unavailable") from e


async def _dispatch_temporal(mission: str, payload: Dict[str, Any]) -> None:
    addr = os.getenv("TEMPORAL_ADDRESS", settings.temporal_address)
    if not addr:
        raise RuntimeError("temporal_unavailable")
    # Lazy, resilient import to support tests that stub only 'temporalio.client'
    import importlib

    mod = importlib.import_module("temporalio.client")
    Client = getattr(mod, "Client")
    task_queue = f"{settings.temporal_task_queue}.{mission}"
    client = await Client.connect(addr, namespace=settings.temporal_namespace)
    await client.start_workflow(
        "JobWorkflow",
        payload,
        id=f"job-{payload.get('job_id')}",
        task_queue=task_queue,
    )


def _format_goal_with_artifacts(goal: str, artifacts: Optional[List[Dict[str, Any]]]) -> str:
    if not artifacts:
        return goal
    lines = [goal, "", "Context Artifacts:"]
    for a in artifacts:
        lines.append(
            f"- {a.get('filename', 'artifact')} (id={a.get('artifact_id')}, sha256={a.get('sha256')}, size={a.get('size')})"
        )
    return "\n".join(lines)


def _sync_execute(goal: str, agents: int, job_id: str) -> Dict[str, Any]:
    if not build_orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator unavailable")
    orch = build_orchestrator(num_agents=agents)
    results = orch.run_goal_sync(goal)
    dag_hash = None
    try:
        # Repeat seed procedure to compute stable hash without re-running plan
        from swarm.planner import Planner
        # removed unused DAGSpec import
        from hashlib import sha256
        seed = int(sha256(goal.encode('utf-8')).hexdigest()[:8], 16)
        dag_hash = Planner().make_dag(goal, seed=seed).compute_hash()
    except Exception:
        pass
    # Reviewer stage (Phase 4 scaffolding): attach validation & confidence
    try:
        from swarm.reviewer import review_results  # lazy import

        results = review_results(results, auto_heal=False)
    except Exception:
        pass
    decision = enforcer.post(goal=goal, results=results)
    lineage.complete_job(job_id=job_id, decision=decision, results=results)
    return {"results": list(results), "decision": decision, "dag_hash": dag_hash}


async def _enqueue_nats(job_id: str, goal: str, agents: int, artifacts: Optional[List[Dict[str, Any]]]) -> None:
    mission = os.getenv("MISSION", settings.env or "default")
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "goal": goal,
        "agents": agents,
        "artifacts": artifacts or [],
    }
    carrier = _trace_carrier()
    if carrier.get("traceparent"):
        payload["traceparent"] = carrier.get("traceparent")
    await _dispatch_nats(mission, payload)


async def _enqueue_temporal(job_id: str, goal: str, agents: int, artifacts: Optional[List[Dict[str, Any]]]) -> None:
    mission = os.getenv("MISSION", settings.env or "default")
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "goal": goal,
        "agents": agents,
        "artifacts": artifacts or [],
    }
    carrier = _trace_carrier()
    if carrier.get("traceparent"):
        payload["traceparent"] = carrier.get("traceparent")
    await _dispatch_temporal(mission, payload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize lineage DB on startup using lifespan (replaces deprecated on_event)
    lineage.init_db()
    mission_id = os.getenv("MISSION", settings.env or "default")
    if _set_default_ctx:
        try:
            _set_default_ctx(mission_id=mission_id)
        except Exception:
            pass
    # Initialize tracing resource attributes (best-effort)
    if _init_tracing is not None:
        try:
            _init_tracing(service_name=os.getenv("SERVICE_NAME", settings.service_name), service_version=os.getenv("SERVICE_VERSION", "0.3.0"), environment=os.getenv("ENV", settings.env))
        except Exception:
            pass
    # Ensure NATS streams exist for jobs/results/HITL if available
    if ensure_streams:
        try:
            await ensure_streams()
        except Exception:
            pass
    yield


app = FastAPI(title="AgentForge Swarm Gateway", version="0.3.0", lifespan=lifespan)
# Instrument & mount metrics
instrument_app(app)
app.include_router(metrics_router)


@app.get("/health")
def health():
    return {"status": "ok", "service": settings.service_name, "dispatch_mode": settings.dispatch_mode}


@app.get("/v1/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/readyz")
def readyz():
    # Basic readiness: lineage DB initialized and (optionally) orchestrator importable
    ready = True
    details: Dict[str, Any] = {"lineage": True}
    details["orchestrator"] = bool(build_orchestrator)
    details["nats"] = bool(ensure_streams is not None and publish_job is not None)
    return {"ready": ready, "details": details}


@app.post("/v1/artifacts/upload", response_model=UploadResponse)
async def upload_artifact(file: UploadFile = File(...)):
    try:
        meta = store.save_file(
            file.file,
            filename=file.filename or "upload.bin",
            content_type=file.content_type,
        )
        presigned = store.presign(meta)
        # lineage
        lineage.record_artifact(meta)
        return UploadResponse(
            artifact_id=meta["artifact_id"],
            backend=meta["backend"],
            filename=meta["filename"],
            size=meta["size"],
            sha256=meta["sha256"],
            presigned_url=presigned,
        )
    except ValueError as ve:
        raise HTTPException(status_code=413, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


async def _prepare_task(spec: TaskSpec, mission: str):  # type: ignore[valid-type]
    pre = enforcer.pre(goal=spec.goal)
    if not pre.get("ok", False):
        if TASK_SUBMIT_ERRORS:
            try: TASK_SUBMIT_ERRORS.labels("pre_enforcement").inc()  # type: ignore
            except Exception: pass
        raise HTTPException(status_code=403, detail="Pre-enforcement failed")
    task_id = lineage.start_job(goal=spec.goal, artifacts=spec.metadata.get("artifacts"))
    if _set_obs_ctx:
        try: _set_obs_ctx(mission_id=mission, task_id=task_id)
        except Exception: pass
    carrier = _trace_carrier()
    payload = {
        "job_id": task_id,
        "goal": spec.goal,
        "priority": getattr(spec, "priority", 0),
        "metadata": spec.metadata,
        "task_spec": spec.model_dump(),
        "traceparent": carrier.get("traceparent"),
    }
    return task_id, payload

async def _attempt_dispatch(mission: str, payload: Dict[str, Any]) -> str:
    try:
        if _tracer:
            with _tracer.start_as_current_span("api.submit_task"):  # type: ignore
                await _dispatch_nats(mission, payload)
        else:
            await _dispatch_nats(mission, payload)
        return "nats"
    except Exception:
        try:
            if os.getenv("TEMPORAL_ADDRESS", settings.temporal_address):
                await _dispatch_temporal(mission, payload)
                return "temporal"
        except Exception:
            if TASK_SUBMIT_ERRORS:
                try:
                    TASK_SUBMIT_ERRORS.labels("dispatch_failure").inc()  # type: ignore
                except Exception:
                    pass
            raise HTTPException(status_code=503, detail="Dispatch failed")
    raise HTTPException(status_code=503, detail="No async dispatcher available (NATS/Temporal)")

# Canonical task submission (always async enqueue)
@app.post("/v1/tasks", response_model=TaskSubmitResponse)
async def submit_task(spec: TaskSpec):  # type: ignore[valid-type]
    if TaskSpec is None:
        raise HTTPException(status_code=500, detail="Protocol models unavailable")
    mission = os.getenv("MISSION", settings.env or "default")
    if _set_obs_ctx:
        try: _set_obs_ctx(mission_id=mission)
        except Exception: pass
    t0 = time.perf_counter()
    task_id, payload = await _prepare_task(spec, mission)
    dispatched_via = await _attempt_dispatch(mission, payload)
    if TASK_SUBMIT and TASK_SUBMIT_LATENCY:
        try:
            TASK_SUBMIT.labels(dispatched_via).inc()  # type: ignore
            TASK_SUBMIT_LATENCY.labels(dispatched_via).observe(time.perf_counter() - t0)  # type: ignore
        except Exception:
            pass
    return TaskSubmitResponse(task_id=task_id, dispatch=dispatched_via)


@app.post("/v1/jobs/submit", response_model=JobResponse)
def submit_job(req: JobRequest):
    pre = enforcer.pre(goal=req.goal)
    if not pre.get("ok", False):
        raise HTTPException(status_code=403, detail="Pre-enforcement failed")

    goal = _format_goal_with_artifacts(req.goal, req.artifacts)
    job_id = lineage.start_job(goal=goal, artifacts=req.artifacts)

    # Resolve dispatch mode per request
    dispatch_mode = os.getenv("DISPATCH_MODE", settings.dispatch_mode).strip().lower()

    if dispatch_mode == "sync":
        start = time.perf_counter()
        out = _sync_execute(goal=goal, agents=req.agents or 2, job_id=job_id)
        latency = time.perf_counter() - start
        # Record latency against default budget (task type classification TBD)
        try:
            b = get_task_budget("default")
            mission = os.getenv("MISSION", settings.env or "default")
            record_task_completion(latency, "default", mission, b.name, b.p99_ms, b.hard_cap_ms)
        except Exception:
            pass
        return JobResponse(
            goal=goal,
            results=out["results"],
            decision=out["decision"],
            dag_hash=out.get("dag_hash"),
        )  # type: ignore[index]

    if dispatch_mode == "nats":
        # Use NATS (JetStream or bare) and fall back gracefully in _dispatch_nats
        asyncio.run(_enqueue_nats(job_id, goal, req.agents or 2, req.artifacts))
        return JobResponse(
            goal=goal,
            results=[{"status": "queued", "job_id": job_id}],
            decision={
                "approved": False,
                "action": "queued",
                "reason": "async_dispatch",
            },
        )

    if dispatch_mode == "temporal":
        asyncio.run(_enqueue_temporal(job_id, goal, req.agents or 2, req.artifacts))
        return JobResponse(
            goal=goal,
            results=[{"status": "queued", "job_id": job_id}],
            decision={
                "approved": False,
                "action": "queued",
                "reason": "async_dispatch",
            },
        )

    raise HTTPException(status_code=400, detail=f"Unknown dispatch mode: {dispatch_mode}")


@app.post("/v1/jobs/submit_ch")
def submit_with_critic_healer(req: CHRequest):
    goal = req.goal
    if req.artifacts:
        goal += "\n\nContext Artifacts:\n"
        for a in req.artifacts:
            goal += (
                f"- {a.get('filename', 'artifact')} (id={a.get('artifact_id')}, sha256={a.get('sha256')}, size={a.get('size')})\n"
            )
    pre = enforcer.pre(goal=goal)
    if not pre.get("ok", False):
        raise HTTPException(status_code=403, detail="Pre-enforcement failed")
    # Record lineage start
    job_id = lineage.start_job(goal=goal, artifacts=req.artifacts)
    ch = CriticHealer()
    out = ch.run(goal=goal, total_agents=req.agents or 6)
    decision = out["decision"]
    results = out["results"]
    lineage.complete_job(job_id=job_id, decision=decision, results=results)
    return {
        "goal": goal,
        "results": results,
        "decision": decision,
        "source": out.get("source", "base"),
    }


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    j = lineage.get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": j.job_id,
        "goal": j.goal,
        "status": j.status,
        "created_at": j.created_at.isoformat(),
        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        "decision": j.decision,
        "results": j.results,
        "events": lineage.list_job_events(job_id),
        "dag_hash": j.dag_hash,
    }


# Mirror for canonical task status
@app.get("/v1/tasks/{task_id}")
def get_task(task_id: str):
    return get_job(task_id)


# Existing analyze endpoint
class AnalyzeRequest(BaseModel):
    artifact: Dict[str, Any]
    query: str = Field(
        ..., description="What you want from the swarm (e.g., 'Full code review and fix plan')."
    )
    agents: Optional[int] = Field(default=4, ge=1, le=2048)


@app.post("/v1/jobs/analyze", response_model=JobResponse)
def analyze_artifact(req: AnalyzeRequest):
    if not build_orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator unavailable")
    artifact = req.artifact
    fname = artifact.get("filename", "artifact")
    goal = (
        f"Analyze artifact '{fname}' (id={artifact.get('artifact_id')}, sha256={artifact.get('sha256')}, size={artifact.get('size')}).\n"
        f"Task: {req.query}\n"
        "Deliver a complete, production-ready result that satisfies the SLA gates."
    )

    pre = enforcer.pre(goal=goal)
    if not pre.get("ok", False):
        raise HTTPException(status_code=403, detail="Pre-enforcement failed")

    job_id = lineage.start_job(goal=goal, artifacts=[artifact])
    dispatch_mode = os.getenv("DISPATCH_MODE", settings.dispatch_mode).strip().lower()
    if dispatch_mode != "sync":
        return submit_job(JobRequest(goal=goal, agents=req.agents or 4, artifacts=[artifact]))

    orch = build_orchestrator(num_agents=req.agents or 4)
    results = orch.run_goal_sync(goal)
    decision = enforcer.post(goal=goal, results=results)
    lineage.complete_job(job_id=job_id, decision=decision, results=results)
    return JobResponse(goal=goal, results=list(results), decision=decision)


@app.on_event("startup")
async def _maybe_start_mesh_snapshots():
    if not DistMemoryMesh or not start_snapshot_task:
        return
    for scope in (
        "global",
        f"project:{settings.env}",
    ):
        mesh = DistMemoryMesh(scope=scope, actor="api")
        start_snapshot_task(mesh, interval_sec=60)


@app.post("/job/sync", response_model=JobResponse)
async def run_job_sync(req: JobRequest):
    goal = _format_goal_with_artifacts(req.goal, req.artifacts)
    from hashlib import sha256
    seed = int(sha256(goal.encode('utf-8')).hexdigest()[:8], 16)
    from swarm.planner import Planner
    dag_hash = Planner().make_dag(goal, seed=seed).compute_hash()
    job_id = lineage.start_job(goal=goal, artifacts=req.artifacts, dag_hash=dag_hash)
    out = _sync_execute(goal, req.agents, job_id)
    return JobResponse(goal=goal, results=out['results'], decision=out['decision'], dag_hash=dag_hash)


@app.post("/job/async")
async def run_job_async(req: JobRequest):
    goal = _format_goal_with_artifacts(req.goal, req.artifacts)
    from hashlib import sha256
    seed = int(sha256(goal.encode('utf-8')).hexdigest()[:8], 16)
    from swarm.planner import Planner
    dag_hash = Planner().make_dag(goal, seed=seed).compute_hash()
    job_id = lineage.start_job(goal=goal, artifacts=req.artifacts, dag_hash=dag_hash)
    try:
        await _enqueue_nats(job_id, goal, req.agents, req.artifacts)
    except Exception:
        raise HTTPException(status_code=503, detail="Dispatch unavailable")
    return {"job_id": job_id, "dag_hash": dag_hash}


@app.get('/job/{job_id}')
async def get_job_status(job_id: str):
    j = lineage.get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail='job not found')
    return {
        'job_id': j.job_id,
        'goal': j.goal,
        'status': j.status,
        'dag_hash': getattr(j, 'dag_hash', None),
        'decision': j.decision,
        'results_summary': j.results,
        'created_at': j.created_at.isoformat() if j.created_at else None,
        'completed_at': j.completed_at.isoformat() if j.completed_at else None,
    }

# New: Retrieve deterministic DAG spec JSON by hash
@app.get('/v1/dag/{dag_hash}')
def get_dag(dag_hash: str):
    path = os.path.join('var', 'artifacts', f'{dag_hash}.dag.json')
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='dag not found')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Echo hash explicitly (file name authoritative)
        data['hash'] = dag_hash
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'failed to load dag: {e}')


@app.get("/v1/fused_tracks/{track_id}")
def get_fused_track(track_id: str):
    if not load_fused_track:
        raise HTTPException(status_code=404, detail="fusion storage unavailable")
    doc = load_fused_track(track_id)
    if not doc:
        raise HTTPException(status_code=404, detail="track not found")
    return doc

@app.get("/v1/evidence/{job_id}")
def get_evidence_bundle(job_id: str):
    bundle = lineage.build_evidence_bundle(job_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="job not found")
    return bundle

import hmac, hashlib, base64

class EvidenceSigner:
    def __init__(self, secret: str | None = None, ttl_seconds: int = 300):
        self.secret = (secret or os.getenv("EVIDENCE_SIGNING_SECRET", "dev-secret")).encode()
        self.ttl = int(os.getenv("EVIDENCE_SIGNING_TTL", str(ttl_seconds)))

    def sign(self, evidence_id: str) -> str:
        # produce a simple HMAC-SHA256 token
        expires = int(time.time()) + self.ttl
        msg = f"{evidence_id}.{expires}".encode()
        sig = hmac.new(self.secret, msg, hashlib.sha256).digest()
        b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")
        return f"{evidence_id}.{expires}.{b64}"

    def verify(self, token: str) -> bool:
        try:
            evidence_id, exp_s, b64 = token.split(".")
            exp = int(exp_s)
            if exp < int(time.time()):
                return False
            msg = f"{evidence_id}.{exp}".encode()
            expected = base64.urlsafe_b64encode(hmac.new(self.secret, msg, hashlib.sha256).digest()).decode().rstrip("=")
            return hmac.compare_digest(expected, b64)
        except Exception:
            return False

_signer = EvidenceSigner()

# Sanitization helper
_def_max_len = 256

def _safe_str(s: str | None, max_len: int = _def_max_len) -> str | None:
    if s is None:
        return None
    cleaned = ''.join(ch for ch in s if ch.isprintable())
    return cleaned[:max_len]

# --- Event Sources (dummy in-memory feed for tests) --------------------------

class _EventSource:
    def __init__(self):
        # maintain a set of subscriber queues for broadcast semantics
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subs.discard(q)
        except Exception:
            pass

    async def publish(self, item: dict):
        # enqueue to all subscribers without blocking
        for q in list(self._subs):
            try:
                q.put_nowait(item)
            except Exception:
                # if a queue is full/broken, drop it
                try: self._subs.discard(q)
                except Exception: pass

    async def next(self, timeout: float = 5.0) -> Optional[dict]:
        # legacy fallback when used without explicit subscription
        # create a temp sub, wait once, then unsubscribe
        q = self.subscribe()
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.unsubscribe(q)

_events = _EventSource()

# --- SSE endpoint ------------------------------------------------------------

# Dynamic limit readers

def _max_clients() -> int:
    try:
        return int(os.getenv("TACTICAL_STREAM_MAX_CLIENTS", "50"))
    except Exception:
        return 50


def _max_per_ip() -> int:
    try:
        return int(os.getenv("TACTICAL_STREAM_MAX_PER_IP", "0"))
    except Exception:
        return 0

_current_clients = 0
_ip_conn_counts: dict[str, int] = defaultdict(int)

# Optional bearer token checker (simple dev mode)
def _bearer_required() -> bool:
    return os.getenv("TACTICAL_REQUIRE_BEARER", "0").lower() in {"1","true","yes","on"}

def _bearer_verified(hdrs) -> bool:
    try:
        auth = hdrs.get("authorization") or hdrs.get("Authorization")
        if not auth or not auth.lower().startswith("bearer "):
            return False
        token = auth.split(" ", 1)[1].strip()
        expected = os.getenv("TACTICAL_BEARER_TOKEN", "dev-token")
        return bool(token) and (token == expected)
    except Exception:
        return False

def _client_verified(hdrs) -> bool:
    try:
        val = (
            hdrs.get("ssl-client-verify")
            or hdrs.get("x-ssl-client-verify")
            or hdrs.get("x-client-verify")
        )
        return str(val).upper() == "SUCCESS"
    except Exception:
        return False

_def_xff = ("x-forwarded-for", "x-real-ip")

def _get_client_ip(headers, fallback_host: str | None) -> str:
    try:
        for h in _def_xff:
            v = headers.get(h)
            if v:
                return v.split(",")[0].strip()
    except Exception:
        pass
    return fallback_host or "unknown"

async def _sse_iter(client_id: str):
    global _current_clients
    q = _events.subscribe()
    try:
        heartbeat = float(os.getenv("TACTICAL_SSE_HEARTBEAT_SECS", "10"))
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=heartbeat)
            except asyncio.TimeoutError:
                item = None
            if item is None:
                yield b": heartbeat\n\n"
                continue
            # sanitize
            props = item.get("properties", {})
            title = _safe_str(props.get("title"))
            description = _safe_str(props.get("description"), 1024)
            evidence_id = props.get("evidence_id")
            evidence_link = None
            if evidence_id:
                token = _signer.sign(str(evidence_id))
                evidence_link = f"/v1/evidence/{token}"
            # Form GeoJSON Feature
            feature = {
                "type": "Feature",
                "geometry": item.get("geometry"),
                "properties": {
                    **{k: v for k, v in props.items() if k not in ("title","description")},
                    "title": title,
                    "description": description,
                    "evidence_link": evidence_link,
                },
            }
            if TACTICAL_ALERTS_PUBLISHED:
                try: TACTICAL_ALERTS_PUBLISHED.labels("sse").inc()  # type: ignore
                except Exception: pass
            data = json.dumps(feature, separators=(",", ":")).encode()
            # Emit only data line so tests can parse with startswith("data: ")
            yield b"data: " + data + b"\n\n"
    finally:
        _events.unsubscribe(q)
        _current_clients = max(0, _current_clients - 1)

@app.get("/events/stream")
async def events_stream(request: Request):
    global _current_clients
    # Optional mTLS requirement enforced via ingress-provided headers
    if os.getenv("TACTICAL_REQUIRE_CLIENT_CERT", "0").lower() in {"1","true","yes","on"}:
        if not _client_verified(request.headers):
            raise HTTPException(status_code=403, detail="client_cert_required")
    # Optional bearer requirement
    if _bearer_required() and not _bearer_verified(request.headers):
        raise HTTPException(status_code=401, detail="unauthorized")
    ip = _get_client_ip(request.headers, getattr(request.client, "host", None))
    if _current_clients >= _max_clients():
        raise HTTPException(status_code=429, detail="too_many_clients")
    m_per_ip = _max_per_ip()
    if m_per_ip and _ip_conn_counts.get(ip, 0) >= m_per_ip:
        raise HTTPException(status_code=429, detail="too_many_clients_ip")
    _current_clients += 1
    _ip_conn_counts[ip] = _ip_conn_counts.get(ip, 0) + 1
    resp = StreamingResponse(_sse_iter(client_id=str(id(request))), media_type="text/event-stream")
    try:
        resp.background = None  # ensure no background task conflict
    except Exception:
        pass
    # Starlette doesn't provide on-close hook here; decrement in iterator finally as well
    return resp

# --- WebSocket endpoint ------------------------------------------------------

@app.websocket("/events/ws")
async def events_ws(ws: WebSocket):
    global _current_clients
    # Optional mTLS header check
    if os.getenv("TACTICAL_REQUIRE_CLIENT_CERT", "0").lower() in {"1","true","yes","on"}:
        if not _client_verified(ws.headers):
            await ws.close(code=4403)
            return
    # Optional bearer requirement
    if _bearer_required() and not _bearer_verified(ws.headers):
        await ws.close(code=4401)
        return
    ip = _get_client_ip(ws.headers, getattr(ws.client, "host", None))
    if _current_clients >= _max_clients():
        await ws.close(code=4408)
        return
    m_per_ip = _max_per_ip()
    if m_per_ip and _ip_conn_counts.get(ip, 0) >= m_per_ip:
        await ws.close(code=4408)
        return
    _current_clients += 1
    _ip_conn_counts[ip] = _ip_conn_counts.get(ip, 0) + 1
    await ws.accept()
    q = _events.subscribe()
    try:
        heartbeat = float(os.getenv("TACTICAL_WS_HEARTBEAT_SECS", "10"))
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=heartbeat)
            except asyncio.TimeoutError:
                item = None
            if item is None:
                await ws.send_json({"event": "heartbeat", "ts": int(time.time())})
                continue
            props = item.get("properties", {})
            title = _safe_str(props.get("title"))
            description = _safe_str(props.get("description"), 1024)
            evidence_id = props.get("evidence_id")
            evidence_link = None
            if evidence_id:
                token = _signer.sign(str(evidence_id))
                evidence_link = f"/v1/evidence/{token}"
            feature = {
                "type": "Feature",
                "geometry": item.get("geometry"),
                "properties": {
                    **{k: v for k, v in props.items() if k not in ("title","description")},
                    "title": title,
                    "description": description,
                    "evidence_link": evidence_link,
                },
            }
            if TACTICAL_ALERTS_PUBLISHED:
                try: TACTICAL_ALERTS_PUBLISHED.labels("ws").inc()  # type: ignore
                except Exception: pass
            await ws.send_json({"event": "marker", "data": feature})
    except WebSocketDisconnect:
        pass
    finally:
        _events.unsubscribe(q)
        _current_clients = max(0, _current_clients - 1)
        _ip_conn_counts[ip] = max(0, _ip_conn_counts.get(ip, 0) - 1)

# Helper for tests to push events
async def _test_emit_geojson(feature_geometry: dict, properties: dict):
    await _events.publish({"geometry": feature_geometry, "properties": properties})

# Dev-only publish endpoint for tests/load
@app.post("/_test/emit")
async def _test_emit(body: dict = Body(...)):
    if os.getenv("TACTICAL_ENABLE_TEST_ENDPOINTS", "0").lower() not in {"1","true","yes","on"}:
        raise HTTPException(status_code=404, detail="not_found")
    geometry = body.get("geometry") or {"type": "Point", "coordinates": body.get("coordinates")}
    props = body.get("properties") or {}
    await _events.publish({"geometry": geometry, "properties": props})
    return {"ok": True}

# ensure references so linters don't strip imports in tests
_ = (Request, StreamingResponse, WebSocket, WebSocketDisconnect, TACTICAL_ALERTS_PUBLISHED)
