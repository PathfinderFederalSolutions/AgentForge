from __future__ import annotations
from typing import Any, Dict, Optional, List
import os
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from swarm.enforcement import enforcer
from swarm.config import settings
from swarm.storage import store
from swarm import lineage
import asyncio
from contextlib import asynccontextmanager

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

class JobRequest(BaseModel):
    goal: str = Field(..., description="Natural language goal or mission order.")
    agents: Optional[int] = Field(default=2, ge=1, le=1024)
    artifacts: Optional[List[Dict[str, Any]]] = Field(default=None, description="Artifact metadata to provide context.")

class JobResponse(BaseModel):
    goal: str
    results: Any
    decision: Dict[str, Any]

class UploadResponse(BaseModel):
    artifact_id: str
    backend: str
    filename: str
    size: int
    sha256: str
    presigned_url: Optional[str] = None

class SubmitAsyncResponse(BaseModel):
    job_id: str
    dispatch: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize lineage DB on startup using lifespan (replaces deprecated on_event)
    lineage.init_db()
    yield

app = FastAPI(title="AgentForge Swarm Gateway", version="0.3.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "service": settings.service_name, "dispatch_mode": settings.dispatch_mode}

@app.post("/v1/artifacts/upload", response_model=UploadResponse)
async def upload_artifact(file: UploadFile = File(...)):
    try:
        meta = store.save_file(file.file, filename=file.filename or "upload.bin", content_type=file.content_type)
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

@app.post("/v1/jobs/submit", response_model=JobResponse)
def submit_job(req: JobRequest):
    pre = enforcer.pre(goal=req.goal)
    if not pre.get("ok", False):
        raise HTTPException(status_code=403, detail="Pre-enforcement failed")

    if not build_orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator unavailable")

    goal = req.goal
    if req.artifacts:
        goal += "\n\nContext Artifacts:\n"
        for a in req.artifacts:
            goal += f"- {a.get('filename','artifact')} (id={a.get('artifact_id')}, sha256={a.get('sha256')}, size={a.get('size')})\n"

    job_id = lineage.start_job(goal=goal, artifacts=req.artifacts)

    # Resolve dispatch mode per request
    dispatch_mode = os.getenv("DISPATCH_MODE", settings.dispatch_mode).strip().lower()

    if dispatch_mode == "sync":
        orch = build_orchestrator(num_agents=req.agents or 2)
        results = orch.run_goal_sync(goal)
        decision = enforcer.post(goal=goal, results=results)
        lineage.complete_job(job_id=job_id, decision=decision, results=results)
        return JobResponse(goal=goal, results=list(results), decision=decision)

    if dispatch_mode == "nats":
        import json, nats  # lazy import

        async def _send():
            nc = await nats.connect(servers=[settings.nats_url])
            payload = {"job_id": job_id, "goal": goal, "agents": req.agents or 2, "artifacts": req.artifacts or []}
            await nc.publish(settings.nats_topic_jobs, json.dumps(payload).encode("utf-8"))
            await nc.drain()

        asyncio.run(_send())
        return JobResponse(goal=goal, results=[{"status": "queued", "job_id": job_id}], decision={"approved": False, "action": "queued", "reason": "async_dispatch"})

    if dispatch_mode == "temporal":
        from temporalio.client import Client

        async def _schedule():
            client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
            # Start by workflow name to avoid importing worker module in API process
            handle = await client.start_workflow(
                "JobWorkflow",  # matches @workflow.defn(name="JobWorkflow") in worker
                {"job_id": job_id, "goal": goal, "agents": req.agents or 2, "artifacts": req.artifacts or []},
                id=f"job-{job_id}",
                task_queue=settings.temporal_task_queue,
            )
            return handle.id
        asyncio.run(_schedule())
        return JobResponse(goal=goal, results=[{"status": "queued", "job_id": job_id}], decision={"approved": False, "action": "queued", "reason": "async_dispatch"})

    raise HTTPException(status_code=400, detail=f"Unknown dispatch mode: {dispatch_mode}")

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
    }

# Existing analyze endpoint
class AnalyzeRequest(BaseModel):
    artifact: Dict[str, Any]
    query: str = Field(..., description="What you want from the swarm (e.g., 'Full code review and fix plan').")
    agents: Optional[int] = Field(default=4, ge=1, le=2048)

@app.post("/v1/jobs/analyze", response_model=JobResponse)
def analyze_artifact(req: AnalyzeRequest):
    if not build_orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator unavailable")

    artifact = req.artifact
    fname = artifact.get("filename", "artifact")
    goal = f"Analyze artifact '{fname}' (id={artifact.get('artifact_id')}, sha256={artifact.get('sha256')}, size={artifact.get('size')}).\nTask: {req.query}\nDeliver a complete, production-ready result that satisfies the SLA gates."

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