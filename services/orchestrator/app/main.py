# services/orchestrator/app/main.py
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
import signal
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Coroutine

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import argparse
import time

# --- Logging ---------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [orchestrator] %(message)s",
)
log = logging.getLogger("orchestrator")

# --- Optional imports to fit your repo ------------------------------------

# NATS/JetStream client (your wrapper)
try:
    from services.orchestrator.app.nats_client import get_nc_and_js  # expected helper
except Exception as e:  # pragma: no cover
    log.warning("Failed to import nats_client.get_nc_and_js: %s", e)
    get_nc_and_js = None

# Protocol message helpers (if you have schema/constructors here)
try:
    from swarm.protocol.messages import dict_to_bytes  # optional convenience
except Exception:
    dict_to_bytes = lambda d: json.dumps(d).encode("utf-8")  # fallback

# Evidence lineage writer (optional)
def _maybe_lineage_add(job_id: str, kind: str, payload: Dict[str, Any]) -> None:
    """
    If swarm.lineage is available, use it; else write JSONL evidence
    under var/artifacts/phase_runs.
    """
    try:
        from swarm.lineage import add_evidence  # type: ignore
        add_evidence(job_id=job_id, kind=kind, payload=payload)
        return
    except Exception:
        pass

    # Fallback JSONL ledger used by your evidence bundler
    outdir = Path("var/artifacts/phase_runs")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{job_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "kind": kind,
            "payload": payload
        }) + "\n")


# --- Configuration ---------------------------------------------------------

JOBS_SUBJECT = os.getenv("AF_JOBS_SUBJECT", "swarm.jobs.staging")
RESULTS_SUBJECT = os.getenv("AF_RESULTS_SUBJECT", "swarm.results.staging")
EDGE_MODE = os.getenv("EDGE_MODE", "false").lower() == "true"

PLANS_PATH = Path(os.getenv("AF_MASTER_PLAN", "plans/master_orchestration.yaml"))

EVIDENCE_KIND_PHASE_SPEC = "phase_spec_applied"
EVIDENCE_KIND_JOB_ENQUEUED = "phase_job_enqueued"
EVIDENCE_KIND_RESULT = "phase_job_result"
EVIDENCE_KIND_SUMMARY = "phase_summary"

PHASES_FILE = "plans/master_orchestration.yaml"
JOBS_TOPIC = "swarm.jobs.staging"
RESULTS_TOPIC = "swarm.results.staging"
EVIDENCE_DIR = "/evidence"

# --- Data models -----------------------------------------------------------

class PhaseTask(BaseModel):
    name: str
    action: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    timeout_s: int = 600  # default 10 minutes per task
    # Optionally require previous job success
    require_success_of: List[str] = Field(default_factory=list)

class PhaseSpec(BaseModel):
    phase: str
    tasks: List[PhaseTask]

class MasterPlan(BaseModel):
    phases: List[PhaseSpec]

# --- PhaseRunner -----------------------------------------------------------

class PhaseRunner:
    """
    Minimal orchestrator that:
      - loads plans/master_orchestration.yaml
      - publishes jobs to swarm.jobs.staging
      - awaits correlated results on swarm.results.staging
      - writes evidence records retrievable at /v1/evidence/{job_id}
    """

    def __init__(self, phases_file=PHASES_FILE):
        self.nc = None
        self.js = None
        self._stop = asyncio.Event()
        with open(phases_file) as f:
            self.phases = yaml.safe_load(f)

    async def start(self):
        if get_nc_and_js is None:
            raise RuntimeError("NATS client helper not available in this image.")
        self.nc, self.js = await get_nc_and_js()
        log.info("PhaseRunner connected to NATS/JS. EDGE_MODE=%s", EDGE_MODE)

        # Handle signals gracefully (CLI mode)
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(s, self._stop.set)
            except NotImplementedError:
                # Windows or restricted env
                pass

    async def close(self):
        try:
            if self.nc is not None:
                await self.nc.drain()
        except Exception as e:  # pragma: no cover
            log.warning("Error draining NATS: %s", e)

    def _load_master_plan(self) -> MasterPlan:
        if not PLANS_PATH.exists():
            raise FileNotFoundError(f"Master plan not found: {PLANS_PATH}")
        with PLANS_PATH.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return MasterPlan(**raw)

    async def run_phase(self, phase_name: str) -> Dict[str, Any]:
        plan = self._load_master_plan()
        phase = next((p for p in plan.phases if p.phase == phase_name), None)
        if phase is None:
            raise ValueError(f"Phase '{phase_name}' not found in {PLANS_PATH}")

        # Record the applied spec as evidence (traceability)
        _maybe_lineage_add(
            job_id=f"phase::{phase_name}",
            kind=EVIDENCE_KIND_PHASE_SPEC,
            payload=phase.model_dump(),
        )

        # Map: task.name -> job_id for dependency checks
        prior_results: Dict[str, Dict[str, Any]] = {}
        for task in phase.tasks:
            # Enforce intra-phase dependencies
            for dep in task.require_success_of:
                if dep not in prior_results or prior_results[dep].get("status") != "ok":
                    raise RuntimeError(
                        f"Task '{task.name}' requires successful completion of '{dep}'"
                    )

            job_id = str(uuid.uuid4())
            msg = {
                "job_id": job_id,
                "phase": phase_name,
                "task": task.name,
                "action": task.action,
                "payload": task.payload,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "edge_mode": EDGE_MODE,
            }

            await self._publish_job(job_id, msg)
            _maybe_lineage_add(job_id, EVIDENCE_KIND_JOB_ENQUEUED, msg)

            # Await correlated result
            res = await self._await_result(job_id, timeout_s=task.timeout_s)
            prior_results[task.name] = res or {"status": "no_result"}
            _maybe_lineage_add(job_id, EVIDENCE_KIND_RESULT, res or {"status": "no_result"})

            # Hard stop on failure
            if not res or res.get("status") != "ok":
                summary = {
                    "phase": phase_name,
                    "failed_task": task.name,
                    "result": res,
                }
                _maybe_lineage_add(f"phase::{phase_name}", EVIDENCE_KIND_SUMMARY, summary)
                return summary

        # Phase successful
        summary = {
            "phase": phase_name,
            "status": "ok",
            "tasks": {k: v.get("status") for k, v in prior_results.items()},
        }
        _maybe_lineage_add(f"phase::{phase_name}", EVIDENCE_KIND_SUMMARY, summary)
        return summary

    async def _publish_job(self, job_id: str, msg: Dict[str, Any]) -> None:
        assert self.nc is not None
        assert self.js is not None
        headers = {
            "Nats-Msg-Id": job_id,  # idempotency on replays
            "Content-Type": "application/json",
        }
        data = dict_to_bytes(msg)
        await self.js.publish(
            subject=JOBS_SUBJECT,
            payload=data,
            headers=headers,
        )
        log.info("Enqueued job %s to %s (task=%s action=%s)",
                 job_id, JOBS_SUBJECT, msg.get("task"), msg.get("action"))

    async def _await_result(self, job_id: str, timeout_s: int = 600) -> Optional[Dict[str, Any]]:
        """
        Wait for a single correlated result on RESULTS_SUBJECT.
        Correlation rule: message body must contain matching 'job_id'.
        Uses ephemeral inbox subscription to avoid interfering with durable consumers.
        """
        assert self.nc is not None

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        inbox = await self.nc.create_inbox()

        async def handler(msg):
            try:
                payload = json.loads(msg.data.decode("utf-8"))
            except Exception:
                return
            if payload.get("job_id") == job_id:
                if not fut.done():
                    fut.set_result(payload)

        sub = await self.nc.subscribe(subject=RESULTS_SUBJECT, queue=None, cb=handler, inbox=inbox)

        try:
            res = await asyncio.wait_for(fut, timeout=timeout_s)
            log.info("Received result for job %s: %s", job_id, res.get("status"))
            return res
        except asyncio.TimeoutError:
            log.error("Timed out waiting for result (job_id=%s, timeout_s=%s)", job_id, timeout_s)
            return None
        finally:
            try:
                await self.nc.unsubscribe(sub)
            except Exception:
                pass

    def run_phase(self, phase_name):
        phase = next((p for p in self.phases if p.get("name") == phase_name), None)
        if not phase:
            print(f"Phase {phase_name} not found.")
            return
        for job in ["render_phase", "apply_k8s", "run_tests"]:
            job_id = f"{phase_name}_{job}_{int(time.time())}"
            # publish_job, wait_for_result, attach_evidence must be implemented in worker_protocol
            from services.orchestrator.app.worker_protocol import publish_job, wait_for_result, attach_evidence
            publish_job(JOBS_TOPIC, job, job_id, phase)
            result = wait_for_result(RESULTS_TOPIC, job_id)
            for key in phase.get("evidence_keys", []):
                attach_evidence(job_id, key, os.path.join(EVIDENCE_DIR, job_id))
            print(f"Job {job} for phase {phase_name} complete. Result: {result}")

# --- FastAPI app + endpoints ----------------------------------------------

app = FastAPI(title="AgentForge Orchestrator", version="0.1.0")

class RunPhaseRequest(BaseModel):
    phase: str

class RunPhaseResponse(BaseModel):
    summary: Dict[str, Any]

@app.on_event("startup")
async def _startup():
    app.state.runner = PhaseRunner()
    await app.state.runner.start()
    log.info("Orchestrator startup complete.")

@app.on_event("shutdown")
async def _shutdown():
    try:
        runner: PhaseRunner = app.state.runner  # type: ignore
        await runner.close()
    except Exception:
        pass
    log.info("Orchestrator shutdown complete.")

@app.post("/v1/phase/run", response_model=RunPhaseResponse)
async def run_phase(req: RunPhaseRequest):
    try:
        runner: PhaseRunner = app.state.runner  # type: ignore
    except Exception:
        raise HTTPException(status_code=500, detail="Runner not initialized")

    try:
        summary = await runner.run_phase(req.phase)
        return RunPhaseResponse(summary=summary)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Phase execution error")
        raise HTTPException(status_code=500, detail=str(e))

# --- CLI entrypoint (optional) --------------------------------------------

async def _cli_main():
    """
    CLI usage:
      python -m services.orchestrator.app.main --run-phase phase01_edge
    """
    parser = argparse.ArgumentParser(description="AgentForge Phase Orchestrator")
    parser.add_argument("--run-phase", type=str, help="Phase name in master_orchestration.yaml")
    args = parser.parse_args()

    if not args.run_phase:
        print("Nothing to do. Use --run-phase <phase_name> or call the HTTP endpoint.")
        return

    runner = PhaseRunner()
    await runner.start()
    try:
        summary = await runner.run_phase(args.run_phase)
        print(json.dumps(summary, indent=2))
    finally:
        await runner.close()

if __name__ == "__main__":
    # If run directly, use CLI mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-phase", type=str, help="Run a specific orchestration phase")
    args = parser.parse_args()
    if args.run_phase:
        runner = PhaseRunner()
        runner.run_phase(args.run_phase)
    else:
        try:
            asyncio.run(_cli_main())
        except KeyboardInterrupt:
            pass
