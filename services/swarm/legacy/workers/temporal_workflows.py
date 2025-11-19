from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List
import os
from concurrent.futures import ThreadPoolExecutor

from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.common import RetryPolicy

try:
    # Optional: if available, relax sandbox for workflows that import non-deterministic modules
    from temporalio.worker import UnsandboxedWorkflowRunner as _UnsandboxedRunner  # type: ignore
except Exception:
    _UnsandboxedRunner = None

from ..enforcement import enforcer
from swarm import lineage
from ..critic_healer import CriticHealer
from ..bkg import store as bkg

# Lazy import to avoid heavy startup costs
_build_orchestrator = None

def _get_build_orchestrator():
    global _build_orchestrator
    if _build_orchestrator is None:
        from orchestrator import build_orchestrator
        _build_orchestrator = build_orchestrator
    return _build_orchestrator

@dataclass
class JobInput:
    job_id: str
    goal: str
    agents: int
    artifacts: List[Dict[str, Any]]
    mission: str = "default"  # shard key

@activity.defn(name="run_orchestration")
def run_orchestration(inp: JobInput) -> Dict[str, Any]:
    activity.heartbeat({"job_id": inp.job_id})
    # Idempotency: if BKG exists and approved, return it
    existing = bkg.get(inp.goal)
    if existing and existing.get("decision", {}).get("approved", False):
        return {"job_id": inp.job_id, "decision": existing["decision"], "results": existing["results"], "source": "bkg"}

    build_orchestrator = _get_build_orchestrator()

    use_ch = os.getenv("CRITIC_HEALER_ENABLED", "1") != "0"
    if use_ch:
        ch = CriticHealer()
        out = ch.run(goal=inp.goal, total_agents=inp.agents)
        results = out["results"]
        decision = out["decision"]
    else:
        orch = build_orchestrator(num_agents=inp.agents)
        results = orch.run_goal_sync(inp.goal)
        decision = enforcer.post(goal=inp.goal, results=results)

    if decision.get("approved"):
        bkg.update(inp.goal, decision, results)
    lineage.complete_job(job_id=inp.job_id, decision=decision, results=results)
    return {"job_id": inp.job_id, "decision": decision, "results": list(results)}

@workflow.defn(name="JobWorkflow")
class JobWorkflow:
    @workflow.run
    async def run(self, inp: JobInput) -> Dict[str, Any]:
        # Versioning support
        v = workflow.get_version("wf-schema", workflow.DEFAULT_VERSION, 2)
        rp = RetryPolicy(maximum_attempts=3, non_retryable_error_types=["ValueError"])
        return await workflow.execute_activity(
            "run_orchestration",
            inp,
            schedule_to_close_timeout=timedelta(minutes=60 if v >= 2 else 30),
            start_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(seconds=30),
            retry_policy=rp,
        )

async def run_worker():
    temporal_addr = os.getenv("TEMPORAL_ADDRESS", "temporalite:7233")
    temporal_ns = os.getenv("TEMPORAL_NAMESPACE", "default")
    mission = os.getenv("MISSION", os.getenv("ENV", "default"))
    base_tq = os.getenv("TEMPORAL_TASK_QUEUE", "agentforge")
    tq = f"{base_tq}.{mission}"

    client = await Client.connect(temporal_addr, namespace=temporal_ns)
    max_workers = int(os.getenv("TEMPORAL_ACTIVITY_WORKERS", "8"))

    worker = Worker(
        client,
        task_queue=tq,
        workflows=[JobWorkflow],
        activities=[run_orchestration],
        activity_executor=ThreadPoolExecutor(max_workers=max_workers),
        max_cached_workflows=1000,
    )
    await worker.run()