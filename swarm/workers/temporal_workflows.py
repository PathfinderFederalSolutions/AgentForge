from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.worker import Worker

from swarm.config import settings
from swarm.enforcement import enforcer
from swarm import lineage

try:
    from orchestrator import build_orchestrator
except Exception:
    build_orchestrator = None

@dataclass
class JobInput:
    job_id: str
    goal: str
    agents: int
    artifacts: List[Dict[str, Any]]

@activity.defn
def run_orchestration(inp: JobInput) -> Dict[str, Any]:
    if not build_orchestrator:
        raise RuntimeError("Orchestrator unavailable")
    orch = build_orchestrator(num_agents=inp.agents)
    results = orch.run_goal_sync(inp.goal)
    decision = enforcer.post(goal=inp.goal, results=results)
    lineage.complete_job(job_id=inp.job_id, decision=decision, results=results)
    return {"job_id": inp.job_id, "decision": decision, "results": list(results)}

@workflow.defn(name="JobWorkflow")
class JobWorkflow:
    @workflow.run
    async def run(self, inp: JobInput) -> Dict[str, Any]:
        return await workflow.execute_activity(
            run_orchestration,
            inp,
            schedule_to_close_timeout=timedelta(minutes=30),
        )

# Worker bootstrap
from datetime import timedelta

async def run_worker():
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[JobWorkflow],
        activities=[run_orchestration],
    )
    await worker.run()