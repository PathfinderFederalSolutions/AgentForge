import asyncio
import json
import logging
import signal
from typing import Any, Dict, List

import nats

from swarm.config import settings
from swarm.enforcement import enforcer
from swarm import lineage

try:
    from orchestrator import build_orchestrator
except Exception:
    build_orchestrator = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nats-worker")

async def handle_job(msg):
    data = json.loads(msg.data.decode("utf-8"))
    job_id: str = data["job_id"]
    goal: str = data["goal"]
    agents: int = int(data.get("agents", 2))
    artifacts: List[Dict[str, Any]] = data.get("artifacts", [])

    log.info("Processing job %s", job_id)

    if not build_orchestrator:
        log.error("Orchestrator unavailable")
        return

    try:
        orch = build_orchestrator(num_agents=agents)
        results = orch.run_goal_sync(goal)
        decision = enforcer.post(goal=goal, results=results)
        lineage.complete_job(job_id=job_id, decision=decision, results=results)

        out = {"job_id": job_id, "goal": goal, "decision": decision, "results": list(results)}
        await msg._client.publish(settings.nats_topic_results, json.dumps(out).encode("utf-8"))
        log.info("Completed job %s", job_id)
    except Exception as e:
        log.exception("Job %s failed: %s", job_id, e)
        lineage.complete_job(job_id=job_id, decision={"approved": False, "action": "hitl", "reason": "worker_error"}, results=[f"Error: {e}"])

async def main():
    nc = await nats.connect(servers=[settings.nats_url])
    sub = await nc.subscribe(settings.nats_topic_jobs, cb=handle_job)
    log.info("Subscribed to %s", settings.nats_topic_jobs)

    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, stop.set)

    await stop.wait()
    await nc.drain()

if __name__ == "__main__":
    asyncio.run(main())