import asyncio
import json
import logging
import os
from typing import Any, Dict

from swarm.jetstream import subscribe_hitl, ensure_streams
from swarm.learning.feedback import record_feedback

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("hitl-worker")

BATCH = int(os.getenv("NATS_FETCH_BATCH", "10"))
WAIT = float(os.getenv("NATS_FETCH_WAIT", "5.0"))

async def process_msg(m):
    try:
        data = json.loads(m.data.decode())
    except Exception:
        log.exception("Invalid HITL JSON; dropping")
        await m.ack()
        return

    goal = data.get("goal")
    decision = data.get("decision", {})
    results = data.get("results", [])
    mission = os.getenv("MISSION", os.getenv("ENV", "default"))

    # Placeholder triage: immediately record feedback; real UI would update decision before writeback
    try:
        record_feedback(goal or "", results if isinstance(results, list) else [results], decision)
        log.info("HITL feedback recorded mission=%s goal_len=%d", mission, len(goal or ""))
    except Exception:
        log.exception("HITL feedback writeback failed")

    await m.ack()

async def main():
    mission = os.getenv("MISSION", os.getenv("ENV", "staging"))
    # Ensure JetStream streams exist before subscribing
    try:
        await ensure_streams()
    except Exception as e:
        logging.getLogger(__name__).warning("ensure_streams failed: %s", e)
    _, _, sub = await subscribe_hitl(consumer_name=f"hitl-{mission}", mission=mission)
    log.info("HITL consumer started mission=%s", mission)
    while True:
        try:
            msgs = await sub.fetch(BATCH, timeout=WAIT)
        except Exception:
            await asyncio.sleep(0.5)
            continue
        if not msgs:
            continue
        try:
            await asyncio.gather(*(process_msg(m) for m in msgs))
        except Exception:
            log.exception("HITL batch processing error")

if __name__ == "__main__":
    asyncio.run(main())
