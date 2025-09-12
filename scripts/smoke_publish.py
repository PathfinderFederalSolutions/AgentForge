import argparse
import asyncio
import json
import os
from nats.aio.client import Client as NATS

NATS_URL = os.getenv("NATS_URL", "nats://nats.agentforge-staging.svc.cluster.local:4222")
MISSION = os.getenv("MISSION", "staging")

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mission", default="staging")
    p.add_argument("--goal", required=True)
    p.add_argument("--agents", type=int, default=2)
    args = p.parse_args()
    nc = NATS()
    await nc.connect(servers=[NATS_URL])
    payload = {"job_id": "smoke", "goal": args.goal, "agents": args.agents}
    headers = {"Nats-Msg-Id": payload["job_id"]}
    await nc.publish(f"swarm.jobs.{args.mission}", json.dumps(payload).encode(), headers=headers)
    await nc.drain()

if __name__ == "__main__":
    asyncio.run(main())
