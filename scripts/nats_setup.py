import asyncio
import os
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig, RetentionPolicy, StorageType, ConsumerConfig, DeliverPolicy, AckPolicy

NATS_URL = os.getenv("NATS_URL", "nats://nats.agentforge-staging.svc.cluster.local:4222")

async def main():
    nc = NATS()
    await nc.connect(servers=[NATS_URL])
    js = nc.jetstream()

    try:
        await js.add_stream(
            StreamConfig(
                name="swarm_jobs",
                subjects=["swarm.jobs.*"],
                retention=RetentionPolicy.WorkQueue,
                storage=StorageType.File,
            )
        )
    except Exception:
        pass

    # Ensure durable consumer aligned to staging mission
    try:
        cfg = ConsumerConfig(
            durable_name="worker-staging",
            deliver_policy=DeliverPolicy.ALL,
            ack_policy=AckPolicy.EXPLICIT,
            filter_subject="swarm.jobs.staging",
            max_ack_pending=128,
        )
        await js.add_consumer("swarm_jobs", cfg)
    except Exception:
        pass

    await nc.drain()

if __name__ == "__main__":
    asyncio.run(main())