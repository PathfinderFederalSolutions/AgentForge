import asyncio
import json
import logging
import os
from datetime import datetime, timedelta

import boto3

logger = logging.getLogger(__name__)

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_USE_SSL = os.getenv("S3_USE_SSL", "1") == "1"
SNAPSHOT_INTERVAL_SEC = int(os.getenv("MESH_SNAPSHOT_INTERVAL_SEC", "300"))

_state = {}

async def subscribe_and_apply(js, mission: str):
    sub = await js.subscribe(f"mesh.ops.{mission}")
    async for msg in sub.messages:
        try:
            op = json.loads(msg.data.decode())
            # Apply op to _state (CRDT ops would go here)
            _state.update(op)
            await msg.ack()
        except Exception:
            logger.exception("apply op failed")
            await msg.ack()

async def snapshot_loop():
    if not (S3_ENDPOINT and S3_BUCKET and S3_ACCESS_KEY and S3_SECRET_KEY):
        logger.warning("S3 not configured; snapshots disabled")
        return
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        use_ssl=S3_USE_SSL,
    )
    while True:
        try:
            payload = json.dumps(_state).encode()
            key = f"snapshots/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=payload, ContentType="application/json")
            logger.info("snapshot saved: %s", key)
        except Exception:
            logger.exception("snapshot failed")
        await asyncio.sleep(SNAPSHOT_INTERVAL_SEC)

def start_mesh_tasks(loop, js, mission: str):
    loop.create_task(subscribe_and_apply(js, mission))
    loop.create_task(snapshot_loop())