from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict

from nats.js.api import (
    StreamConfig,
    ConsumerConfig,
    RetentionPolicy,
    StorageType,
    DeliverPolicy,
    AckPolicy,
    DiscardPolicy,
)
from nats.js.errors import NotFoundError

from .nats_conn import get_nc_async, get_js_async

log = logging.getLogger("jetstream")
logger = logging.getLogger(__name__)

STREAM_JOBS = "swarm_jobs"
STREAM_RESULTS = "swarm_results"
STREAM_MESH = "mesh_ops"
STREAM_HITL = "swarm_hitl"

SUBJ_JOBS = "swarm.jobs.*"
SUBJ_RESULTS = "swarm.results.*"
SUBJ_MESH = "mesh.ops.>"
SUBJ_HITL = "swarm.hitl.*"

# EDGE mode config
EDGE_MODE = os.getenv("EDGE_MODE", "0").lower() in {"1", "true", "yes", "on"}
EDGE_JS_MAX_MSGS_PER_SUBJECT = int(os.getenv("EDGE_JS_MAX_MSGS_PER_SUBJECT", "10000"))
EDGE_JS_MAX_BYTES = int(os.getenv("EDGE_JS_MAX_BYTES", str(1 * 1024 * 1024 * 1024)))  # 1 GiB default
EDGE_ACK_WAIT_S = int(os.getenv("EDGE_ACK_WAIT_S", "120"))  # 120s default
_ACK_WAIT_NS = EDGE_ACK_WAIT_S * 1_000_000_000


async def _ensure_stream(js, name: str, cfg: StreamConfig) -> None:
    try:
        await js.stream_info(name)
        logger.debug("Stream %s exists", name)
    except NotFoundError:
        await js.add_stream(cfg)
        logger.info("Created stream %s", name)

async def ensure_streams():
    js = await get_js_async()
    # Jobs: work-queue retention
    await _ensure_stream(
        js,
        STREAM_JOBS,
        StreamConfig(
            name=STREAM_JOBS,
            subjects=[SUBJ_JOBS],
            retention=RetentionPolicy.WORK_QUEUE,
            storage=StorageType.File if hasattr(StorageType, 'File') else StorageType.FILE,
            max_consumers=-1,
            max_msgs=-1 if not EDGE_MODE else -1,  # keep unlimited overall
            max_msgs_per_subject=0 if not EDGE_MODE else EDGE_JS_MAX_MSGS_PER_SUBJECT,
            max_bytes=-1 if not EDGE_MODE else EDGE_JS_MAX_BYTES,
            duplicate_window=60_000_000_000,  # 60s in ns
            allow_direct=True,
            discard=DiscardPolicy.Old,
            num_replicas=1,
        ),
    )
    # Results: limits
    await _ensure_stream(
        js,
        STREAM_RESULTS,
        StreamConfig(
            name=STREAM_RESULTS,
            subjects=[SUBJ_RESULTS],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.File if hasattr(StorageType, 'File') else StorageType.FILE,
            max_consumers=-1,
            max_msgs=-1 if not EDGE_MODE else -1,
            max_msgs_per_subject=0 if not EDGE_MODE else EDGE_JS_MAX_MSGS_PER_SUBJECT,
            max_bytes=-1 if not EDGE_MODE else EDGE_JS_MAX_BYTES,
            duplicate_window=60_000_000_000,
            allow_direct=True,
            discard=DiscardPolicy.Old,
            num_replicas=1,
        ),
    )
    # Mesh ops: limits
    await _ensure_stream(
        js,
        STREAM_MESH,
        StreamConfig(
            name=STREAM_MESH,
            subjects=[SUBJ_MESH],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.File if hasattr(StorageType, 'File') else StorageType.FILE,
            max_consumers=-1,
            max_msgs=-1 if not EDGE_MODE else -1,
            max_msgs_per_subject=0 if not EDGE_MODE else EDGE_JS_MAX_MSGS_PER_SUBJECT,
            max_bytes=-1 if not EDGE_MODE else EDGE_JS_MAX_BYTES,
            duplicate_window=60_000_000_000,
            allow_direct=True,
            discard=DiscardPolicy.Old,
            num_replicas=1,
        ),
    )
    # HITL: limits
    await _ensure_stream(
        js,
        STREAM_HITL,
        StreamConfig(
            name=STREAM_HITL,
            subjects=[SUBJ_HITL],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.File if hasattr(StorageType, 'File') else StorageType.FILE,
            max_consumers=-1,
            max_msgs=-1 if not EDGE_MODE else -1,
            max_msgs_per_subject=0 if not EDGE_MODE else EDGE_JS_MAX_MSGS_PER_SUBJECT,
            max_bytes=-1 if not EDGE_MODE else EDGE_JS_MAX_BYTES,
            duplicate_window=60_000_000_000,
            allow_direct=True,
            discard=DiscardPolicy.Old,
            num_replicas=1,
        ),
    )

async def _publish_with_id(js, subj: str, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    # Prefer explicit id header for JS duplicate suppression
    msg_id = (
        payload.get("job_id")
        or payload.get("invocation_id")
        or payload.get("id")
        or payload.get("request_id")
    )
    headers = None
    if msg_id:
        headers = {"Nats-Msg-Id": str(msg_id)}
    await js.publish(subj, data, headers=headers)

async def publish_job(mission: str, payload: Dict[str, Any]) -> None:
    js = await get_js_async()
    subj = f"swarm.jobs.{mission}"
    await _publish_with_id(js, subj, payload)
    # keep connection for subscribers

async def publish_result(mission: str, payload: Dict[str, Any]) -> None:
    js = await get_js_async()
    subj = f"swarm.results.{mission}"
    await _publish_with_id(js, subj, payload)
    # keep connection for subscribers

async def publish_hitl(mission: str, payload: Dict[str, Any]) -> None:
    js = await get_js_async()
    subj = f"swarm.hitl.{mission}"
    await _publish_with_id(js, subj, payload)
    # keep connection for subscribers

async def subscribe_jobs(consumer_name: str, mission: str):
    nc = await get_nc_async()
    js = await get_js_async()
    durable = consumer_name or f"worker-{mission}"
    subject = f"swarm.jobs.{mission}"

    # Ensure stream exists (idempotent)
    try:
        await js.add_stream(
            StreamConfig(
                name=STREAM_JOBS,
                subjects=[SUBJ_JOBS],
                storage=StorageType.File if hasattr(StorageType, 'File') else StorageType.FILE,
                retention=RetentionPolicy.WORK_QUEUE,
            )
        )
    except Exception:
        pass

    # Ensure durable, subject-filtered consumer exists (idempotent)
    if EDGE_MODE:
        cfg = ConsumerConfig(
            durable_name=durable,
            deliver_policy=DeliverPolicy.ALL,
            ack_policy=AckPolicy.EXPLICIT,
            filter_subject=subject,
            max_ack_pending=256,
            ack_wait=_ACK_WAIT_NS,
        )
    else:
        cfg = ConsumerConfig(
            durable_name=durable,
            deliver_policy=DeliverPolicy.ALL,
            ack_policy=AckPolicy.EXPLICIT,
            filter_subject=subject,
            max_ack_pending=128,
        )
    try:
        await js.add_consumer(STREAM_JOBS, cfg)
    except Exception:
        pass

    # Explicit stream avoids discovery errors
    sub = await js.pull_subscribe(subject, durable=durable, stream=STREAM_JOBS)
    logger.info("Subscribed durable consumer for mission=%s", mission)
    return nc, js, sub

async def subscribe_hitl(consumer_name: str, mission: str):
    nc = await get_nc_async()
    js = await get_js_async()
    durable = consumer_name or f"hitl-{mission}"
    subject = f"swarm.hitl.{mission}"

    # Ensure stream exists (idempotent)
    try:
        await js.add_stream(
            StreamConfig(
                name=STREAM_HITL,
                subjects=[SUBJ_HITL],
                storage=StorageType.File if hasattr(StorageType, 'File') else StorageType.FILE,
                retention=RetentionPolicy.LIMITS,
            )
        )
    except Exception:
        pass

    if EDGE_MODE:
        cfg = ConsumerConfig(
            durable_name=durable,
            deliver_policy=DeliverPolicy.ALL,
            ack_policy=AckPolicy.EXPLICIT,
            filter_subject=subject,
            max_ack_pending=512,
            ack_wait=_ACK_WAIT_NS,
        )
    else:
        cfg = ConsumerConfig(
            durable_name=durable,
            deliver_policy=DeliverPolicy.ALL,
            ack_policy=AckPolicy.EXPLICIT,
            filter_subject=subject,
            max_ack_pending=256,
        )
    try:
        await js.add_consumer(STREAM_HITL, cfg)
    except Exception:
        pass

    sub = await js.pull_subscribe(subject, durable=durable, stream=STREAM_HITL)
    logger.info(f"Subscribed HITL durable consumer for mission={mission}")
    return nc, js, sub