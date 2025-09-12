import asyncio
import json
import os
import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio

EDGE_ENV = {
    "EDGE_MODE": "1",
    "EDGE_JS_MAX_MSGS_PER_SUBJECT": "1000",
    "EDGE_JS_MAX_BYTES": str(10 * 1024 * 1024),
}

async def _nats_available() -> bool:
    try:
        import nats  # type: ignore
        nc = await nats.connect(os.getenv("NATS_URL", "nats://localhost:4222"), connect_timeout=0.5)
        await nc.close()
        return True
    except Exception:
        return False

async def _connect_nats():
    import nats
    nc = await nats.connect(os.getenv("NATS_URL", "nats://localhost:4222"))
    return nc, nc.jetstream()

async def _add_stream(js):
    from nats.js.api import StreamConfig, RetentionPolicy, StorageType
    try:
        await js.add_stream(StreamConfig(
            name="swarm_jobs",
            subjects=["swarm.jobs.test"],
            retention=RetentionPolicy.WORK_QUEUE,
            storage=StorageType.FILE,
        ))
    except Exception:
        pass

async def _purge(js):
    try:
        await js.purge_stream("swarm_jobs")
    except Exception:
        pass

async def _recreate_stream(js):
    from nats.js.api import StreamConfig, RetentionPolicy, StorageType
    try:
        await js.delete_stream("swarm_jobs")
    except Exception:
        pass
    await js.add_stream(StreamConfig(
        name="swarm_jobs",
        subjects=["swarm.jobs.test"],
        retention=RetentionPolicy.WORK_QUEUE,
        storage=StorageType.FILE,
    ))

async def publish_with_id(js, subj, payload):
    msg_id = payload.get("job_id") or payload.get("invocation_id") or payload.get("id") or payload.get("request_id")
    headers = {"Nats-Msg-Id": str(msg_id)} if msg_id else None
    return await js.publish(subj, json.dumps(payload).encode(), headers=headers)

async def _subscribe_pull(js, subj, durable):
    from nats.js.api import ConsumerConfig, DeliverPolicy, AckPolicy
    try:
        await js.add_consumer("swarm_jobs", ConsumerConfig(
            durable_name=durable,
            filter_subject=subj,
            deliver_policy=DeliverPolicy.ALL,
            ack_policy=AckPolicy.EXPLICIT,
            max_ack_pending=256,
        ))
    except Exception:
        pass
    return await js.pull_subscribe(subj, durable=durable, stream="swarm_jobs")

async def _recv_batch(sub, n=10, timeout=1.0):
    msgs = await sub.fetch(n, timeout=timeout)
    return msgs

async def _await_pending(js, stream: str, durable: str, at_least: int, timeout_s: float = 5.0):
    start = asyncio.get_event_loop().time()
    while True:
        try:
            info = await js.consumer_info(stream, durable)
            if getattr(info, 'num_pending', 0) >= at_least:
                return True
        except Exception:
            pass
        if asyncio.get_event_loop().time() - start > timeout_s:
            return False
        await asyncio.sleep(0.1)

@pytest_asyncio.fixture(autouse=True)
async def set_edge_env(monkeypatch):
    for k,v in EDGE_ENV.items():
        monkeypatch.setenv(k, v)
    yield

async def test_persistence_and_replay_without_duplication(monkeypatch):
    print("[test] start")
    if not await _nats_available():
        pytest.skip("NATS not available on localhost:4222")

    nc, js = await _connect_nats()
    print("[test] connected")
    await _recreate_stream(js)
    print("[test] stream recreated")

    subj = "swarm.jobs.test"
    durable = "edge_tester"
    await _subscribe_pull(js, subj, durable)
    print("[test] subscribed")

    # Publish 5 messages with ids
    payloads = []
    for i in range(5):
        p = {"job_id": f"edge-{i}", "goal": "edge test", "i": i}
        await publish_with_id(js, subj, p)
        payloads.append(p)
    print("[test] published 5")

    # Disconnect link simulation: close connection, then reconnect
    await nc.close()
    print("[test] closed initial nc")
    # Wait briefly to ensure disconnect
    await asyncio.sleep(0.5)

    # Reconnect
    nc2, js2 = await _connect_nats()
    print("[test] reconnected")
    sub2 = await _subscribe_pull(js2, subj, durable)
    print("[test] resubscribed")

    # Wait for pending messages to be visible to the durable after reconnect
    await _await_pending(js2, "swarm_jobs", durable, at_least=5, timeout_s=5.0)

    # Fetch all 5
    msgs = await _recv_batch(sub2, n=10, timeout=2.0)
    print(f"[test] fetched {len(msgs)} initial")
    ids = set()
    for m in msgs:
        d = json.loads(m.data.decode())
        ids.add(d.get("job_id"))
        # Prefer synchronous ack to ensure server processes ack before next phase
        try:
            await m.ack_sync(timeout=1.5)  # type: ignore[attr-defined]
        except Exception:
            await m.ack()
    # Ensure all acks are flushed to the server before continuing
    try:
        await nc2.flush(timeout=2)
    except Exception:
        pass

    assert ids == {f"edge-{i}" for i in range(5)}
    print("[test] ids ok")

    # Give the server a brief moment to settle ack state
    await asyncio.sleep(0.1)

    # Replay publish same IDs to ensure no duplication (dedupe window default 60-120s)
    dup_flags = []
    for p in payloads:
        ack = await publish_with_id(js2, subj, p)
        try:
            # nats-py PubAck has attribute 'duplicate' in recent versions
            if hasattr(ack, 'duplicate'):
                dup_flags.append(bool(getattr(ack, 'duplicate')))
            elif isinstance(ack, dict) and 'duplicate' in ack:
                dup_flags.append(bool(ack['duplicate']))
        except Exception:
            pass
    print("[test] republished duplicates")

    if dup_flags:
        print(f"[test] dup flags: {dup_flags}")
        assert all(dup_flags), "Republish PubAck must indicate duplicate"
        # Optional: still attempt fetch to ensure nothing pending
        try:
            msgs2 = await sub2.fetch(1, timeout=0.5)
            got = len(msgs2)
        except Exception:
            got = 0
        print(f"[test] duplicate fetch got={got}")
        assert got == 0
    else:
        # Fallback path for older clients without duplicate flag
        try:
            msgs2 = await sub2.fetch(1, timeout=1.0)
            got = len(msgs2)
        except Exception:
            got = 0
        print(f"[test] duplicate fetch got={got}")
        assert got == 0

async def test_edge_link_status_metric_and_gauge_toggle(monkeypatch):
    # Import orchestrator bus and metrics registry
    from services.orchestrator.app.nats_client import bus, EDGE_LINK_GAUGE, SITE
    try:
        from swarm.api import metrics as m  # type: ignore
    except Exception:
        pytest.skip("metrics module unavailable")

    if EDGE_LINK_GAUGE is None:
        pytest.skip("prometheus_client not installed")

    if not await _nats_available():
        pytest.skip("NATS not available on localhost:4222")

    # Connect
    await bus.connect()
    # Gauge should be 1
    val = EDGE_LINK_GAUGE.labels(site=SITE)._value.get()  # type: ignore
    assert float(val) == 1.0
    # Metrics payload should contain edge_link_status 1
    if getattr(m, "generate_latest", None) and getattr(m, "_REGISTRY", None):
        payload = m.generate_latest(m._REGISTRY).decode()
        assert f'edge_link_status{{site="{SITE}"}} 1' in payload

    # Force disconnect
    if bus.nc:
        await bus.nc.close()
    await asyncio.sleep(0.5)

    # Wait for gauge to flip
    for _ in range(20):
        val = EDGE_LINK_GAUGE.labels(site=SITE)._value.get()  # type: ignore
        if float(val) == 0.0:
            break
        await asyncio.sleep(0.2)
    assert float(val) == 0.0
    # Metrics payload should reflect 0
    if getattr(m, "generate_latest", None) and getattr(m, "_REGISTRY", None):
        payload = m.generate_latest(m._REGISTRY).decode()
        assert f'edge_link_status{{site="{SITE}"}} 0' in payload

    # Reconnect and confirm 1 again
    await bus.connect()
    val = EDGE_LINK_GAUGE.labels(site=SITE)._value.get()  # type: ignore
    assert float(val) == 1.0
    if getattr(m, "generate_latest", None) and getattr(m, "_REGISTRY", None):
        payload = m.generate_latest(m._REGISTRY).decode()
        assert f'edge_link_status{{site="{SITE}"}} 1' in payload
