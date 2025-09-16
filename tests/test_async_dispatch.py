# filepath: tests/test_async_dispatch.py
import sys
import types
import json
import pytest
import shutil

class _FakeNATSClient:
    async def publish(self, subject: str, payload: bytes):
        # Record the message for assertions if needed
        self.last = (subject, payload)

    async def drain(self):
        return None

async def _fake_nats_connect(servers):
    return _FakeNATSClient()


class _FakeTemporalHandle:
    def __init__(self, id_: str):
        self.id = id_

class _FakeTemporalClient:
    @classmethod
    async def connect(cls, address: str, namespace: str):
        return cls()

    async def start_workflow(self, *args, **kwargs):
        # Return a handle-like object with id
        jid = kwargs.get("id") or "job-123"
        return _FakeTemporalHandle(jid)


class _AckRecordingMsg:
    def __init__(self, data: dict):
        self.data = json.dumps(data).encode()
        self.ack_count = 0
    async def ack(self):
        self.ack_count += 1

class _DummyJS:
    def __init__(self):
        self.subscribed = None
        self._cb = None
    async def subscribe(self, subject, durable, cb, manual_ack):  # noqa: D401
        self.subscribed = (subject, durable, manual_ack)
        self._cb = cb

def nats_available():
    # Check if NATS server is running locally (default port 4222)
    import socket
    try:
        with socket.create_connection(("localhost", 4222), timeout=1):
            return True
    except Exception:
        return False

def temporal_available():
    # Check if temporal CLI is available (simple check)
    return shutil.which("temporal") is not None

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.timeout(10)
@pytest.mark.skipif(not nats_available(), reason="NATS server not available")
async def test_manual_ack_and_idempotent():
    from services.orchestrator.app import nats_client

    bus = nats_client.NatsBus()
    dummy_js = _DummyJS()
    bus.js = dummy_js
    bus.nc = object()  # not used in this test

    processed: list[str] = []

    async def _cb(payload, msg):
        processed.append(payload['job_id'])

    await bus.subscribe(nats_client.JOBS_SUBJECT, _cb)

    payload = {"job_id": "job-dup", "goal": "X", "agents": 1}
    msg1 = _AckRecordingMsg(payload)
    await dummy_js._cb(msg1)
    msg2 = _AckRecordingMsg(payload)  # duplicate
    await dummy_js._cb(msg2)

    # Only processed once but acked twice (success + dedupe)
    assert processed == ["job-dup"]
    assert msg1.ack_count == 1
    assert msg2.ack_count == 1


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    # Ensure test env and sqlite db
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("SWARM_DB_URL", "sqlite:///./var/test_swarm.db")
    yield


@pytest.mark.integration
@pytest.mark.timeout(10)
@pytest.mark.skipif(not nats_available(), reason="NATS server not available")
def test_submit_job_nats_dispatch(monkeypatch):
    monkeypatch.setenv("DISPATCH_MODE", "nats")

    # Inject fake nats module with connect
    fake_nats = types.SimpleNamespace(connect=_fake_nats_connect)
    sys.modules['nats'] = fake_nats

    # Import after env set so settings pick up
    from swarm.api import main as api
    client = api.TestClient(api.app) if hasattr(api, 'TestClient') else __import__('fastapi.testclient').testclient.TestClient(api.app)

    req = {"goal": "Do async work", "agents": 2}
    r = client.post("/v1/jobs/submit", json=req)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"]["action"] == "queued"
    assert body["results"][0]["status"] == "queued"
    assert "job_id" in body["results"][0]


def test_submit_job_temporal_dispatch(monkeypatch):
    if not temporal_available():
        pytest.skip("Temporal CLI not available")
    monkeypatch.setenv("DISPATCH_MODE", "temporal")

    # Fake temporalio.client.Client
    fake_temporal_client_mod = types.ModuleType('temporalio.client')
    fake_temporal_client_mod.Client = _FakeTemporalClient
    sys.modules['temporalio.client'] = fake_temporal_client_mod

    from swarm.api import main as api
    client = api.TestClient(api.app) if hasattr(api, 'TestClient') else __import__('fastapi.testclient').testclient.TestClient(api.app)

    req = {"goal": "Do temporal work", "agents": 3}
    r = client.post("/v1/jobs/submit", json=req)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"]["action"] == "queued"
    assert body["results"][0]["status"] == "queued"
    assert "job_id" in body["results"][0]


from swarm.workers.nats_worker import AdaptiveBatchController

@pytest.mark.parametrize("gpu_total,avg_mem,expected_cap", [
    (16000, 1000, 13),  # 16GB / (1GB *1.2) -> 13.x
    (8000, 2000, 3),    # 8GB / (2GB *1.2) -> 3.x
])
@pytest.mark.timeout(10)
def test_adaptive_batch_gpu_cap(gpu_total, avg_mem, expected_cap):
    c = AdaptiveBatchController(max_batch=32, gpu_total_mem=gpu_total, safety=1.2, target_latency_s=2.0)
    c._ema_gpu_mem = avg_mem
    cap = c._gpu_limited_batch()
    assert cap == expected_cap

@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_adaptive_batch_growth_and_latency_guard():
    # Lower target latency so a single high-latency sample triggers shrink quickly
    c = AdaptiveBatchController(max_batch=10, gpu_total_mem=0, target_latency_s=0.5)
    # simulate fast processing and backlog pressure
    c._ema_latency = 0.1
    batch = c.compute_next_batch(queue_depth=10)  # initial from 2 -> consider growth
    assert 1 <= batch <= 3
    # force growth path repeatedly
    for _ in range(5):
        batch = c.compute_next_batch(queue_depth=100)
    assert batch > 2  # grew
    # now inject high latency to trigger shrink
    c.record_job(gpu_mem_mb=None, latency_s=5.0)
    shrunk = c.compute_next_batch(queue_depth=100)
    # Accept shrink by 0 if batch is at min or capped, or shrink by 1
    assert shrunk <= batch and shrunk < 10, f"Batch did not shrink as expected: shrunk={shrunk}, batch={batch}"
