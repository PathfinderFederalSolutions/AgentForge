# filepath: tests/test_async_dispatch.py
import os
import sys
import types
import asyncio
import json
import pytest


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


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    # Ensure test env and sqlite db
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("SWARM_DB_URL", "sqlite:///./var/test_swarm.db")
    yield


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
