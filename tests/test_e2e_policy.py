import os
from fastapi.testclient import TestClient

# Force test env and sync dispatch
os.environ["ENV"] = "test"
os.environ["DISPATCH_MODE"] = "sync"
os.environ["SWARM_DB_URL"] = "sqlite:///./var/test_swarm.db"

from swarm.api import main as api  # noqa: E402

# Fake orchestrator to produce approved results
class FakeOrchestrator:
    def __init__(self, num_agents=2):
        self.num_agents = num_agents

    def run_goal_sync(self, goal):
        return [
            {"id": "st1", "provider": "test", "result": "All good."},
            {"id": "st2", "provider": "test", "result": "All good too."},
        ]

def fake_build_orchestrator(num_agents=2):
    return FakeOrchestrator(num_agents=num_agents)

# Patch orchestrator and create client
api.build_orchestrator = fake_build_orchestrator
client = TestClient(api.app)

def test_upload_and_submit_approved(tmp_path):
    # Upload a small “archive”
    data = b"print('hello')"
    files = {"file": ("code.zip", data, "application/zip")}
    r = client.post("/v1/artifacts/upload", files=files)
    assert r.status_code == 200
    artifact = r.json()

    req = {
        "goal": "Review this code and produce a secure, production-ready result.",
        "agents": 2,
        "artifacts": [artifact],
    }
    r = client.post("/v1/jobs/submit", json=req)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"]["approved"] is True
    assert body["decision"]["metrics"]["error_rate"] == 0.0

def test_submit_rejected_when_errors(tmp_path, monkeypatch):
    # Orchestrator emitting an error result triggers rejection
    class ErrOrch(FakeOrchestrator):
        def run_goal_sync(self, goal):
            return [
                {"id": "st1", "provider": "test", "result": "Error: failed A"},
                {"id": "st2", "provider": "test", "result": "ok"},
            ]

    api.build_orchestrator = lambda num_agents=2: ErrOrch(num_agents)

    req = {"goal": "Do work", "agents": 2}
    r = client.post("/v1/jobs/submit", json=req)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"]["approved"] is False
    assert body["decision"]["metrics"]["error_count"] == 1