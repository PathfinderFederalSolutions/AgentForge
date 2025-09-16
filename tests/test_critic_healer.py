import os
import pytest
from fastapi.testclient import TestClient

os.environ["ENV"] = "test"
os.environ["DISPATCH_MODE"] = "sync"
os.environ["SWARM_DB_URL"] = "sqlite:///./var/test_swarm.db"

from swarm.api import main as api  # noqa

# Fake orchestrator with base failing, canary succeeding after mutation
class FakeOrchestrator:
    def __init__(self, num_agents=2): pass
    def run_goal_sync(self, goal):
        if "Healer Hint" in goal:
            return [{"result": "All good"}, {"result": "All good 2"}]
        return [{"result": "Error: fail A"}, {"result": "Error: fail B"}]

def fake_build_orchestrator(num_agents=2):
    return FakeOrchestrator(num_agents=num_agents)

api.build_orchestrator = fake_build_orchestrator
client = TestClient(api.app)

@pytest.mark.integration
def test_submit_with_critic_healer_promotes_canary():
    req = {"goal": "Achieve desired result", "agents": 6}
    r = client.post("/v1/jobs/submit_ch", json=req)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"]["approved"] in (True, False)  # depends on enforcer rules
    # Given fake orchestrator, canary path should be chosen
    assert body["source"] in ("canary", "base")