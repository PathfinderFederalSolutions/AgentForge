from __future__ import annotations
import os, sys
if os.path.abspath('.') not in sys.path:
    sys.path.insert(0, os.path.abspath('.'))
from swarm.canary import router as canary  # type: ignore
from fastapi.testclient import TestClient
from swarm.api.main import app  # type: ignore

def test_canary_eval_latency_budget_violation(monkeypatch):
    # Extremely low budget so evaluation passes threshold
    monkeypatch.setenv('CANARY_EVAL_LATENCY_BUDGET_MS', '0.00001')
    canary.start_canary(target_fraction=0.1)
    # Provide minimal observations to allow evaluation logic to execute quickly
    for _ in range(35):
        canary.record_observation(False, 50, False, 0.95)
        canary.record_observation(True, 50, False, 0.95)
    res = canary.maybe_progress()
    assert 'phase' in res
    client = TestClient(app)
    metrics = client.get('/metrics').text
    assert 'canary_eval_latency_budget_violations_total' in metrics
