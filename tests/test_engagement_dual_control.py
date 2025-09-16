# tests/test_engagement_dual_control.py
import os
import pytest
from fastapi.testclient import TestClient
from services.engagement.app.main import app, ARTIFACTS_DIR

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Remove created artifacts after test
    for root, dirs, files in os.walk(ARTIFACTS_DIR):
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except Exception:
                pass


def test_dual_approval_success(monkeypatch):
    # Simulate dual approval
    monkeypatch.setattr("services.hitl.app.require_dual_approval", lambda req, pid: True)
    payload = {
        "target_metadata": {"target": "A"},
        "recommended_coa": "COA1",
        "roe_checks": {"check": True},
        "evidence_list": ["ev1", "ev2"]
    }
    resp = client.post("/engage", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["approved"] is True
    assert data["signed"] is True
    assert data["engagement_time_to_decision_seconds"] >= 0


def test_dual_approval_denied(monkeypatch):
    # Simulate denial
    monkeypatch.setattr("services.hitl.app.require_dual_approval", lambda req, pid: False)
    payload = {
        "target_metadata": {"target": "B"},
        "recommended_coa": "COA2",
        "roe_checks": {"check": False},
        "evidence_list": ["ev3"]
    }
    resp = client.post("/engage", json=payload)
    assert resp.status_code == 403
    assert "Dual approval required" in resp.text
