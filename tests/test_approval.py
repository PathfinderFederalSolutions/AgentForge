import pytest
from approval import ApprovalManager
from fastapi.testclient import TestClient
from swarm.api.main import app
from swarm import lineage

client = TestClient(app)

def test_disabled_returns_approved(monkeypatch):
    monkeypatch.setenv("AF_APPROVAL_ENABLE", "0")
    mgr = ApprovalManager()
    d = mgr.check_and_gate({"id": "t1", "description": "safe op"}, "ok", "general")
    assert d["approved"] is True
    assert d["escalated"] is False


def test_escalates_high_impact_low_conf_and_autoapproves(monkeypatch):
    monkeypatch.setenv("AF_APPROVAL_ENABLE", "1")
    monkeypatch.setenv("AF_HITL_AUTOAPPROVE", "1")
    monkeypatch.setenv("AF_APPROVAL_REQUIRE_STRICT", "0")
    mgr = ApprovalManager()
    task = {"id": "t2", "description": "payment deploy", "metadata": {"impact": "high"}}
    decision = mgr.check_and_gate(task, "Error: failed", "Dynamic Agent Lifecycle")
    assert decision["escalated"] is True
    assert decision["approved"] is True
    assert decision["reason"] == "autoapproved"


def test_strict_blocks_without_autoapprove(monkeypatch):
    monkeypatch.setenv("AF_APPROVAL_ENABLE", "1")
    monkeypatch.setenv("AF_HITL_AUTOAPPROVE", "0")
    monkeypatch.setenv("AF_APPROVAL_REQUIRE_STRICT", "1")
    mgr = ApprovalManager()
    task = {"id": "t3", "description": "prod payment deploy", "metadata": {"impact": "critical"}}
    with pytest.raises(ValueError):
        mgr.check_and_gate(task, "Error: boom", "Memory Mesh")


def test_evidence_bundle_endpoint(monkeypatch):
    monkeypatch.setenv("AF_APPROVAL_ENABLE", "0")
    # Start and complete a synthetic job to ensure lineage populated
    job_id = lineage.start_job(goal="Test evidence bundle goal")
    lineage.complete_job(job_id, {"approved": True, "metrics": {"confidence": 0.87}, "citations": ["ref1"]}, [{"capability": "x", "result": "ok"}])
    resp = client.get(f"/v1/evidence/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["confidence"] == 0.87
    assert data["citations"] == ["ref1"]
    assert isinstance(data["events"], list) and any(e["event_type"] == "job_completed" for e in data["events"])
    # reproducibility info present
    assert "reproducibility" in data and "dag_path_exists" in data["reproducibility"]
