import os
import pytest
from approval import ApprovalManager


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
