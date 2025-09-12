from __future__ import annotations
from typing import Any
import uuid

from swarm.reviewer import review_results, review_tool_result
from fastapi.testclient import TestClient
from services.hitl.app import app as hitl_app  # type: ignore
from swarm.learning.feedback import list_feedback  # type: ignore


def test_review_results_annotations_no_heal():
    results = [
        {"result": "All good"},
        {"result": "Error: something went wrong"},
    ]
    out = review_results(results, auto_heal=False)
    assert len(out) == 2
    assert out[0]["validation"]["ok"] is True
    assert out[0]["validation"]["confidence"] == 0.9
    assert out[1]["validation"]["ok"] is False
    assert out[1]["validation"]["confidence"] == 0.4
    # no healing when auto_heal=False
    assert out[1]["result"].startswith("Error:")


def test_review_results_auto_heal():
    results = [
        {"output": "Error: fail"},
    ]
    out = review_results(results, auto_heal=True)
    assert out[0]["validation"]["ok"] is True
    assert out[0]["validation"]["confidence"] == 0.6
    assert out[0]["result"].startswith("Healed:")


class DummyRes:
    def __init__(self, output: Any):
        self.output = output
        self.metadata = {}


def test_review_tool_result_annotations():
    r = DummyRes("Hello")
    rr = review_tool_result(r, auto_heal=False)
    assert rr.metadata["review"]["ok"] is True
    assert rr.metadata["review"]["confidence"] == 0.9

    r2 = DummyRes("Error: nope")
    rr2 = review_tool_result(r2, auto_heal=True)
    assert rr2.metadata["review"]["ok"] is True
    assert rr2.metadata["review"]["confidence"] == 0.6
    assert str(rr2.output).startswith("Healed:")


def test_adjudication_persists_feedback():
    client = TestClient(hitl_app)
    before = len(list_feedback())
    subj = f"track-{uuid.uuid4().hex[:8]}"
    resp = client.post('/adjudications', json={
        'subject_id': subj,
        'decision': 'approve',
        'rationale': 'Looks good',
        'user_id': 'tester',
        'evidence_refs': ['ev1','ev2']
    })
    assert resp.status_code == 200
    after = len(list_feedback())
    assert after == before + 1
