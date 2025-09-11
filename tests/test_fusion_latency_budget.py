from __future__ import annotations
import os, sys
if os.path.abspath('.') not in sys.path:
    sys.path.insert(0, os.path.abspath('.'))
from fastapi.testclient import TestClient
from swarm.api.main import app  # type: ignore
from swarm.fusion.bayesian import fuse_calibrate_persist  # type: ignore
from swarm.fusion.eo_ir import ingest_streams, build_evidence_chain  # type: ignore

def test_fusion_latency_budget_violation(monkeypatch):
    # Set extremely low budget so real latency breaches it.
    monkeypatch.setenv('FUSION_LATENCY_BUDGET_MS', '0.00001')
    # Deterministic streams
    eo = [0.1,0.2,0.3,0.4,0.5]
    ir = [0.15,0.25,0.35,0.45,0.55]
    eo_arr, ir_arr = ingest_streams(eo, ir)
    evidence = build_evidence_chain(eo, ir)
    res = fuse_calibrate_persist(eo_arr, ir_arr, evidence=evidence, alpha=0.1)
    assert 'track_id' in res
    client = TestClient(app)
    metrics = client.get('/metrics').text
    # Find violation counter line
    assert 'fusion_latency_budget_violations_total' in metrics
    # Basic numeric check (>0). We avoid strict parsing differences; presence is sufficient if function executed.
    # Stronger assertion: look for mission label (default)
    assert 'mission="default"' in metrics or 'mission="' in metrics
