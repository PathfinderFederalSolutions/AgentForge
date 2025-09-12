import os, sys
# Ensure project root on path for direct pytest invocation environments
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from swarm.fusion.eo_ir import ingest_streams, build_evidence_chain
from swarm.fusion.bayesian import bayesian_fuse, fuse_calibrate_persist
from swarm.fusion.conformal import conformal_validate
from swarm.storage import load_fused_track
from swarm.api.metrics import FUSION_LATENCY_MS, FUSION_CONFIDENCE, FUSION_EVIDENCE_LEN, FUSION_FUSED_TRACKS  # type: ignore
from fastapi.testclient import TestClient
from swarm.api.main import app

def test_fusion_and_conformal_pipeline():
    eo = [1,2,3,4,5,6,7,8]
    ir = [2,3,4,5,6,7,8,9]
    eo_arr, ir_arr = ingest_streams(eo, ir)
    mu, var = bayesian_fuse(eo_arr, ir_arr)
    res = (np.asarray(eo_arr) - mu).tolist()
    lo, hi = conformal_validate(res, alpha=0.1)
    assert var > 0
    assert hi >= 0 and lo <= 0


def test_fuse_calibrate_persist_track():
    eo = [0.1, 0.2, 0.25, 0.3, 0.5, 0.55, 0.6]
    ir = [0.05, 0.15, 0.22, 0.28, 0.52, 0.58, 0.62]
    eo_arr, ir_arr = ingest_streams(eo, ir)
    evidence = build_evidence_chain(eo, ir)
    fused = fuse_calibrate_persist(eo_arr, ir_arr, evidence=evidence, alpha=0.1)
    assert 'track_id' in fused
    assert fused['confidence'] <= 1.0 and fused['confidence'] > 0.0
    assert isinstance(fused['covariance'], list)
    assert fused['covariance'][0][0] == fused['var']
    assert fused['latency_ms'] < 1000  # sanity latency budget
    track = load_fused_track(fused['track_id'])
    assert track is not None
    assert track['track_id'] == fused['track_id']
    assert len(track['evidence']) == len(evidence)
    # Metrics sanity (if prometheus client available)
    if FUSION_LATENCY_MS:
        client = TestClient(app)
        metrics_body = client.get('/metrics').text
        assert 'fusion_pipeline_latency_ms' in metrics_body
        assert 'fusion_fused_tracks_total' in metrics_body
        assert 'fusion_confidence_bucket' in metrics_body
        assert 'fusion_evidence_length_bucket' in metrics_body