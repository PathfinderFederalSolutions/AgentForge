# tests/test_cds_hash_verification.py
import hashlib
import json
import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../services/cds-bridge/app')))
import main

client = TestClient(main.app)

@pytest.fixture
def evidence_bundle():
    bundle = {"id": "test123", "data": "evidence"}
    banner = "TOP SECRET"
    sha = hashlib.sha256(json.dumps(bundle).encode()).hexdigest()
    return bundle, banner, sha

def test_cds_transfer_and_verify(evidence_bundle):
    bundle, banner, sha = evidence_bundle
    # Transfer
    resp = client.post("/transfer", json={"bundle": bundle, "banner": banner})
    assert resp.status_code == 200
    receipt = resp.json()
    assert receipt["sha256"] == sha
    # Verify
    resp2 = client.post("/verify", json={"bundle": bundle, "sha256": sha})
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "verified"

def test_cds_verify_corrupted_payload(evidence_bundle):
    bundle, banner, sha = evidence_bundle
    corrupted = dict(bundle)
    corrupted["data"] = "tampered"
    resp = client.post("/verify", json={"bundle": corrupted, "sha256": sha})
    assert resp.status_code == 400
    assert "SHA256 verification failed" in resp.text
