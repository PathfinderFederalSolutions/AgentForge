import uuid
from fastapi.testclient import TestClient
from swarm.api.main import app
from swarm import lineage

client = TestClient(app)

def test_evidence_bundle_not_found():
    missing = uuid.uuid4().hex
    resp = client.get(f"/v1/evidence/{missing}")
    assert resp.status_code == 404


def test_evidence_bundle_reproducibility_flags():
    # Start job with a synthetic dag_hash but do NOT persist DAG file
    dag_hash = "deadbeefcafebabe"
    job_id = lineage.start_job(goal="Check reproducibility flags", dag_hash=dag_hash)
    lineage.complete_job(job_id, {"approved": True, "metrics": {"confidence": 0.5}}, [{"ok": True}])
    resp = client.get(f"/v1/evidence/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dag_hash"] == dag_hash
    repro = data.get("reproducibility", {})
    assert repro.get("dag_path_exists") is False
    # Ensure structure fields present
    assert "citations" in data
    assert "confidence" in data


def test_evidence_bundle_with_persisted_dag():
    # Persist an actual DAG and ensure reproducibility flags reflect existence
    import hashlib
    from swarm.planner import Planner

    goal = "Evidence dag persistence test"
    seed = int(hashlib.sha256(goal.encode('utf-8')).hexdigest()[:8], 16)
    dag = Planner().make_dag(goal, seed=seed)
    dag_hash = dag.compute_hash()
    job_id = lineage.start_job(goal=goal, dag_hash=dag_hash)
    # Persist DAG linked to job
    lineage.persist_dag(dag, job_id=job_id)
    lineage.complete_job(job_id, {"approved": True, "metrics": {"confidence": 0.9}}, [{"ok": True}])
    resp = client.get(f"/v1/evidence/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dag_hash"] == dag_hash
    repro = data.get("reproducibility", {})
    assert repro.get("dag_path_exists") is True
    assert repro.get("dag_path", "").endswith(f"{dag_hash}.dag.json")
    # Ensure dag_persisted event captured
    assert any(ev.get("event_type") == "dag_persisted" for ev in data.get("events", []))
