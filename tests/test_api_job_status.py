import pytest
from fastapi.testclient import TestClient
from swarm.api.main import app
import socket

client = TestClient(app)

def _is_nats_running(host="127.0.0.1", port=4222):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except Exception:
        return False

@pytest.mark.skipif(not _is_nats_running(), reason="Skipping job status test: NATS not running locally")
@pytest.mark.integration
@pytest.mark.timeout(10)
def test_job_status_includes_dag_hash():
    payload = {"goal": "Test deterministic DAG goal", "agents": 2}
    r = client.post("/job/sync", json=payload)
    assert r.status_code == 200
    data = r.json()
    dag_hash = data.get('dag_hash')
    assert dag_hash
    # Sync endpoint doesn't return job_id directly, so create async job to test status endpoint
    r_async = client.post("/job/async", json=payload)
    assert r_async.status_code == 200
    job_id = r_async.json().get('job_id')
    assert job_id
    status = client.get(f"/job/{job_id}")
    assert status.status_code == 200
    js = status.json()
    assert js.get('dag_hash')
