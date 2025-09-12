from fastapi.testclient import TestClient
from swarm.api.main import app
from swarm.observability.task_latency import record_dag_completion, dag_budget_violation  # type: ignore


def test_metrics_endpoint():
    client = TestClient(app)
    resp = client.get("/metrics")
    # prometheus_client should be installed per requirements; accept 200 or 503 if registry unavailable
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    body = resp.text
    assert ("python_" in body) or ("process_" in body)


def test_dag_latency_budget_metric():
    before = dag_budget_violation._value.get()  # type: ignore[attr-defined]
    record_dag_completion(1500, 1000)
    after = dag_budget_violation._value.get()  # type: ignore[attr-defined]
    assert after == before + 1
