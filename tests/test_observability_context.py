# filepath: tests/test_observability_context.py
import pytest
from fastapi.testclient import TestClient
from swarm.api.main import app
from swarm.observability.task_latency import record_task_completion  # type: ignore
from sla_kpi_config import get_task_budget  # type: ignore

metrics_mod = pytest.importorskip("swarm.observability.costs")
otel_mod = pytest.importorskip("swarm.observability.otel")

set_ctx = getattr(metrics_mod, "set_observability_context")
wrap_llm_call = getattr(metrics_mod, "wrap_llm_call")

# Minimal fake response
class _Resp:
    def __init__(self):
        self.prompt_tokens = 3
        self.completion_tokens = 7
        self.cost_usd = 0.001


def test_cost_metrics_with_context():
    def _call():
        return _Resp()
    wrapped = wrap_llm_call("openai","gpt-x", _call)
    # set only mission first
    set_ctx(mission_id="m-alpha")
    wrapped()
    # then set mission+task
    set_ctx(mission_id="m-alpha", task_id="task-123")
    wrapped()
    # No assertion on metrics registry values (would require parsing)
    # Ensure wrapper returns object
    assert isinstance(wrapped(), _Resp)
    # Scrape metrics and ensure mission_id/task_id label keys appear for llm metrics
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "llm_tokens_total" in body
    assert "mission_id=\"m-alpha\"" in body  # mission label emitted
    assert "task_id=\"task-123\"" in body  # task label emitted


def test_tracing_tag_helper():
    tag_span = getattr(otel_mod, "tag_span")
    # Should be no-op without active span
    tag_span(mission_id="m1", task_id="t1")
    # Just ensure callable
    assert callable(tag_span)


def test_task_latency_metrics_recording():
    b = get_task_budget("default")
    record_task_completion(0.2, "default", "m-beta", b.name, b.p99_ms, b.hard_cap_ms)
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "task_latency_seconds" in body
    assert "task_latency_budget_violations_total" in body  # may be zero counter
