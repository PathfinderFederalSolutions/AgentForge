# tests/test_imports.py (expand)
import pytest
import forge_types  # Custom types
import router  # New
import orchestrator
import agents
from router import DynamicRouter

def test_no_circular_imports():
    router_obj = DynamicRouter()
    assert router_obj
    from agents import Agent
    agent = Agent(forge_types.AgentContract(name="test", capabilities=[], memory_scopes=[], tools=[], budget=1000))
    assert agent.router  # Uses DynamicRouter without circle

# --- New policy-driven routing tests for MoERouter v2 ---
from router_v2 import MoERouter, Provider

@pytest.fixture(autouse=True)
def _seed():
    import random
    random.seed(42)


def _make_router():
    r = MoERouter(epsilon=0.0)
    p_local = Provider(key="mock", model="mock-model", capabilities={"general"}, cost_per_1k=0.0)
    p_local.client = type("C", (), {"invoke": lambda self, msgs=None: type("R", (), {"content": "ok", "prompt_tokens": 10, "completion_tokens": 20})()})()
    p_local.instrument()
    p_paid = Provider(key="openai", model="gpt-4o-mini", capabilities={"general"}, cost_per_1k=0.01)
    p_paid.client = type("C", (), {"invoke": lambda self, msgs=None: type("R", (), {"content": "ok", "prompt_tokens": 1000, "completion_tokens": 0})()})()
    p_paid.instrument()
    r.register(p_local)
    r.register(p_paid)
    return r


def test_policy_local_sensitive(tmp_path, monkeypatch):
    policy_path = tmp_path / "policies.json"
    policy_path.write_text('[{"name":"local_only","task_regex":"(?i)sensitive secret","sensitive_local_only":true,"allowed_providers":["mock"],"priority":50}]')
    monkeypatch.setenv("ROUTER_POLICY_PATH", str(policy_path))
    from importlib import reload
    import swarm.router_policy_loader as loader
    reload(loader)
    r = _make_router()
    chosen = r.route("This is a Sensitive Secret task")
    assert chosen == "mock"


def test_policy_cost_cap(monkeypatch, tmp_path):
    policy_path = tmp_path / "policies.json"
    policy_path.write_text('[{"name":"cap","task_regex":".*","max_cost_usd":0.005,"priority":1}]')
    monkeypatch.setenv("ROUTER_POLICY_PATH", str(policy_path))
    from importlib import reload
    import swarm.router_policy_loader as loader
    reload(loader)
    r = _make_router()
    # Simulate heavy usage on paid provider to exceed cap
    r.providers["openai"].cost_spent = 0.01
    chosen2 = r.route("another generic task")
    assert chosen2 == "mock"


def test_latency_filter(monkeypatch, tmp_path):
    policy_path = tmp_path / "policies.json"
    policy_path.write_text('[{"name":"lat","task_regex":".*","max_latency_ms":10,"priority":5}]')
    monkeypatch.setenv("ROUTER_POLICY_PATH", str(policy_path))
    from importlib import reload
    import swarm.router_policy_loader as loader
    reload(loader)
    r = _make_router()
    r.providers["openai"].avg_latency_ms = 50
    chosen = r.route("generic task for latency")
    assert chosen == "mock"


def test_router_span_tagging():
    r = MoERouter(epsilon=0.0)
    r.register(Provider(key='mock', model='mock', capabilities={'general'}))
    out = r.route('simple general task')
    assert out == 'mock'
