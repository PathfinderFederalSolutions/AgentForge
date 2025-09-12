import pytest
metrics_mod = pytest.importorskip("swarm.observability.costs", reason="metrics module not present")
wrap_llm_call = getattr(metrics_mod, "wrap_llm_call")

class Resp:
    def __init__(self, prompt_tokens, completion_tokens, cost_usd):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost_usd = cost_usd

def test_cost_and_tokens_counters():
    def fake_call(x): return Resp(10, 20, 0.005)
    wrapped = wrap_llm_call("openai", "gpt-5", fake_call)
    r = wrapped("hi")
    assert isinstance(r, Resp)

# --- New test for router v2 policy cost enforcement ---
from router_v2 import MoERouter, Provider

def _make_router():
    r = MoERouter(epsilon=0.0)
    mock = Provider(key="mock", model="mock", capabilities={"general"}, cost_per_1k=0.0)
    mock.client = type("C", (), {"invoke": lambda self, msgs=None: type("R", (), {"content": "m", "prompt_tokens": 5, "completion_tokens": 5})()})()
    mock.instrument()
    paid = Provider(key="paid", model="m1", capabilities={"general"}, cost_per_1k=1.0)
    paid.client = type("C", (), {"invoke": lambda self, msgs=None: type("R", (), {"content": "p", "prompt_tokens": 80, "completion_tokens": 20})()})()
    paid.instrument()
    r.register(mock)
    r.register(paid)
    return r

def test_policy_cost_budget_enforced(monkeypatch, tmp_path):
    policy_path = tmp_path / "policies.json"
    policy_path.write_text('[{"name":"tight","task_regex":".*","max_cost_usd":0.25,"priority":5}]')
    monkeypatch.setenv("ROUTER_POLICY_PATH", str(policy_path))
    from importlib import reload
    import swarm.router_policy_loader as loader
    reload(loader)
    r = _make_router()
    paid = r.providers["paid"]
    for _ in range(3):  # each ~0.1
        paid.call([{"role":"user","content":"hi"}])
    chosen = r.route("generic task")
    assert chosen == "mock"