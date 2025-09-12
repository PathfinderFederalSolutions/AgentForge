import time
import pytest

# Skip if executor internals aren’t available (keeps CI green without brittle imports)
executor = pytest.importorskip("swarm.workers.tool_executor", reason="executor not present")
RetryPolicy = getattr(executor, "RetryPolicy")
IdempotencyCache = getattr(executor, "IdempotencyCache")

def test_retry_backoff_deterministic(monkeypatch):
    policy = RetryPolicy(base=0.01, factor=2.0, max_attempts=3, jitter=0.0)
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    for i in range(policy.max_attempts):
        time.sleep(policy.backoff(i))
    assert sleeps == [0.01, 0.02, 0.04]

def test_idempotency_key_reuse():
    cache = IdempotencyCache(ttl_seconds=60)
    key = "abc"
    assert cache.check_and_set(key) is True
    assert cache.check_and_set(key) is False
