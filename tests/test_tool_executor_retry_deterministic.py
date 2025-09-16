# filepath: tests/test_tool_executor_retry_deterministic.py
from __future__ import annotations
import asyncio
import json
import types
from typing import Any, Dict

import pytest


class DummyMsg:
    def __init__(self, data: bytes):
        self.data = data
        self.acked = False
    async def ack(self):
        self.acked = True


class DummyJS:
    def __init__(self):
        self.published: list[tuple[str, bytes]] = []
        self.subscribed = None
        self.info = types.SimpleNamespace(num_pending=0)
    async def publish(self, subj: str, data: bytes, headers=None):
        self.published.append((subj, data))
    async def add_stream(self, *args, **kwargs):
        return None
    async def subscribe(self, subj: str, durable: str, cb, manual_ack: bool, ack_wait: int):
        self.subscribed = (subj, durable, cb, manual_ack, ack_wait)
        self._cb = cb
        return types.SimpleNamespace(messages=asyncio.Queue())
    async def consumer_info(self, stream: str, consumer: str):
        return self.info


@pytest.mark.asyncio
async def test_deterministic_retry_flow(monkeypatch):
    import swarm.workers.tool_executor as te

    async def _connect(self, servers):
        return None

    def _jetstream(self):
        return js

    js = DummyJS()
    monkeypatch.setattr(te.NATS, 'connect', _connect, raising=False)
    monkeypatch.setattr(te.NATS, 'jetstream', _jetstream, raising=False)

    # Deterministic hooks
    te.set_backoff_override(lambda attempt: 1)
    async def _no_sleep(sec: float):
        return None
    te.set_sleep_override(_no_sleep)

    calls = {"n": 0}
    @te.ToolRegistry.register('flaky2')
    async def _flaky2(args: Dict[str, Any]):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return {"ok": True}

    inv = {
        "invocation_id": "xyz-123",
        "task_id": "t-d1",
        "tool": "flaky2",
        "args": {},
        "attempt": 1,
        "max_attempts": 3,
    }

    task = asyncio.create_task(te.main())
    await asyncio.sleep(0.05)
    cb = js._cb

    # First attempt triggers retry
    msg1 = DummyMsg(json.dumps(inv).encode())
    await cb(msg1)
    assert msg1.acked
    # A retry publish should have happened to tools.invocations.flaky2
    assert any(s.startswith('tools.invocations.flaky2') for s,_ in js.published)

    # Second attempt
    inv2 = dict(inv)
    inv2["attempt"] = 2
    msg2 = DummyMsg(json.dumps(inv2).encode())
    await cb(msg2)
    # Should schedule a second retry
    assert any(s.startswith('tools.invocations.flaky2') for s,_ in js.published)

    # Third attempt succeeds and publishes result
    inv3 = dict(inv)
    inv3["attempt"] = 3
    msg3 = DummyMsg(json.dumps(inv3).encode())
    await cb(msg3)
    assert any(s.startswith('tools.results.') for s,_ in js.published)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
