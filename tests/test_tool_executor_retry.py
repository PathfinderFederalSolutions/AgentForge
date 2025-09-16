from __future__ import annotations
import asyncio
import json
import types
from typing import Any, Dict

import pytest

# We will import the handler by executing the module and capturing inner functions via monkeypatching


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
    async def add_stream(self, *args, **kwargs):  # no-op
        return None
    async def subscribe(self, subj: str, durable: str, cb, manual_ack: bool, ack_wait: int):
        self.subscribed = (subj, durable, cb, manual_ack, ack_wait)
        # keep a reference for tests
        self._cb = cb
        return types.SimpleNamespace(messages=asyncio.Queue())
    async def consumer_info(self, stream: str, consumer: str):
        return self.info


class DummyNC:
    def __init__(self, js: DummyJS):
        self._js = js
    async def drain(self):
        return None


@pytest.mark.asyncio
async def test_retry_and_idempotency(monkeypatch):
    # Import module and patch NATS and jetstream
    import swarm.workers.tool_executor as te

    # Patch NATS client connect to return our DummyJS
    async def _connect(self, servers):
        return None

    def _jetstream(self):
        return js

    js = DummyJS()

    monkeypatch.setattr(te.NATS, 'connect', _connect, raising=False)
    monkeypatch.setattr(te.NATS, 'jetstream', _jetstream, raising=False)

    # Register a flaky tool
    calls = {"n": 0}

    @te.ToolRegistry.register('flaky')
    async def _flaky(args: Dict[str, Any]):
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("transient")
        return {"ok": True}

    # Prepare a ToolInvocation message
    inv = {
        "invocation_id": "abc",
        "task_id": "t1",
        "tool": "flaky",
        "args": {},
        "attempt": 1,
        "max_attempts": 3,
    }

    # Start main(), but we only need the handler closure – so run until subscribe sets cb
    task = asyncio.create_task(te.main())
    # Give it time to initialize
    await asyncio.sleep(0.1)
    # Access handler via patched JS
    cb = js._cb

    # First attempt should schedule a retry and ack
    msg1 = DummyMsg(json.dumps(inv).encode())
    await cb(msg1)
    assert msg1.acked
    # The retry should be scheduled; accelerate sleep by patching asyncio.sleep to no-op for the retry task
    async def _fast_sleep(sec):
        return None
    monkeypatch.setattr(asyncio, 'sleep', _fast_sleep)

    # Let the background retry publish
    await asyncio.sleep(0)

    # Simulate second attempt message delivery
    inv2 = dict(inv)
    inv2["attempt"] = 2
    msg2 = DummyMsg(json.dumps(inv2).encode())
    await cb(msg2)

    # On success, a result should be published
    assert any(subj.startswith('tools.results.') for subj, _ in js.published)

    # Now ensure idempotent replay: send the same first message again; handler should publish cached result and ack
    msg3 = DummyMsg(json.dumps(inv).encode())
    await cb(msg3)
    assert msg3.acked
    assert any(subj.startswith('tools.results.') for subj, _ in js.published)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
