# filepath: tests/test_tool_executor_dlq.py
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
    async def add_stream(self, *args, **kwargs):  # no-op
        return None
    async def subscribe(self, subj: str, durable: str, cb, manual_ack: bool, ack_wait: int):
        self.subscribed = (subj, durable, cb, manual_ack, ack_wait)
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
async def test_dlq_on_non_retryable_failure(monkeypatch):
    import swarm.workers.tool_executor as te

    async def _connect(self, servers):
        return None

    def _jetstream(self):
        return js

    js = DummyJS()

    monkeypatch.setattr(te.NATS, 'connect', _connect, raising=False)
    monkeypatch.setattr(te.NATS, 'jetstream', _jetstream, raising=False)

    # Register a tool that fails with non-retryable error (ValueError -> validation)
    @te.ToolRegistry.register('always_fail')
    async def _always_fail(args: Dict[str, Any]):
        raise ValueError('bad input')

    inv = {
        'invocation_id': 'xyz',
        'task_id': 't2',
        'tool': 'always_fail',
        'args': {},
        'attempt': 1,
        'max_attempts': 3,
    }

    task = asyncio.create_task(te.main())
    await asyncio.sleep(0.1)
    cb = js._cb

    msg = DummyMsg(json.dumps(inv).encode())
    await cb(msg)

    # Should publish a failure result and a DLQ message
    subjects = [s for s,_ in js.published]
    assert any(s.startswith('tools.results.') for s in subjects)
    assert any(s.startswith('tools.dlq.') for s in subjects)
    assert msg.acked

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
