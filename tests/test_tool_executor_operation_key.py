# filepath: tests/test_tool_executor_operation_key.py
from __future__ import annotations
import asyncio
import json
from typing import Any, Dict
import types
import pytest

# This test validates idempotency via deterministic operation key: two distinct
# invocations (different invocation_id) with identical task_id/tool/args should
# only cause the underlying tool side effect to happen once; second publish is a replay.

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
async def test_operation_key_idempotent_replay(monkeypatch):
    import swarm.workers.tool_executor as te

    executions: list[Dict[str, Any]] = []

    @te.ToolRegistry.register('side_effect')
    async def _tool(args: Dict[str, Any]):  # noqa: D401
        executions.append({'ran': True, 'args': args})
        return {'value': args.get('x')}

    js = DummyJS()

    async def _connect(self, servers):
        return None
    def _jetstream(self):
        return js

    monkeypatch.setattr(te.NATS, 'connect', _connect, raising=False)
    monkeypatch.setattr(te.NATS, 'jetstream', _jetstream, raising=False)

    # Two invocations differing only by invocation_id
    inv1 = {
        'invocation_id': 'id1',
        'task_id': 'task-123',
        'tool': 'side_effect',
        'args': {'x': 7},
        'attempt': 1,
        'max_attempts': 2,
    }
    inv2 = {
        'invocation_id': 'id2',  # different
        'task_id': 'task-123',
        'tool': 'side_effect',
        'args': {'x': 7},  # identical args => same op key
        'attempt': 1,
        'max_attempts': 2,
    }

    task = asyncio.create_task(te.main())
    await asyncio.sleep(0.1)
    cb = js._cb

    # First invocation executes tool
    await cb(DummyMsg(json.dumps(inv1).encode()))
    await asyncio.sleep(0.05)
    # Second invocation should replay without executing tool again
    await cb(DummyMsg(json.dumps(inv2).encode()))
    await asyncio.sleep(0.05)

    # Exactly one real execution
    assert len(executions) == 1

    result_msgs = [json.loads(d.decode()) for s,d in js.published if s.startswith('tools.results.')]
    # Two results published (first true, second replay)
    assert len(result_msgs) >= 2
    # Find replay (has metadata.idempotent_replay)
    replay_msgs = [r for r in result_msgs if r.get('metadata', {}).get('idempotent_replay')]
    assert replay_msgs, 'Expected a replay result with idempotent_replay metadata'
    # Ensure replay uses second invocation id
    assert any(r['invocation_id'] == 'id2' for r in replay_msgs)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
