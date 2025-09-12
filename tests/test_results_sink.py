# filepath: tests/test_results_sink.py
import os
import asyncio
import json
import time
from pathlib import Path
import socket

import pytest

# Skip if nats server not reachable
NATS_URL = os.getenv('NATS_URL','nats://localhost:4222')
ENV = os.getenv('ENV','staging')
RESULTS_SUBJECT_PREFIX = os.getenv('RESULTS_SUBJECT_PREFIX','swarm.results')

@pytest.mark.asyncio
async def test_results_sink_jsonl_fallback(tmp_path, monkeypatch):
    # Basic connectivity check
    host_port = NATS_URL.replace('nats://','').split(',')[0]
    host, port = host_port.split(':')
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect((host,int(port)))
    except Exception:
        pytest.skip('NATS not available')
    finally:
        s.close()

    # Use tmp dir for sink
    monkeypatch.setenv('RESULTS_SINK_DIR', str(tmp_path))
    monkeypatch.setenv('DATABASE_URL','')  # force jsonl
    monkeypatch.setenv('ENV', ENV)
    monkeypatch.setenv('RESULTS_ACK_WAIT_SEC','2')
    # Simulate one failure then success to test redelivery
    monkeypatch.setenv('RESULTS_SINK_TEST_FAIL_ONCE','1')

    from nats.aio.client import Client as NATS
    from swarm.workers import results_sink

    stop = asyncio.Event()

    async def run_sink():
        await results_sink.main(stop_event=stop)

    task = asyncio.create_task(run_sink())

    # Allow subscription setup
    await asyncio.sleep(0.5)

    nc = NATS()
    await nc.connect(servers=[NATS_URL])
    js = nc.jetstream()

    subj = f"{RESULTS_SUBJECT_PREFIX}.{ENV}"

    # Publish a synthetic ToolResult
    payload = {
        'invocation_id':'abc123', 'task_id':'t1','tool':'echo','success':True,
        'output': {'value': 1}, 'error': None, 'attempt':1, 'latency_ms': 12.3,
        'trace_id': None, 'started_ts': time.time(), 'completed_ts': time.time(), 'metadata': {}
    }
    await js.publish(subj, json.dumps(payload).encode())

    # Wait for processing (include redelivery) - ack wait is 2 sec; allow 5 sec
    for _ in range(10):
        await asyncio.sleep(0.5)
        path = tmp_path / f'swarm_results_{ENV}.jsonl'
        if path.exists():
            lines = path.read_text().strip().splitlines()
            if lines:
                obj = json.loads(lines[-1])
                if obj.get('invocation_id') == 'abc123':
                    break
    else:
        stop.set()
        await task
        pytest.fail('Result not persisted')

    # Validate file content
    path = tmp_path / f'swarm_results_{ENV}.jsonl'
    data = path.read_text().strip().splitlines()
    assert any(json.loads(l)['invocation_id']=='abc123' for l in data)

    # Ensure at least two persistence attempts due to injected failure
    assert results_sink._PERSIST_ATTEMPTS >= 2

    stop.set()
    await asyncio.sleep(0)  # let loop notice
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except Exception:
        pass
