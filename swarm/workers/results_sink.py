# filepath: swarm/workers/results_sink.py
from __future__ import annotations
import os
import json
import asyncio
from typing import Optional

from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig, RetentionPolicy
from prometheus_client import Gauge, Counter, start_http_server

from swarm.protocol.messages import ToolResult

# Environment / config
ENV = os.getenv("ENV", "staging")
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
RESULTS_SUBJECT_PREFIX = os.getenv("RESULTS_SUBJECT_PREFIX", "swarm.results")
RESULTS_STREAM = os.getenv("RESULTS_STREAM", "SWARM_RESULTS")
RESULTS_CONSUMER_PREFIX = os.getenv("RESULTS_CONSUMER_PREFIX", "results")
ACK_WAIT_SEC = int(os.getenv("RESULTS_ACK_WAIT_SEC", "10"))
BACKLOG_SAMPLE_SEC = float(os.getenv("RESULTS_BACKLOG_SAMPLE_SEC", "5"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "9002"))
SINK_DIR = os.getenv("RESULTS_SINK_DIR", "var")
DATABASE_URL = os.getenv("DATABASE_URL")

# Test hooks
_TEST_FAIL_ONCE = os.getenv("RESULTS_SINK_TEST_FAIL_ONCE") == "1"
_PERSIST_ATTEMPTS = 0  # exposed for tests

# Metrics
RESULTS_BACKLOG = Gauge('results_backlog_gauge', 'Pending messages for results sink JetStream consumer', ['stream','consumer'])
RESULTS_PERSISTED = Counter('results_persisted_total', 'Results persisted', ['backend','mission_id','task_id'])
RESULTS_PERSIST_FAIL = Counter('results_persist_fail_total', 'Result persistence failures', ['backend','reason'])

# DB lazy init
_engine = None
_table_ready = False

async def _init_db():
    global _engine, _table_ready
    if not DATABASE_URL:
        return
    if _engine is None:
        try:
            from sqlalchemy import create_engine
            _engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
        except Exception:
            _engine = None
            return
    if _engine is not None and not _table_ready:
        try:
            from sqlalchemy import text
            ddl = """
            CREATE TABLE IF NOT EXISTS swarm_results (
              id SERIAL PRIMARY KEY,
              env TEXT,
              received_at TIMESTAMPTZ DEFAULT NOW(),
              invocation_id TEXT,
              task_id TEXT,
              tool TEXT,
              success BOOLEAN,
              attempt INT,
              payload JSONB
            );
            CREATE INDEX IF NOT EXISTS idx_swarm_results_task ON swarm_results(task_id);
            CREATE INDEX IF NOT EXISTS idx_swarm_results_env ON swarm_results(env);
            """
            with _engine.begin() as conn:
                for stmt in ddl.strip().split(';'):
                    s = stmt.strip()
                    if s:
                        conn.execute(text(s))
            _table_ready = True
        except Exception:
            # Leave table creation best-effort
            pass

async def _persist_result(res: ToolResult):
    global _TEST_FAIL_ONCE, _PERSIST_ATTEMPTS
    _PERSIST_ATTEMPTS += 1
    if _TEST_FAIL_ONCE:
        _TEST_FAIL_ONCE = False
        raise RuntimeError("simulated_persist_failure")

    # Prefer DB if configured and initialized
    if DATABASE_URL and _engine is not None and _table_ready:
        try:
            from sqlalchemy import text
            payload = res.model_dump()
            with _engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO swarm_results (env, invocation_id, task_id, tool, success, attempt, payload)
                    VALUES (:env,:invocation_id,:task_id,:tool,:success,:attempt, :payload::jsonb)
                """), {
                    'env': ENV,
                    'invocation_id': res.invocation_id,
                    'task_id': res.task_id,
                    'tool': res.tool,
                    'success': res.success,
                    'attempt': res.attempt,
                    'payload': json.dumps(payload)
                })
            RESULTS_PERSISTED.labels('db', res.metadata.get('mission_id','unknown'), res.task_id).inc()
            return
        except Exception as e:
            RESULTS_PERSIST_FAIL.labels('db', type(e).__name__).inc()
            # fall back to file
    # File persistence (JSONL)
    try:
        os.makedirs(SINK_DIR, exist_ok=True)
        path = os.path.join(SINK_DIR, f'swarm_results_{ENV}.jsonl')
        line = json.dumps(res.model_dump()) + "\n"
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line)
        RESULTS_PERSISTED.labels('jsonl', res.metadata.get('mission_id','unknown'), res.task_id).inc()
    except Exception as e:
        RESULTS_PERSIST_FAIL.labels('jsonl', type(e).__name__).inc()
        raise

async def main(stop_event: Optional[asyncio.Event] = None):
    # Tracing best-effort
    try:
        from swarm.observability.otel import init_tracing
        init_tracing(service_name='results-sink', service_version=os.getenv('SERVICE_VERSION','0.3.0'), environment=ENV)
    except Exception:
        pass
    # Metrics server
    try:
        start_http_server(METRICS_PORT)
    except Exception:
        pass

    await _init_db()

    nc = NATS()
    await nc.connect(servers=[NATS_URL])
    js = nc.jetstream()

    # Ensure stream exists
    try:
        await js.add_stream(StreamConfig(name=RESULTS_STREAM, subjects=[f'{RESULTS_SUBJECT_PREFIX}.>'], retention=RetentionPolicy.Limits, max_age=0))
    except Exception:
        pass

    subject = f'{RESULTS_SUBJECT_PREFIX}.{ENV}'
    durable = f'{RESULTS_CONSUMER_PREFIX}-{ENV}'

    async def _sample_backlog():
        while True:
            try:
                info = await js.consumer_info(RESULTS_STREAM, durable)
                pending = getattr(info, 'num_pending', None)
                if pending is None:
                    pending = getattr(info, 'num_waiting', 0)
                RESULTS_BACKLOG.labels(RESULTS_STREAM, durable).set(float(pending or 0))
            except Exception:
                pass
            await asyncio.sleep(BACKLOG_SAMPLE_SEC)

    asyncio.create_task(_sample_backlog())

    async def handler(msg):
        # Parse ToolResult
        try:
            data = json.loads(msg.data.decode())
            res = ToolResult(**data)
        except Exception:
            # Malformed => ack to avoid poison pill loop
            await msg.ack()
            return
        try:
            await _persist_result(res)
            await msg.ack()
        except Exception:
            # Leave unacked for redelivery
            return

    await js.subscribe(subject, durable=durable, cb=handler, manual_ack=True, ack_wait=ACK_WAIT_SEC)

    # Run until stop_event signaled or forever
    while True:
        if stop_event and stop_event.is_set():
            break
        await asyncio.sleep(1)

    try:
        await nc.drain()
    except Exception:
        pass

if __name__ == '__main__':
    asyncio.run(main())
