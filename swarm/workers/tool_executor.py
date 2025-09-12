from __future__ import annotations
import os
import json
import time
import asyncio
import traceback
from typing import Callable, Dict, Any, Awaitable, Tuple, Optional
from dataclasses import dataclass
import hashlib

from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig, RetentionPolicy

from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import start_http_server

from swarm.protocol.messages import ToolInvocation, ToolResult
from swarm import lineage

TOOLS: Dict[str, Callable[[Dict[str,Any]], Awaitable[Any] | Any]] = {}

# Deterministic testing hooks
_BACKOFF_OVERRIDE: Optional[Callable[[int], int]] = None
_SLEEP_OVERRIDE: Optional[Callable[[float], Awaitable[None]]] = None

def set_backoff_override(fn: Optional[Callable[[int], int]]) -> None:
    global _BACKOFF_OVERRIDE
    _BACKOFF_OVERRIDE = fn

def set_sleep_override(fn: Optional[Callable[[float], Awaitable[None]]]) -> None:
    global _SLEEP_OVERRIDE
    _SLEEP_OVERRIDE = fn

async def _sleep(sec: float) -> None:
    if _SLEEP_OVERRIDE is not None:
        await _SLEEP_OVERRIDE(sec)
    else:
        await asyncio.sleep(sec)

# Metrics
ACTIVE = Gauge('tool_executor_active','Active tool executions')
ATTEMPTS = Counter('tool_executor_attempts_total','Attempts per tool', ['tool'])
FAILURES = Counter('tool_executor_failures_total','Failures per tool', ['tool','reason'])
LATENCY = Histogram('tool_executor_latency_seconds','Latency per tool', ['tool'])
RETRIES = Counter('tool_executor_retries_total','Retries scheduled', ['tool','reason'])
REPLAYS = Counter('tool_executor_idempotent_replays_total','Idempotent result replays', ['tool'])
# Also expose generic counter name requested by observability plan
EXECUTOR_RETRY_TOTAL = Counter('executor_retry_total','Retries scheduled', ['tool','reason'])
# Backlog gauge for JetStream consumer lag (tools stream)
BACKLOG = Gauge('tool_queue_backlog_gauge','Pending messages for tool executor JetStream consumer', ['stream','consumer'])

MAX_CONCURRENCY = int(os.getenv('TOOL_EXECUTOR_CONCURRENCY','8'))
SUBJECT = os.getenv('TOOLS_INVOCATION_SUBJECT','tools.invocations')
RESULT_SUBJECT = os.getenv('TOOLS_RESULT_SUBJECT','tools.results')
DLQ_SUBJECT = os.getenv('TOOLS_DLQ_SUBJECT','tools.dlq')
RETRY_BASE_MS = int(os.getenv('TOOLS_RETRY_BASE_MS','200'))
RETRY_MAX_MS = int(os.getenv('TOOLS_RETRY_MAX_MS','5000'))
IDEMP_TTL_SEC = int(os.getenv('TOOLS_IDEMP_TTL_SEC','900'))
METRICS_PORT = int(os.getenv('METRICS_PORT','9000'))
# Offload large ToolResult.output payloads above this threshold (in bytes)
RESULT_MAX_BYTES = int(os.getenv('TOOLS_RESULT_MAX_BYTES','262144'))  # 256KB default

# Stream and consumer identifiers
TOOLS_STREAM = os.getenv('TOOLS_STREAM','TOOLS')
TOOLS_CONSUMER = os.getenv('TOOLS_CONSUMER','tool-exec')

# In-memory idempotency store: key -> (stored_ts, result_dict)
_idemp_store: Dict[str, Tuple[float, Dict[str, Any]]] = {}

@dataclass
class RetryPolicy:
    base: float = 0.1
    factor: float = 2.0
    max_attempts: int = 3
    jitter: float = 0.2  # proportion of jitter
    def backoff(self, attempt: int) -> float:
        if attempt <= 0: return self.base
        d = min(self.base * (self.factor ** attempt), self.base * (self.factor ** (self.max_attempts-1)))
        if self.jitter <= 0: return d
        import random
        jitter_amt = d * self.jitter
        return max(0.0, d - jitter_amt + random.random()*jitter_amt)

class IdempotencyCache:
    def __init__(self, ttl_seconds: int = 900):
        self.ttl = ttl_seconds
        self._store: Dict[str, float] = {}
    def check_and_set(self, key: str) -> bool:
        now = time.time()
        # prune
        to_del = [k for k,v in self._store.items() if now - v > self.ttl]
        for k in to_del: self._store.pop(k, None)
        if key in self._store:
            return False
        self._store[key] = now
        return True

class ToolRegistry:
    @staticmethod
    def register(name: str):
        def deco(fn: Callable[[Dict[str,Any]], Awaitable[Any] | Any]):
            TOOLS[name] = fn
            return fn
        return deco

@ToolRegistry.register('echo')
async def tool_echo(args: Dict[str,Any]):
    await _sleep(0.01)
    return {'echo': args}

# Simplified error classifier (reduced branching complexity)
def _classify_error(exc: Exception) -> Tuple[str, bool]:
    et = type(exc).__name__.lower()
    transient_tokens = ('timeout','rate','throttle','temporary')
    network_tokens = ('connection','network')
    if any(tok in et for tok in transient_tokens):
        return ('transient', True)
    if any(tok in et for tok in network_tokens):
        return ('network', True)
    if 'value' in et or 'validation' in et:
        return ('validation', False)
    return ('runtime', False)

def _backoff_ms(attempt: int) -> int:
    if _BACKOFF_OVERRIDE is not None:
        try:
            return int(_BACKOFF_OVERRIDE(attempt))
        except Exception:
            # fall through to default
            pass
    base = RETRY_BASE_MS
    delay = min(RETRY_MAX_MS, base * (2 ** max(0, attempt-1)))
    # jitter +/- 20%
    jitter = int(0.2 * delay)
    return max(0, delay - jitter + int(jitter * (time.time() % 1)))

# Deterministic operation key computation (task_id + tool + canonical args)
def _operation_key(inv: 'ToolInvocation') -> str:
    try:
        # Canonical serialization of args (sorted keys, no whitespace)
        raw = json.dumps({'task_id': inv.task_id, 'tool': inv.tool, 'args': inv.args}, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()
    except Exception:
        # Fallback: best-effort concatenation
        return f"{inv.task_id}:{inv.tool}:{hash(str(inv.args))}"

async def _execute(inv: ToolInvocation) -> ToolResult:
    tool_fn = TOOLS.get(inv.tool)
    if not tool_fn:
        return ToolResult(invocation_id=inv.invocation_id, task_id=inv.task_id, tool=inv.tool, success=False, error='tool_not_found')
    ATTEMPTS.labels(inv.tool).inc()
    t0 = time.perf_counter()
    ACTIVE.inc()
    lineage.record_event('tool_execution_started', {"invocation_id": inv.invocation_id, "tool": inv.tool, "attempt": inv.attempt}, job_id=inv.task_id)
    try:
        if asyncio.iscoroutinefunction(tool_fn):
            out = await tool_fn(inv.args)
        else:
            loop = asyncio.get_running_loop()
            out = await loop.run_in_executor(None, tool_fn, inv.args)
        latency_ms = (time.perf_counter()-t0)*1000.0
        LATENCY.labels(inv.tool).observe(latency_ms/1000.0)
        res = ToolResult(invocation_id=inv.invocation_id, task_id=inv.task_id, tool=inv.tool, success=True, output=out, attempt=inv.attempt, latency_ms=latency_ms)
        lineage.record_event('tool_execution_succeeded', {"invocation_id": inv.invocation_id, "tool": inv.tool, "attempt": inv.attempt, "latency_ms": latency_ms}, job_id=inv.task_id)
        return res
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        latency_ms = (time.perf_counter()-t0)*1000.0
        LATENCY.labels(inv.tool).observe(latency_ms/1000.0)
        reason, retryable = _classify_error(e)
        FAILURES.labels(inv.tool, reason).inc()
        lineage.record_event('tool_execution_failed', {"invocation_id": inv.invocation_id, "tool": inv.tool, "attempt": inv.attempt, "latency_ms": latency_ms, "reason": reason, "error": str(e), "traceback": tb}, job_id=inv.task_id)
        return ToolResult(invocation_id=inv.invocation_id, task_id=inv.task_id, tool=inv.tool, success=False, error=str(e), attempt=inv.attempt, latency_ms=latency_ms, metadata={'traceback': tb, 'reason': reason, 'retryable': retryable})
    finally:
        ACTIVE.dec()

async def main():
    # Initialize tracing (best-effort)
    try:
        from swarm.observability.otel import init_tracing
        init_tracing(service_name='tool-executor', service_version=os.getenv('SERVICE_VERSION','0.3.0'), environment=os.getenv('ENV','development'))
    except Exception:
        pass

    # Start Prometheus metrics server
    try:
        start_http_server(METRICS_PORT)
    except Exception:
        pass

    nc = NATS()
    await nc.connect(servers=[os.getenv('NATS_URL','nats://localhost:4222')])
    js = nc.jetstream()
    # Ensure streams exist (invocations and DLQ)
    try:
        await js.add_stream(StreamConfig(name=TOOLS_STREAM, subjects=[f'{SUBJECT}.>'], retention=RetentionPolicy.Limits, max_age=0))
    except Exception:
        pass
    try:
        await js.add_stream(StreamConfig(name='TOOLS_DLQ', subjects=[f'{DLQ_SUBJECT}.>'], retention=RetentionPolicy.Limits, max_age=0))
    except Exception:
        pass

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    # Periodic backlog sampler for consumer info
    async def _sample_backlog():
        while True:
            try:
                info = await js.consumer_info(TOOLS_STREAM, TOOLS_CONSUMER)
                # Pending for this consumer (num_pending) reflects backlog to be delivered
                pending = getattr(info, 'num_pending', None)
                if pending is None:
                    # Fallback: compute from delivered + num_ack_pending
                    pending = getattr(info, 'num_waiting', 0)
                BACKLOG.labels(TOOLS_STREAM, TOOLS_CONSUMER).set(float(pending or 0))
            except asyncio.CancelledError:
                # Allow task to be cancelled cleanly
                break
            except Exception:
                # On error, avoid crashing sampler
                pass
            # Respect test sleep override but avoid CPU spin by yielding briefly
            await _sleep(float(os.getenv('TOOLS_BACKLOG_SAMPLE_SEC','5')))
            if _SLEEP_OVERRIDE is not None:
                await asyncio.sleep(0.01)

    sampler_task = asyncio.create_task(_sample_backlog())

    # Local import to avoid heavy deps if not needed
    from io import BytesIO
    from swarm.storage import store as _artifact_store

    async def _publish_result(res: ToolResult):
        # Potentially offload large output payloads to artifact storage
        try:
            payload = res.output
            # Serialize output only to measure size; avoid serializing whole result twice
            out_bytes = json.dumps(payload).encode()
            if len(out_bytes) > RESULT_MAX_BYTES:
                meta = _artifact_store.save_file(BytesIO(out_bytes), filename=f"tool-output-{res.invocation_id}.json", content_type="application/json")
                # Replace output with a pointer
                res.output = {"artifact_ref": meta}
                # mark metadata flag
                if not isinstance(res.metadata, dict):
                    res.metadata = {}
                res.metadata["artifact_offloaded"] = True
        except Exception:
            # Best-effort: if offload fails, proceed with original payload
            pass
        data_dict = res.model_dump()
        data_bytes = json.dumps(data_dict, separators=(",", ":")).encode()
        # Add stream-level dedupe header using invocation_id or operation key
        op_key = getattr(res, '_operation_key', None)
        msg_id = getattr(res, 'invocation_id', None) or op_key or getattr(res, 'task_id', None)
        headers = {"Nats-Msg-Id": str(msg_id)} if msg_id else None
        await js.publish(f'{RESULT_SUBJECT}.{res.task_id}', data_bytes, headers=headers)
        # store idempotent cache under invocation_id; op key mapping added later in handler
        entry = (time.time(), data_dict)
        _idemp_store[res.invocation_id] = entry
        if hasattr(res, '_operation_key') and getattr(res, '_operation_key'):
            _idemp_store[getattr(res, '_operation_key')] = entry

    async def _publish_dlq(inv: ToolInvocation, res: ToolResult | None):
        envelope = {"invocation": inv.model_dump(), "result": res.model_dump() if res else None}
        # Use invocation_id as dedupe key for DLQ entries to avoid duplicates on crash/replay
        msg_id = getattr(inv, 'invocation_id', None) or getattr(inv, 'task_id', None)
        headers = {"Nats-Msg-Id": str(msg_id)} if msg_id else None
        await js.publish(f'{DLQ_SUBJECT}.{inv.tool}', json.dumps(envelope, separators=(",", ":")).encode(), headers=headers)

    def _prune_idemp_now():
        now = time.time()
        to_del = [k for k,(ts,_) in _idemp_store.items() if now - ts > IDEMP_TTL_SEC]
        for k in to_del:
            _idemp_store.pop(k, None)

    async def handler(msg):
        data = json.loads(msg.data.decode())
        inv = ToolInvocation(**data)
        op_key = _operation_key(inv)

        async def _replay_if_cached() -> bool:
            cached = _idemp_store.get(inv.invocation_id)
            alt_entry = _idemp_store.get(op_key) if not cached else None
            entry_ref = cached or alt_entry
            if not entry_ref:
                return False
            ts, result_dict = entry_ref
            publish_dict = result_dict.copy()
            if publish_dict.get('invocation_id') != inv.invocation_id:
                orig_id = publish_dict.get('invocation_id')
                publish_dict['invocation_id'] = inv.invocation_id
                md = publish_dict.setdefault('metadata', {})
                md['idempotent_replay'] = True
                md['replayed_from_invocation'] = orig_id
            js_data = json.dumps(publish_dict, separators=(",", ":")).encode()
            # Use current invocation_id or operation key as message id for dedupe
            msg_id = inv.invocation_id or op_key or inv.task_id
            headers = {"Nats-Msg-Id": str(msg_id)} if msg_id else None
            await js.publish(f'{RESULT_SUBJECT}.{inv.task_id}', js_data, headers=headers)
            entry = (ts, result_dict)
            _idemp_store[inv.invocation_id] = entry
            _idemp_store[op_key] = entry
            REPLAYS.labels(inv.tool).inc()
            await msg.ack()
            _prune_idemp_now()
            return True

        # Attempt replay first
        if await _replay_if_cached():
            return

        async with sem:
            res = await _execute(inv)
        setattr(res, '_operation_key', op_key)

        def _should_retry(r: ToolResult) -> bool:
            if r.success:
                return False
            meta = r.metadata if isinstance(r.metadata, dict) else {}
            return bool(meta.get('retryable')) and inv.attempt < getattr(inv, 'max_attempts', 3)

        if _should_retry(res):
            meta = res.metadata if isinstance(res.metadata, dict) else {}
            reason = meta.get('reason', 'transient')
            delay_ms = _backoff_ms(inv.attempt)
            RETRIES.labels(inv.tool, reason).inc()
            EXECUTOR_RETRY_TOTAL.labels(inv.tool, reason).inc()
            async def _republish_later():
                await _sleep(delay_ms/1000.0)
                again = inv.model_copy(update={"attempt": inv.attempt + 1})
                # No dedupe header for invocation messages; they must be re-enqueued
                await js.publish(f'{SUBJECT}.{inv.tool}', again.model_dump_json().encode())
            # In deterministic test mode, publish inline to avoid race with immediate assertions
            if _SLEEP_OVERRIDE is not None:
                again = inv.model_copy(update={"attempt": inv.attempt + 1})
                await js.publish(f'{SUBJECT}.{inv.tool}', again.model_dump_json().encode())
            else:
                asyncio.create_task(_republish_later())
            await msg.ack()
            _prune_idemp_now()
            return

        await _publish_result(res)
        if op_key and res.invocation_id in _idemp_store:
            _idemp_store[op_key] = _idemp_store[res.invocation_id]
        if not res.success:
            await _publish_dlq(inv, res)
        await msg.ack()
        _prune_idemp_now()

    # Assert reference so static analyzers treat handler as used
    assert callable(handler)

    await js.subscribe(f'{SUBJECT}.>', durable=TOOLS_CONSUMER, cb=handler, manual_ack=True, ack_wait=30)
    # Keep running; avoid tight busy-loop when _SLEEP_OVERRIDE short‑circuits sleeps in tests
    try:
        while True:
            # Main idle wait; overridden _sleep returns immediately in deterministic tests
            await _sleep(60)
            if _SLEEP_OVERRIDE is not None:
                # Yield to event loop to prevent CPU spinning when override makes _sleep a no-op
                await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        # Cancel background sampler and re-raise to satisfy tests expecting cancellation
        try:
            sampler_task.cancel()
        finally:
            raise

if __name__ == '__main__':
    asyncio.run(main())
