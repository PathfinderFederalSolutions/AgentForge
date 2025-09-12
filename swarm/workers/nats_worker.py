import asyncio
import json
import logging
import os
import random
import time
from collections import OrderedDict
from typing import Optional

try:
    from prometheus_client import Gauge, Histogram, Counter  # type: ignore
except Exception:  # pragma: no cover
    Gauge = Histogram = Counter = None  # type: ignore

# Add explicit import for JetStream timeout
try:
    from nats.errors import TimeoutError as NatsTimeoutError  # type: ignore
except Exception:  # pragma: no cover
    class NatsTimeoutError(Exception):
        pass

from swarm.config import settings
from swarm.enforcement import enforcer
from swarm import lineage
from swarm.jetstream import subscribe_jobs, publish_result, ensure_streams
from swarm.critic_healer import CriticHealer
from swarm.bkg import store as bkg
from swarm.memory_mesh import start_mesh_tasks

try:
    from orchestrator import build_orchestrator
except Exception:
    build_orchestrator = None

try:
    from swarm.observability.costs import set_observability_context as _set_obs_ctx  # type: ignore
    from swarm.observability.otel import tag_span as _tag_span  # type: ignore
except Exception:  # pragma: no cover
    _set_obs_ctx = None  # type: ignore
    _tag_span = None  # type: ignore

from sla_kpi_config import get_task_budget  # type: ignore
from swarm.observability.task_latency import record_task_completion  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH = int(os.getenv("NATS_FETCH_BATCH", "10"))
WAIT = float(os.getenv("NATS_FETCH_WAIT", "5.0"))
MAX_JOB_ATTEMPTS = int(os.getenv("NATS_JOB_ATTEMPTS", "3"))
RETRY_BASE = float(os.getenv("NATS_RETRY_BASE", "0.25"))
DEDUP_CACHE_SIZE = int(os.getenv("NATS_DEDUP_CACHE", "2048"))

# Fast-path: When set, worker will ack messages without executing business logic.
WORKER_NOOP = os.getenv("WORKER_NOOP", "0") in {"1", "true", "yes", "on"}

# In‑memory idempotency (LRU) for processed job_ids
_processed_jobs: OrderedDict[str, float] = OrderedDict()

def _dedupe_seen(job_id: str) -> bool:
    if not job_id:
        return False
    seen = job_id in _processed_jobs
    now = time.time()
    _processed_jobs[job_id] = now
    # Maintain LRU size
    if len(_processed_jobs) > DEDUP_CACHE_SIZE:
        _processed_jobs.popitem(last=False)
    return seen

# Adaptive batching controller -------------------------------------------------
class AdaptiveBatchController:
    """Controls dynamic fetch batch size based on:
    - Observed per‑job latency (EMA)
    - Observed per‑job GPU memory usage (EMA from job payload gpu_mem)
    - Queue depth (pending messages) hint
    - Configured GPU capacity (GPU_TOTAL_MEM_MB)

    Strategy:
    1. GPU bin‑packing: limit batch so that avg_gpu_mem * batch * SAFETY <= GPU_TOTAL_MEM_MB.
    2. Latency guard: if EMA latency > TARGET_LATENCY reduce batch by 1 (floor 1).
    3. Backlog pressure: if queue_depth >= current_batch * 2 and latency within budget, allow +1 step growth.
    4. Never exceed static MAX (env NATS_FETCH_BATCH) nor drop below 1.
    """
    def __init__(self, max_batch: int, gpu_total_mem: int, safety: float = 1.2, target_latency_s: float = 2.0):
        self.max_batch = max_batch
        self.gpu_total_mem = gpu_total_mem
        self.safety = safety
        self.target_latency_s = target_latency_s
        self._ema_latency: Optional[float] = None
        self._ema_gpu_mem: Optional[float] = None
        self._alpha = 0.2  # smoothing factor
        self._last_batch = max(1, min(2, max_batch))

    def record_job(self, gpu_mem_mb: Optional[float], latency_s: Optional[float]):
        if latency_s is not None:
            if self._ema_latency is None:
                self._ema_latency = latency_s
            else:
                self._ema_latency = self._alpha * latency_s + (1 - self._alpha) * self._ema_latency
        if gpu_mem_mb is not None and gpu_mem_mb > 0:
            if self._ema_gpu_mem is None:
                self._ema_gpu_mem = gpu_mem_mb
            else:
                self._ema_gpu_mem = self._alpha * gpu_mem_mb + (1 - self._alpha) * self._ema_gpu_mem

    def _gpu_limited_batch(self) -> int:
        if self.gpu_total_mem <= 0 or not self._ema_gpu_mem:
            return self.max_batch
        cap = int(self.gpu_total_mem / (self._ema_gpu_mem * self.safety))
        return max(1, min(cap, self.max_batch))

    def compute_next_batch(self, queue_depth: int) -> int:
        batch = self._last_batch
        gpu_cap = self._gpu_limited_batch()
        # Adjust upward if backlog large and under gpu cap
        if queue_depth >= batch * 2 and batch < gpu_cap:
            batch += 1
        # Latency guard
        if self._ema_latency and self._ema_latency > self.target_latency_s and batch > 1:
            batch = max(1, batch - 1)
        # Ensure not above gpu cap
        batch = min(batch, gpu_cap)
        self._last_batch = batch
        return batch

# Metrics (created lazily to avoid overhead when prometheus_client absent)
if Gauge and Histogram and Counter:
    _BATCH_SIZE_G = Gauge("worker_adaptive_batch_size", "Current adaptive fetch batch size", ["mission"])  # type: ignore
    _GPU_AVG_MEM_G = Gauge("worker_gpu_avg_mem_mb", "EMA of observed gpu_mem per job (MB)", ["mission"])  # type: ignore
    _QUEUE_DEPTH_G = Gauge("worker_queue_depth", "Last observed JetStream consumer pending messages", ["mission"])  # type: ignore
    _BATCH_LAT_H = Histogram("worker_batch_latency_seconds", "Latency to process a batch", ["mission"], buckets=(0.1,0.25,0.5,1,2,5,10,30))  # type: ignore
    _JOBS_PROCESSED = Counter("worker_jobs_processed_total", "Jobs processed by worker", ["mission"])  # type: ignore
else:  # pragma: no cover
    _BATCH_SIZE_G = _GPU_AVG_MEM_G = _QUEUE_DEPTH_G = _BATCH_LAT_H = _JOBS_PROCESSED = None  # type: ignore

# Instantiate controller (values may be overridden in tests by reassigning)
_GPU_TOTAL_MEM_MB = int(os.getenv("GPU_TOTAL_MEM_MB", "0"))
_GPU_SAFETY = float(os.getenv("GPU_BINPACK_SAFETY", "1.2"))
_TARGET_JOB_LAT_S = float(os.getenv("WORKER_TARGET_JOB_LATENCY_SEC", "2.0"))
_controller = AdaptiveBatchController(BATCH, _GPU_TOTAL_MEM_MB, _GPU_SAFETY, _TARGET_JOB_LAT_S)

# --- Helpers to reduce complexity --------------------------------------------

def _execute_mission(goal: str, agents: int):
    use_ch = os.getenv("CRITIC_HEALER_ENABLED", "1") != "0"
    if use_ch:
        ch = CriticHealer()
        out = ch.run(goal=goal, total_agents=agents)
        results = out.get("results", [])
        decision = out.get("decision", {})
    else:
        if not build_orchestrator:
            raise RuntimeError("Orchestrator unavailable")
        orch = build_orchestrator(num_agents=agents)
        results = orch.run_goal_sync(goal)
        decision = enforcer.post(goal=goal, results=results)
    return decision, results

def _post_completion(job_id: str, goal: str, mission: str, decision: dict, results: list):
    # Persist best-known-good on approvals
    try:
        if decision.get("approved"):
            bkg.update(goal, decision, results)
    except Exception:
        logger.debug("BKG update skipped")

    # Lineage completion
    try:
        lineage.complete_job(job_id=job_id, decision=decision, results=results)
    except Exception:
        logger.debug("lineage completion failed for job %s", job_id)

    # After completion attempt to record latency budget metrics
    try:
        j = lineage.get_job(job_id)
        if j and j.created_at and j.completed_at:
            latency_sec = (j.completed_at - j.created_at).total_seconds()
            b = get_task_budget("default")
            record_task_completion(latency_sec, "default", mission, b.name, b.p99_ms, b.hard_cap_ms)
    except Exception:
        pass

async def _record_job_obs(goal: str, mission: str, start_ts: float):
    try:
        lat = time.perf_counter() - start_ts
        gpu_mem = None
        if isinstance(goal, str) and goal.startswith("{"):
            try:
                g_obj = json.loads(goal)
                gpu_mem = g_obj.get("gpu_mem")
            except Exception:
                pass
        _controller.record_job(gpu_mem_mb=gpu_mem, latency_s=lat)
        if _JOBS_PROCESSED:
            _JOBS_PROCESSED.labels(mission).inc()  # type: ignore
        if _GPU_AVG_MEM_G and _controller._ema_gpu_mem:
            _GPU_AVG_MEM_G.labels(mission).set(_controller._ema_gpu_mem)  # type: ignore
    except Exception:
        pass

async def run_job(goal: str, agents: int, mission: str, job_id: str):
    decision, results = _execute_mission(goal, agents)
    _post_completion(job_id, goal, mission, decision, results)

    # Publish results to mission-sharded results subject
    try:
        await publish_result(mission, {"job_id": job_id, "goal": goal, "decision": decision, "results": results})
    except Exception:
        logger.debug("result publish failed for job %s", job_id)

    return decision, results

def _parse_goal(data: dict) -> Optional[str]:
    goal = data.get("goal") or data.get("payload") or data.get("text") or data.get("message")
    if isinstance(goal, (dict, list)):
        goal = json.dumps(goal)
    return goal if goal else None

async def _resolve_agents_mission(data: dict):
    agents = int(data.get("agents") or os.getenv("DEFAULT_AGENTS", 2))
    mission = os.getenv("MISSION") or (settings.env or "default")
    return agents, mission

async def _tag_context(mission: str, job_id: str) -> None:
    if _set_obs_ctx:
        try:
            _set_obs_ctx(mission_id=mission, task_id=job_id)
        except Exception:
            pass
    if _tag_span:
        try:
            _tag_span(mission_id=mission, task_id=job_id)
        except Exception:
            pass

async def _decode_json_or_ack(m) -> Optional[dict]:
    try:
        return json.loads(m.data.decode())
    except Exception:
        logger.exception("Invalid JSON; dropping")
        await m.ack()
        return None

async def _process_attempts(goal: str, agents: int, mission: str, job_id: str, m) -> None:
    start_ts = time.perf_counter()
    attempt = 0
    while attempt < MAX_JOB_ATTEMPTS:
        attempt += 1
        try:
            decision, _ = await run_job(goal, agents, mission, job_id)
            await _record_job_obs(goal, mission, start_ts)
            logger.info(
                "job completed id=%s mission=%s approved=%s reason=%s attempts=%d",
                job_id,
                mission,
                bool(decision.get("approved")),
                decision.get("reason"),
                attempt,
            )
            await m.ack()  # ACK only on success
            return
        except Exception as e:
            logger.exception("job %s attempt %d failed: %s", job_id, attempt, e)
            if attempt >= MAX_JOB_ATTEMPTS:
                logger.warning("exhausted retries for job_id=%s; will be redelivered", job_id)
                return
            sleep_for = RETRY_BASE * (2 ** (attempt - 1)) + random.random() * 0.1
            await asyncio.sleep(min(sleep_for, 5.0))

async def process_msg(m):
    # Fast-path for drain tests: immediately ack messages without heavy work.
    if WORKER_NOOP:
        try:
            await m.ack()
        except Exception:
            pass
        return

    data = await _decode_json_or_ack(m)
    if data is None:
        return
    job_id = data.get("job_id", "unknown")
    goal = _parse_goal(data)
    if not goal:
        logger.warning("Dropping message without goal/payload: %s", data)
        await m.ack()
        return
    agents, mission = await _resolve_agents_mission(data)
    await _tag_context(mission, job_id)

    if _dedupe_seen(job_id):
        logger.info("duplicate job_id=%s ignored (idempotent)", job_id)
        await m.ack()
        return

    logger.info("job received id=%s mission=%s agents=%d", job_id, mission, agents)
    await _process_attempts(goal, agents, mission, job_id, m)

# Main loop -------------------------------------------------------------------

async def _compute_adaptive_batch(js, sub, mission: str):
    adaptive_batch = BATCH
    queue_depth = -1
    try:
        info = await js.consumer_info("swarm_jobs", sub._consumer)  # type: ignore
        queue_depth = getattr(info, "num_pending", -1)
    except Exception:
        pass
    try:
        adaptive_batch = _controller.compute_next_batch(queue_depth if queue_depth >= 0 else 0)
    except Exception:
        adaptive_batch = BATCH
    if _BATCH_SIZE_G:
        try:
            _BATCH_SIZE_G.labels(mission).set(adaptive_batch)  # type: ignore
        except Exception:
            pass
    if _QUEUE_DEPTH_G and queue_depth >= 0:
        try:
            _QUEUE_DEPTH_G.labels(mission).set(queue_depth)  # type: ignore
        except Exception:
            pass
    return adaptive_batch, queue_depth

async def _fetch_and_process(sub, adaptive_batch: int, mission: str):
    batch_start = time.perf_counter()
    try:
        msgs = await sub.fetch(adaptive_batch, timeout=WAIT)
    except (asyncio.TimeoutError, NatsTimeoutError):
        return
    except Exception as e:
        logger.exception("Fetch error: %s", e)
        await asyncio.sleep(min(5.0, 0.5 + random.random()))
        return

    if not msgs:
        return

    try:
        await asyncio.gather(*(process_msg(m) for m in msgs))
    except Exception:
        logger.exception("Batch processing error")
    finally:
        batch_lat = time.perf_counter() - batch_start
        if _BATCH_LAT_H:
            try:
                _BATCH_LAT_H.labels(mission).observe(batch_lat)  # type: ignore
            except Exception:
                pass

async def main():
    mission = os.getenv("MISSION", "staging")
    # Ensure JetStream streams exist before subscribing
    try:
        await ensure_streams()
    except Exception as e:
        logging.getLogger(__name__).warning("ensure_streams failed: %s", e)
    _nc, js, sub = await subscribe_jobs(consumer_name=f"worker-{mission}", mission=mission)
    start_mesh_tasks(asyncio.get_running_loop(), js, mission)
    logger.info("Subscribed durable consumer for mission=%s", mission)

    while True:
        try:
            adaptive_batch, _ = await _compute_adaptive_batch(js, sub, mission)  # type: ignore
        except Exception:
            adaptive_batch = BATCH
        await _fetch_and_process(sub, adaptive_batch, mission)

if __name__ == "__main__":
    asyncio.run(main())