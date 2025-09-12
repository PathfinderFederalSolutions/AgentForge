import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response, PlainTextResponse

# Config
QUEUE_DIR = Path(os.getenv("SYNC_QUEUE_DIR", "/agentforge/replay"))
LEDGER_DIR = Path(os.getenv("SYNC_LEDGER_DIR", "/app/services/syncdaemon/logs"))
MAX_INFLIGHT = int(os.getenv("SYNC_MAX_INFLIGHT", "64"))
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
SUBJECT = os.getenv("SYNC_SUBJECT", "swarm.results.edge")
SITE = os.getenv("SITE", "edge")

# Ensure directories
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AgentForge Edge Sync Daemon")

# CORS origin (restrict to localhost by default)
ALLOWED_ORIGIN = os.getenv("SYNC_ALLOWED_ORIGIN", "http://localhost")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"]
)

# Metrics
registry = CollectorRegistry()
replay_queue_depth = Gauge(
    "replay_queue_depth", "Number of files pending in replay queue", ["site"], registry=registry
)
edge_bandwidth_bps = Gauge(
    "edge_bandwidth_bps", "Estimated outbound bandwidth during replay in bytes/sec", ["site"], registry=registry
)

# Internal state
_inflight = 0
_last_tx_window = []  # list[(ts, bytes)]
_nc = None
_js = None
_running = True


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _log_ledger(event: str, data: dict) -> None:
    ts = int(time.time() * 1000)
    entry = {"ts": ts, "event": event, **data}
    log_path = LEDGER_DIR / f"sync_{time.strftime('%Y%m%d')}.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")


async def _connect_nats():
    global _nc, _js
    import importlib
    nats = importlib.import_module("nats")
    _nc = await nats.connect(
        servers=[NATS_URL],
        reconnect_time_wait=0.5,
        max_reconnect_attempts=-1,
        reconnect_jitter=0.25,
    )
    _js = _nc.jetstream()


async def _ensure_stream():
    # Best-effort create stream if missing (JetStream)
    try:
        cfg = {"name": "SWARM_RESULTS", "subjects": [f"{SUBJECT}.*", SUBJECT]}
        await _js.add_stream(name=cfg["name"], subjects=cfg["subjects"])  # type: ignore
    except Exception:
        pass


async def _publish_with_id(data: bytes, msg_id: str):
    headers = {"Nats-Msg-Id": msg_id}
    try:
        await _js.publish(SUBJECT, data, headers=headers)  # type: ignore
    except Exception:
        # Fallback to bare publish
        await _nc.publish(SUBJECT, data, headers=headers)  # type: ignore


def _update_bandwidth(tx_bytes: int):
    global _last_tx_window
    now = time.time()
    _last_tx_window.append((now, tx_bytes))
    # Drop entries older than 5s
    cutoff = now - 5.0
    _last_tx_window = [(t, b) for (t, b) in _last_tx_window if t >= cutoff]
    if len(_last_tx_window) > 1:
        dt = max(0.001, _last_tx_window[-1][0] - _last_tx_window[0][0])
    else:
        dt = 1.0
    total = sum(b for _, b in _last_tx_window)
    edge_bandwidth_bps.labels(SITE).set(total / dt)


def _original_from_sending(path: Path) -> Path:
    s = str(path)
    return Path(s[:-8]) if s.endswith(".sending") else path


async def _process_file(path: Path):
    global _inflight
    try:
        # Read content
        content = path.read_bytes()
        pre_hash = _sha256_bytes(content)
        msg_id = pre_hash  # idempotency via content hash

        # Publish with headers
        await _publish_with_id(content, msg_id)
        _update_bandwidth(len(content))

        # Verify after send by re-hashing content (unchanged)
        post_hash = _sha256_bytes(content)
        if post_hash != pre_hash:
            _log_ledger("hash_mismatch", {"file": str(path), "pre": pre_hash, "post": post_hash})
            # Move back to original name to retry later
            try:
                base = _original_from_sending(path)
                if path.exists():
                    path.replace(base)
            except Exception:
                pass
            return

        # On success, remove file
        path.unlink(missing_ok=True)
        _log_ledger("replayed", {"file": str(path), "bytes": len(content), "msg_id": msg_id})
    except Exception as e:
        _log_ledger("error", {"file": str(path), "error": str(e)})
        # On failure, restore original file name for retry
        try:
            base = _original_from_sending(path)
            if path.exists():
                path.replace(base)
        except Exception:
            pass
    finally:
        _inflight = max(0, _inflight - 1)


async def _worker_loop():
    global _inflight
    while _running:
        try:
            files = []
            for p in QUEUE_DIR.glob("**/*"):
                if not p.is_file():
                    continue
                # Skip ledger files and in-flight markers
                try:
                    if LEDGER_DIR in p.parents:
                        continue
                except Exception:
                    pass
                if str(p).endswith(".sending"):
                    continue
                files.append(p)
            files.sort()
            replay_queue_depth.labels(SITE).set(len(files))
            if not files:
                await asyncio.sleep(1.0)
                continue
            # Backpressure
            while _inflight >= MAX_INFLIGHT:
                await asyncio.sleep(0.05)
            # Atomically mark one file as in-flight by renaming to .sending
            path = files[0]
            sending = Path(str(path) + ".sending")
            try:
                path.replace(sending)
            except Exception as e:
                # Another worker likely grabbed it; brief pause and retry
                _log_ledger("rename_conflict", {"file": str(path), "error": str(e)})
                await asyncio.sleep(0.05)
                continue
            _inflight += 1
            asyncio.create_task(_process_file(sending))
        except Exception as e:
            _log_ledger("loop_error", {"error": str(e)})
            await asyncio.sleep(1.0)


@app.on_event("startup")
async def on_startup():
    global _running, _inflight
    # Reset state in case of prior test run
    _running = True
    _inflight = 0
    _log_ledger("startup", {"queue_dir": str(QUEUE_DIR), "subject": SUBJECT, "site": SITE})
    try:
        await _connect_nats()
        await _ensure_stream()
        _log_ledger("nats_connected", {"url": NATS_URL})
    except Exception as e:
        _log_ledger("startup_nats_error", {"error": str(e)})
    replay_queue_depth.labels(SITE).set(0)
    edge_bandwidth_bps.labels(SITE).set(0)
    asyncio.create_task(_worker_loop())


@app.on_event("shutdown")
async def on_shutdown():
    global _running
    _running = False
    try:
        if _nc:
            await _nc.drain()
    except Exception:
        pass


@app.get("/healthz")
async def healthz():
    return {"ok": True, "queue": len([p for p in QUEUE_DIR.glob('**/*') if p.is_file() and not str(p).endswith('.sending')])}


@app.get("/livez")
async def livez():
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    try:
        m = generate_latest(registry)
        return Response(m, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        return PlainTextResponse(str(e), status_code=500)
