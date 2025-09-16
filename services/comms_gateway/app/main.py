import asyncio
import base64
import json
import logging
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
try:
    from aiolimiter import AsyncLimiter  # type: ignore
except Exception:  # pragma: no cover
    AsyncLimiter = None  # type: ignore

# Crypto (AES-256-GCM)
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
except Exception:  # pragma: no cover
    AESGCM = None  # type: ignore

logger = logging.getLogger("comms-gateway")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Storage and logs
BASE_DIR = os.getenv("COMMS_DATA_DIR", "var/comms")
LOG_DIR = os.path.join(BASE_DIR, "logs", "delivery")
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
STORE_DIR = os.path.join(BASE_DIR, "store")
os.makedirs(STORE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, "devices.sqlite")

app = FastAPI(title="Soldier Comms Gateway")
# Use allowlist from env to avoid wildcard '*' in CORS; default to localhost only
_cors = [o.strip() for o in os.getenv("COMMS_CORS_ORIGINS", "http://localhost").split(",") if o.strip()]
REQUIRE_MTLS = os.getenv("COMMS_REQUIRE_MTLS", "0") == "1"
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Device-ID"],
)

# Metrics
COMMS_RATE_LIMITED = Counter(
    "comms_device_rate_limited_total",
    "Total messages dropped or delayed due to per-device rate limiting",
    labelnames=("device_id",),
)
COMMS_DELIVERY_S = Histogram(
    "comms_alert_delivery_seconds",
    "End-to-end time from publish to device receipt",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 10),
    labelnames=("device_id", "priority"),
)

# Jitter defaults
JITTER_MS = int(os.getenv("COMMS_JITTER_MS", "100"))
DEFAULT_RPS = float(os.getenv("COMMS_DEFAULT_RPS", "2"))  # per-device
DEFAULT_BURST = int(os.getenv("COMMS_DEFAULT_BURST", "5"))

# AES-256-GCM key
_key_b64 = os.getenv("COMMS_AES_KEY")
if _key_b64:
    try:
        AES_KEY = base64.b64decode(_key_b64)
    except Exception as e:  # pragma: no cover
        logger.warning("Invalid COMMS_AES_KEY, generating ephemeral: %s", e)
        AES_KEY = AESGCM.generate_key(bit_length=256) if AESGCM else os.urandom(32)
else:
    AES_KEY = AESGCM.generate_key(bit_length=256) if AESGCM else os.urandom(32)


@dataclass
class DeviceState:
    token: str
    subject: str  # from client cert, if available
    limiter: Any
    ws: Optional[WebSocket] = None
    queue: "asyncio.PriorityQueue[tuple[int, float, dict]]" = field(default_factory=asyncio.PriorityQueue)
    consumer_task: Optional[asyncio.Task] = None
    # custom rate limiter state
    _rate_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS devices(device_id TEXT PRIMARY KEY, token TEXT, subject TEXT, created_at REAL)"
    )
    return conn


def _load_devices() -> Dict[str, DeviceState]:
    devices: Dict[str, DeviceState] = {}
    if not os.path.exists(DB_PATH):
        return devices
    conn = _conn()
    cur = conn.cursor()
    for device_id, token, subject, _ in cur.execute("SELECT device_id, token, subject, created_at FROM devices"):
        limiter = AsyncLimiter(DEFAULT_RPS, DEFAULT_BURST) if AsyncLimiter else None
        devices[device_id] = DeviceState(token=token, subject=subject or "", limiter=limiter)
    return devices


DEVICES: Dict[str, DeviceState] = _load_devices()


class EnrollmentRequest(BaseModel):
    device_id: str


class EnrollmentResponse(BaseModel):
    device_id: str
    token: str


class Alert(BaseModel):
    device_id: str
    priority: int = 5  # 0 highest
    payload: Dict[str, Any]
    evidence_links: List[str] | None = None
    id: str | None = None
    ts: float | None = None


class MultiAlert(BaseModel):
    alerts: List[Alert]


async def require_device(request: Request, authorization: str = Header(default="")) -> str:
    # Expect Bearer <token>
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    device_id = request.query_params.get("device_id") or request.headers.get("X-Device-ID")
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id required")
    ds = DEVICES.get(device_id)
    if not ds or ds.token != token:
        raise HTTPException(status_code=403, detail="invalid device token")
    return device_id


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/enroll", response_model=EnrollmentResponse)
async def enroll(
    req: EnrollmentRequest,
    x_forwarded_client_cert: str | None = Header(default=None),
    ssl_client_cert: str | None = Header(default=None),
    ssl_client_subject_dn: str | None = Header(default=None),
):
    # If mTLS ingress is configured, the client cert subject may be injected via headers
    subject = ""
    # Prefer explicit subject DN if present
    if ssl_client_subject_dn:
        subject = ssl_client_subject_dn[:512]
    elif x_forwarded_client_cert:
        subject = x_forwarded_client_cert[:512]
    elif ssl_client_cert:
        # Fall back to raw cert header if provided (truncated)
        subject = ssl_client_cert[:512]
    if REQUIRE_MTLS and not subject:
        raise HTTPException(status_code=400, detail="client cert subject required")
    token = secrets.token_urlsafe(24)
    limiter = AsyncLimiter(DEFAULT_RPS, DEFAULT_BURST) if AsyncLimiter else None
    DEVICES[req.device_id] = DeviceState(token=token, subject=subject, limiter=limiter)
    conn = _conn()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO devices(device_id, token, subject, created_at) VALUES (?,?,?,?)",
            (req.device_id, token, subject, time.time()),
        )
    return EnrollmentResponse(device_id=req.device_id, token=token)


async def _persist_encrypted(filename: str, data: bytes) -> None:
    if AESGCM is None:
        # Fallback: best-effort store cleartext if crypto unavailable
        with open(filename, "wb") as f:
            f.write(data)
        return
    aes = AESGCM(AES_KEY)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, data, None)
    with open(filename, "wb") as f:
        f.write(nonce + ct)


async def _write_receipt(device_id: str, receipt: dict):
    path = os.path.join(LOG_DIR, f"{device_id}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(receipt) + "\n")


async def _allow_send_now(ds: DeviceState) -> bool:
    """Return True if within rate and burst; else False and record a limit event."""
    # Custom limiter: allow DEFAULT_BURST within any short window, and cap to DEFAULT_RPS
    now = time.time()
    # Drop old entries beyond 1 second
    cutoff = now - 1.0
    while ds._rate_timestamps and ds._rate_timestamps[0] < cutoff:
        ds._rate_timestamps.popleft()
    # Enforce burst
    if len(ds._rate_timestamps) >= DEFAULT_BURST:
        return False
    # Enforce RPS (approximate): if last allowed time makes average interval < 1/RPS
    if len(ds._rate_timestamps) >= 1:
        min_interval = 1.0 / max(DEFAULT_RPS, 0.0001)
        if (now - ds._rate_timestamps[-1]) < min_interval:
            return False
    ds._rate_timestamps.append(now)
    return True


async def _device_consumer(device_id: str, ds: DeviceState):
    try:
        while True:
            # priority lower int means higher priority
            prio, enqueue_ts, alert = await ds.queue.get()
            # Jitter window to allow reordering/dedup before delivery
            now = time.time()
            delay = (enqueue_ts + JITTER_MS / 100.0) - now
            if delay > 0:
                await asyncio.sleep(delay)
            # Rate limiting (non-blocking)
            allowed = await _allow_send_now(ds)
            if not allowed:
                COMMS_RATE_LIMITED.labels(device_id=device_id).inc()
                # requeue slightly later with same priority
                await asyncio.sleep(0.05)
                await ds.queue.put((prio, time.time(), alert))
                continue
            if not ds.ws:
                # Store offline for later delivery (encrypted)
                filename = os.path.join(STORE_DIR, f"{device_id}_{int(time.time()*1000)}.bin")
                await _persist_encrypted(filename, json.dumps(alert).encode("utf-8"))
                continue
            # Send to device
            try:
                await ds.ws.send_json({"type": "alert", "alert": alert})
                # Measure latency if publish_ts present
                if alert.get("ts"):
                    COMMS_DELIVERY_S.labels(device_id=device_id, priority=str(prio)).observe(max(0.0, time.time() - float(alert["ts"])))
            except Exception as e:  # pragma: no cover
                logger.warning("send failed to %s: %s", device_id, e)
                # Put back for retry with slight penalty
                await asyncio.sleep(0.2)
                await ds.queue.put((min(prio + 1, 9), time.time(), alert))
    except asyncio.CancelledError:  # pragma: no cover
        pass


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    device_id = websocket.query_params.get("device_id")
    token = websocket.headers.get("authorization", "").replace("Bearer ", "")
    if not device_id:
        await websocket.close(code=4400)
        return
    ds = DEVICES.get(device_id)
    if not ds or ds.token != token:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    ds.ws = websocket
    if ds.consumer_task is None or ds.consumer_task.done():
        ds.consumer_task = asyncio.create_task(_device_consumer(device_id, ds))
    try:
        while True:
            msg = await websocket.receive_json()
            # Expect receipts: {"type":"receipt","id":...,"rcv_ts":...}
            if msg.get("type") == "receipt":
                await _write_receipt(device_id, msg)
            # else ignore for now
    except WebSocketDisconnect:
        ds.ws = None
    finally:
        # Keep consumer alive to buffer offline
        pass


@app.post("/publish")
async def publish(alert: Alert):
    # Assign id and ts if missing
    alert.id = alert.id or secrets.token_hex(8)
    alert.ts = alert.ts or time.time()
    ds = DEVICES.get(alert.device_id)
    if not ds:
        raise HTTPException(status_code=404, detail="unknown device")
    await ds.queue.put((max(0, min(alert.priority, 9)), time.time(), alert.model_dump()))
    return {"status": "queued", "id": alert.id}


@app.post("/publish/batch")
async def publish_batch(batch: MultiAlert):
    enq = 0
    for a in batch.alerts:
        a.id = a.id or secrets.token_hex(8)
        a.ts = a.ts or time.time()
        ds = DEVICES.get(a.device_id)
        if not ds:
            continue
        await ds.queue.put((max(0, min(a.priority, 9)), time.time(), a.model_dump()))
        enq += 1
    return {"status": "queued", "count": enq}


# Optional offline TTS stub (no heavy deps). Stores SSML for device retrieval.
class TTSRequest(BaseModel):
    device_id: str
    text: str


@app.post("/tts/offline")
async def tts_offline(req: TTSRequest):
    ssml = f"<speak>{json.dumps(req.text)[1:-1]}</speak>"
    filename = os.path.join(STORE_DIR, f"tts_{req.device_id}_{int(time.time()*1000)}.ssml")
    await _persist_encrypted(filename, ssml.encode("utf-8"))
    return {"status": "stored", "path": filename}
