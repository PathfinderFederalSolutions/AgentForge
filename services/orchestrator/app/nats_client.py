import asyncio, json, os, random
from collections import OrderedDict
from nats.aio.client import Client as NATS
from nats.js import JetStreamContext

# Metrics (graceful if prometheus_client not installed)
try:  # pragma: no cover
    from prometheus_client import Gauge  # type: ignore
except Exception:  # pragma: no cover
    Gauge = None  # type: ignore

ENV = os.getenv("ENV", "staging")
SITE = os.getenv("SITE", ENV)
JOBS_SUBJECT = f"swarm.jobs.{ENV}"
RESULTS_SUBJECT = f"swarm.results.{ENV}"
DEDUP_SIZE = int(os.getenv("NATS_DEDUP_CACHE", "2048"))
RETRY_BASE = float(os.getenv("NATS_RETRY_BASE", "0.25"))
MAX_ATTEMPTS = int(os.getenv("NATS_MAX_ATTEMPTS", "3"))
MAX_RECONNECT_WAIT = float(os.getenv("NATS_MAX_RECONNECT_WAIT", "15"))

_seen: OrderedDict[str, float] = OrderedDict()

# Prom gauge for link status
EDGE_LINK_GAUGE = None
if Gauge:
    try:
        EDGE_LINK_GAUGE = Gauge("edge_link_status", "Edge link connectivity (1 up, 0 down)", ["site"])  # type: ignore
    except Exception:
        EDGE_LINK_GAUGE = None


def _mark_seen(msg_id: str) -> bool:
    if not msg_id:
        return False
    existed = msg_id in _seen
    _seen[msg_id] = asyncio.get_event_loop().time()
    if len(_seen) > DEDUP_SIZE:
        _seen.popitem(last=False)
    return existed

NATS_MONITOR_STREAM = os.getenv("SWARM_STREAM","swarm_jobs")
RESULT_PREFIX = "swarm_results."

class NatsBus:
    def __init__(self):
        self.nc: NATS | None = None
        self.js: JetStreamContext | None = None
        self._connecting = asyncio.Lock()

    async def connect(self):
        async with self._connecting:
            # Already connected
            if self.nc and getattr(self.nc, "is_connected", False):
                return
            servers = [os.getenv("NATS_URL","nats://nats:4222")]

            async def _on_disconnect():  # pragma: no cover - simple gauge flip
                if EDGE_LINK_GAUGE:
                    try:
                        EDGE_LINK_GAUGE.labels(site=SITE).set(0)  # type: ignore
                    except Exception:
                        pass

            async def _on_reconnect():  # pragma: no cover - simple gauge flip
                if EDGE_LINK_GAUGE:
                    try:
                        EDGE_LINK_GAUGE.labels(site=SITE).set(1)  # type: ignore
                    except Exception:
                        pass

            async def _on_closed():  # pragma: no cover
                if EDGE_LINK_GAUGE:
                    try:
                        EDGE_LINK_GAUGE.labels(site=SITE).set(0)  # type: ignore
                    except Exception:
                        pass

            # Exponential backoff with jitter
            attempt = 0
            while True:
                try:
                    self.nc = NATS()
                    await self.nc.connect(
                        servers=servers,
                        reconnect_time_wait=2,
                        max_reconnect_attempts=-1,
                        allow_reconnect=True,
                        connect_timeout=5,
                        disconnected_cb=_on_disconnect,
                        reconnected_cb=_on_reconnect,
                        closed_cb=_on_closed,
                        name=os.getenv("NATS_CLIENT_NAME", "orchestrator")
                    )
                    self.js = self.nc.jetstream()
                    if EDGE_LINK_GAUGE:
                        try:
                            EDGE_LINK_GAUGE.labels(site=SITE).set(1)  # type: ignore
                        except Exception:
                            pass
                    return
                except Exception:
                    attempt += 1
                    # Jittered exponential backoff
                    backoff = min((2 ** min(attempt, 6)) * RETRY_BASE, MAX_RECONNECT_WAIT)
                    jitter = random.uniform(0, backoff / 2.0)
                    sleep_for = backoff + jitter
                    if EDGE_LINK_GAUGE:
                        try:
                            EDGE_LINK_GAUGE.labels(site=SITE).set(0)  # type: ignore
                        except Exception:
                            pass
                    await asyncio.sleep(sleep_for)

    async def publish_json(self, subject: str, data: dict):
        # Attach idempotency header if present
        msg_id = data.get("job_id") or data.get("invocation_id") or data.get("id") or data.get("request_id")
        headers = {"Nats-Msg-Id": str(msg_id)} if msg_id else None
        # Ensure connected (reconnect if needed)
        if not self.nc or not getattr(self.nc, "is_connected", False):
            await self.connect()
        await self.js.publish(subject, json.dumps(data).encode(), headers=headers)

    async def subscribe(self, subject: str, cb):
        # JetStream durable subscription with explicit ack
        durable = subject.replace('.', '_') + "_orch"
        # Ensure connection
        if not self.nc or not getattr(self.nc, "is_connected", False):
            await self.connect()

        async def _handler(msg):
            try:
                payload = json.loads(msg.data.decode())
            except Exception:
                await msg.ack()  # drop invalid
                return
            msg_id = payload.get('job_id') or payload.get('invocation_id') or payload.get('id') or payload.get('request_id')
            if _mark_seen(msg_id):
                await msg.ack()
                return
            attempt = 0
            while attempt < MAX_ATTEMPTS:
                attempt += 1
                try:
                    await cb(payload, msg)
                    await msg.ack()
                    return
                except Exception as e:
                    # only log & retry; on final attempt allow redelivery (no ack)
                    print(f"[orchestrator] handler error attempt={attempt} id={msg_id} {e}")
                    if attempt >= MAX_ATTEMPTS:
                        return
                    backoff = min(2 ** (attempt -1) * RETRY_BASE, 2.0)
                    await asyncio.sleep(backoff)
        await self.js.subscribe(subject, durable=durable, cb=_handler, manual_ack=True)

bus = NatsBus()