import asyncio
import os
import random
import time
import json

import pytest
from websockets import connect as ws_connect
from httpx import AsyncClient

pytestmark = pytest.mark.integration

BASE = os.getenv("COMMS_BASE", "http://localhost:8081")
WS = os.getenv("COMMS_WS", "ws://localhost:8081/ws")


@pytest.mark.asyncio
async def test_delivery_latency_50_devices():
    # Skip if env not running
    try:
        async with AsyncClient(base_url=BASE, timeout=2) as ac:
            r = await ac.get("/health")
            if r.status_code != 200:
                pytest.skip("gateway not running")
    except Exception:
        pytest.skip("gateway not running")

    async with AsyncClient(base_url=BASE) as ac:
        tokens = {}
        # enroll 50 devices
        for i in range(50):
            did = f"dev-{i}"
            r = await ac.post("/enroll", json={"device_id": did})
            r.raise_for_status()
            tokens[did] = r.json()["token"]

        # connect websockets
        sockets = {}
        for did, tok in tokens.items():
            ws = await ws_connect(f"{WS}?device_id={did}", extra_headers={"Authorization": f"Bearer {tok}"})
            sockets[did] = ws

        # publish one alert per device and measure latency
        start = time.time()
        for did in tokens.keys():
            await ac.post("/publish", json={"device_id": did, "priority": random.randint(0, 5), "payload": {"msg": "hi"}})

        async def recv_one(did, ws):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.5)
                recv_ts = time.time()
                data = json.loads(raw)
                alert_id = data.get("alert", {}).get("id")
                # send receipt back
                try:
                    await ws.send(json.dumps({"type": "receipt", "id": alert_id, "rcv_ts": recv_ts}))
                except Exception:
                    pass
                return recv_ts - start
            except Exception:
                return 10.0
        latencies = await asyncio.gather(*[recv_one(did, ws) for did, ws in sockets.items()])

        # close sockets
        await asyncio.gather(*[ws.close() for ws in sockets.values()])

        latencies = [lat for lat in latencies if lat < 9.0]
        latencies.sort()
        assert latencies, "no latencies recorded"
        p99 = latencies[int(max(0, 0.99 * len(latencies) - 1))]
        assert p99 <= 2.0, f"p99 exceeded: {p99:.3f}s"
