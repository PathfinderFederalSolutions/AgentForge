import os
import time

import pytest
from websockets import connect as ws_connect
from httpx import AsyncClient

pytestmark = pytest.mark.integration

BASE = os.getenv("COMMS_BASE", "http://localhost:8081")
WS = os.getenv("COMMS_WS", "ws://localhost:8081/ws")


@pytest.mark.asyncio
async def test_rate_limit_enforced():
    try:
        async with AsyncClient(base_url=BASE, timeout=2) as ac:
            r = await ac.get("/health")
            if r.status_code != 200:
                pytest.skip("gateway not running")
    except Exception:
        pytest.skip("gateway not running")

    async with AsyncClient(base_url=BASE) as ac:
        did = "rl-dev"
        tok = (await ac.post("/enroll", json={"device_id": did})).json()["token"]
        ws = await ws_connect(f"{WS}?device_id={did}", extra_headers={"Authorization": f"Bearer {tok}"})

        # blast 20 publishes quickly
        for i in range(20):
            await ac.post("/publish", json={"device_id": did, "priority": 1, "payload": {"n": i}})
        # receive with expected spacing due to limiter (DEFAULT_RPS=2)
        first = None
        recv_times = []
        for _ in range(10):
            await ws.recv()
            t = time.time()
            recv_times.append(t)
            if first is None:
                first = t
        await ws.close()
        # verify average rps <= 3 within the window
        if len(recv_times) > 1:
            duration = recv_times[-1] - recv_times[0]
            avg_rps = len(recv_times) / (duration + 1e-6)
            assert avg_rps <= 3.0
