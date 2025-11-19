from __future__ import annotations
import asyncio
import os
from typing import Optional, Tuple

import nats
from nats.aio.client import Client as NATS
from nats.js import JetStreamContext

from .config import settings

_NC: Optional[NATS] = None
_JS: Optional[JetStreamContext] = None
_LOCK = asyncio.Lock()

async def _connect() -> Tuple[NATS, JetStreamContext]:
    global _NC, _JS
    if _NC is not None and _JS is not None and getattr(_NC, "is_connected", False):
        return _NC, _JS

    async with _LOCK:
        if (_NC is not None and _JS is not None and getattr(_NC, "is_connected", False)):
            return _NC, _JS

        servers = [os.getenv("NATS_URL", settings.nats_url)]
        user = os.getenv("NATS_USER") or None
        password = os.getenv("NATS_PASSWORD") or None

        _NC = await nats.connect(
            servers=servers,
            user=user,
            password=password,
            connect_timeout=5,
            allow_reconnect=True,
            max_reconnect_attempts=-1,
            reconnect_time_wait=2,
            ping_interval=20,
            max_outstanding_pings=5,
            drain_timeout=5,
            name=os.getenv("NATS_CLIENT_NAME", "agentforge-worker"),
        )
        _JS = _NC.jetstream()
        return _NC, _JS

async def get_nc_async() -> NATS:
    nc, _ = await _connect()
    return nc

async def get_js_async() -> JetStreamContext:
    _, js = await _connect()
    return js

async def close():
    global _NC, _JS
    try:
        if _NC and not _NC.is_closed:
            await _NC.drain()
    finally:
        _NC = None
        _JS = None
