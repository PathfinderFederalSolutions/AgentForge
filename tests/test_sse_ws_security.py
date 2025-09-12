# filepath: /Users/baileymahoney/AgentForge/tests/test_sse_ws_security.py
import os
import pytest
from swarm.api.main import app
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def test_sse_requires_client_cert_when_enabled(monkeypatch):
    monkeypatch.setenv("TACTICAL_REQUIRE_CLIENT_CERT", "1")
    with TestClient(app) as client:
        # Without header -> forbidden
        resp = client.get("/events/stream", stream=True)
        assert resp.status_code == 403
        # With verification header -> allowed
        with client.stream("GET", "/events/stream", headers={"ssl-client-verify": "SUCCESS"}) as resp2:
            assert resp2.status_code == 200


def test_ws_requires_client_cert_when_enabled(monkeypatch):
    monkeypatch.setenv("TACTICAL_REQUIRE_CLIENT_CERT", "1")
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            client.websocket_connect("/events/ws")
        # With header should connect
        with client.websocket_connect("/events/ws", headers={"ssl-client-verify": "SUCCESS"}) as ws:
            # Expect either heartbeat or marker; we won't block, just ensure connected then close
            pass


def test_sse_global_rate_limit(monkeypatch):
    monkeypatch.setenv("TACTICAL_REQUIRE_CLIENT_CERT", "0")
    monkeypatch.setenv("TACTICAL_STREAM_MAX_CLIENTS", "1")
    with TestClient(app) as c1, TestClient(app) as c2:
        with c1.stream("GET", "/events/stream") as r1:
            assert r1.status_code == 200
            # Second should be over the limit
            r2 = c2.get("/events/stream", stream=True)
            assert r2.status_code == 429


def test_ws_global_rate_limit(monkeypatch):
    monkeypatch.setenv("TACTICAL_REQUIRE_CLIENT_CERT", "0")
    monkeypatch.setenv("TACTICAL_STREAM_MAX_CLIENTS", "1")
    with TestClient(app) as c1, TestClient(app) as c2:
        with c1.websocket_connect("/events/ws") as ws1:
            # Second should be rejected
            with pytest.raises(WebSocketDisconnect):
                c2.websocket_connect("/events/ws")
