import os
import pytest
from swarm.api.main import app
from starlette.testclient import TestClient


def test_sse_requires_bearer_when_enabled(monkeypatch):
    monkeypatch.setenv("TACTICAL_REQUIRE_CLIENT_CERT", "0")
    monkeypatch.setenv("TACTICAL_REQUIRE_BEARER", "1")
    monkeypatch.setenv("TACTICAL_BEARER_TOKEN", "dev-token")
    with TestClient(app) as client:
        # Without bearer -> unauthorized
        resp = client.get("/events/stream", stream=True)
        assert resp.status_code == 401
        # With wrong bearer -> unauthorized
        resp2 = client.get("/events/stream", headers={"Authorization": "Bearer wrong"}, stream=True)
        assert resp2.status_code == 401
        # With correct bearer -> allowed
        with client.stream("GET", "/events/stream", headers={"Authorization": "Bearer dev-token"}) as resp3:
            assert resp3.status_code == 200


def test_ws_requires_bearer_when_enabled(monkeypatch):
    monkeypatch.setenv("TACTICAL_REQUIRE_CLIENT_CERT", "0")
    monkeypatch.setenv("TACTICAL_REQUIRE_BEARER", "1")
    monkeypatch.setenv("TACTICAL_BEARER_TOKEN", "dev-token")
    with TestClient(app) as client:
        with pytest.raises(Exception):
            client.websocket_connect("/events/ws")
        # Wrong bearer rejected
        with pytest.raises(Exception):
            client.websocket_connect("/events/ws", headers={"Authorization": "Bearer wrong"})
        # Correct connects
        with client.websocket_connect("/events/ws", headers={"Authorization": "Bearer dev-token"}):
            pass
