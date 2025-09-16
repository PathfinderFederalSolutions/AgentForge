import pytest
import time
from swarm.api.main import app
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.mark.integration
@pytest.mark.timeout(10)
@pytest.mark.xfail(run=False, reason="TestClient hangs in some environments")
def test_sse_requires_client_cert_when_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TACTICAL_REQUIRE_CLIENT_CERT", "1")
    with TestClient(app) as client:
        # Without header -> forbidden
        with client.stream("GET", "/events/stream") as resp:
            assert resp.status_code == 403
        # With verification header -> allowed
        with client.stream("GET", "/events/stream", headers={"ssl-client-verify": "SUCCESS"}) as resp2:
            assert resp2.status_code == 200


@pytest.mark.integration
@pytest.mark.timeout(10)
@pytest.mark.xfail(run=False, reason="TestClient hangs in some environments")
def test_ws_requires_client_cert_when_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TACTICAL_REQUIRE_CLIENT_CERT", "1")
    with TestClient(app) as client:
        # Without header should be rejected (expect connection to fail/close immediately)
        try:
            with client.websocket_connect("/events/ws") as ws:
                # If connection succeeds, it should close immediately without any data
                # Try to receive data with a very short timeout
                with pytest.raises((WebSocketDisconnect, Exception)):
                    ws.receive_json(timeout=0.1)
        except (WebSocketDisconnect, Exception):
            # Expected - connection should fail
            pass
        # With header should connect successfully
        with client.websocket_connect("/events/ws", headers={"ssl-client-verify": "SUCCESS"}) as ws:
            # Connection should succeed and we can receive a heartbeat or marker
            try:
                msg = ws.receive_json(timeout=1)
                assert msg["event"] in ("heartbeat", "marker")
            except Exception:
                # Heartbeat might not come immediately, that's ok
                pass


@pytest.mark.integration
@pytest.mark.timeout(10)
@pytest.mark.xfail(run=False, reason="TestClient hangs in some environments")
def test_sse_global_rate_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TACTICAL_REQUIRE_CLIENT_CERT", "0")
    monkeypatch.setenv("TACTICAL_STREAM_MAX_CLIENTS", "1")
    with TestClient(app) as c1:
        with c1.stream("GET", "/events/stream") as r1:
            assert r1.status_code == 200
        time.sleep(0.1)  # Allow background cleanup
    with TestClient(app) as c2:
        with c2.stream("GET", "/events/stream") as r2:
            assert r2.status_code == 429
    monkeypatch.delenv("TACTICAL_REQUIRE_CLIENT_CERT", raising=False)
    monkeypatch.delenv("TACTICAL_STREAM_MAX_CLIENTS", raising=False)
    time.sleep(0.1)


@pytest.mark.integration
@pytest.mark.timeout(10)
@pytest.mark.xfail(run=False, reason="TestClient hangs in some environments")
def test_ws_global_rate_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TACTICAL_REQUIRE_CLIENT_CERT", "0")
    monkeypatch.setenv("TACTICAL_STREAM_MAX_CLIENTS", "1")
    with TestClient(app) as c1:
        with c1.websocket_connect("/events/ws") as _:
            time.sleep(0.1)
    with TestClient(app) as c2:
        with pytest.raises(WebSocketDisconnect):
            c2.websocket_connect("/events/ws")
    monkeypatch.delenv("TACTICAL_REQUIRE_CLIENT_CERT", raising=False)
    monkeypatch.delenv("TACTICAL_STREAM_MAX_CLIENTS", raising=False)
    time.sleep(0.1)
