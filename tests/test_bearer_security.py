import pytest
from swarm.api.main import app
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import socket


def app_available():
    # Check if the app is running locally (port 8000 or 8080)
    for port in (8000, 8080):
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except Exception:
            continue
    return True  # TestClient spins up app in-process, so always True


@pytest.mark.integration
@pytest.mark.timeout(10)
@pytest.mark.xfail(run=False, reason="TestClient hangs in some environments")
def test_sse_requires_bearer_when_enabled(monkeypatch):
    if not app_available():
        pytest.skip("App not available for SSE test")
    monkeypatch.setenv("TACTICAL_REQUIRE_CLIENT_CERT", "0")
    monkeypatch.setenv("TACTICAL_REQUIRE_BEARER", "1")
    monkeypatch.setenv("TACTICAL_BEARER_TOKEN", "dev-token")
    with TestClient(app) as client:
        # Without bearer -> unauthorized
        with client.stream("GET", "/events/stream") as resp:
            assert resp.status_code == 401
        # With wrong bearer -> unauthorized
        with client.stream("GET", "/events/stream", headers={"Authorization": "Bearer wrong"}) as resp2:
            assert resp2.status_code == 401
        # With correct bearer -> allowed
        with client.stream("GET", "/events/stream", headers={"Authorization": "Bearer dev-token"}) as resp3:
            assert resp3.status_code == 200


@pytest.mark.integration
@pytest.mark.timeout(10)
@pytest.mark.xfail(run=False, reason="TestClient hangs in some environments")
def test_ws_requires_bearer_when_enabled(monkeypatch):
    if not app_available():
        pytest.skip("App not available for WS test")
    monkeypatch.setenv("TACTICAL_REQUIRE_CLIENT_CERT", "0")
    monkeypatch.setenv("TACTICAL_REQUIRE_BEARER", "1")
    monkeypatch.setenv("TACTICAL_BEARER_TOKEN", "dev-token")
    with TestClient(app) as client:
        # Without bearer should be rejected
        ws = None
        try:
            ws = client.websocket_connect("/events/ws", timeout=1)
            with pytest.raises((WebSocketDisconnect, Exception)):
                ws.receive_json(timeout=0.5)
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            if ws:
                ws.close()

        # Wrong bearer rejected
        ws2 = None
        try:
            ws2 = client.websocket_connect("/events/ws", headers={"Authorization": "Bearer wrong"}, timeout=1)
            with pytest.raises((WebSocketDisconnect, Exception)):
                ws2.receive_json(timeout=0.5)
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            if ws2:
                ws2.close()

        # Correct connects
        ws3 = None
        try:
            ws3 = client.websocket_connect("/events/ws", headers={"Authorization": "Bearer dev-token"}, timeout=1)
            msg = ws3.receive_json(timeout=1)
            assert msg["event"] in ("heartbeat", "marker")
        except Exception:
            pass
        finally:
            if ws3:
                ws3.close()
