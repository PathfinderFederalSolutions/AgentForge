import json
import time
from swarm.api.main import app, _test_emit_geojson
from starlette.testclient import TestClient


def _read_sse_lines(resp):
    buf = b""
    for chunk in resp.iter_bytes():
        buf += chunk
        while b"\n\n" in buf:
            packet, buf = buf.split(b"\n\n", 1)
            yield packet.decode()


def test_sse_stream_emits_marker_and_heartbeat():
    with TestClient(app) as client:
        # Start SSE
        with client.stream("GET", "/events/stream") as resp:
            assert resp.status_code == 200
            # Emit a marker on the app loop
            client.portal.call(_test_emit_geojson, {"type": "Point", "coordinates": [10, 20]}, {"title": "A", "evidence_id": "abc"})
            # Read until we get marker
            started = time.time()
            got_marker = False
            while time.time() - started < 5:
                for line in _read_sse_lines(resp):
                    if line.startswith("data: "):
                        payload = json.loads(line[len("data: "):])
                        assert payload["type"] == "Feature"
                        assert payload["geometry"]["type"] == "Point"
                        assert "evidence_link" in payload["properties"]
                        got_marker = True
                        break
                if got_marker:
                    break
                time.sleep(0.05)
            assert got_marker


def test_ws_stream_emits_marker_and_heartbeat():
    with TestClient(app) as client:
        with client.websocket_connect("/events/ws") as ws:
            client.portal.call(_test_emit_geojson, {"type": "Point", "coordinates": [30, 40]}, {"title": "B", "evidence_id": "def"})
            msg = ws.receive_json(timeout=2)
            assert msg["event"] in ("marker", "heartbeat")
            if msg["event"] == "heartbeat":
                msg = ws.receive_json(timeout=2)
            assert msg["event"] == "marker"
            feature = msg["data"]
            assert feature["type"] == "Feature"
            assert feature["geometry"]["type"] == "Point"
            assert "evidence_link" in feature["properties"]
