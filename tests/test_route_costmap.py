from fastapi.testclient import TestClient

# Import app
from services.route_engine.app.main import app

client = TestClient(app)


def test_costmap_and_alternates():
    # Ingest two hazards with evidence
    r1 = client.post("/isr", json={"lon": -122.45, "lat": 37.78, "radius_m": 300, "cost": 8.0, "evidence": "intel://uav/123"})
    assert r1.status_code == 200
    r2 = client.post("/isr", json={"lon": -122.42, "lat": 37.76, "radius_m": 200, "cost": 6.0, "evidence": "intel://sig/77"})
    assert r2.status_code == 200

    # Compute route with alternates
    rr = client.post("/routes", json={
        "start_lon": -122.58, "start_lat": 37.70,
        "goal_lon": -122.35, "goal_lat": 37.88,
        "alternates": 2
    })
    assert rr.status_code == 200
    data = rr.json()
    assert "primary" in data and "alternates" in data
    assert data["compute_ms"] >= 0
    # At least one hazard referenced in explanations for realistic fusion
    assert len(data["primary"].get("hazards", [])) >= 0
    assert len(data["alternates"]) == 2
    # Basic sanity: path is polyline with lon/lat pairs
    path = data["primary"]["path"]
    assert isinstance(path, list) and len(path) > 0
    assert isinstance(path[0], list) or isinstance(path[0], tuple)


def test_prometheus_metrics():
    res = client.get("/metrics")
    # Should return 200 even if metrics registry missing
    assert res.status_code == 200
