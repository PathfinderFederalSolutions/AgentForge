import time
from fastapi.testclient import TestClient
from services.route_engine.app.main import app

client = TestClient(app)


def test_recompute_under_one_second():
    # Warm up a baseline route
    base = client.post("/routes", json={
        "start_lon": -122.58, "start_lat": 37.70,
        "goal_lon": -122.35, "goal_lat": 37.88,
        "alternates": 0
    })
    assert base.status_code == 200

    # Inject ISR and recompute
    r = client.post("/isr", json={"lon": -122.50, "lat": 37.80, "radius_m": 500, "cost": 10.0})
    assert r.status_code == 200

    t0 = time.time()
    rr = client.post("/routes", json={
        "start_lon": -122.58, "start_lat": 37.70,
        "goal_lon": -122.35, "goal_lat": 37.88,
        "alternates": 1
    })
    dt = time.time() - t0
    assert rr.status_code == 200
    assert dt < 1.0, f"Recompute took {dt}s"
