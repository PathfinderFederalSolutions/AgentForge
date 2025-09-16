from __future__ import annotations
import os
import math
import time
import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import Response
from contextlib import asynccontextmanager
import contextlib

# Prometheus metrics (graceful degrade)
try:
    from prometheus_client import CollectorRegistry, Histogram, CONTENT_TYPE_LATEST, generate_latest
except Exception:  # pragma: no cover
    CollectorRegistry = None  # type: ignore
    Histogram = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain"  # type: ignore
    def generate_latest(_r):  # type: ignore
        return b""

# Optional NATS subscription for ISR updates
NATS_URL = os.getenv("NATS_URL", "")
NATS_SUBJECT = os.getenv("ROUTE_NATS_SUBJECT", "isr.events")

# AOI bounding box (lon_min, lat_min, lon_max, lat_max)
AOI = tuple(
    float(x) for x in os.getenv("ROUTE_AOI_BBOX", "-122.6,37.6,-122.3,37.9").split(",")
)  # San Francisco area by default
GRID_SIZE = int(os.getenv("ROUTE_GRID_SIZE", "128"))
ALT_COUNT = int(os.getenv("ROUTE_ALTERNATES", "2"))
MAX_REPLAN_S = float(os.getenv("ROUTE_MAX_REPLAN_SECONDS", "1.0"))

# Initialize FastAPI app early so decorators below have a defined symbol
app = FastAPI(title="route-engine", version="0.1.0")

_registry = CollectorRegistry() if CollectorRegistry else None
_route_hist = (
    Histogram(
        "route_compute_seconds",
        "Route computation time in seconds",
        ["result"],
        registry=_registry,
        buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5),
    )
    if Histogram and _registry
    else None
)


@dataclass
class Hazard:
    lon: float
    lat: float
    radius_m: float
    cost: float = 5.0
    evidence: Optional[str] = None
    seen_at: float = field(default_factory=time.time)


@dataclass
class Grid:
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    n: int
    base_cost: float = 1.0

    def __post_init__(self):
        self.costs = [self.base_cost] * (self.n * self.n)  # type: List[float]
        self.obstacles: Set[int] = set()
        self.hazards: List[Hazard] = []

    # Coordinate transforms
    def to_idx(self, lon: float, lat: float) -> int:
        i, j = self.to_ij(lon, lat)
        return self.ij_to_idx(i, j)

    def to_ij(self, lon: float, lat: float) -> Tuple[int, int]:
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min)
        y = (lat - self.lat_min) / (self.lat_max - self.lat_min)
        i = max(0, min(self.n - 1, int(x * self.n)))
        j = max(0, min(self.n - 1, int((1 - y) * self.n)))  # lat inversed
        return i, j

    def ij_to_idx(self, i: int, j: int) -> int:
        return j * self.n + i

    def idx_to_ij(self, idx: int) -> Tuple[int, int]:
        return idx % self.n, idx // self.n

    def ij_to_lonlat(self, i: int, j: int) -> Tuple[float, float]:
        x = (i + 0.5) / self.n
        y = (j + 0.5) / self.n
        lon = self.lon_min + x * (self.lon_max - self.lon_min)
        lat = self.lat_min + (1 - y) * (self.lat_max - self.lat_min)
        return lon, lat

    # Hazard fusion into costmap
    def add_hazard(self, hz: Hazard):
        self.hazards.append(hz)
        self._apply_hazard(hz)

    def _apply_hazard(self, hz: Hazard):
        # Approx meters per degree latitude/longitude at mid-latitude
        meters_per_deg_lat = 111_000.0
        meters_per_deg_lon = 88_000.0 * math.cos(math.radians((self.lat_min + self.lat_max) / 2.0))
        radius_lon = hz.radius_m / max(1.0, meters_per_deg_lon)
        radius_lat = hz.radius_m / meters_per_deg_lat
        # Determine bounding ij box for speed
        lon0, lat0 = hz.lon, hz.lat
        i_min, j_min = self.to_ij(lon0 - radius_lon, lat0 + radius_lat)
        i_max, j_max = self.to_ij(lon0 + radius_lon, lat0 - radius_lat)
        for j in range(min(j_min, j_max), max(j_min, j_max) + 1):
            for i in range(min(i_min, i_max), max(i_min, i_max) + 1):
                idx = self.ij_to_idx(i, j)
                lon, lat = self.ij_to_lonlat(i, j)
                # Elliptical distance heuristic
                dx = (lon - lon0) * meters_per_deg_lon
                dy = (lat - lat0) * meters_per_deg_lat
                d = math.hypot(dx, dy)
                if d <= hz.radius_m:
                    # Smooth bump: closer -> higher additive cost
                    k = 1.0 - (d / max(1.0, hz.radius_m))
                    self.costs[idx] += hz.cost * (0.25 + 0.75 * k * k)

    # A* search on 8-connected grid
    def neighbors(self, idx: int) -> List[Tuple[int, float]]:
        i, j = self.idx_to_ij(idx)
        out: List[Tuple[int, float]] = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.n and 0 <= nj < self.n:
                    nidx = self.ij_to_idx(ni, nj)
                    # Diagonal has sqrt(2) distance weight
                    step = math.hypot(di, dj)
                    # Edge cost as mean of node costs
                    cost = step * 0.5 * (self.costs[idx] + self.costs[nidx])
                    out.append((nidx, cost))
        return out

    def heuristic(self, a: int, b: int) -> float:
        ia, ja = self.idx_to_ij(a)
        ib, jb = self.idx_to_ij(b)
        return math.hypot(ia - ib, ja - jb)

    def a_star(self, start_idx: int, goal_idx: int) -> Tuple[List[int], float]:
        import heapq
        openq: List[Tuple[float, int]] = []
        heapq.heappush(openq, (0.0, start_idx))
        came: Dict[int, int] = {}
        g: Dict[int, float] = {start_idx: 0.0}
        goal = goal_idx
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == goal:
                # Reconstruct
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path, g[goal]
            for nb, w in self.neighbors(cur):
                cand = g[cur] + w
                if cand < g.get(nb, 1e18):
                    g[nb] = cand
                    came[nb] = cur
                    f = cand + self.heuristic(nb, goal)
                    heapq.heappush(openq, (f, nb))
        raise RuntimeError("no path")

    def route(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Dict[str, Any]:
        s_idx = self.to_idx(*start)
        g_idx = self.to_idx(*goal)
        path, cost = self.a_star(s_idx, g_idx)
        # Convert to lon/lat list
        coords = [self.ij_to_lonlat(*self.idx_to_ij(p)) for p in path]
        # Summarize hazards intersected/avoided near path
        touched: List[Hazard] = []
        for hz in self.hazards:
            if self._path_touches_hazard(coords, hz):
                touched.append(hz)
        return {
            "path": coords,
            "cost": cost,
            "hazards": [
                {"lon": h.lon, "lat": h.lat, "radius_m": h.radius_m, "evidence": h.evidence}
                for h in touched
            ],
        }

    def _path_touches_hazard(self, coords: List[Tuple[float, float]], hz: Hazard) -> bool:
        # Simple nearest-point distance check
        meters_per_deg_lat = 111_000.0
        meters_per_deg_lon = 88_000.0 * math.cos(math.radians((self.lat_min + self.lat_max) / 2.0))
        for lon, lat in coords:
            dx = (lon - hz.lon) * meters_per_deg_lon
            dy = (lat - hz.lat) * meters_per_deg_lat
            if math.hypot(dx, dy) <= hz.radius_m:
                return True
        return False


# Global grid instance
_grid = Grid(AOI[0], AOI[1], AOI[2], AOI[3], GRID_SIZE)

# Explanations log
LOG_DIR = os.getenv("ROUTE_LOG_DIR", "var/logs/route_engine")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "route_explanations.jsonl")


def _log_explanation(data: Dict[str, Any]) -> None:
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **data}, separators=(",", ":")) + "\n")
    except Exception:
        pass


class RouteRequest(BaseModel):
    start_lon: float
    start_lat: float
    goal_lon: float
    goal_lat: float
    alternates: int = Field(default=ALT_COUNT, ge=0, le=5)


class RouteResponse(BaseModel):
    primary: Dict[str, Any]
    alternates: List[Dict[str, Any]]
    compute_ms: float


class ISRUpdate(BaseModel):
    lon: float
    lat: float
    radius_m: float = Field(ge=1.0, le=5_000.0, default=200.0)
    cost: float = Field(ge=0.1, le=50.0, default=5.0)
    evidence: Optional[str] = None


@app.get("/metrics")
async def metrics():
    if not _registry:
        return Response(b"", media_type=CONTENT_TYPE_LATEST)
    try:
        return Response(generate_latest(_registry), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        return Response(str(e).encode(), media_type="text/plain", status_code=500)


@app.post("/isr")
async def ingest_isr(upd: ISRUpdate):
    hz = Hazard(lon=upd.lon, lat=upd.lat, radius_m=upd.radius_m, cost=upd.cost, evidence=upd.evidence)
    _grid.add_hazard(hz)
    _log_explanation({
        "event": "isr_ingest",
        "lon": upd.lon,
        "lat": upd.lat,
        "radius_m": upd.radius_m,
        "cost": upd.cost,
        "evidence": upd.evidence,
    })
    return {"status": "ok", "hazards": len(_grid.hazards)}


@app.post("/routes", response_model=RouteResponse)
async def compute_routes(req: RouteRequest):
    start = (req.start_lon, req.start_lat)
    goal = (req.goal_lon, req.goal_lat)
    t0 = time.time()
    try:
        primary = _grid.route(start, goal)
        alts: List[Dict[str, Any]] = []
        # Simple alternates by penalizing used cells progressively
        used_idxs = {
            _grid.to_idx(lon, lat) for lon, lat in primary["path"]
        }
        for k in range(max(0, req.alternates)):
            # Apply mild penalty to used cells to encourage diversity
            penalty = 0.25 * (k + 1)
            _apply_penalty(used_idxs, penalty)
            alt = _grid.route(start, goal)
            alts.append(alt)
            # Update used set
            used_idxs |= {
                _grid.to_idx(lon, lat) for lon, lat in alt["path"]
            }
        dt = time.time() - t0
        if _route_hist:
            try:
                _route_hist.labels(result="ok").observe(dt)
            except Exception:
                pass
        # Explanation with evidence refs
        evidence_refs = [h.get("evidence") for h in primary.get("hazards", []) if h.get("evidence")]
        _log_explanation({
            "event": "route_compute",
            "ms": dt * 1000.0,
            "primary_cost": primary.get("cost"),
            "alternates": len(alts),
            "evidence_refs": evidence_refs,
        })
        return RouteResponse(primary=primary, alternates=alts, compute_ms=dt * 1000.0)
    except Exception as e:
        dt = time.time() - t0
        if _route_hist:
            try:
                _route_hist.labels(result="error").observe(dt)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))


# Helper to apply transient penalties (not persisted across process restarts)
def _apply_penalty(idxs: set[int], k: float) -> None:
    for idx in idxs:
        try:
            _grid.costs[idx] += k
        except Exception:
            pass


# Background: optional NATS ISR subscription
async def _nats_consumer():  # pragma: no cover - best effort dependency
    if not NATS_URL:
        return
    try:
        import nats
        nc = await nats.connect(servers=[NATS_URL], max_reconnect_attempts=-1, reconnect_time_wait=0.5)
        async def handler(msg):
            try:
                data = json.loads(msg.data.decode())
                upd = ISRUpdate(**data)
                await ingest_isr(upd)  # reuse HTTP handler logic
            except Exception:
                pass
        await nc.subscribe(NATS_SUBJECT, cb=handler)
        # Keep running
        while True:
            await asyncio.sleep(60)
    except Exception:
        # Silent fail to avoid crashing app if NATS not present
        return


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[override]
    # Warm-up: ensure basic route computes within threshold
    try:
        s = (AOI[0] + 0.01, AOI[1] + 0.01)
        g = (AOI[2] - 0.01, AOI[3] - 0.01)
        _ = _grid.route(s, g)
    except Exception:
        pass
    # Start NATS consumer
    task = None
    if NATS_URL:
        task = asyncio.create_task(_nats_consumer())
    try:
        yield
    finally:
        if task:
            task.cancel()
            with contextlib.suppress(Exception):
                await task

# Attach lifespan to app (set after definition to avoid reordering large blocks)
app.router.lifespan_context = lifespan


@app.get("/healthz")
async def healthz():
    return {"ok": True, "hazards": len(_grid.hazards), "grid": _grid.n}


# Uvicorn entry when running as module: python -m services.route_engine.app.main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.route_engine.app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8010")), reload=False)
