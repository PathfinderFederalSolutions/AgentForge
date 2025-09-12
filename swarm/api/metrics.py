"""Prometheus metrics & instrumentation utilities for FastAPI services.

Centralizes /metrics route, multiprocess handling, and common counters/histograms
so that API + orchestrator (and future services) share a single registration point.

Usage (in FastAPI app module):
    from swarm.api.metrics import metrics_router, instrument_app, TASK_SUBMIT, TASK_SUBMIT_LATENCY
    instrument_app(app)
    app.include_router(metrics_router)

Supports gunicorn/uvicorn multiprocess mode when PROMETHEUS_MULTIPROC_DIR is set.
"""
from __future__ import annotations
import os
import time
from fastapi import APIRouter, Response, Request

# Attempt to import prometheus_client; degrade gracefully if unavailable
try:  # pragma: no cover - import path differences not critical for tests
    from prometheus_client import (
        CollectorRegistry,
        multiprocess,
        generate_latest,
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        REGISTRY,
        Gauge,  # added Gauge for drift metrics
    )  # type: ignore
except Exception:  # pragma: no cover
    CollectorRegistry = None  # type: ignore
    multiprocess = None  # type: ignore
    generate_latest = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    REGISTRY = None  # type: ignore
    Gauge = None  # type: ignore

__all__ = [
    "metrics_router",
    "instrument_app",
    "TASK_SUBMIT",
    "TASK_SUBMIT_ERRORS",
    "TASK_SUBMIT_LATENCY",
    "ORCH_JOB_SUBMIT",
    "FUSION_FUSED_TRACKS",
    "FUSION_LATENCY_MS",
    "FUSION_CONFIDENCE",
    "FUSION_EVIDENCE_LEN",
    # Canary metrics
    "CANARY_TRAFFIC_FRACTION",
    "CANARY_REGRESSIONS",
    "CANARY_PROMOTIONS",
    "CANARY_ROLLBACKS",
    # Added budget / eval metrics
    "FUSION_LATENCY_BUDGET_VIOLATIONS",
    "CANARY_EVAL_LATENCY_MS",
    "CANARY_EVAL_LATENCY_BUDGET_VIOLATIONS",
    # Drift metrics
    "DRIFT_PSI",
    "DRIFT_KL",
    "DRIFT_ALERTS",
    # Tactical
    "TACTICAL_ALERTS_PUBLISHED",
]

# ----------------------------------------------------------------------------
# Registry (multiprocess aware) ------------------------------------------------
# ----------------------------------------------------------------------------

if CollectorRegistry and os.getenv("PROMETHEUS_MULTIPROC_DIR"):
    _REGISTRY = CollectorRegistry()
    try:  # pragma: no cover
        multiprocess.MultiProcessCollector(_REGISTRY)  # type: ignore
    except Exception:  # pragma: no cover
        _REGISTRY = REGISTRY  # fallback
else:
    _REGISTRY = REGISTRY if REGISTRY else None  # type: ignore

# ----------------------------------------------------------------------------
# Metric definitions (shared) --------------------------------------------------
# ----------------------------------------------------------------------------
if Counter and Histogram and _REGISTRY:
    TASK_SUBMIT = Counter(
        "task_submit_total",
        "Tasks submitted via API",
        ["dispatch"],
        registry=_REGISTRY,
    )
    TASK_SUBMIT_ERRORS = Counter(
        "task_submit_errors_total",
        "Task submission errors",
        ["reason"],
        registry=_REGISTRY,
    )
    TASK_SUBMIT_LATENCY = Histogram(
        "task_submit_latency_seconds",
        "Task submission latency",
        ["dispatch"],
        registry=_REGISTRY,
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1200),
    )

    API_REQUEST_LATENCY = Histogram(
        "api_request_latency_seconds",
        "Overall API request latency by method & path",
        ["method", "path", "status"],
        registry=_REGISTRY,
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    )

    ORCH_JOB_SUBMIT = Counter(
        "orchestrator_job_submit_total",
        "Jobs submitted via orchestrator API",
        registry=_REGISTRY,
    )

    # New fusion metrics
    FUSION_FUSED_TRACKS = Counter(
        "fusion_fused_tracks_total",
        "Number of fused tracks produced",
        ["mission"],
        registry=_REGISTRY,
    )
    FUSION_LATENCY_MS = Histogram(
        "fusion_pipeline_latency_ms",
        "Fusion + calibration pipeline latency (ms)",
        ["mission"],
        registry=_REGISTRY,
        buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
    )
    FUSION_CONFIDENCE = Histogram(
        "fusion_confidence",
        "Calibrated confidence distribution",
        ["mission"],
        registry=_REGISTRY,
        buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0),
    )
    FUSION_EVIDENCE_LEN = Histogram(
        "fusion_evidence_length",
        "Count of evidence records per fused track",
        ["mission"],
        registry=_REGISTRY,
        buckets=(1,2,4,8,16,32,64,128),
    )
    # Canary metrics definitions
    CANARY_TRAFFIC_FRACTION = Histogram(
        "canary_traffic_fraction",
        "Assigned traffic fraction to canary",
        ["phase"],
        registry=_REGISTRY,
        buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0),
    )
    CANARY_REGRESSIONS = Counter(
        "canary_regressions_total",
        "Detected canary regressions",
        ["reason"],
        registry=_REGISTRY,
    )
    CANARY_PROMOTIONS = Counter(
        "canary_promotions_total",
        "Successful promotions",
        registry=_REGISTRY,
    )
    CANARY_ROLLBACKS = Counter(
        "canary_rollbacks_total",
        "Rollback events",
        registry=_REGISTRY,
    )
    # Fusion latency budget violations (uses env FUSION_LATENCY_BUDGET_MS)
    FUSION_LATENCY_BUDGET_VIOLATIONS = Counter(
        "fusion_latency_budget_violations_total",
        "Fusion pipeline latency budget violations",
        ["mission"],
        registry=_REGISTRY,
    )
    # Canary evaluation latency + budget violations (env CANARY_EVAL_LATENCY_BUDGET_MS)
    CANARY_EVAL_LATENCY_MS = Histogram(
        "canary_eval_latency_ms",
        "Latency of canary evaluation cycles (ms)",
        [],
        registry=_REGISTRY,
        buckets=(1,2,5,10,20,50,100,200,500,1000,2000),
    )
    CANARY_EVAL_LATENCY_BUDGET_VIOLATIONS = Counter(
        "canary_eval_latency_budget_violations_total",
        "Canary evaluation latency budget violations",
        ["threshold"],
        registry=_REGISTRY,
    )
    # Drift monitoring metrics
    DRIFT_PSI = Gauge(
        "drift_psi",
        "Population Stability Index (PSI) for monitored feature",
        ["feature"],
        registry=_REGISTRY,
    )
    DRIFT_KL = Gauge(
        "drift_kl_divergence",
        "KL divergence for monitored feature",
        ["feature"],
        registry=_REGISTRY,
    )
    DRIFT_ALERTS = Counter(
        "drift_alerts_total",
        "Count of drift alerts fired (threshold exceedances)",
        ["feature", "metric", "threshold"],
        registry=_REGISTRY,
    )
    # Tactical metrics
    TACTICAL_ALERTS_PUBLISHED = Counter(
        "tactical_alerts_published_total",
        "Number of tactical GeoJSON alerts published",
        ["channel"],
        registry=_REGISTRY,
    )
else:  # graceful fallbacks
    TASK_SUBMIT = None  # type: ignore
    TASK_SUBMIT_ERRORS = None  # type: ignore
    TASK_SUBMIT_LATENCY = None  # type: ignore
    API_REQUEST_LATENCY = None  # type: ignore
    ORCH_JOB_SUBMIT = None  # type: ignore
    FUSION_FUSED_TRACKS = None  # type: ignore
    FUSION_LATENCY_MS = None  # type: ignore
    FUSION_CONFIDENCE = None  # type: ignore
    FUSION_EVIDENCE_LEN = None  # type: ignore
    CANARY_TRAFFIC_FRACTION = None  # type: ignore
    CANARY_REGRESSIONS = None  # type: ignore
    CANARY_PROMOTIONS = None  # type: ignore
    CANARY_ROLLBACKS = None  # type: ignore
    FUSION_LATENCY_BUDGET_VIOLATIONS = None  # type: ignore
    CANARY_EVAL_LATENCY_MS = None  # type: ignore
    CANARY_EVAL_LATENCY_BUDGET_VIOLATIONS = None  # type: ignore
    DRIFT_PSI = None  # type: ignore
    DRIFT_KL = None  # type: ignore
    DRIFT_ALERTS = None  # type: ignore
    TACTICAL_ALERTS_PUBLISHED = None  # type: ignore


metrics_router = APIRouter()


@metrics_router.get("/metrics")
def metrics_endpoint():  # type: ignore
    if not generate_latest or not _REGISTRY:
        return Response(status_code=503, content=b"prometheus_client_unavailable", media_type=CONTENT_TYPE_LATEST)
    try:
        payload = generate_latest(_REGISTRY)  # type: ignore
    except ValueError:
        # Registry collect race (rare with multiprocess); retry once
        payload = generate_latest(_REGISTRY)  # type: ignore
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


# ----------------------------------------------------------------------------
# Instrumentation --------------------------------------------------------------
# ----------------------------------------------------------------------------

def instrument_app(app):  # type: ignore
    """Attach lightweight latency middleware.
    Skips /metrics path to avoid self-scrape noise.
    """
    if API_REQUEST_LATENCY is None:
        return

    @app.middleware("http")  # type: ignore
    async def _latency_mw(request: Request, call_next):  # pragma: no cover - timing logic simple
        if request.url.path == "/metrics":
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        try:
            API_REQUEST_LATENCY.labels(request.method, request.url.path, str(response.status_code)).observe(
                time.perf_counter() - start
            )  # type: ignore
        except Exception:
            pass
        return response


# Convenience re-export for type checkers
instrument_app.__all__ = []
