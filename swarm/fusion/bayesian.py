from __future__ import annotations
# Simplified & refactored to reduce cyclomatic complexity and line counts for lint/static analysis.
from typing import Tuple, List, Dict, Any, Optional
import time, os
import numpy as np

# Optional imports (best-effort)
try:
    from .conformal import conformal_interval_and_confidence  # type: ignore
except Exception:  # pragma: no cover
    conformal_interval_and_confidence = None  # type: ignore
try:
    from swarm.storage import persist_fused_track  # type: ignore
except Exception:  # pragma: no cover
    persist_fused_track = None  # type: ignore
try:  # metrics
    from swarm.api.metrics import (
        FUSION_FUSED_TRACKS,
        FUSION_LATENCY_MS,
        FUSION_CONFIDENCE,
        FUSION_EVIDENCE_LEN,
        FUSION_LATENCY_BUDGET_VIOLATIONS,
    )  # type: ignore
except Exception:  # pragma: no cover
    FUSION_FUSED_TRACKS = FUSION_LATENCY_MS = FUSION_CONFIDENCE = FUSION_EVIDENCE_LEN = FUSION_LATENCY_BUDGET_VIOLATIONS = None  # type: ignore
try:  # ROC
    from .roc_det import compute_roc, compute_det, eer  # type: ignore
except Exception:  # pragma: no cover
    compute_roc = compute_det = eer = None  # type: ignore
try:  # tracing
    from opentelemetry import trace  # type: ignore
    _fusion_tracer = trace.get_tracer("swarm.fusion")
except Exception:  # pragma: no cover
    _fusion_tracer = None

# --- Core fusion primitives --------------------------------------------------

def bayesian_fuse(eo: np.ndarray, ir: np.ndarray) -> Tuple[float, float]:
    mu1, var1 = float(eo.mean()), float(eo.var() + 1e-6)
    mu2, var2 = float(ir.mean()), float(ir.var() + 1e-6)
    prec1, prec2 = 1.0 / var1, 1.0 / var2
    mu_post = (mu1 * prec1 + mu2 * prec2) / (prec1 + prec2)
    var_post = 1.0 / (prec1 + prec2)
    return mu_post, var_post

def bayesian_fuse_with_covariance(eo: np.ndarray, ir: np.ndarray) -> Dict[str, Any]:
    mu, var = bayesian_fuse(eo, ir)
    return {"mu": mu, "var": var, "covariance": [[var]]}

# --- Helpers -----------------------------------------------------------------

def _conformal(residuals: np.ndarray, alpha: float) -> Tuple[List[float], float]:
    if conformal_interval_and_confidence:
        try:
            lo, hi, conf = conformal_interval_and_confidence(residuals.tolist(), alpha=alpha)
            return [lo, hi], conf
        except Exception:
            pass
    # fallback
    return [float(residuals.min()), float(residuals.max())], 1 - alpha

def _emit_metrics(mission: str, latency_ms: float, confidence: float, evidence_count: int) -> None:
    try:
        if FUSION_LATENCY_MS: FUSION_LATENCY_MS.labels(mission).observe(latency_ms)
        if FUSION_CONFIDENCE: FUSION_CONFIDENCE.labels(mission).observe(confidence)
        if FUSION_EVIDENCE_LEN: FUSION_EVIDENCE_LEN.labels(mission).observe(evidence_count)
    except Exception:
        pass

def _check_latency_budget(mission: str, latency_ms: float) -> None:
    try:
        budget_ms = float(os.getenv("FUSION_LATENCY_BUDGET_MS", "0"))
        if budget_ms > 0 and latency_ms > budget_ms and FUSION_LATENCY_BUDGET_VIOLATIONS:
            FUSION_LATENCY_BUDGET_VIOLATIONS.labels(mission).inc()  # type: ignore
    except Exception:
        pass

def _persist(fused: Dict[str, Any], interval: List[float], evidence: List[Dict[str, Any]], confidence: float, out: Dict[str, Any]) -> None:
    if not persist_fused_track:
        return
    try:
        track_state = {"mu": fused["mu"], "var": fused["var"], "interval": interval}
        track_id = persist_fused_track(track_state, fused["covariance"], confidence, evidence)  # type: ignore[arg-type]
        out["track_id"] = track_id
        mission = os.getenv("MISSION", "default")
        if FUSION_FUSED_TRACKS:
            try: FUSION_FUSED_TRACKS.labels(mission).inc()
            except Exception: pass
    except Exception:
        pass

def _trace(latency_ms: float, confidence: float, evidence: List[Dict[str, Any]], out: Dict[str, Any]) -> None:
    if not _fusion_tracer:
        return
    try:
        with _fusion_tracer.start_as_current_span("fusion.fuse_calibrate_persist") as span:  # type: ignore
            span.set_attribute("fusion.latency_ms", latency_ms)
            span.set_attribute("fusion.confidence", float(confidence))
            span.set_attribute("fusion.evidence_count", len(evidence))
            if "track_id" in out:
                span.set_attribute("fusion.track_id", out["track_id"])
    except Exception:
        pass

def _maybe_eer(residuals: np.ndarray) -> None:
    if not (compute_roc and compute_det and eer):
        return
    try:
        mags = np.abs(residuals)
        if mags.size < 6:  # need minimal variety
            return
        med = float(np.median(mags))
        pts = [{"score": float(x), "label": 1 if x > med else 0} for x in mags.tolist()]
        fpr, _, _ = compute_roc(pts)  # type: ignore
        fpr2, fnr = compute_det(pts)  # type: ignore
        if fpr2 and fnr:
            _ = eer(fpr2, fnr)  # side-effect metric
    except Exception:
        pass

# --- Public API --------------------------------------------------------------

def fuse_calibrate_persist(
    eo: np.ndarray,
    ir: np.ndarray,
    evidence: Optional[List[Dict[str, Any]]] = None,
    alpha: float = 0.1,
    include_residuals: bool = False,
) -> Dict[str, Any]:
    start = time.perf_counter()
    fused = bayesian_fuse_with_covariance(eo, ir)
    mu = fused["mu"]
    residuals = np.concatenate([(eo - mu), (ir - mu)])
    interval, confidence = _conformal(residuals, alpha)
    latency_ms = (time.perf_counter() - start) * 1000.0
    mission = os.getenv("MISSION", "default")
    out: Dict[str, Any] = {**fused, "interval": interval, "confidence": confidence, "latency_ms": latency_ms, "evidence": evidence or []}
    _emit_metrics(mission, latency_ms, confidence, len(evidence or []))
    _check_latency_budget(mission, latency_ms)
    if include_residuals:
        out["residuals"] = residuals.tolist()
    _persist(fused, interval, evidence or [], confidence, out)
    _trace(latency_ms, confidence, evidence or [], out)
    _maybe_eer(residuals)
    return out

__all__ = ["bayesian_fuse", "bayesian_fuse_with_covariance", "fuse_calibrate_persist"]