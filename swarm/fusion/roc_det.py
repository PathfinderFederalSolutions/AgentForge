from __future__ import annotations
__doc__ = """ROC/DET utilities for fusion confidence calibration validation.

Provides:
  compute_roc(points) -> fpr,tpr,thresholds
  compute_det(points) -> fpr,fnr
  eer(fpr,fnr) -> equal error rate

Input format:
  points: list of dicts with keys {'score': float, 'label': int}
    label = 1 for positive (true detection), 0 for negative.

Lightweight implementation (no sklearn dependency) to keep test environment minimal.
"""
from typing import List, Tuple
import math
try:  # metrics optional
    from swarm.api.metrics import Histogram  # type: ignore
except Exception:  # pragma: no cover
    Histogram = None  # type: ignore
try:  # tracing optional
    from opentelemetry import trace  # type: ignore
    _roc_tracer = trace.get_tracer("swarm.fusion.roc")
except Exception:  # pragma: no cover
    _roc_tracer = None

if Histogram:
    try:
        ROC_EER_METRIC = Histogram(
            "fusion_roc_eer","Equal error rate distribution",[],buckets=(0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5))  # type: ignore
    except Exception:  # pragma: no cover
        ROC_EER_METRIC = None  # type: ignore
else:
    ROC_EER_METRIC = None  # type: ignore

def _sorted(points: List[dict]) -> List[dict]:
    return sorted(points, key=lambda d: d['score'], reverse=True)

def compute_roc(points: List[dict]) -> Tuple[List[float], List[float], List[float]]:
    if not points:
        return [], [], []
    pts = _sorted(points)
    P = sum(1 for p in pts if p['label'] == 1)
    N = sum(1 for p in pts if p['label'] == 0)
    if P == 0 or N == 0:
        return [], [], []
    tp = fp = 0
    fpr: List[float] = []
    tpr: List[float] = []
    thresholds: List[float] = []
    last_score = None
    for p in pts:
        if last_score is None or p['score'] != last_score:
            if last_score is not None:
                fpr.append(fp / N)
                tpr.append(tp / P)
                thresholds.append(last_score)
            last_score = p['score']
        if p['label'] == 1:
            tp += 1
        else:
            fp += 1
    # final point
    fpr.append(fp / N)
    tpr.append(tp / P)
    thresholds.append(last_score if last_score is not None else 0.0)
    return fpr, tpr, thresholds

def compute_det(points: List[dict]) -> Tuple[List[float], List[float]]:
    fpr,tpr,_ = compute_roc(points)
    fnr = [1 - x for x in tpr]
    return fpr, fnr

def eer(fpr: List[float], fnr: List[float]) -> float:
    if not fpr or not fnr or len(fpr) != len(fnr):
        return math.nan
    best = 1.0
    for a,b in zip(fpr, fnr):
        best = min(best, abs(a-b))
    idx = min(range(len(fpr)), key=lambda i: abs(fpr[i]-fnr[i]))
    val = (fpr[idx] + fnr[idx]) / 2.0
    if ROC_EER_METRIC:
        try: ROC_EER_METRIC.observe(val)
        except Exception: pass
    if _roc_tracer:
        try:
            with _roc_tracer.start_as_current_span("fusion.roc.eer") as span:  # type: ignore
                span.set_attribute("fusion.roc.eer", float(val))
        except Exception:  # pragma: no cover
            pass
    return val

__all__ = ["compute_roc","compute_det","eer","ROC_EER_METRIC"]
