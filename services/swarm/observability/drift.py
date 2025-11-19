from __future__ import annotations
"""Drift monitoring utilities (PSI & KL divergence) with in-memory baselines.

Provides:
  - update_baseline(feature, values)
  - compute_psi(feature, new_values, bins=10)
  - compute_kl(feature, new_values, bins=10)
  - evaluate_drift(feature, new_values, psi_threshold=0.2, kl_threshold=0.5)

Design:
  Baselines stored as histogram counts (fixed bins derived from initial snapshot).
  Subsequent evaluations reuse bin edges for comparability. If baseline absent,
  it is initialized from first call (cold start) and no alert generated.

Notes:
  - PSI > ~0.1 indicates moderate shift; >0.2 often treated as significant.
  - KL divergence thresholds are domain dependent; default 0.5 is conservative.
  - For sparse data (<5 samples) returns None to avoid noisy alerts.

Metrics:
  Exposes gauge updates & alert counters via swarm.api.metrics (if available).
"""
import math
from typing import Dict, List, Tuple, Optional

try:  # Metrics optional
    from ..api.metrics import DRIFT_PSI, DRIFT_KL, DRIFT_ALERTS  # type: ignore
except Exception:  # pragma: no cover
    DRIFT_PSI = DRIFT_KL = DRIFT_ALERTS = None  # type: ignore

_baselines: Dict[str, Dict[str, List[float]]] = {}
# structure: feature -> { 'edges': [...], 'counts': [...], 'total': int }

def _build_hist(values: List[float], bins: int) -> Tuple[List[float], List[int]]:
    if bins < 2:
        bins = 2
    vmin, vmax = min(values), max(values)
    if vmax == vmin:  # all same value -> single wide bin
        vmax = vmin + 1.0
    width = (vmax - vmin) / bins
    edges = [vmin + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        # last edge inclusive
        idx = min(int((v - vmin) / width), bins - 1)
        counts[idx] += 1
    return edges, counts

def update_baseline(feature: str, values: List[float], bins: int = 10) -> None:
    if not values:
        return
    edges, counts = _build_hist(values, bins)
    _baselines[feature] = {"edges": edges, "counts": counts, "total": len(values)}

def _hist_from_edges(values: List[float], edges: List[float]) -> List[int]:
    bins = len(edges) - 1
    counts = [0] * bins
    vmin = edges[0]
    vmax = edges[-1]
    width = (vmax - vmin) / bins
    for v in values:
        if v < vmin:
            idx = 0
        elif v > vmax:
            idx = bins - 1
        else:
            idx = min(int((v - vmin) / width), bins - 1)
        counts[idx] += 1
    return counts

def compute_psi(feature: str, new_values: List[float], bins: int = 10) -> Optional[float]:
    if len(new_values) < 5:
        return None
    base = _baselines.get(feature)
    if base is None:
        update_baseline(feature, new_values, bins)
        return None
    edges = base["edges"]
    base_counts = base["counts"]
    base_total = max(base["total"], 1)
    new_counts = _hist_from_edges(new_values, edges)
    new_total = max(len(new_values), 1)
    psi = 0.0
    for b, n in zip(base_counts, new_counts):
        p = b / base_total
        q = n / new_total
        # apply small floor to avoid div/0 & log blowups
        p = max(p, 1e-6)
        q = max(q, 1e-6)
        psi += (p - q) * math.log(p / q)
    return psi

def compute_kl(feature: str, new_values: List[float], bins: int = 10) -> Optional[float]:
    if len(new_values) < 5:
        return None
    base = _baselines.get(feature)
    if base is None:
        update_baseline(feature, new_values, bins)
        return None
    edges = base["edges"]
    base_counts = base["counts"]
    base_total = max(base["total"], 1)
    new_counts = _hist_from_edges(new_values, edges)
    new_total = max(len(new_values), 1)
    kl = 0.0
    for b, n in zip(base_counts, new_counts):
        p = b / base_total
        q = n / new_total
        p = max(p, 1e-6)
        q = max(q, 1e-6)
        kl += p * math.log(p / q)
    return kl

def evaluate_drift(feature: str, new_values: List[float], *, bins: int = 10, psi_threshold: float = 0.2, kl_threshold: float = 0.5) -> Dict[str, Optional[float]]:
    # If baseline does not yet exist, initialize and return neutral result
    if feature not in _baselines:
        update_baseline(feature, new_values, bins)
        return {"psi": None, "kl": None, "alert": False}
    psi = compute_psi(feature, new_values, bins)
    kl = compute_kl(feature, new_values, bins)
    alert = False
    if psi is not None and psi > psi_threshold:
        alert = True
        if DRIFT_ALERTS:
            try: DRIFT_ALERTS.labels(feature, "psi", str(psi_threshold)).inc()  # type: ignore
            except Exception: pass
    if kl is not None and kl > kl_threshold:
        alert = True
        if DRIFT_ALERTS:
            try: DRIFT_ALERTS.labels(feature, "kl", str(kl_threshold)).inc()  # type: ignore
            except Exception: pass
    if DRIFT_PSI and psi is not None:
        try: DRIFT_PSI.labels(feature).set(psi)  # type: ignore
        except Exception: pass
    if DRIFT_KL and kl is not None:
        try: DRIFT_KL.labels(feature).set(kl)  # type: ignore
        except Exception: pass
    return {"psi": psi, "kl": kl, "alert": alert}

__all__ = [
    "update_baseline",
    "compute_psi",
    "compute_kl",
    "evaluate_drift",
]
