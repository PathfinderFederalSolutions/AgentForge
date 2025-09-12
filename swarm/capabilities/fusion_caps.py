from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from swarm.capabilities.registry import registry
from swarm.fusion.eo_ir import ingest_streams, build_evidence_chain
from swarm.fusion.bayesian import bayesian_fuse, fuse_calibrate_persist
from swarm.fusion.conformal import conformal_validate as _conformal_validate

@registry.decorator(
    name="bayesian_fusion",
    inputs={"eo": "List[float]", "ir": "List[float]"},
    outputs={"mu": "float", "var": "float", "covariance": "List[List[float]]"},
    desc="Precision-weighted EO/IR Gaussian fusion (with covariance)",
    meta={"cost": 1, "latency_ms": 5, "throughput": "high", "trust": 0.9, "modalities": ["eo", "ir"], "sla": {"max_latency_ms": 100}},
)
def cap_bayesian_fusion(eo: List[float], ir: List[float]) -> Dict[str, Any]:
    eo_arr, ir_arr = ingest_streams(eo, ir)
    mu, var = bayesian_fuse(eo_arr, ir_arr)
    return {"mu": mu, "var": var, "covariance": [[var]]}

@registry.decorator(
    name="conformal_validate",
    inputs={"residuals": "List[float]", "alpha": "float"},
    outputs={"lo": "float", "hi": "float"},
    desc="Empirical conformal interval from residuals",
    meta={"cost": 1, "latency_ms": 2, "throughput": "high", "trust": 0.95, "modalities": ["numeric"], "sla": {"max_latency_ms": 50}},
)
def cap_conformal_validate(residuals: List[float], alpha: float = 0.1) -> Dict[str, float]:
    lo, hi = _conformal_validate(residuals, alpha)
    return {"lo": lo, "hi": hi}

@registry.decorator(
    name="fuse_and_persist_track",
    inputs={"eo": "List[float]", "ir": "List[float]", "alpha": "float"},
    outputs={"track_id": "str", "mu": "float", "var": "float", "covariance": "List[List[float]]", "confidence": "float"},
    desc="EO/IR fusion + conformal calibration + persistence of fused track with evidence chain",
    meta={"cost": 2, "latency_ms": 10, "throughput": "med", "trust": 0.9, "modalities": ["eo","ir"], "sla": {"max_latency_ms": 150}},
)
def cap_fuse_and_persist_track(eo: List[float], ir: List[float], alpha: float = 0.1) -> Dict[str, Any]:
    eo_arr, ir_arr = ingest_streams(eo, ir)
    evidence = build_evidence_chain(eo, ir)
    fused = fuse_calibrate_persist(eo_arr, ir_arr, evidence=evidence, alpha=alpha)
    return fused