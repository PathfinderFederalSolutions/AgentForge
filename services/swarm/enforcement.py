from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os
import math
from datetime import datetime
from services.orchestrator.app.enforcement_bridge import load_sla, make_enforcer

# NOTE: Complexity in this module is intentional due to metric extraction logic.
# ruff: noqa: C901,E402

class SwarmEnforcer:
    def __init__(self) -> None:
        self.sla = load_sla()
        self.enforcer = make_enforcer(self.sla)

    def pre(self, goal: str) -> Dict[str, Any]:
        # Hook for pre-task policy; extend as needed (rate-limit, authz, etc.).
        return {"ok": True}

    def post(self, goal: str, results: Any) -> Dict[str, Any]:
        return self.enforcer.enforce(goal=goal, results=results)

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _extract_signals(results: List[Dict[str, Any]]) -> Dict[str, Any]:  # noqa: C901
    errs = 0
    total = max(1, len(results))
    completeness_by_mod: Dict[str, float] = {}
    fp = fn = tp = tn = 0
    unc_vals: List[float] = []
    calib_errs: List[float] = []

    for r in results:
        # Error heuristic
        s = str(r.get("result", "")) + " " + " ".join(f"{k}:{v}" for k, v in r.items())
        if "error" in s.lower() or r.get("error"):
            errs += 1

        # Modality completeness (0..1)
        modality = (r.get("modality") or r.get("provider") or r.get("capability") or "generic").lower()
        completeness = _safe_float(r.get("completeness", 1.0), 1.0)
        completeness_by_mod[modality] = max(completeness_by_mod.get(modality, 0.0), completeness)

        # Uncertainty proxies
        for k in ("uncertainty", "sigma", "std", "var"):
            if k in r:
                v = _safe_float(r[k])
                if v is not None:
                    unc_vals.append(math.sqrt(v) if k == "var" else v)

        # Calibration error (if mu/var and a reference y_hat or residuals)
        if "mu" in r and ("var" in r or "sigma" in r):
            mu = _safe_float(r.get("mu"), 0.0)
            var = _safe_float(r.get("var", (r.get("sigma", 0.0) or 0.0) ** 2), 1.0)
            if "y" in r:
                y = _safe_float(r["y"], mu)
                calib_errs.append(abs(y - mu) / max(1e-6, math.sqrt(var)))
            elif "residual" in r:
                resid = _safe_float(r["residual"], 0.0)
                calib_errs.append(abs(resid) / max(1e-6, math.sqrt(var)))

        # FP/FN if labels exist
        pred = r.get("pred_label")
        gt = r.get("gt_label")
        if pred is not None and gt is not None:
            if pred and gt:
                tp += 1
            elif pred and not gt:
                fp += 1
            elif not pred and gt:
                fn += 1
            else:
                tn += 1

    error_rate = errs / total
    completeness = sum(completeness_by_mod.values()) / max(1, len(completeness_by_mod))
    avg_uncertainty = (sum(unc_vals) / len(unc_vals)) if unc_vals else None
    calib_error = (sum(calib_errs) / len(calib_errs)) if calib_errs else None
    fp_rate = fp / max(1, fp + tn)
    fn_rate = fn / max(1, fn + tp)

    return {
        "error_rate": error_rate,
        "errors": errs,
        "completeness": round(completeness, 4),
        "completeness_by_modality": completeness_by_mod,
        "avg_uncertainty": avg_uncertainty,
        "calibration_error": calib_error,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
    }

def _meets_thresholds(m: Dict[str, Any]) -> Tuple[bool, str]:  # noqa: C901
    # Defaults can be tuned via env
    max_error_rate = float(os.getenv("SLA_MAX_ERROR_RATE", "0.0"))
    max_unc = float(os.getenv("SLA_MAX_UNCERTAINTY", "1e9"))
    max_calib = float(os.getenv("SLA_MAX_CALIB_ERROR", "1.0"))
    min_complete = float(os.getenv("SLA_MIN_COMPLETENESS", "0.8"))
    max_fp = float(os.getenv("SLA_MAX_FP_RATE", "0.1"))
    max_fn = float(os.getenv("SLA_MAX_FN_RATE", "0.1"))

    if m["error_rate"] > max_error_rate:
        return False, "error_rate"
    if m["completeness"] < min_complete:
        return False, "completeness"
    if m["fp_rate"] > max_fp:
        return False, "fp_rate"
    if m["fn_rate"] > max_fn:
        return False, "fn_rate"
    if m["avg_uncertainty"] is not None and m["avg_uncertainty"] > max_unc:
        return False, "uncertainty"
    if m["calibration_error"] is not None and m["calibration_error"] > max_calib:
        return False, "calibration"
    return True, "ok"

def post(goal: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate SLA gates and decide approval. Extends metrics for scale-out fusion.
    """
    metrics = _extract_signals(results)
    ok, reason = _meets_thresholds(metrics)
    decision = {
        "approved": bool(ok),
        "action": "approve" if ok else "hitl",
        "reason": reason,
        "metrics": metrics,
    }
    # Learning feedback to influence planner scoring
    try:
        from ..learning.feedback import record_feedback
        record_feedback(goal, results, decision)
    except Exception:
        pass

    # HITL routing for edge cases (publish to NATS HITL)
    if not ok:
        try:
            import asyncio
            mission = os.getenv("MISSION") or os.getenv("ENV", "default")
            from ..jetstream import publish_hitl
            payload = {"goal": goal, "decision": decision, "results": results}
            asyncio.run(publish_hitl(mission, payload))
        except Exception:
            pass
    return decision

enforcer = SwarmEnforcer()