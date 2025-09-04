from __future__ import annotations
from typing import Any, Dict, List, Optional

def _load_sla_module():
    # Load your SLA/KPI config module if present
    try:
        import sla_kpi_config as _sla  # user-provided file
        return _sla
    except Exception:
        return None

def load_sla() -> Dict[str, Any]:
    """
    Load SLA/KPI configuration from sla_kpi_config.py with safe defaults.
    Defaults enforce strict integrity: zero error_rate, >=0.95 completeness.
    """
    cfg: Dict[str, Any] = {"name": "default", "thresholds": {}, "policies": {}}
    mod = _load_sla_module()
    if mod:
        for attr in ("load", "load_sla_kpi_config"):
            if hasattr(mod, attr):
                try:
                    got = getattr(mod, attr)()
                    if isinstance(got, dict):
                        cfg.update(got)
                        break
                except Exception:
                    pass
        if not cfg.get("thresholds"):
            for attr in ("SLA_KPI", "CONFIG", "DEFAULTS"):
                if hasattr(mod, attr):
                    got = getattr(mod, attr)
                    if isinstance(got, dict):
                        cfg.update(got)
                        break

    thresholds = cfg.setdefault("thresholds", {})
    thresholds.setdefault("error_rate", 0.0)
    thresholds.setdefault("completeness", 0.95)

    policies = cfg.setdefault("policies", {})
    policies.setdefault("strict_unknown_kpis", True)
    policies.setdefault("require_hitl_on_violation", True)
    return cfg

def _import_enforcer_cls():
    """
    Import your enforcer class from ochestrator_enforcer.py (user spelling),
    or fallback to orchestrator_enforcer.py if present.
    Accepts common class names.
    """
    module = None
    try:
        import ochestrator_enforcer as oe  # user file name
        module = oe
    except Exception:
        try:
            import orchestrator_enforcer as oe2  # optional fallback
            module = oe2
        except Exception:
            module = None

    if not module:
        return None

    for name in ("OrchestratorEnforcer", "Enforcer", "PolicyEnforcer"):
        if hasattr(module, name):
            return getattr(module, name)
    return None

class DefaultEnforcer:
    """
    Safe default: compute basic KPIs from results, require HITL on violations.
    A task result is considered an error if it’s None or starts with 'Error:'.
    """
    def __init__(self, sla: Dict[str, Any]):
        self.sla = sla or {}

    def _is_error(self, val: Any) -> bool:
        if val is None:
            return True
        s = str(val).strip().lower()
        return s.startswith("error:")

    def enforce(self, goal: str, results: Any) -> Dict[str, Any]:
        thresholds = (self.sla.get("thresholds") or {})
        policies = (self.sla.get("policies") or {})
        strict_unknown = bool(policies.get("strict_unknown_kpis", True))
        require_hitl = bool(policies.get("require_hitl_on_violation", True))

        task_count = 0
        error_count = 0
        metrics: Dict[str, Any] = {}
        violations: List[Dict[str, Any]] = []

        if isinstance(results, list):
            task_count = len(results)
            for item in results:
                out = item.get("result") if isinstance(item, dict) else item
                if self._is_error(out):
                    error_count += 1
        elif isinstance(results, dict) and "metrics" in results:
            metrics.update(results.get("metrics") or {})
            task_count = int(metrics.get("task_count") or 0)
            error_count = int(metrics.get("error_count") or 0)
        else:
            if strict_unknown:
                violations.append({"kpi": "results_shape", "value": "unknown", "threshold": "known"})

        metrics.setdefault("task_count", task_count)
        metrics.setdefault("error_count", error_count)
        metrics.setdefault("error_rate", (error_count / task_count) if task_count else (1.0 if strict_unknown else 0.0))
        metrics.setdefault("completeness", 1.0 - metrics["error_rate"])

        for k, thr in thresholds.items():
            val = metrics.get(k)
            if isinstance(val, (int, float)) and isinstance(thr, (int, float)):
                if (k == "error_rate" and val > thr) or (k != "error_rate" and val < thr):
                    violations.append({"kpi": k, "value": val, "threshold": thr})
            elif val is None and strict_unknown:
                violations.append({"kpi": k, "value": None, "threshold": thr})

        approved = len(violations) == 0
        action = "approve" if approved else ("hitl" if require_hitl else "retry")
        return {
            "approved": approved,
            "action": action,
            "reason": "ok" if approved else "sla_kpi_violations_or_errors",
            "violations": violations,
            "metrics": metrics,
            "policy": {
                "strict_unknown_kpis": strict_unknown,
                "require_hitl_on_violation": require_hitl
            },
        }

def make_enforcer(sla: Dict[str, Any]):
    EnforcerCls = _import_enforcer_cls()
    if EnforcerCls is None:
        return DefaultEnforcer(sla)
    try:
        return EnforcerCls(sla)  # prefer ctor(sla)
    except TypeError:
        inst = EnforcerCls()
        if hasattr(inst, "configure"):
            try:
                inst.configure(sla=sla)
            except Exception:
                pass
        return inst