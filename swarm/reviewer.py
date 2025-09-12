# filepath: swarm/reviewer.py
from __future__ import annotations
from typing import List, Dict, Any

# Minimal reviewer scaffold: annotate results with validation and confidence
# to support Phase 4 without changing existing tests.

def review_results(results: List[Dict[str, Any]], auto_heal: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in results:
        rr = dict(r)
        txt = str(rr.get("result", rr.get("output", "")))
        rr["validation"] = {
            "ok": not txt.lower().startswith("error"),
            "confidence": 0.9 if not txt.lower().startswith("error") else 0.4,
        }
        if auto_heal and not rr["validation"]["ok"]:
            rr["result"] = rr.get("result", rr.get("output", "")).replace("Error:", "Healed:")
            rr["validation"] = {"ok": True, "confidence": 0.6}
        out.append(rr)
    return out


def review_tool_result(res, auto_heal: bool = False):
    """
    Annotate a ToolResult-like object with a validation section and optional auto-heal.
    The object is expected to have attributes: output, metadata (dict-like).
    Returns the same object instance for convenience.
    """
    try:
        txt = str(getattr(res, "output", ""))
        ok = not txt.lower().startswith("error")
        confidence = 0.9 if ok else 0.4
        md = dict(getattr(res, "metadata", {}) or {})
        md["review"] = {"ok": ok, "confidence": confidence}
        if auto_heal and not ok:
            healed = txt.replace("Error:", "Healed:")
            setattr(res, "output", healed)
            md["review"] = {"ok": True, "confidence": 0.6, "healed": True}
        setattr(res, "metadata", md)
        return res
    except Exception:
        return res
