from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple

def debate(responses: List[str], rounds: int = 2) -> str:
    cur = responses
    for _ in range(rounds):
        critique = [f"Critique: {r}" for r in cur]
        cur = [f"Refined: {r}" for r in critique]
    return max(cur, key=len)

def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for r in results:
        for k, v in r.items():
            out.setdefault(k, []).append(v)
    # basic reduce: pick last or average numbers
    agg = {}
    for k, vals in out.items():
        if all(isinstance(x, (int, float)) for x in vals):
            agg[k] = sum(vals) / max(1, len(vals))
        else:
            agg[k] = vals[-1]
    return agg