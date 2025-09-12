from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np

def ingest_streams(eo: List[float], ir: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder ingest: convert lists to arrays and basic normalize.
    """
    eo_arr = np.asarray(eo, dtype=float)
    ir_arr = np.asarray(ir, dtype=float)
    if eo_arr.size == 0 or ir_arr.size == 0:
        raise ValueError("empty streams")
    eo_arr = (eo_arr - eo_arr.mean()) / (eo_arr.std() + 1e-8)
    ir_arr = (ir_arr - ir_arr.mean()) / (ir_arr.std() + 1e-8)
    return eo_arr, ir_arr


def build_evidence_chain(eo: List[float], ir: List[float], base_subject: str = "eo_ir_sample") -> List[Dict[str, Any]]:
    """Build a simple evidence chain referencing source sample indices.

    Each sample becomes an evidence record with a synthetic message_id (deterministic) and subject.
    This is a lightweight stand-in until upstream messaging IDs are available.
    """
    chain: List[Dict[str, Any]] = []
    for i, v in enumerate(eo):
        chain.append({"modality": "eo", "idx": i, "value": float(v), "message_id": f"eo-{i}", "subject": base_subject})
    for i, v in enumerate(ir):
        chain.append({"modality": "ir", "idx": i, "value": float(v), "message_id": f"ir-{i}", "subject": base_subject})
    return chain

__all__ = ['ingest_streams', 'build_evidence_chain']