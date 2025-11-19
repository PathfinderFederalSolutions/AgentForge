from __future__ import annotations
import hashlib
import json
from typing import List

def hash_event(prev_hash: str, event: dict) -> str:
    payload = json.dumps({"prev": prev_hash, "event": event}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def merkle_root(hashes: List[str]) -> str:
    nodes = hashes
    if not nodes:
        return hashlib.sha256(b"").hexdigest()
    while len(nodes) > 1:
        it = iter(nodes)
        paired = []
        for a in it:
            b = next(it, a)
            paired.append(hashlib.sha256((a + b).encode("utf-8")).hexdigest())
        nodes = paired
    return nodes[0]

def cosign_sign_stub(digest: str) -> str:
    # Placeholder for cosign integration
    return f"sig:{digest[:12]}"