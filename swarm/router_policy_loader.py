# filepath: swarm/router_policy_loader.py
"""Router policy loader.

Loads JSON policy file defining routing constraints. Environment variable
ROUTER_POLICY_PATH can override default path (config/router_policies.json).

Policy schema (list of objects):
  {
    "name": "secure_source_code",
    "task_regex": "(?i)source code|confidential",
    "max_cost_usd": 0.50,             # cumulative per provider cost ceiling (optional)
    "max_latency_ms": 15000,          # soft average latency ceiling (optional)
    "sensitive_local_only": true,     # restrict to local providers
    "allowed_providers": ["mock"],    # override allowlist when sensitive_local_only
    "priority": 50                    # larger = higher precedence
  }

Policies are matched by first regex (highest priority first). If multiple match, the highest
priority one is used.
"""
from __future__ import annotations
import os, json, re, threading
from typing import List, Dict, Optional

_POLICY_PATH = os.getenv("ROUTER_POLICY_PATH", "config/router_policies.json")
_lock = threading.Lock()
_policies: List[Dict] | None = None

# Precompile regex patterns for speed
_DEF_CACHE: Dict[int, re.Pattern] = {}

_DEFAULT_POLICIES = [
    {
        "name": "default_low_cost_guardrail",
        "task_regex": ".*",
        "max_cost_usd": 25.0,
        "max_latency_ms": 60_000,
        "sensitive_local_only": False,
        "priority": 0,
    },
    {
        "name": "sensitive_confidential_local",
        "task_regex": "(?i)confidential|secret|internal|pii|personally identifiable",
        "sensitive_local_only": True,
        "allowed_providers": ["mock"],
        "max_cost_usd": 5.0,
        "max_latency_ms": 30_000,
        "priority": 100,
    },
]

def _load() -> List[Dict]:
    global _policies
    with _lock:
        if _policies is not None:
            return _policies
        try:
            if os.path.exists(_POLICY_PATH):
                with open(_POLICY_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    _policies = data
                else:
                    _policies = _DEFAULT_POLICIES
            else:
                _policies = _DEFAULT_POLICIES
        except Exception:
            _policies = _DEFAULT_POLICIES
        # Sort by priority descending
        _policies.sort(key=lambda p: p.get("priority", 0), reverse=True)
        return _policies

def list_policies() -> List[Dict]:
    return list(_load())

def match_policy(task_description: str) -> Optional[Dict]:
    task_description = task_description or ""
    for p in _load():
        pattern = p.get("task_regex") or ".*"
        pid = id(pattern)
        comp = _DEF_CACHE.get(pid)
        if comp is None:
            try:
                comp = re.compile(pattern)
            except Exception:
                comp = re.compile(r".*")
            _DEF_CACHE[pid] = comp
        if comp.search(task_description):
            return p
    return None

if __name__ == "__main__":
    import pprint
    pprint.pp(list_policies())
    print("Match example:", match_policy("Confidential source code refactor"))
