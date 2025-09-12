from __future__ import annotations
import os, sys
if os.path.abspath('.') not in sys.path:
    sys.path.insert(0, os.path.abspath('.'))
from swarm.canary import router as canary

def test_canary_progress_and_promotion():
    canary.start_canary(target_fraction=0.2)
    # feed healthy observations
    for _ in range(60):
        canary.record_observation(is_canary=False, latency_ms=50, error=False, completeness=0.95)
        canary.record_observation(is_canary=True, latency_ms=52, error=False, completeness=0.95)
    result = canary.maybe_progress()
    assert result["phase"] in ("canary","promote")
    # accelerate promotion by meeting stability criteria
    for _ in range(60):
        canary.record_observation(is_canary=True, latency_ms=51, error=False, completeness=0.95)
    result = canary.maybe_progress()
    # Eventually reach promote phase
    for _ in range(20):
        if result["phase"] == "promote":
            break
        result = canary.maybe_progress()
    assert result["phase"] == "promote"


def test_canary_regression_triggers_rollback():
    canary.start_canary(target_fraction=0.1)
    # base good metrics
    for _ in range(60):
        canary.record_observation(False, 50, False, 0.98)
    # canary worse error + latency
    for _ in range(60):
        canary.record_observation(True, 120, True, 0.80)
    res = canary.maybe_progress()
    assert res["phase"] in ("rollback","canary")
    if res["phase"] != "rollback":
        # second evaluation after enough samples should roll back
        res = canary.maybe_progress()
    assert res["phase"] == "rollback"
