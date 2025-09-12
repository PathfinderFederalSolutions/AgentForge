from __future__ import annotations
import random
from swarm.observability import drift

random.seed(42)  # Set random seed for reproducibility

def _gen_values(mean: float, n: int = 500, spread: float = 1.0):
    return [random.gauss(mean, spread) for _ in range(n)]


def test_drift_initializes_baseline():
    vals = _gen_values(0.0)
    res = drift.evaluate_drift("feature_a", vals)
    # first call initializes baseline; metrics None
    assert res["psi"] is None and res["kl"] is None and res["alert"] is False


def test_drift_no_alert_small_shift():
    base = _gen_values(0.0)
    drift.evaluate_drift("feature_b", base)  # init baseline
    shifted = _gen_values(0.1)  # small shift
    res = drift.evaluate_drift("feature_b", shifted)
    assert res["psi"] is not None
    assert res["kl"] is not None
    assert res["alert"] is False  # small shift


def test_drift_alert_large_shift():
    base = _gen_values(0.0)
    drift.evaluate_drift("feature_c", base)
    shifted = _gen_values(3.5)  # large mean shift
    res = drift.evaluate_drift("feature_c", shifted)
    assert res["alert"] is True
    # at least one metric should exceed threshold
    assert (res["psi"] and res["psi"] > 0.2) or (res["kl"] and res["kl"] > 0.5)
