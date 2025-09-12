from __future__ import annotations
import os, sys
if os.path.abspath('.') not in sys.path:
    sys.path.insert(0, os.path.abspath('.'))
import math
from swarm.fusion.roc_det import compute_roc, compute_det, eer, ROC_EER_METRIC

def test_roc_det_basic():
    pts = [
        {"score": 0.9, "label": 1},
        {"score": 0.8, "label": 1},
        {"score": 0.7, "label": 0},
        {"score": 0.6, "label": 1},
        {"score": 0.3, "label": 0},
        {"score": 0.2, "label": 0},
    ]
    fpr, tpr, th = compute_roc(pts)
    assert len(fpr) == len(tpr) == len(th) > 0
    fpr2, fnr = compute_det(pts)
    assert fpr2 == fpr
    assert all(abs((1 - t) - n) < 1e-9 for t, n in zip(tpr, fnr))
    rate = eer(fpr, fnr)
    if ROC_EER_METRIC:
        # call again to ensure observation path does not raise
        _ = eer(fpr, fnr)
    assert not math.isnan(rate)
    assert 0 <= rate <= 1
