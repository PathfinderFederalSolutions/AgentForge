from __future__ import annotations
from typing import List, Tuple
import numpy as np

def conformal_validate(residuals: List[float], alpha: float = 0.1) -> Tuple[float, float]:
    """
    Return two-sided empirical prediction interval half-width using residual quantile.
    (Original helper preserved for backward compatibility.)
    """
    arr = np.abs(np.asarray(residuals, dtype=float))
    q = float(np.quantile(arr, 1 - alpha))
    return -q, q


def conformal_interval_and_confidence(residuals: List[float], alpha: float = 0.1) -> Tuple[float, float, float]:
    """Return (lo, hi, calibrated_confidence).

    Empirical interval derived from absolute residual quantile. Calibrated confidence is 1 - alpha.
    (Future: compute empirical coverage on calibration split.)
    """
    lo, hi = conformal_validate(residuals, alpha=alpha)
    confidence = 1 - alpha
    return lo, hi, confidence

__all__ = ['conformal_validate', 'conformal_interval_and_confidence']