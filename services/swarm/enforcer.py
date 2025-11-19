from dataclasses import dataclass

@dataclass
class Signals:
    error_rate: float
    completeness: float
    uncertainty: float
    calib_error: float

def _extract_signals(result) -> Signals:
    """
    Extract normalized signals from a capability result.
    Returns:
        Signals: error_rate in [0,1], completeness in [0,1], uncertainty in [0,1], calib_error in [0,1]
    """
    return Signals(
        error_rate=float(result.get("error_rate", 0.0)),
        completeness=float(result.get("completeness", 0.0)),
        uncertainty=float(result.get("uncertainty", 0.0)),
        calib_error=float(result.get("calib_error", 0.0)),
    )

def _meets_thresholds(s: Signals, cfg) -> bool:
    """
    Decide if a result passes configured thresholds.
    cfg must provide: sla_max_error_rate, sla_min_completeness, sla_max_uncertainty, sla_max_calib_error
    """
    if s.error_rate > cfg.sla_max_error_rate:
        return False
    if s.completeness < cfg.sla_min_completeness:
        return False
    if s.uncertainty > cfg.sla_max_uncertainty:
        return False
    if s.calib_error > cfg.sla_max_calib_error:
        return False
    return True