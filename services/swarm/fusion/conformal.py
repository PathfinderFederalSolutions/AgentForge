"""
Conformal Prediction Implementation for AgentForge
Provides uncertainty quantification and prediction intervals
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any

log = logging.getLogger("conformal-prediction")

def conformal_validate(residuals: List[float], alpha: float = 0.1) -> Tuple[float, float]:
    """
    Perform conformal prediction to generate prediction intervals
    
    Args:
        residuals: List of residual errors from calibration set
        alpha: Significance level (e.g., 0.1 for 90% confidence)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for prediction interval
    """
    try:
        residuals_array = np.array(residuals)
        
        # Calculate quantiles for conformal prediction
        lower_quantile = alpha / 2
        upper_quantile = 1 - (alpha / 2)
        
        # Compute prediction intervals
        lower_bound = np.quantile(residuals_array, lower_quantile)
        upper_bound = np.quantile(residuals_array, upper_quantile)
        
        log.debug(f"Conformal prediction: alpha={alpha}, interval=[{lower_bound:.3f}, {upper_bound:.3f}]")
        
        return float(lower_bound), float(upper_bound)
        
    except Exception as e:
        log.error(f"Conformal prediction failed: {e}")
        # Return conservative wide interval
        return -2.0, 2.0

def adaptive_conformal_prediction(
    residuals: List[float], 
    recent_errors: List[float],
    alpha: float = 0.1,
    adaptation_rate: float = 0.1
) -> Tuple[float, float]:
    """
    Adaptive conformal prediction that adjusts based on recent performance
    
    Args:
        residuals: Historical residual errors
        recent_errors: Recent prediction errors for adaptation
        alpha: Base significance level
        adaptation_rate: How quickly to adapt to recent performance
        
    Returns:
        Tuple of (adaptive_lower_bound, adaptive_upper_bound)
    """
    try:
        # Base conformal prediction
        base_lower, base_upper = conformal_validate(residuals, alpha)
        
        if not recent_errors:
            return base_lower, base_upper
        
        # Calculate recent error statistics
        recent_mean_error = np.mean(np.abs(recent_errors))
        historical_mean_error = np.mean(np.abs(residuals))
        
        # Adaptation factor based on recent vs historical performance
        if recent_mean_error > historical_mean_error:
            # Recent performance worse - widen intervals
            adaptation_factor = 1.0 + (adaptation_rate * (recent_mean_error / historical_mean_error - 1.0))
        else:
            # Recent performance better - narrow intervals slightly
            adaptation_factor = 1.0 - (adaptation_rate * 0.5 * (1.0 - recent_mean_error / historical_mean_error))
        
        # Apply adaptation
        interval_width = base_upper - base_lower
        center = (base_upper + base_lower) / 2
        adapted_width = interval_width * adaptation_factor
        
        adapted_lower = center - adapted_width / 2
        adapted_upper = center + adapted_width / 2
        
        log.debug(f"Adaptive conformal: adaptation_factor={adaptation_factor:.3f}, interval=[{adapted_lower:.3f}, {adapted_upper:.3f}]")
        
        return float(adapted_lower), float(adapted_upper)
        
    except Exception as e:
        log.error(f"Adaptive conformal prediction failed: {e}")
        return conformal_validate(residuals, alpha)

def calibrate_fusion_confidence(
    fusion_results: List[Dict[str, Any]], 
    ground_truth: List[float],
    alpha: float = 0.1
) -> Dict[str, Any]:
    """
    Calibrate fusion confidence using conformal prediction
    
    Args:
        fusion_results: List of fusion results with confidence scores
        ground_truth: Corresponding ground truth values
        alpha: Significance level for calibration
        
    Returns:
        Calibration results with adjusted confidence intervals
    """
    try:
        if len(fusion_results) != len(ground_truth):
            raise ValueError("Fusion results and ground truth must have same length")
        
        # Extract predictions and confidence scores
        predictions = [result.get("fused_value", 0.0) for result in fusion_results]
        confidences = [result.get("confidence", 0.5) for result in fusion_results]
        
        # Calculate residuals
        residuals = [pred - truth for pred, truth in zip(predictions, ground_truth)]
        
        # Perform conformal calibration
        lower_bound, upper_bound = conformal_validate(residuals, alpha)
        
        # Calculate calibration metrics
        coverage_count = sum(
            1 for pred, truth in zip(predictions, ground_truth)
            if lower_bound <= (pred - truth) <= upper_bound
        )
        coverage_rate = coverage_count / len(predictions)
        
        # Calculate confidence calibration
        mean_confidence = np.mean(confidences)
        confidence_variance = np.var(confidences)
        
        calibration_result = {
            "prediction_interval": [lower_bound, upper_bound],
            "coverage_rate": coverage_rate,
            "target_coverage": 1.0 - alpha,
            "calibration_quality": abs(coverage_rate - (1.0 - alpha)),
            "mean_confidence": mean_confidence,
            "confidence_variance": confidence_variance,
            "total_samples": len(fusion_results),
            "residual_statistics": {
                "mean": np.mean(residuals),
                "std": np.std(residuals),
                "min": np.min(residuals),
                "max": np.max(residuals)
            }
        }
        
        log.info(f"Fusion calibration: coverage={coverage_rate:.3f}, target={1.0-alpha:.3f}, quality={calibration_result['calibration_quality']:.3f}")
        
        return calibration_result
        
    except Exception as e:
        log.error(f"Fusion calibration failed: {e}")
        return {
            "error": str(e),
            "prediction_interval": [-1.0, 1.0],
            "coverage_rate": 0.5,
            "calibration_quality": 0.5
        }
