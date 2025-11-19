"""
Adaptive Conformal Prediction for Intelligence Data Streams
Implements time-varying conformal prediction with concept drift detection and handling
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import logging
import time
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import warnings

log = logging.getLogger("adaptive-conformal")

class DriftDetectionMethod(Enum):
    """Methods for detecting concept drift"""
    ADWIN = "adwin"
    CUSUM = "cusum"
    PAGE_HINKLEY = "page_hinkley"
    STATISTICAL_TEST = "statistical_test"
    ENSEMBLE = "ensemble"

class AdaptationStrategy(Enum):
    """Strategies for adapting to concept drift"""
    SLIDING_WINDOW = "sliding_window"
    EXPONENTIAL_FORGETTING = "exponential_forgetting"
    WEIGHTED_ENSEMBLE = "weighted_ensemble"
    CHANGE_POINT_RESET = "change_point_reset"

@dataclass
class ConformalCalibrationSet:
    """Calibration set for conformal prediction"""
    residuals: deque = field(default_factory=deque)
    timestamps: deque = field(default_factory=deque)
    predictions: deque = field(default_factory=deque)
    ground_truth: deque = field(default_factory=deque)
    max_size: int = 1000
    
    def add_sample(self, residual: float, timestamp: float, 
                   prediction: float, truth: float):
        """Add new calibration sample"""
        if len(self.residuals) >= self.max_size:
            self.residuals.popleft()
            self.timestamps.popleft()
            self.predictions.popleft()
            self.ground_truth.popleft()
        
        self.residuals.append(residual)
        self.timestamps.append(timestamp)
        self.predictions.append(prediction)
        self.ground_truth.append(truth)
    
    def get_recent_residuals(self, window_size: int) -> List[float]:
        """Get most recent residuals"""
        return list(self.residuals)[-window_size:]
    
    def get_weighted_residuals(self, decay_rate: float) -> Tuple[List[float], List[float]]:
        """Get residuals with exponential weights"""
        residuals = list(self.residuals)
        current_time = time.time()
        timestamps = list(self.timestamps)
        
        weights = []
        for ts in timestamps:
            age = current_time - ts
            weight = np.exp(-decay_rate * age)
            weights.append(weight)
        
        return residuals, weights

@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis"""
    drift_detected: bool
    detection_method: str
    drift_magnitude: float
    detection_timestamp: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ADWINDriftDetector:
    """ADWIN (Adaptive Windowing) drift detector"""
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta  # Confidence parameter
        self.window = deque()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        
    def add_element(self, value: float) -> bool:
        """Add new element and check for drift"""
        self.window.append(value)
        self.width += 1
        
        if self.width == 1:
            self.total = value
            self.variance = 0.0
            return False
        
        # Update statistics
        self.total += value
        
        # Check for drift using ADWIN algorithm
        drift_detected = self._detect_change()
        
        if drift_detected:
            # Trim window from the left
            self._trim_window()
        
        return drift_detected
    
    def _detect_change(self) -> bool:
        """Detect change using ADWIN algorithm"""
        if self.width < 2:
            return False
        
        # Simplified ADWIN implementation
        # In practice, this would be more sophisticated
        n = self.width
        mean = self.total / n
        
        # Calculate variance incrementally
        if n >= 10:  # Need sufficient samples
            # Split window and compare means
            mid = n // 2
            left_sum = sum(list(self.window)[:mid])
            right_sum = sum(list(self.window)[mid:])
            
            left_mean = left_sum / mid
            right_mean = right_sum / (n - mid)
            
            # Statistical test for difference
            diff = abs(left_mean - right_mean)
            threshold = np.sqrt((2 * np.log(2 / self.delta)) / min(mid, n - mid))
            
            return diff > threshold
        
        return False
    
    def _trim_window(self):
        """Trim window after drift detection"""
        # Keep only recent half of the window
        new_size = max(1, self.width // 2)
        while len(self.window) > new_size:
            removed = self.window.popleft()
            self.total -= removed
            self.width -= 1

class CUSUMDriftDetector:
    """CUSUM (Cumulative Sum) drift detector"""
    
    def __init__(self, threshold: float = 5.0, drift_threshold: float = 10.0):
        self.threshold = threshold
        self.drift_threshold = drift_threshold
        self.sum_pos = 0.0
        self.sum_neg = 0.0
        self.mean_estimate = 0.0
        self.n_samples = 0
        
    def add_element(self, value: float) -> bool:
        """Add element and check for drift"""
        self.n_samples += 1
        
        # Update mean estimate
        self.mean_estimate += (value - self.mean_estimate) / self.n_samples
        
        # CUSUM statistics
        deviation = value - self.mean_estimate
        self.sum_pos = max(0, self.sum_pos + deviation - self.threshold)
        self.sum_neg = max(0, self.sum_neg - deviation - self.threshold)
        
        # Check for drift
        drift_detected = (self.sum_pos > self.drift_threshold or 
                         self.sum_neg > self.drift_threshold)
        
        if drift_detected:
            self._reset()
        
        return drift_detected
    
    def _reset(self):
        """Reset detector after drift"""
        self.sum_pos = 0.0
        self.sum_neg = 0.0
        self.n_samples = 0
        self.mean_estimate = 0.0

class EnsembleDriftDetector:
    """Ensemble of drift detectors for robust detection"""
    
    def __init__(self):
        self.detectors = {
            'adwin': ADWINDriftDetector(),
            'cusum': CUSUMDriftDetector()
        }
        self.detection_history = deque(maxlen=100)
        
    def add_element(self, value: float) -> DriftDetectionResult:
        """Add element to all detectors and get ensemble result"""
        detections = {}
        
        for name, detector in self.detectors.items():
            drift_detected = detector.add_element(value)
            detections[name] = drift_detected
        
        # Ensemble decision (majority vote)
        detection_count = sum(detections.values())
        ensemble_drift = detection_count >= len(self.detectors) / 2
        
        # Calculate confidence based on agreement
        agreement = detection_count / len(self.detectors)
        confidence = agreement if ensemble_drift else 1.0 - agreement
        
        result = DriftDetectionResult(
            drift_detected=ensemble_drift,
            detection_method="ensemble",
            drift_magnitude=agreement,
            detection_timestamp=time.time(),
            confidence=confidence,
            metadata=detections
        )
        
        self.detection_history.append(result)
        return result

class AdaptiveConformalPredictor:
    """Adaptive conformal predictor with concept drift handling"""
    
    def __init__(self,
                 alpha: float = 0.1,
                 adaptation_strategy: AdaptationStrategy = AdaptationStrategy.SLIDING_WINDOW,
                 drift_detection_method: DriftDetectionMethod = DriftDetectionMethod.ENSEMBLE,
                 calibration_window_size: int = 500):
        
        self.alpha = alpha  # Miscoverage level
        self.adaptation_strategy = adaptation_strategy
        self.drift_detection_method = drift_detection_method
        self.calibration_window_size = calibration_window_size
        
        # Calibration set
        self.calibration_set = ConformalCalibrationSet(max_size=calibration_window_size)
        
        # Drift detection
        if drift_detection_method == DriftDetectionMethod.ENSEMBLE:
            self.drift_detector = EnsembleDriftDetector()
        elif drift_detection_method == DriftDetectionMethod.ADWIN:
            self.drift_detector = ADWINDriftDetector()
        elif drift_detection_method == DriftDetectionMethod.CUSUM:
            self.drift_detector = CUSUMDriftDetector()
        else:
            self.drift_detector = None
        
        # Adaptation parameters
        self.forgetting_factor = 0.99
        self.concept_drift_history = deque(maxlen=100)
        self.coverage_history = deque(maxlen=1000)
        
        # Performance tracking
        self.prediction_intervals = deque(maxlen=1000)
        self.coverage_violations = 0
        self.total_predictions = 0
        
        log.info(f"Adaptive conformal predictor initialized: alpha={alpha}, "
                f"adaptation={adaptation_strategy.value}, drift_detection={drift_detection_method.value}")
    
    def update_calibration(self, prediction: float, ground_truth: float, timestamp: float):
        """Update calibration set with new observation"""
        residual = abs(prediction - ground_truth)
        
        # Add to calibration set
        self.calibration_set.add_sample(residual, timestamp, prediction, ground_truth)
        
        # Drift detection
        if self.drift_detector:
            if hasattr(self.drift_detector, 'add_element'):
                if isinstance(self.drift_detector, EnsembleDriftDetector):
                    drift_result = self.drift_detector.add_element(residual)
                else:
                    drift_detected = self.drift_detector.add_element(residual)
                    drift_result = DriftDetectionResult(
                        drift_detected=drift_detected,
                        detection_method=self.drift_detection_method.value,
                        drift_magnitude=1.0 if drift_detected else 0.0,
                        detection_timestamp=timestamp,
                        confidence=0.8
                    )
                
                if drift_result.drift_detected:
                    self._handle_concept_drift(drift_result)
        
        # Update coverage statistics
        self._update_coverage_statistics()
    
    def predict_interval(self, 
                        point_prediction: float,
                        prediction_timestamp: float,
                        feature_vector: Optional[np.ndarray] = None) -> Tuple[float, float, Dict[str, Any]]:
        """
        Generate prediction interval using adaptive conformal prediction
        
        Args:
            point_prediction: Point prediction from underlying model
            prediction_timestamp: Timestamp of prediction
            feature_vector: Optional feature vector for local adaptation
            
        Returns:
            Tuple of (lower_bound, upper_bound, metadata)
        """
        try:
            if len(self.calibration_set.residuals) == 0:
                # No calibration data - return wide default interval
                width = 2.0
                return (point_prediction - width, point_prediction + width, 
                       {"method": "default", "confidence": 0.1})
            
            # Get adaptive quantile based on strategy
            quantile_level = 1 - self.alpha
            
            if self.adaptation_strategy == AdaptationStrategy.SLIDING_WINDOW:
                residuals = self.calibration_set.get_recent_residuals(
                    min(len(self.calibration_set.residuals), self.calibration_window_size)
                )
                quantile = np.quantile(residuals, quantile_level)
                
            elif self.adaptation_strategy == AdaptationStrategy.EXPONENTIAL_FORGETTING:
                residuals, weights = self.calibration_set.get_weighted_residuals(
                    1 - self.forgetting_factor
                )
                quantile = self._weighted_quantile(residuals, weights, quantile_level)
                
            elif self.adaptation_strategy == AdaptationStrategy.WEIGHTED_ENSEMBLE:
                # Combine multiple strategies
                sliding_residuals = self.calibration_set.get_recent_residuals(
                    min(len(self.calibration_set.residuals), self.calibration_window_size // 2)
                )
                exponential_residuals, weights = self.calibration_set.get_weighted_residuals(
                    1 - self.forgetting_factor
                )
                
                sliding_quantile = np.quantile(sliding_residuals, quantile_level)
                exponential_quantile = self._weighted_quantile(exponential_residuals, weights, quantile_level)
                
                # Weighted combination based on recent performance
                recent_coverage = self._get_recent_coverage()
                target_coverage = 1 - self.alpha
                
                if recent_coverage > target_coverage:
                    # Over-covering, prefer tighter intervals
                    quantile = 0.3 * sliding_quantile + 0.7 * exponential_quantile
                else:
                    # Under-covering, prefer wider intervals
                    quantile = 0.7 * sliding_quantile + 0.3 * exponential_quantile
                    
            else:
                # Default to simple quantile
                residuals = list(self.calibration_set.residuals)
                quantile = np.quantile(residuals, quantile_level)
            
            # Local adaptation based on feature similarity (if provided)
            if feature_vector is not None:
                quantile = self._locally_adapt_quantile(quantile, feature_vector)
            
            # Temporal adaptation based on recent drift
            quantile = self._temporally_adapt_quantile(quantile, prediction_timestamp)
            
            # Generate interval
            lower_bound = point_prediction - quantile
            upper_bound = point_prediction + quantile
            
            # Store prediction interval
            interval_info = {
                "prediction": point_prediction,
                "lower": lower_bound,
                "upper": upper_bound,
                "width": 2 * quantile,
                "timestamp": prediction_timestamp,
                "method": self.adaptation_strategy.value,
                "alpha": self.alpha
            }
            
            self.prediction_intervals.append(interval_info)
            self.total_predictions += 1
            
            # Metadata
            metadata = {
                "method": self.adaptation_strategy.value,
                "quantile_level": quantile_level,
                "quantile_value": quantile,
                "calibration_samples": len(self.calibration_set.residuals),
                "recent_coverage": self._get_recent_coverage(),
                "drift_detected_recently": self._recent_drift_detected(),
                "confidence": min(0.99, 0.5 + 0.4 * len(self.calibration_set.residuals) / self.calibration_window_size)
            }
            
            return lower_bound, upper_bound, metadata
            
        except Exception as e:
            log.error(f"Adaptive conformal prediction failed: {e}")
            # Fallback to wide interval
            width = 2.0
            return (point_prediction - width, point_prediction + width,
                   {"method": "fallback", "error": str(e), "confidence": 0.1})
    
    def _weighted_quantile(self, values: List[float], weights: List[float], quantile_level: float) -> float:
        """Calculate weighted quantile"""
        try:
            if not values or not weights:
                return 1.0
            
            values = np.array(values)
            weights = np.array(weights)
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Sort by values
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_weights = weights[sorted_indices]
            
            # Calculate cumulative weights
            cumulative_weights = np.cumsum(sorted_weights)
            
            # Find quantile
            quantile_idx = np.searchsorted(cumulative_weights, quantile_level)
            quantile_idx = min(quantile_idx, len(sorted_values) - 1)
            
            return float(sorted_values[quantile_idx])
            
        except Exception as e:
            log.warning(f"Weighted quantile calculation failed: {e}")
            return np.quantile(values, quantile_level)
    
    def _locally_adapt_quantile(self, base_quantile: float, feature_vector: np.ndarray) -> float:
        """Adapt quantile based on local feature similarity"""
        try:
            # Simple local adaptation - could be enhanced with more sophisticated methods
            # For now, just add small perturbation based on feature characteristics
            
            if len(feature_vector) == 0:
                return base_quantile
            
            # Feature-based adaptation factor
            feature_variance = np.var(feature_vector)
            adaptation_factor = 1.0 + 0.1 * np.tanh(feature_variance - 1.0)
            
            return base_quantile * adaptation_factor
            
        except Exception as e:
            log.warning(f"Local adaptation failed: {e}")
            return base_quantile
    
    def _temporally_adapt_quantile(self, base_quantile: float, timestamp: float) -> float:
        """Adapt quantile based on temporal patterns"""
        try:
            # Check for recent concept drift
            if self._recent_drift_detected():
                # Increase interval width after drift detection
                return base_quantile * 1.2
            
            # Check recent coverage performance
            recent_coverage = self._get_recent_coverage()
            target_coverage = 1 - self.alpha
            
            if recent_coverage < target_coverage * 0.9:
                # Under-covering, increase interval width
                return base_quantile * 1.1
            elif recent_coverage > target_coverage * 1.1:
                # Over-covering, decrease interval width
                return base_quantile * 0.9
            
            return base_quantile
            
        except Exception as e:
            log.warning(f"Temporal adaptation failed: {e}")
            return base_quantile
    
    def _handle_concept_drift(self, drift_result: DriftDetectionResult):
        """Handle detected concept drift"""
        try:
            log.info(f"Concept drift detected: {drift_result.detection_method}, "
                    f"magnitude={drift_result.drift_magnitude:.3f}")
            
            self.concept_drift_history.append(drift_result)
            
            if self.adaptation_strategy == AdaptationStrategy.CHANGE_POINT_RESET:
                # Reset calibration set
                self.calibration_set = ConformalCalibrationSet(max_size=self.calibration_window_size)
                log.info("Calibration set reset due to concept drift")
                
            elif self.adaptation_strategy == AdaptationStrategy.SLIDING_WINDOW:
                # Reduce calibration window size temporarily
                self.calibration_window_size = max(50, self.calibration_window_size // 2)
                log.info(f"Calibration window reduced to {self.calibration_window_size}")
                
            elif self.adaptation_strategy == AdaptationStrategy.EXPONENTIAL_FORGETTING:
                # Increase forgetting rate
                self.forgetting_factor = max(0.9, self.forgetting_factor * 0.95)
                log.info(f"Forgetting factor increased to {self.forgetting_factor}")
            
        except Exception as e:
            log.error(f"Drift handling failed: {e}")
    
    def _update_coverage_statistics(self):
        """Update coverage statistics"""
        if len(self.prediction_intervals) < 2:
            return
        
        # Check coverage for recent predictions that have ground truth
        recent_intervals = list(self.prediction_intervals)[-100:]  # Last 100 predictions
        recent_calibration = list(self.calibration_set.ground_truth)[-100:]
        
        if len(recent_calibration) == 0:
            return
        
        # Match predictions with ground truth by timestamp (simplified)
        covered = 0
        total = 0
        
        for interval in recent_intervals:
            # Find corresponding ground truth (simplified matching)
            if total < len(recent_calibration):
                truth = recent_calibration[total]
                if interval["lower"] <= truth <= interval["upper"]:
                    covered += 1
                total += 1
        
        if total > 0:
            coverage_rate = covered / total
            self.coverage_history.append(coverage_rate)
    
    def _get_recent_coverage(self) -> float:
        """Get recent coverage rate"""
        if len(self.coverage_history) == 0:
            return 1.0 - self.alpha  # Target coverage
        
        # Return average of recent coverage rates
        recent_window = min(50, len(self.coverage_history))
        recent_rates = list(self.coverage_history)[-recent_window:]
        return np.mean(recent_rates)
    
    def _recent_drift_detected(self) -> bool:
        """Check if drift was detected recently"""
        if len(self.concept_drift_history) == 0:
            return False
        
        # Check if drift detected in last 10 detections
        recent_drifts = list(self.concept_drift_history)[-10:]
        current_time = time.time()
        
        for drift in recent_drifts:
            if current_time - drift.detection_timestamp < 300:  # 5 minutes
                return True
        
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            metrics = {
                "total_predictions": self.total_predictions,
                "calibration_samples": len(self.calibration_set.residuals),
                "recent_coverage": self._get_recent_coverage(),
                "target_coverage": 1 - self.alpha,
                "concept_drifts_detected": len(self.concept_drift_history),
                "adaptation_strategy": self.adaptation_strategy.value,
                "drift_detection_method": self.drift_detection_method.value,
                "current_forgetting_factor": self.forgetting_factor,
                "current_calibration_window": self.calibration_window_size
            }
            
            if len(self.prediction_intervals) > 0:
                recent_intervals = list(self.prediction_intervals)[-100:]
                widths = [interval["width"] for interval in recent_intervals]
                metrics["average_interval_width"] = np.mean(widths)
                metrics["interval_width_std"] = np.std(widths)
            
            if len(self.coverage_history) > 0:
                metrics["coverage_stability"] = np.std(list(self.coverage_history))
            
            return metrics
            
        except Exception as e:
            log.error(f"Performance metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def reset_after_drift(self):
        """Manual reset after major concept drift"""
        log.info("Manual reset triggered")
        self.calibration_set = ConformalCalibrationSet(max_size=self.calibration_window_size)
        self.coverage_history.clear()
        self.prediction_intervals.clear()
        self.concept_drift_history.clear()
        
        # Reset adaptation parameters
        self.forgetting_factor = 0.99
        self.calibration_window_size = 500
        self.coverage_violations = 0
        self.total_predictions = 0

# Utility function for creating adaptive conformal predictors
def create_intelligence_conformal_predictor(
    coverage_level: float = 0.9,
    intelligence_domain: str = "general"
) -> AdaptiveConformalPredictor:
    """
    Create conformal predictor optimized for intelligence applications
    
    Args:
        coverage_level: Desired coverage level (e.g., 0.9 for 90% coverage)
        intelligence_domain: Domain-specific optimizations
    
    Returns:
        Configured AdaptiveConformalPredictor
    """
    alpha = 1.0 - coverage_level
    
    # Domain-specific configurations
    if intelligence_domain == "real_time":
        return AdaptiveConformalPredictor(
            alpha=alpha,
            adaptation_strategy=AdaptationStrategy.EXPONENTIAL_FORGETTING,
            drift_detection_method=DriftDetectionMethod.CUSUM,
            calibration_window_size=200
        )
    elif intelligence_domain == "high_accuracy":
        return AdaptiveConformalPredictor(
            alpha=alpha,
            adaptation_strategy=AdaptationStrategy.WEIGHTED_ENSEMBLE,
            drift_detection_method=DriftDetectionMethod.ENSEMBLE,
            calibration_window_size=1000
        )
    elif intelligence_domain == "adversarial":
        return AdaptiveConformalPredictor(
            alpha=alpha,
            adaptation_strategy=AdaptationStrategy.CHANGE_POINT_RESET,
            drift_detection_method=DriftDetectionMethod.ENSEMBLE,
            calibration_window_size=300
        )
    else:
        return AdaptiveConformalPredictor(
            alpha=alpha,
            adaptation_strategy=AdaptationStrategy.SLIDING_WINDOW,
            drift_detection_method=DriftDetectionMethod.ENSEMBLE,
            calibration_window_size=500
        )
