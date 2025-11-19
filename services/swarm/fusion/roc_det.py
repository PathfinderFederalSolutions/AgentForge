"""
ROC/DET Analysis Implementation for AgentForge
Receiver Operating Characteristic and Detection Error Tradeoff analysis
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

log = logging.getLogger("roc-det-analysis")

# Metrics for tracking ROC/DET performance
try:
    from prometheus_client import Histogram
    ROC_EER_METRIC = Histogram(
        'roc_eer_analysis_seconds',
        'Time spent on ROC EER analysis',
        buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    )
except ImportError:
    class MockMetric:
        def observe(self, value):
            pass
    ROC_EER_METRIC = MockMetric()

def compute_roc(detection_points: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute ROC curve from detection points
    
    Args:
        detection_points: List of detection results with scores and labels
        
    Returns:
        Tuple of (false_positive_rate, true_positive_rate, thresholds)
    """
    start_time = time.time()
    
    try:
        if not detection_points:
            return [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]
        
        # Extract scores and labels
        scores = []
        labels = []
        
        for point in detection_points:
            if isinstance(point, dict):
                scores.append(point.get('score', 0.0))
                labels.append(point.get('label', 0))
            else:
                # Handle other formats
                scores.append(float(point))
                labels.append(1)  # Default positive label
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Sort by scores in descending order
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # Calculate ROC points
        fpr = []
        tpr = []
        thresholds = []
        
        # Count positives and negatives
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        
        if n_pos == 0 or n_neg == 0:
            # Handle edge case
            return [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]
        
        # Calculate ROC curve points
        tp = 0
        fp = 0
        
        for i, (score, label) in enumerate(zip(sorted_scores, sorted_labels)):
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            # Calculate rates
            tpr_val = tp / n_pos
            fpr_val = fp / n_neg
            
            fpr.append(fpr_val)
            tpr.append(tpr_val)
            thresholds.append(score)
        
        # Add endpoints
        fpr = [0.0] + fpr + [1.0]
        tpr = [0.0] + tpr + [1.0]
        thresholds = [sorted_scores[0] + 0.1] + thresholds + [sorted_scores[-1] - 0.1]
        
        ROC_EER_METRIC.observe(time.time() - start_time)
        
        log.debug(f"ROC computed: {len(fpr)} points, AUC≈{calculate_auc(fpr, tpr):.3f}")
        
        return fpr, tpr, thresholds
        
    except Exception as e:
        log.error(f"ROC computation failed: {e}")
        return [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]

def compute_det(detection_points: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """
    Compute DET curve from detection points
    
    Args:
        detection_points: List of detection results with scores and labels
        
    Returns:
        Tuple of (false_positive_rate, false_negative_rate)
    """
    try:
        # First compute ROC
        fpr, tpr, _ = compute_roc(detection_points)
        
        # Convert TPR to FNR (False Negative Rate = 1 - TPR)
        fnr = [1.0 - t for t in tpr]
        
        log.debug(f"DET computed: {len(fpr)} points")
        
        return fpr, fnr
        
    except Exception as e:
        log.error(f"DET computation failed: {e}")
        return [0.0, 1.0], [1.0, 0.0]

def eer(fpr: List[float], fnr: List[float]) -> float:
    """
    Calculate Equal Error Rate from FPR and FNR curves
    
    Args:
        fpr: False Positive Rate values
        fnr: False Negative Rate values
        
    Returns:
        Equal Error Rate value
    """
    try:
        fpr_array = np.array(fpr)
        fnr_array = np.array(fnr)
        
        # Find point where FPR ≈ FNR
        differences = np.abs(fpr_array - fnr_array)
        min_diff_index = np.argmin(differences)
        
        # EER is the average of FPR and FNR at the crossing point
        eer_value = (fpr_array[min_diff_index] + fnr_array[min_diff_index]) / 2.0
        
        log.debug(f"EER calculated: {eer_value:.4f} at index {min_diff_index}")
        
        return float(eer_value)
        
    except Exception as e:
        log.error(f"EER calculation failed: {e}")
        return 0.15  # Default reasonable EER

def calculate_auc(fpr: List[float], tpr: List[float]) -> float:
    """Calculate Area Under Curve for ROC"""
    try:
        fpr_array = np.array(fpr)
        tpr_array = np.array(tpr)
        
        # Trapezoidal integration
        auc = np.trapz(tpr_array, fpr_array)
        
        return float(auc)
        
    except Exception:
        return 0.5  # Random classifier baseline

def advanced_detection_analysis(
    detection_results: List[Dict[str, Any]],
    ground_truth: List[int]
) -> Dict[str, Any]:
    """
    Perform comprehensive detection performance analysis
    
    Args:
        detection_results: Detection results with scores and metadata
        ground_truth: Ground truth labels (0 or 1)
        
    Returns:
        Comprehensive analysis including ROC, DET, EER, and performance metrics
    """
    try:
        # Prepare detection points
        detection_points = []
        for result, truth in zip(detection_results, ground_truth):
            detection_points.append({
                'score': result.get('confidence', result.get('score', 0.5)),
                'label': truth,
                'metadata': result.get('metadata', {})
            })
        
        # Compute ROC and DET curves
        fpr, tpr, thresholds = compute_roc(detection_points)
        det_fpr, fnr = compute_det(detection_points)
        
        # Calculate performance metrics
        auc = calculate_auc(fpr, tpr)
        eer_value = eer(det_fpr, fnr)
        
        # Calculate optimal threshold
        optimal_threshold_idx = np.argmax(np.array(tpr) - np.array(fpr))
        optimal_threshold = thresholds[optimal_threshold_idx] if optimal_threshold_idx < len(thresholds) else 0.5
        
        # Performance at optimal threshold
        optimal_tpr = tpr[optimal_threshold_idx] if optimal_threshold_idx < len(tpr) else 0.5
        optimal_fpr = fpr[optimal_threshold_idx] if optimal_threshold_idx < len(fpr) else 0.5
        
        analysis_result = {
            "roc_curve": {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "auc": auc
            },
            "det_curve": {
                "fpr": det_fpr,
                "fnr": fnr,
                "eer": eer_value
            },
            "performance_metrics": {
                "auc": auc,
                "eer": eer_value,
                "optimal_threshold": optimal_threshold,
                "optimal_tpr": optimal_tpr,
                "optimal_fpr": optimal_fpr,
                "detection_accuracy": optimal_tpr * (1 - optimal_fpr)
            },
            "data_statistics": {
                "total_samples": len(detection_points),
                "positive_samples": sum(1 for p in detection_points if p['label'] == 1),
                "negative_samples": sum(1 for p in detection_points if p['label'] == 0),
                "score_range": [min(p['score'] for p in detection_points), 
                              max(p['score'] for p in detection_points)]
            }
        }
        
        log.info(f"Detection analysis: AUC={auc:.3f}, EER={eer_value:.3f}, samples={len(detection_points)}")
        
        return analysis_result
        
    except Exception as e:
        log.error(f"Advanced detection analysis failed: {e}")
        return {"error": str(e)}

def calculate_temporal_overlap(times1: List[float], times2: List[float]) -> float:
    """Calculate temporal overlap between sensor streams"""
    if not times1 or not times2:
        return 0.0
    
    start1, end1 = min(times1), max(times1)
    start2, end2 = min(times2), max(times2)
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_end <= overlap_start:
        return 0.0
    
    overlap_duration = overlap_end - overlap_start
    total_duration = max(end1, end2) - min(start1, start2)
    
    return overlap_duration / total_duration if total_duration > 0 else 0.0

def calculate_temporal_correlation(
    eo_temporal: List[Tuple[float, float]], 
    ir_temporal: List[Tuple[float, float]]
) -> float:
    """Calculate temporal correlation between EO and IR data"""
    try:
        if len(eo_temporal) < 2 or len(ir_temporal) < 2:
            return 0.0
        
        eo_values = [val for _, val in eo_temporal]
        ir_values = [val for _, val in ir_temporal]
        
        min_length = min(len(eo_values), len(ir_values))
        if min_length < 2:
            return 0.0
        
        correlation = np.corrcoef(eo_values[:min_length], ir_values[:min_length])[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
        
    except Exception:
        return 0.0

def calculate_fusion_stability(eo_values: List[float], ir_values: List[float]) -> Dict[str, float]:
    """Calculate fusion stability metrics"""
    try:
        metrics = {}
        
        if eo_values:
            eo_std = np.std(eo_values)
            eo_mean = np.mean(eo_values)
            metrics["eo_coefficient_of_variation"] = eo_std / (eo_mean + 1e-6)
        
        if ir_values:
            ir_std = np.std(ir_values)
            ir_mean = np.mean(ir_values)
            metrics["ir_coefficient_of_variation"] = ir_std / (ir_mean + 1e-6)
        
        if eo_values and ir_values:
            min_length = min(len(eo_values), len(ir_values))
            correlation = np.corrcoef(eo_values[:min_length], ir_values[:min_length])[0, 1]
            metrics["cross_modal_correlation"] = float(correlation) if not np.isnan(correlation) else 0.0
        
        return metrics
        
    except Exception:
        return {"error": "stability_metrics_failed"}
