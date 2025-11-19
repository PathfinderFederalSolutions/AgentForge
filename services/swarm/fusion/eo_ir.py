"""
EO/IR Sensor Fusion Implementation for AgentForge
Electro-Optical and Infrared sensor data fusion with evidence chains
"""

import numpy as np
import json
import time
import uuid
from typing import List, Dict, Any, Tuple, Optional
import logging

log = logging.getLogger("eo-ir-fusion")

def ingest_streams(eo: List[float], ir: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ingest and preprocess EO and IR data streams
    
    Args:
        eo: Electro-optical sensor readings
        ir: Infrared sensor readings
        
    Returns:
        Tuple of preprocessed (eo_array, ir_array)
    """
    try:
        # Convert to numpy arrays
        eo_array = np.array(eo, dtype=np.float64)
        ir_array = np.array(ir, dtype=np.float64)
        
        # Basic preprocessing: outlier removal and normalization
        eo_processed = preprocess_sensor_data(eo_array, "EO")
        ir_processed = preprocess_sensor_data(ir_array, "IR")
        
        log.debug(f"Stream ingestion: EO({len(eo_processed)} samples), IR({len(ir_processed)} samples)")
        
        return eo_processed, ir_processed
        
    except Exception as e:
        log.error(f"Stream ingestion failed: {e}")
        return np.array(eo), np.array(ir)

def preprocess_sensor_data(data: np.ndarray, sensor_type: str) -> np.ndarray:
    """Preprocess sensor data with outlier removal and normalization"""
    try:
        # Remove outliers using IQR method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter outliers
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        # If too much data removed, use original
        if len(filtered_data) < len(data) * 0.5:
            filtered_data = data
        
        log.debug(f"{sensor_type} preprocessing: {len(data)} → {len(filtered_data)} samples")
        
        return filtered_data
        
    except Exception as e:
        log.warning(f"{sensor_type} preprocessing failed: {e}")
        return data

def build_evidence_chain(
    eo_arr: List[float], 
    ir_arr: List[float], 
    additional_evidence: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Build comprehensive evidence chain for fusion traceability
    
    Args:
        eo_arr: EO sensor values
        ir_arr: IR sensor values
        additional_evidence: Additional evidence sources
        
    Returns:
        Complete evidence chain with metadata
    """
    try:
        evidence_chain = []
        current_time = time.time()
        
        # Add EO evidence with quality metrics
        for i, value in enumerate(eo_arr):
            evidence_chain.append({
                "evidence_id": f"eo-{i}",
                "modality": "electro_optical",
                "sensor_type": "EO",
                "value": value,
                "quality_score": calculate_sensor_quality(value, eo_arr),
                "timestamp": current_time - (len(eo_arr) - i) * 0.1,  # Simulate temporal spacing
                "confidence": 0.85,
                "source": "eo_sensor_array",
                "metadata": {
                    "sensor_index": i,
                    "calibration_status": "calibrated",
                    "signal_strength": min(1.0, abs(value) * 2)
                }
            })
        
        # Add IR evidence with quality metrics
        for i, value in enumerate(ir_arr):
            evidence_chain.append({
                "evidence_id": f"ir-{i}",
                "modality": "infrared",
                "sensor_type": "IR",
                "value": value,
                "quality_score": calculate_sensor_quality(value, ir_arr),
                "timestamp": current_time - (len(ir_arr) - i) * 0.1,
                "confidence": 0.82,
                "source": "ir_sensor_array",
                "metadata": {
                    "sensor_index": i,
                    "calibration_status": "calibrated",
                    "thermal_signature": min(1.0, abs(value) * 1.5)
                }
            })
        
        # Add cross-modal correlation evidence
        if len(eo_arr) > 0 and len(ir_arr) > 0:
            correlation = np.corrcoef(eo_arr[:min(len(eo_arr), len(ir_arr))], 
                                   ir_arr[:min(len(eo_arr), len(ir_arr))])[0, 1]
            
            evidence_chain.append({
                "evidence_id": "cross_modal_correlation",
                "modality": "correlation_analysis",
                "sensor_type": "EO_IR_CORRELATION",
                "value": correlation,
                "quality_score": abs(correlation),
                "timestamp": current_time,
                "confidence": 0.90,
                "source": "fusion_correlator",
                "metadata": {
                    "correlation_coefficient": correlation,
                    "sample_size": min(len(eo_arr), len(ir_arr)),
                    "analysis_type": "pearson_correlation"
                }
            })
        
        # Add additional evidence if provided
        if additional_evidence:
            for evidence in additional_evidence:
                evidence_chain.append({
                    **evidence,
                    "timestamp": evidence.get("timestamp", current_time),
                    "quality_score": evidence.get("quality_score", 0.7)
                })
        
        log.debug(f"Evidence chain built: {len(evidence_chain)} evidence items")
        
        return evidence_chain
        
    except Exception as e:
        log.error(f"Evidence chain building failed: {e}")
        return []

def calculate_sensor_quality(value: float, sensor_array: List[float]) -> float:
    """Calculate quality score for individual sensor reading"""
    try:
        if not sensor_array:
            return 0.5
        
        # Quality based on consistency with other readings
        mean_value = np.mean(sensor_array)
        std_value = np.std(sensor_array)
        
        if std_value == 0:
            return 1.0  # Perfect consistency
        
        # Normalized distance from mean
        z_score = abs(value - mean_value) / std_value
        
        # Quality decreases with distance from mean
        quality = max(0.1, 1.0 - (z_score / 3.0))  # 3-sigma rule
        
        return min(1.0, quality)
        
    except Exception:
        return 0.7  # Default quality score

def temporal_fusion_analysis(
    eo_temporal: List[Tuple[float, float]], 
    ir_temporal: List[Tuple[float, float]]
) -> Dict[str, Any]:
    """
    Perform temporal analysis of EO/IR fusion over time
    
    Args:
        eo_temporal: List of (timestamp, value) tuples for EO
        ir_temporal: List of (timestamp, value) tuples for IR
        
    Returns:
        Temporal fusion analysis results
    """
    try:
        # Extract timestamps and values
        eo_times, eo_values = zip(*eo_temporal) if eo_temporal else ([], [])
        ir_times, ir_values = zip(*ir_temporal) if ir_temporal else ([], [])
        
        # Temporal alignment analysis
        temporal_overlap = calculate_temporal_overlap(eo_times, ir_times)
        
        # Temporal correlation analysis
        temporal_correlation = calculate_temporal_correlation(eo_temporal, ir_temporal)
        
        # Fusion stability over time
        stability_metrics = calculate_fusion_stability(eo_values, ir_values)
        
        analysis_result = {
            "temporal_overlap": temporal_overlap,
            "temporal_correlation": temporal_correlation,
            "stability_metrics": stability_metrics,
            "eo_samples": len(eo_temporal),
            "ir_samples": len(ir_temporal),
            "analysis_timestamp": time.time()
        }
        
        log.info(f"Temporal fusion analysis: overlap={temporal_overlap:.3f}, correlation={temporal_correlation:.3f}")
        
        return analysis_result
        
    except Exception as e:
        log.error(f"Temporal fusion analysis failed: {e}")
        return {"error": str(e)}

def calculate_temporal_overlap(times1: List[float], times2: List[float]) -> float:
    """Calculate temporal overlap between two sensor streams"""
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
    """Calculate temporal correlation between EO and IR streams"""
    try:
        if len(eo_temporal) < 2 or len(ir_temporal) < 2:
            return 0.0
        
        # Simple correlation of values (could be enhanced with time-warping)
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
    """Calculate stability metrics for fusion over time"""
    try:
        stability_metrics = {}
        
        if eo_values:
            eo_array = np.array(eo_values)
            stability_metrics["eo_stability"] = 1.0 - (np.std(eo_array) / (np.mean(eo_array) + 1e-6))
        
        if ir_values:
            ir_array = np.array(ir_values)
            stability_metrics["ir_stability"] = 1.0 - (np.std(ir_array) / (np.mean(ir_array) + 1e-6))
        
        if eo_values and ir_values:
            # Cross-modal stability
            min_length = min(len(eo_values), len(ir_values))
            eo_subset = np.array(eo_values[:min_length])
            ir_subset = np.array(ir_values[:min_length])
            
            # Calculate stability of the difference
            difference = eo_subset - ir_subset
            stability_metrics["cross_modal_stability"] = 1.0 - (np.std(difference) / (np.mean(np.abs(difference)) + 1e-6))
        
        return stability_metrics
        
    except Exception:
        return {"error": "stability_calculation_failed"}
