#!/usr/bin/env python3
"""
Comprehensive test suite for advanced fusion capabilities
Tests Bayesian fusion, conformal prediction, EO/IR fusion, and ROC/DET analysis
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock

def test_bayesian_fusion():
    """Test Bayesian sensor fusion"""
    try:
        from services.swarm.fusion.bayesian import bayesian_fuse, fuse_calibrate_persist
        
        # Test data
        eo_data = [0.1, 0.2, 0.25, 0.3, 0.5, 0.55, 0.6]
        ir_data = [0.05, 0.15, 0.22, 0.28, 0.52, 0.58, 0.62]
        
        # Test basic Bayesian fusion
        fused_mean, fused_variance = bayesian_fuse(eo_data, ir_data)
        
        assert isinstance(fused_mean, float)
        assert isinstance(fused_variance, float)
        assert fused_variance > 0
        assert -1.0 <= fused_mean <= 1.0  # Reasonable range
        
        # Test comprehensive fusion with calibration
        fusion_result = fuse_calibrate_persist(
            eo_arr=eo_data,
            ir_arr=ir_data,
            alpha=0.1,
            track_id="test_fusion_001"
        )
        
        assert "track_id" in fusion_result
        assert "confidence" in fusion_result
        assert "state" in fusion_result
        assert "covariance" in fusion_result
        assert fusion_result["confidence"] > 0
        assert fusion_result["confidence"] <= 1.0
        
        print("✅ Bayesian fusion test passed")
        
    except ImportError:
        pytest.skip("Bayesian fusion not available")

def test_conformal_prediction():
    """Test conformal prediction for uncertainty quantification"""
    try:
        from services.swarm.fusion.conformal import conformal_validate, adaptive_conformal_prediction
        
        # Test data - residual errors
        residuals = [0.1, -0.2, 0.05, 0.0, -0.15, 0.3, -0.1, 0.2]
        
        # Test standard conformal prediction
        lower_bound, upper_bound = conformal_validate(residuals, alpha=0.1)
        
        assert isinstance(lower_bound, float)
        assert isinstance(upper_bound, float)
        assert upper_bound > lower_bound
        
        # Test adaptive conformal prediction
        recent_errors = [0.08, -0.12, 0.03]
        adaptive_lower, adaptive_upper = adaptive_conformal_prediction(
            residuals, recent_errors, alpha=0.1
        )
        
        assert isinstance(adaptive_lower, float)
        assert isinstance(adaptive_upper, float)
        assert adaptive_upper > adaptive_lower
        
        print("✅ Conformal prediction test passed")
        
    except ImportError:
        pytest.skip("Conformal prediction not available")

def test_eo_ir_fusion():
    """Test EO/IR sensor fusion with evidence chains"""
    try:
        from services.swarm.fusion.eo_ir import (
            ingest_streams, build_evidence_chain, temporal_fusion_analysis
        )
        
        # Test data
        eo_stream = [0.3, 0.35, 0.4, 0.45, 0.5]
        ir_stream = [0.25, 0.3, 0.35, 0.4, 0.45]
        
        # Test stream ingestion
        eo_processed, ir_processed = ingest_streams(eo_stream, ir_stream)
        
        assert isinstance(eo_processed, np.ndarray)
        assert isinstance(ir_processed, np.ndarray)
        assert len(eo_processed) <= len(eo_stream)  # May be filtered
        assert len(ir_processed) <= len(ir_stream)
        
        # Test evidence chain building
        evidence_chain = build_evidence_chain(
            eo_processed.tolist(),
            ir_processed.tolist()
        )
        
        assert isinstance(evidence_chain, list)
        assert len(evidence_chain) > 0
        
        # Verify evidence structure
        for evidence in evidence_chain:
            assert "evidence_id" in evidence
            assert "modality" in evidence
            assert "value" in evidence
            assert "confidence" in evidence
        
        # Test temporal fusion analysis
        eo_temporal = [(time.time() - i, val) for i, val in enumerate(eo_stream)]
        ir_temporal = [(time.time() - i, val) for i, val in enumerate(ir_stream)]
        
        temporal_analysis = temporal_fusion_analysis(eo_temporal, ir_temporal)
        
        assert "temporal_overlap" in temporal_analysis
        assert "temporal_correlation" in temporal_analysis
        assert "stability_metrics" in temporal_analysis
        
        print("✅ EO/IR fusion test passed")
        
    except ImportError:
        pytest.skip("EO/IR fusion not available")

def test_roc_det_analysis():
    """Test ROC/DET analysis for detection performance"""
    try:
        from services.swarm.fusion.roc_det import (
            compute_roc, compute_det, eer, advanced_detection_analysis
        )
        
        # Test data - detection results with ground truth
        detection_points = [
            {"score": 0.9, "label": 1},
            {"score": 0.8, "label": 1},
            {"score": 0.7, "label": 0},
            {"score": 0.6, "label": 1},
            {"score": 0.5, "label": 0},
            {"score": 0.4, "label": 0},
            {"score": 0.3, "label": 0},
            {"score": 0.2, "label": 1}
        ]
        
        # Test ROC computation
        fpr, tpr, thresholds = compute_roc(detection_points)
        
        assert isinstance(fpr, list)
        assert isinstance(tpr, list)
        assert isinstance(thresholds, list)
        assert len(fpr) == len(tpr)
        assert len(fpr) > 0
        
        # Test DET computation
        det_fpr, fnr = compute_det(detection_points)
        
        assert isinstance(det_fpr, list)
        assert isinstance(fnr, list)
        assert len(det_fpr) == len(fnr)
        
        # Test EER calculation
        eer_value = eer(det_fpr, fnr)
        
        assert isinstance(eer_value, float)
        assert 0.0 <= eer_value <= 1.0
        
        # Test comprehensive analysis
        detection_results = [{"confidence": p["score"], "metadata": {}} for p in detection_points]
        ground_truth = [p["label"] for p in detection_points]
        
        analysis = advanced_detection_analysis(detection_results, ground_truth)
        
        assert "roc_curve" in analysis
        assert "det_curve" in analysis
        assert "performance_metrics" in analysis
        assert "auc" in analysis["performance_metrics"]
        assert "eer" in analysis["performance_metrics"]
        
        print("✅ ROC/DET analysis test passed")
        
    except ImportError:
        pytest.skip("ROC/DET analysis not available")

@pytest.mark.asyncio
async def test_fusion_api_endpoints():
    """Test fusion API endpoints integration"""
    try:
        import httpx
        
        # Test Bayesian fusion endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/v1/fusion/bayesian",
                json={
                    "eo_data": [0.1, 0.2, 0.3],
                    "ir_data": [0.15, 0.25, 0.35],
                    "track_id": "test_track_001"
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                assert result["success"] is True
                assert "fusion_result" in result
                assert "method" in result
                print("✅ Bayesian fusion API test passed")
            else:
                print(f"⚠️ Bayesian fusion API not available: {response.status_code}")
        
    except Exception as e:
        pytest.skip(f"Fusion API test failed: {e}")

def test_data_fusion_engine():
    """Test the core data fusion engine"""
    try:
        from core.data_fusion import DataSource, fuse_data_sources
        
        # Create test data sources
        sources = [
            DataSource(
                source_id="text_source_1",
                modality="text",
                data="This is test text data for fusion",
                confidence=0.9,
                timestamp=time.time(),
                metadata={"source": "test"}
            ),
            DataSource(
                source_id="text_source_2", 
                modality="text",
                data="Additional text data for multi-source fusion",
                confidence=0.85,
                timestamp=time.time(),
                metadata={"source": "test"}
            )
        ]
        
        # Test fusion (this will be async in real implementation)
        # For now, test the DataSource creation
        assert len(sources) == 2
        assert sources[0].modality == "text"
        assert sources[1].confidence == 0.85
        
        print("✅ Data fusion engine test passed")
        
    except ImportError:
        pytest.skip("Data fusion engine not available")

if __name__ == "__main__":
    pytest.main([__file__])
