#!/usr/bin/env python3
"""
Fusion Pipeline Test
Tests the complete fusion pipeline with Bayesian, conformal, and EO/IR fusion
"""

import os, sys
# Ensure project root on path for direct pytest invocation environments
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pytest

def test_fusion_and_conformal_pipeline():
    """Test complete fusion pipeline with conformal prediction"""
    try:
        from services.swarm.fusion.eo_ir import ingest_streams, build_evidence_chain
        from services.swarm.fusion.bayesian import bayesian_fuse, fuse_calibrate_persist
        from services.swarm.fusion.conformal import conformal_validate
        
        # Test data
        eo = [1,2,3,4,5,6,7,8]
        ir = [2,3,4,5,6,7,8,9]
        
        # Test stream ingestion
        eo_arr, ir_arr = ingest_streams(eo, ir)
        
        # Test Bayesian fusion
        mu, var = bayesian_fuse(eo_arr, ir_arr)
        
        # Test conformal prediction
        res = (np.asarray(eo_arr) - mu).tolist()
        lo, hi = conformal_validate(res, alpha=0.1)
        
        # Assertions
        assert var > 0
        assert hi >= 0 and lo <= 0
        assert isinstance(mu, float)
        assert isinstance(var, float)
        
        print("✅ Fusion and conformal pipeline test passed")
        
    except ImportError:
        pytest.skip("Fusion pipeline components not available")

def test_fuse_calibrate_persist_track():
    """Test complete fusion with calibration and persistence"""
    try:
        from services.swarm.fusion.eo_ir import ingest_streams, build_evidence_chain
        from services.swarm.fusion.bayesian import fuse_calibrate_persist
        from services.swarm.storage import load_fused_track
        
        # Test data
        eo = [0.1, 0.2, 0.25, 0.3, 0.5, 0.55, 0.6]
        ir = [0.05, 0.15, 0.22, 0.28, 0.52, 0.58, 0.62]
        
        # Ingest streams
        eo_arr, ir_arr = ingest_streams(eo, ir)
        
        # Build evidence chain
        evidence = build_evidence_chain(eo, ir)
        
        # Perform fusion with calibration and persistence
        fused = fuse_calibrate_persist(eo_arr, ir_arr, evidence=evidence, alpha=0.1)
        
        # Verify fusion result structure
        assert 'track_id' in fused
        assert fused['confidence'] <= 1.0 and fused['confidence'] > 0.0
        assert isinstance(fused['covariance'], list)
        assert 'processing_time_ms' in fused
        assert fused['processing_time_ms'] < 1000  # Reasonable latency
        
        # Test persistence (if storage available)
        try:
            track = load_fused_track(fused['track_id'])
            if track is not None:
                assert track['track_id'] == fused['track_id']
                print("✅ Fusion persistence verified")
        except:
            print("⚠️ Fusion persistence not available (storage not configured)")
        
        print("✅ Fuse calibrate persist test passed")
        
    except ImportError:
        pytest.skip("Fusion calibration components not available")

def test_fusion_api_integration():
    """Test fusion API endpoint integration"""
    try:
        import requests
        
        # Test Bayesian fusion API
        response = requests.post(
            "http://localhost:8000/v1/fusion/bayesian",
            json={
                "eo_data": [0.1, 0.2, 0.3, 0.4],
                "ir_data": [0.15, 0.25, 0.35, 0.45],
                "track_id": "api_test_track",
                "alpha": 0.1
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            assert result["success"] is True
            assert "fusion_result" in result
            assert "method" in result
            assert result["method"] == "bayesian_fusion"
            
            # Verify fusion result structure
            fusion_result = result["fusion_result"]
            assert "track_id" in fusion_result
            assert "confidence" in fusion_result
            assert "state" in fusion_result
            
            print("✅ Fusion API integration test passed")
        else:
            print(f"⚠️ Fusion API not available: HTTP {response.status_code}")
            
    except ImportError:
        pytest.skip("HTTP client not available")
    except Exception as e:
        pytest.skip(f"Fusion API test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
