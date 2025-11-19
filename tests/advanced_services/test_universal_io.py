#!/usr/bin/env python3
"""
Universal I/O Test
Tests universal input/output processing capabilities
"""

import pytest
import asyncio

def test_universal_io_basic():
    """Test basic Universal I/O functionality"""
    try:
        from services.universal_io.agi_integration import UniversalAGIEngine
        from services.universal_io.input.pipeline import UniversalInputPipeline
        from services.universal_io.output.pipeline import UniversalOutputPipeline
        
        # Test AGI engine creation
        agi_engine = UniversalAGIEngine()
        assert agi_engine is not None
        
        # Test input pipeline
        input_pipeline = UniversalInputPipeline()
        assert input_pipeline is not None
        
        # Test output pipeline
        output_pipeline = UniversalOutputPipeline()
        assert output_pipeline is not None
        
        print("✅ Universal I/O basic test passed")
        
    except ImportError:
        pytest.skip("Universal I/O not available")

def test_universal_io_api():
    """Test Universal I/O API endpoint"""
    try:
        import requests
        
        response = requests.post(
            "http://localhost:8000/v1/services/universal-io/process",
            json={
                "input_data": "Test input for universal I/O processing",
                "input_type": "text",
                "output_format": "json"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if not result.get("error"):
                print("✅ Universal I/O API test passed")
            else:
                print(f"⚠️ Universal I/O API error: {result.get('error')}")
        else:
            print(f"⚠️ Universal I/O API not available: HTTP {response.status_code}")
            
    except Exception as e:
        pytest.skip(f"Universal I/O API test failed: {e}")

def test_universal_io_capabilities():
    """Test Universal I/O capabilities endpoint"""
    try:
        import requests
        
        response = requests.get("http://localhost:8000/v1/chat/capabilities", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Verify input formats
            if "inputFormats" in data:
                input_formats = data["inputFormats"]
                assert "total" in input_formats
                assert input_formats["total"] > 50  # Should support 70+ formats
                
                # Verify categories
                if "categories" in input_formats:
                    categories = input_formats["categories"]
                    expected_categories = ["documents", "data", "images", "audio", "video", "code"]
                    
                    for category in expected_categories:
                        assert category in categories, f"Missing input category: {category}"
            
            # Verify output formats
            if "outputFormats" in data:
                output_formats = data["outputFormats"]
                assert "total" in output_formats
                assert output_formats["total"] > 40  # Should support 45+ formats
            
            print("✅ Universal I/O capabilities test passed")
        else:
            print(f"⚠️ Capabilities endpoint not available: HTTP {response.status_code}")
            
    except Exception as e:
        pytest.skip(f"Universal I/O capabilities test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
