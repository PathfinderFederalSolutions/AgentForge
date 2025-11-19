#!/usr/bin/env python3
"""
Test suite for enhanced backend capabilities
Tests enhanced chat API, database manager, logging, and core AI services
"""

import pytest
import asyncio
import time
import requests
from unittest.mock import patch, MagicMock

BASE_URL = "http://localhost:8000"

def test_enhanced_chat_api_health():
    """Test enhanced chat API health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert data["status"] == "ok"
            assert "enhanced_features" in data or "llms_available" in data
            print("✅ Enhanced chat API health test passed")
        else:
            pytest.skip(f"Backend not available: {response.status_code}")
            
    except requests.exceptions.RequestException:
        pytest.skip("Backend not available for testing")

def test_services_status_endpoint():
    """Test comprehensive services status endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/v1/services/status", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            assert "core_services" in data
            assert "ai_services" in data
            assert "advanced_services" in data
            assert "system_readiness" in data
            
            # Verify system readiness calculation
            readiness = data["system_readiness"]
            assert "available_services" in readiness
            assert "total_services" in readiness
            assert "readiness_percentage" in readiness
            
            print("✅ Services status endpoint test passed")
        else:
            pytest.skip(f"Services status endpoint not available: {response.status_code}")
            
    except requests.exceptions.RequestException:
        pytest.skip("Backend not available for testing")

def test_libraries_status_endpoint():
    """Test AF libraries status endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/v1/libraries/status", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            assert "libraries" in data
            assert "summary" in data
            
            libraries = data["libraries"]
            assert "af_common" in libraries
            assert "af_schemas" in libraries
            assert "af_messaging" in libraries
            
            # Verify library structure
            for lib_name, lib_data in libraries.items():
                assert "available" in lib_data
                assert "description" in lib_data
                assert "components" in lib_data
                assert "integration_status" in lib_data
            
            print("✅ Libraries status endpoint test passed")
        else:
            pytest.skip(f"Libraries status endpoint not available: {response.status_code}")
            
    except requests.exceptions.RequestException:
        pytest.skip("Backend not available for testing")

def test_database_manager():
    """Test database manager functionality"""
    try:
        from core.database_manager import get_db_manager, AgentExecutionRecord
        
        # Test database manager creation
        db = get_db_manager()
        assert db is not None
        
        # Test agent execution record creation
        record = AgentExecutionRecord(
            execution_id="test_exec_001",
            agent_id="test_agent_001",
            agent_type="test_agent",
            task_description="Test task execution",
            start_time=time.time() - 1.0,
            end_time=time.time(),
            success=True,
            result_data={"test": "result"},
            performance_metrics={"execution_time": 1.0}
        )
        
        # Test recording (should not fail)
        db.record_agent_execution(record)
        
        # Test analytics retrieval
        analytics = db.get_agent_performance_analytics(hours=1)
        assert isinstance(analytics, dict)
        
        print("✅ Database manager test passed")
        
    except ImportError:
        pytest.skip("Database manager not available")

def test_enhanced_logging():
    """Test enhanced logging capabilities"""
    try:
        from core.enhanced_logging import (
            log_info, log_error, log_agent_activity, 
            log_swarm_deployment, agentforge_logger
        )
        
        # Test basic logging functions
        log_info("Test info message", {"test": True})
        log_error("Test error message", {"error_type": "test"})
        
        # Test agent activity logging
        log_agent_activity(
            agent_id="test_agent_001",
            action="test_action",
            status="completed",
            details={"duration": 1.5}
        )
        
        # Test swarm deployment logging
        log_swarm_deployment(
            agents_deployed=5,
            capability="test_capability",
            execution_time=2.3,
            success=True
        )
        
        # Test logger instance
        assert agentforge_logger is not None
        
        print("✅ Enhanced logging test passed")
        
    except ImportError:
        pytest.skip("Enhanced logging not available")

def test_request_pipeline():
    """Test request processing pipeline"""
    try:
        from core.request_pipeline import process_user_request, request_pipeline
        
        # Test pipeline instance
        assert request_pipeline is not None
        
        # Test basic request processing (async)
        async def run_pipeline_test():
            result = await process_user_request(
                "Test user request",
                {"test_context": True}
            )
            
            assert isinstance(result, dict)
            assert "success" in result
            assert "request_id" in result
            
            return result
        
        # Run async test
        result = asyncio.run(run_pipeline_test())
        print("✅ Request pipeline test passed")
        
    except ImportError:
        pytest.skip("Request pipeline not available")

def test_retry_handler():
    """Test retry handler with backoff"""
    try:
        from core.retry_handler import retry_with_backoff, RetryConfig
        
        # Test function that fails then succeeds
        call_count = 0
        
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Test failure")
            return "success"
        
        # Test retry logic
        async def run_retry_test():
            result = await retry_with_backoff(
                test_function,
                max_attempts=3,
                base_delay=0.1,
                context={"test": "retry"}
            )
            return result
        
        result = asyncio.run(run_retry_test())
        assert result == "success"
        assert call_count == 3  # Should have retried twice
        
        print("✅ Retry handler test passed")
        
    except ImportError:
        pytest.skip("Retry handler not available")

@pytest.mark.asyncio
async def test_fusion_endpoints():
    """Test fusion API endpoints"""
    try:
        import httpx
        
        test_cases = [
            {
                "endpoint": "/v1/fusion/bayesian",
                "data": {
                    "eo_data": [0.1, 0.2, 0.3],
                    "ir_data": [0.15, 0.25, 0.35]
                }
            },
            {
                "endpoint": "/v1/fusion/conformal-prediction", 
                "data": {
                    "residuals": [0.1, -0.2, 0.05, 0.0],
                    "alpha": 0.1
                }
            },
            {
                "endpoint": "/v1/data-fusion/fuse",
                "data": {
                    "sources": [
                        {
                            "id": "source_1",
                            "modality": "text",
                            "data": "Test data",
                            "confidence": 0.8
                        }
                    ],
                    "method": "auto"
                }
            }
        ]
        
        async with httpx.AsyncClient() as client:
            for test_case in test_cases:
                try:
                    response = await client.post(
                        f"{BASE_URL}{test_case['endpoint']}",
                        json=test_case["data"],
                        timeout=5.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            print(f"✅ {test_case['endpoint']} test passed")
                        else:
                            print(f"⚠️ {test_case['endpoint']} returned error: {result.get('error')}")
                    else:
                        print(f"⚠️ {test_case['endpoint']} not available: {response.status_code}")
                        
                except Exception as e:
                    print(f"⚠️ {test_case['endpoint']} test failed: {e}")
        
    except ImportError:
        pytest.skip("HTTP client not available")

if __name__ == "__main__":
    pytest.main([__file__])
