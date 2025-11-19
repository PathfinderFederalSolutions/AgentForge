#!/usr/bin/env python3
"""
Consolidated test configuration for AgentForge
Supports testing of all current platform capabilities
"""

import os
import sys
import pathlib
import pytest
import asyncio
from typing import Generator

# Enable Python's faulthandler for better debugging
try:
    import faulthandler
    faulthandler.enable()
except Exception:
    pass

# Asyncio debug mode for better error reporting
os.environ.setdefault("PYTHONASYNCIODEBUG", "1")

# Test configuration
ENABLE_INTEGRATION = os.getenv("ENABLE_INTEGRATION", "0") == "1"
ENABLE_BACKEND_TESTS = os.getenv("ENABLE_BACKEND_TESTS", "1") == "1"
ENABLE_FUSION_TESTS = os.getenv("ENABLE_FUSION_TESTS", "1") == "1"
ENABLE_SERVICE_TESTS = os.getenv("ENABLE_SERVICE_TESTS", "1") == "1"

# Ensure project root is on sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Test environment setup
os.environ.setdefault("AF_ENVIRONMENT", "testing")
os.environ.setdefault("AF_LOG_LEVEL", "WARNING")  # Reduce noise in tests

# Mock API keys for testing
os.environ.setdefault("OPENAI_API_KEY", "test_key_openai")
os.environ.setdefault("ANTHROPIC_API_KEY", "test_key_anthropic")
os.environ.setdefault("GOOGLE_API_KEY", "test_key_google")

# Disable external service calls in tests
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment flags"""
    
    # Skip integration tests if not enabled
    if not ENABLE_INTEGRATION:
        skip_integration = pytest.mark.skip(reason="Integration tests disabled (set ENABLE_INTEGRATION=1)")
        for item in items:
            if "integration" in item.keywords or "integration" in str(item.fspath):
                item.add_marker(skip_integration)
    
    # Skip backend tests if not enabled
    if not ENABLE_BACKEND_TESTS:
        skip_backend = pytest.mark.skip(reason="Backend tests disabled (set ENABLE_BACKEND_TESTS=1)")
        for item in items:
            if "backend" in item.keywords or "enhanced_backend" in str(item.fspath):
                item.add_marker(skip_backend)
    
    # Skip fusion tests if not enabled
    if not ENABLE_FUSION_TESTS:
        skip_fusion = pytest.mark.skip(reason="Fusion tests disabled (set ENABLE_FUSION_TESTS=1)")
        for item in items:
            if "fusion" in item.keywords or "fusion" in str(item.fspath):
                item.add_marker(skip_fusion)

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment for each test"""
    # Set test-specific environment variables
    os.environ["AF_TEST_MODE"] = "1"
    os.environ["AF_MOCK_LLM_ENABLED"] = "1"
    
    yield
    
    # Cleanup after test
    os.environ.pop("AF_TEST_MODE", None)

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing"""
    return {
        "response": "This is a mock LLM response for testing",
        "confidence": 0.9,
        "processing_time": 0.5,
        "tokens_used": 100
    }

@pytest.fixture
def sample_agent_data():
    """Sample agent data for testing"""
    return {
        "agent_id": "test_agent_001",
        "name": "Test Agent",
        "type": "neural-mesh",
        "status": "active",
        "capabilities": ["data_analysis", "pattern_recognition"],
        "performance_metrics": {
            "tasks_completed": 15,
            "success_rate": 0.93,
            "average_task_time": 2.5
        }
    }

@pytest.fixture
def sample_fusion_data():
    """Sample fusion data for testing"""
    return {
        "eo_data": [0.1, 0.2, 0.25, 0.3, 0.5, 0.55, 0.6],
        "ir_data": [0.05, 0.15, 0.22, 0.28, 0.52, 0.58, 0.62],
        "expected_confidence_range": (0.7, 0.95),
        "expected_processing_time_ms": (10, 1000)
    }

@pytest.fixture
def sample_chat_request():
    """Sample chat request for testing"""
    return {
        "message": "Test message for AgentForge capabilities",
        "context": {
            "userId": "test_user_001",
            "sessionId": "test_session_001",
            "conversationHistory": [],
            "dataSources": [],
            "userPreferences": {}
        },
        "capabilities": ["general_intelligence"]
    }

@pytest.fixture
def backend_url():
    """Backend URL for testing"""
    return BASE_URL

@pytest.fixture
def admin_dashboard_url():
    """Admin dashboard URL for testing"""
    return ADMIN_DASHBOARD_URL

@pytest.fixture
def individual_frontend_url():
    """Individual frontend URL for testing"""
    return INDIVIDUAL_FRONTEND_URL

# Helper functions for tests
def is_backend_available() -> bool:
    """Check if backend is available for testing"""
    try:
        import requests
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def is_service_available(service_path: str) -> bool:
    """Check if specific service endpoint is available"""
    try:
        import requests
        response = requests.get(f"{BASE_URL}{service_path}", timeout=2)
        return response.status_code in [200, 404]  # 404 means server is up but endpoint might not exist
    except:
        return False

def skip_if_backend_unavailable(func):
    """Decorator to skip test if backend is not available"""
    def wrapper(*args, **kwargs):
        if not is_backend_available():
            pytest.skip("Backend not available for testing")
        return func(*args, **kwargs)
    return wrapper

def skip_if_service_unavailable(service_path: str):
    """Decorator to skip test if specific service is not available"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_service_available(service_path):
                pytest.skip(f"Service {service_path} not available for testing")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Test markers
pytest.mark.backend = pytest.mark.skipif(
    not ENABLE_BACKEND_TESTS, 
    reason="Backend tests disabled"
)

pytest.mark.fusion = pytest.mark.skipif(
    not ENABLE_FUSION_TESTS,
    reason="Fusion tests disabled"
)

pytest.mark.integration = pytest.mark.skipif(
    not ENABLE_INTEGRATION,
    reason="Integration tests disabled"
)

pytest.mark.services = pytest.mark.skipif(
    not ENABLE_SERVICE_TESTS,
    reason="Service tests disabled"
)

# Async test configuration
@pytest.fixture(scope="session")
def anyio_backend():
    """Configure anyio backend for async tests"""
    return "asyncio"