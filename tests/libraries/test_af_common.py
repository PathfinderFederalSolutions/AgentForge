#!/usr/bin/env python3
"""
Test suite for AF-Common library integration
Tests types, logging, configuration, settings, errors, and tracing
"""

import pytest
import asyncio
import time
import json
from unittest.mock import patch, MagicMock

# Test af-common types
def test_af_common_types():
    """Test AF-Common type definitions"""
    try:
        from libs.af_common.types import Task, AgentContract, TaskResult, AgentStatus
        
        # Test Task creation
        task = Task(
            description="Test task",
            capabilities_required=["test_capability"],
            priority=5
        )
        assert task.description == "Test task"
        assert task.priority == 5
        assert len(task.capabilities_required) == 1
        
        # Test AgentContract creation
        contract = AgentContract(
            name="test_agent",
            capabilities=["test_capability"],
            budget=1000
        )
        assert contract.name == "test_agent"
        assert contract.budget == 1000
        
        print("✅ AF-Common types test passed")
        
    except ImportError:
        pytest.skip("AF-Common types not available")

def test_af_common_logging():
    """Test AF-Common logging functionality"""
    try:
        from libs.af_common.logging import setup_logging, get_logger, log_performance
        
        # Test logger setup
        setup_logging("test_service", log_level="INFO")
        
        # Test logger creation
        logger = get_logger("test_logger", service_name="test_service")
        assert logger is not None
        
        # Test performance logging
        log_performance("test_operation", 100.5, test_context="test")
        
        print("✅ AF-Common logging test passed")
        
    except ImportError:
        pytest.skip("AF-Common logging not available")

def test_af_common_settings():
    """Test AF-Common settings and feature flags"""
    try:
        from libs.af_common.settings import get_settings, is_feature_enabled
        
        # Test settings access
        settings = get_settings()
        assert settings is not None
        
        # Test feature flag checking
        enabled_features = settings.get_enabled_features()
        assert isinstance(enabled_features, list)
        
        # Test environment detection
        env_name = settings.environment.name
        assert env_name in ["development", "testing", "production"]
        
        print("✅ AF-Common settings test passed")
        
    except ImportError:
        pytest.skip("AF-Common settings not available")

def test_af_common_errors():
    """Test AF-Common error handling"""
    try:
        from libs.af_common.errors import (
            AgentForgeError, TaskExecutionError, create_error_context,
            ErrorSeverity, ErrorCategory
        )
        
        # Test error context creation
        context = create_error_context(
            service_name="test_service",
            component="test_component", 
            operation="test_operation"
        )
        assert context.service_name == "test_service"
        assert context.component == "test_component"
        
        # Test custom error creation
        error = TaskExecutionError(
            message="Test task failed",
            task_id="test_task_123",
            severity=ErrorSeverity.HIGH
        )
        assert error.message == "Test task failed"
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.TASK
        
        print("✅ AF-Common errors test passed")
        
    except ImportError:
        pytest.skip("AF-Common errors not available")

@pytest.mark.asyncio
async def test_af_common_tracing():
    """Test AF-Common distributed tracing"""
    try:
        from libs.af_common.tracing import get_tracer, trace_operation
        
        # Test tracer creation
        tracer = get_tracer("test_service")
        assert tracer is not None
        
        # Test span creation and context manager
        with trace_operation("test_operation", tags={"test": True}) as span:
            span.add_tag("operation_type", "test")
            span.add_log("Test log message")
            time.sleep(0.1)  # Simulate work
        
        # Verify span was recorded
        finished_spans = tracer.get_finished_spans(limit=1)
        assert len(finished_spans) >= 1
        assert finished_spans[-1].operation_name == "test_operation"
        
        print("✅ AF-Common tracing test passed")
        
    except ImportError:
        pytest.skip("AF-Common tracing not available")

def test_af_common_config():
    """Test AF-Common configuration management"""
    try:
        from libs.af_common.config import get_config, BaseConfig
        
        # Test config access
        config = get_config(BaseConfig)
        assert config is not None
        assert hasattr(config, 'service_name')
        assert hasattr(config, 'environment')
        
        # Test configuration validation
        from libs.af_common.config import validate_config
        issues = validate_config(config)
        assert isinstance(issues, list)
        
        print("✅ AF-Common config test passed")
        
    except ImportError:
        pytest.skip("AF-Common config not available")

if __name__ == "__main__":
    pytest.main([__file__])
