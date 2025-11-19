#!/usr/bin/env python3
"""
Core Systems Infrastructure Test
Tests imports, routing, and core infrastructure components
"""

import pytest
import sys
import os

def test_core_imports():
    """Test that all core imports work correctly"""
    try:
        # Test core agent imports
        from core.agents import Agent, AgentSwarm
        from services.swarm.forge_types import Task, AgentContract
        
        # Test contract creation
        contract = AgentContract(
            name="test_agent",
            capabilities=["test_capability"],
            memory_scopes=[],
            tools=[],
            budget=1000
        )
        
        # Test agent creation
        agent = Agent(contract)
        assert agent.contract.name == "test_agent"
        assert agent.contract.budget == 1000
        
        print("✅ Core imports test passed")
        
    except ImportError as e:
        pytest.fail(f"Core imports failed: {e}")

def test_router_functionality():
    """Test router functionality"""
    try:
        from router import DynamicRouter
        
        # Test router creation
        router = DynamicRouter()
        assert router is not None
        
        # Test basic routing (if methods available)
        if hasattr(router, 'route'):
            route_result = router.route("test task description")
            assert route_result is not None
        
        print("✅ Router functionality test passed")
        
    except ImportError:
        pytest.skip("Router not available")

def test_orchestrator_integration():
    """Test orchestrator integration"""
    try:
        from orchestrator import build_orchestrator
        
        # Test orchestrator creation
        orch = build_orchestrator(num_agents=2)
        assert orch is not None
        assert orch.num_agents >= 2
        
        print("✅ Orchestrator integration test passed")
        
    except ImportError:
        pytest.skip("Orchestrator not available")

def test_enhanced_core_systems():
    """Test enhanced core systems"""
    try:
        # Test enhanced logging
        from core.enhanced_logging import log_info, log_error, agentforge_logger
        
        log_info("Test info message", {"test": True})
        log_error("Test error message", {"error_type": "test"})
        assert agentforge_logger is not None
        
        # Test database manager
        from core.database_manager import get_db_manager
        
        db = get_db_manager()
        assert db is not None
        
        # Test retry handler
        from core.retry_handler import retry_handler
        
        assert retry_handler is not None
        
        print("✅ Enhanced core systems test passed")
        
    except ImportError:
        pytest.skip("Enhanced core systems not available")

def test_configuration_systems():
    """Test configuration and settings systems"""
    try:
        # Test enhanced configuration
        from config.agent_config import get_config, get_server_config, get_agent_config
        
        config = get_config()
        server_config = get_server_config()
        agent_config = get_agent_config()
        
        assert config is not None
        assert server_config is not None
        assert agent_config is not None
        
        # Test AF-Common config if available
        try:
            from libs.af_common.config import get_config as get_af_config, BaseConfig
            af_config = get_af_config(BaseConfig)
            assert af_config is not None
            print("✅ AF-Common configuration available")
        except ImportError:
            print("⚠️ AF-Common configuration not available")
        
        print("✅ Configuration systems test passed")
        
    except ImportError:
        pytest.skip("Configuration systems not available")

if __name__ == "__main__":
    pytest.main([__file__])
