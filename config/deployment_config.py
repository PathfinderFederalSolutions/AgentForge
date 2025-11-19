#!/usr/bin/env python3
"""
Deployment Configuration for AgentForge Cloud Migration
Environment-specific settings for different deployment editions
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class DeploymentConfig:
    """Deployment configuration for different environments"""
    
    # Environment identification
    environment: str  # development, staging, production
    edition: str      # commercial, fedciv, dod, private
    
    # CORS configuration
    cors_origins: List[str]
    
    # Security settings
    require_authentication: bool
    require_https: bool
    enable_rate_limiting: bool
    rate_limit_rpm: int
    
    # Feature flags
    enable_admin_dashboard: bool
    enable_individual_frontend: bool
    enable_websocket_streaming: bool
    enable_fusion_capabilities: bool
    enable_quantum_scheduling: bool
    
    # Compliance settings
    fips_mode: bool
    audit_logging_required: bool
    data_residency_region: str
    
    # Performance settings
    max_concurrent_agents: int
    agent_timeout_seconds: int
    request_timeout_seconds: int
    
    # Infrastructure
    database_url: str
    redis_url: str
    nats_url: str

# Deployment configurations for different editions
DEPLOYMENT_CONFIGS = {
    "commercial": {
        "development": DeploymentConfig(
            environment="development",
            edition="commercial",
            cors_origins=["http://localhost:3001", "http://localhost:3002", "http://localhost:3000"],
            require_authentication=False,
            require_https=False,
            enable_rate_limiting=False,
            rate_limit_rpm=1000,
            enable_admin_dashboard=True,
            enable_individual_frontend=True,
            enable_websocket_streaming=True,
            enable_fusion_capabilities=True,
            enable_quantum_scheduling=True,
            fips_mode=False,
            audit_logging_required=False,
            data_residency_region="us-east-1",
            max_concurrent_agents=100,
            agent_timeout_seconds=300,
            request_timeout_seconds=30,
            database_url="sqlite:///./agentforge.db",
            redis_url="redis://localhost:6379/0",
            nats_url="nats://localhost:4222"
        ),
        "production": DeploymentConfig(
            environment="production",
            edition="commercial",
            cors_origins=["https://agentforge.com", "https://app.agentforge.com", "https://admin.agentforge.com"],
            require_authentication=True,
            require_https=True,
            enable_rate_limiting=True,
            rate_limit_rpm=100,
            enable_admin_dashboard=True,
            enable_individual_frontend=True,
            enable_websocket_streaming=True,
            enable_fusion_capabilities=True,
            enable_quantum_scheduling=True,
            fips_mode=False,
            audit_logging_required=True,
            data_residency_region="us-east-1",
            max_concurrent_agents=1000,
            agent_timeout_seconds=600,
            request_timeout_seconds=60,
            database_url="postgresql://agentforge:password@db:5432/agentforge",
            redis_url="redis://redis:6379/0",
            nats_url="nats://nats:4222"
        )
    },
    "fedciv": {
        "production": DeploymentConfig(
            environment="production",
            edition="fedciv",
            cors_origins=["https://agentforge.gov", "https://app.agentforge.gov"],
            require_authentication=True,
            require_https=True,
            enable_rate_limiting=True,
            rate_limit_rpm=50,
            enable_admin_dashboard=True,
            enable_individual_frontend=True,
            enable_websocket_streaming=True,
            enable_fusion_capabilities=True,
            enable_quantum_scheduling=True,
            fips_mode=True,
            audit_logging_required=True,
            data_residency_region="us-gov-east-1",
            max_concurrent_agents=500,
            agent_timeout_seconds=300,
            request_timeout_seconds=30,
            database_url="postgresql://agentforge:password@db:5432/agentforge",
            redis_url="redis://redis:6379/0",
            nats_url="nats://nats:4222"
        )
    },
    "dod": {
        "production": DeploymentConfig(
            environment="production",
            edition="dod",
            cors_origins=["https://agentforge.mil", "https://app.agentforge.mil"],
            require_authentication=True,
            require_https=True,
            enable_rate_limiting=True,
            rate_limit_rpm=30,
            enable_admin_dashboard=True,
            enable_individual_frontend=False,  # Admin only for DoD
            enable_websocket_streaming=False,  # Disabled for security
            enable_fusion_capabilities=True,
            enable_quantum_scheduling=True,
            fips_mode=True,
            audit_logging_required=True,
            data_residency_region="us-gov-east-1",
            max_concurrent_agents=200,
            agent_timeout_seconds=180,
            request_timeout_seconds=20,
            database_url="postgresql://agentforge:password@db:5432/agentforge",
            redis_url="redis://redis:6379/0",
            nats_url="nats://nats:4222"
        )
    },
    "private": {
        "production": DeploymentConfig(
            environment="production",
            edition="private",
            cors_origins=["https://agentforge.local", "https://app.agentforge.local"],
            require_authentication=True,
            require_https=True,
            enable_rate_limiting=True,
            rate_limit_rpm=200,
            enable_admin_dashboard=True,
            enable_individual_frontend=True,
            enable_websocket_streaming=True,
            enable_fusion_capabilities=True,
            enable_quantum_scheduling=True,
            fips_mode=False,
            audit_logging_required=True,
            data_residency_region="us-east-1",
            max_concurrent_agents=2000,
            agent_timeout_seconds=600,
            request_timeout_seconds=60,
            database_url="postgresql://agentforge:password@db:5432/agentforge",
            redis_url="redis://redis:6379/0",
            nats_url="nats://nats:4222"
        )
    }
}

def get_deployment_config() -> DeploymentConfig:
    """Get deployment configuration based on environment variables"""
    environment = os.getenv("AF_ENVIRONMENT", "development").lower()
    edition = os.getenv("AF_EDITION", "commercial").lower()
    
    # Get base configuration
    if edition in DEPLOYMENT_CONFIGS and environment in DEPLOYMENT_CONFIGS[edition]:
        config = DEPLOYMENT_CONFIGS[edition][environment]
    else:
        # Fallback to commercial development
        config = DEPLOYMENT_CONFIGS["commercial"]["development"]
    
    # Override with environment variables
    config = _override_with_env_vars(config)
    
    return config

def _override_with_env_vars(config: DeploymentConfig) -> DeploymentConfig:
    """Override configuration with environment variables"""
    
    # Security overrides
    if os.getenv("AF_REQUIRE_AUTH"):
        config.require_authentication = os.getenv("AF_REQUIRE_AUTH").lower() == "true"
    
    if os.getenv("AF_REQUIRE_HTTPS"):
        config.require_https = os.getenv("AF_REQUIRE_HTTPS").lower() == "true"
    
    if os.getenv("AF_FIPS_MODE"):
        config.fips_mode = os.getenv("AF_FIPS_MODE").lower() == "true"
    
    # Performance overrides
    if os.getenv("AF_MAX_AGENTS"):
        config.max_concurrent_agents = int(os.getenv("AF_MAX_AGENTS"))
    
    if os.getenv("AF_AGENT_TIMEOUT"):
        config.agent_timeout_seconds = int(os.getenv("AF_AGENT_TIMEOUT"))
    
    if os.getenv("AF_RATE_LIMIT_RPM"):
        config.rate_limit_rpm = int(os.getenv("AF_RATE_LIMIT_RPM"))
    
    # Infrastructure overrides
    if os.getenv("DATABASE_URL"):
        config.database_url = os.getenv("DATABASE_URL")
    
    if os.getenv("REDIS_URL"):
        config.redis_url = os.getenv("REDIS_URL")
    
    if os.getenv("NATS_URL"):
        config.nats_url = os.getenv("NATS_URL")
    
    return config

def get_cors_origins() -> List[str]:
    """Get CORS origins for current deployment"""
    config = get_deployment_config()
    return config.cors_origins

def is_production() -> bool:
    """Check if running in production"""
    config = get_deployment_config()
    return config.environment == "production"

def is_fips_mode() -> bool:
    """Check if FIPS mode is enabled"""
    config = get_deployment_config()
    return config.fips_mode

def get_security_settings() -> Dict[str, Any]:
    """Get security settings for current deployment"""
    config = get_deployment_config()
    
    return {
        "require_authentication": config.require_authentication,
        "require_https": config.require_https,
        "enable_rate_limiting": config.enable_rate_limiting,
        "rate_limit_rpm": config.rate_limit_rpm,
        "fips_mode": config.fips_mode,
        "audit_logging_required": config.audit_logging_required,
        "data_residency_region": config.data_residency_region
    }

def get_feature_flags() -> Dict[str, bool]:
    """Get feature flags for current deployment"""
    config = get_deployment_config()
    
    return {
        "admin_dashboard": config.enable_admin_dashboard,
        "individual_frontend": config.enable_individual_frontend,
        "websocket_streaming": config.enable_websocket_streaming,
        "fusion_capabilities": config.enable_fusion_capabilities,
        "quantum_scheduling": config.enable_quantum_scheduling
    }
