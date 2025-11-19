"""
Shared configuration management for AgentForge services
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Type, TypeVar, Union
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

T = TypeVar('T', bound=BaseSettings)

class BaseConfig(BaseSettings):
    """Base configuration for all AgentForge services"""
    
    # Core service settings
    service_name: str = Field(default="agentforge-service")
    environment: str = Field(default="development")
    version: str = Field(default="0.1.0")
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    
    # HTTP Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    
    # Database
    database_url: str = Field(default="sqlite:///./agentforge.db")
    database_pool_size: int = Field(default=10)
    database_max_overflow: int = Field(default=20)
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = None
    
    # NATS/JetStream
    nats_url: str = Field(default="nats://localhost:4222")
    nats_cluster: Optional[str] = None
    nats_max_reconnect_attempts: int = Field(default=60)
    nats_reconnect_wait: float = Field(default=2.0)
    
    # Observability
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    tracing_enabled: bool = Field(default=False)
    tracing_endpoint: Optional[str] = None
    
    # Security
    api_key_required: bool = Field(default=False)
    api_keys: list[str] = Field(default_factory=list)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)
    
    # Agent settings
    max_agents: int = Field(default=10)
    agent_timeout_seconds: int = Field(default=300)
    task_queue_size: int = Field(default=1000)
    
    # Memory settings
    memory_ttl_seconds: int = Field(default=604800)  # 7 days
    memory_prune_threshold: int = Field(default=10000)
    embeddings_backend: str = Field(default="hash")  # hash, sentence-transformers
    
    # LLM Provider settings
    openai_api_key: Optional[str] = None
    openai_model: str = Field(default="gpt-4")
    openai_max_tokens: int = Field(default=4096)
    
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022")
    anthropic_max_tokens: int = Field(default=4096)
    
    google_api_key: Optional[str] = None
    google_model: str = Field(default="gemini-1.5-pro")
    
    cohere_api_key: Optional[str] = None
    cohere_model: str = Field(default="command-r-plus")
    
    mistral_api_key: Optional[str] = None
    mistral_model: str = Field(default="mistral-large-latest")
    
    # Vector database settings
    pinecone_enabled: bool = Field(default=False)
    pinecone_api_key: Optional[str] = None
    pinecone_index: str = Field(default="agentforge")
    pinecone_environment: str = Field(default="us-west1-gcp")
    
    # Development/Testing
    mock_llm_enabled: bool = Field(default=False)
    debug_mode: bool = Field(default=False)
    test_mode: bool = Field(default=False)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

class SwarmConfig(BaseConfig):
    """Configuration specific to swarm services"""
    
    # Swarm-specific settings
    swarm_size: int = Field(default=3)
    swarm_scaling_enabled: bool = Field(default=True)
    swarm_max_size: int = Field(default=20)
    swarm_min_size: int = Field(default=1)
    
    # Task distribution
    task_distribution_strategy: str = Field(default="round_robin")  # round_robin, capability_based, load_based
    task_retry_attempts: int = Field(default=3)
    task_retry_delay_seconds: float = Field(default=1.0)
    
    # Agent lifecycle
    agent_idle_timeout_seconds: int = Field(default=300)
    agent_health_check_interval: int = Field(default=30)
    agent_restart_on_failure: bool = Field(default=True)

class OrchestratorConfig(BaseConfig):
    """Configuration specific to orchestrator service"""
    
    # Orchestration settings
    max_concurrent_jobs: int = Field(default=50)
    job_timeout_seconds: int = Field(default=3600)
    planning_timeout_seconds: int = Field(default=60)
    
    # SLA/KPI enforcement
    sla_enforcement_enabled: bool = Field(default=True)
    sla_error_rate_threshold: float = Field(default=0.05)
    sla_latency_threshold_ms: float = Field(default=5000.0)
    
    # HITL (Human-in-the-Loop)
    hitl_enabled: bool = Field(default=True)
    hitl_auto_approve: bool = Field(default=True)
    hitl_timeout_seconds: int = Field(default=300)

class MemoryConfig(BaseConfig):
    """Configuration specific to memory services"""
    
    # Memory mesh settings
    mesh_mode: str = Field(default="local")  # local, distributed
    mesh_sync_interval_seconds: int = Field(default=30)
    mesh_conflict_resolution: str = Field(default="last_write_wins")
    
    # Vector storage
    vector_dimensions: int = Field(default=384)
    vector_similarity_threshold: float = Field(default=0.7)
    vector_max_results: int = Field(default=10)
    
    # Persistence
    persistence_enabled: bool = Field(default=True)
    persistence_backend: str = Field(default="sqlite")  # sqlite, postgresql, redis
    backup_enabled: bool = Field(default=True)
    backup_interval_hours: int = Field(default=24)

# Global configuration instance
_config_cache: Dict[str, BaseSettings] = {}

def get_config(config_class: Type[T] = BaseConfig, reload: bool = False) -> T:
    """
    Get configuration instance with caching
    
    Args:
        config_class: Configuration class to instantiate
        reload: Force reload from environment
        
    Returns:
        Configuration instance
    """
    cache_key = config_class.__name__
    
    if reload or cache_key not in _config_cache:
        _config_cache[cache_key] = config_class()
        
    return _config_cache[cache_key]  # type: ignore

def load_config_from_file(file_path: Union[str, Path], config_class: Type[T] = BaseConfig) -> T:
    """
    Load configuration from JSON file
    
    Args:
        file_path: Path to configuration file
        config_class: Configuration class to instantiate
        
    Returns:
        Configuration instance
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
    with open(path, 'r') as f:
        config_data = json.load(f)
        
    return config_class(**config_data)

def save_config_to_file(config: BaseSettings, file_path: Union[str, Path]) -> None:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration instance to save
        file_path: Path to save configuration file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config.model_dump(), f, indent=2, default=str)

def get_environment_config() -> Dict[str, str]:
    """Get all environment variables with AF_ prefix"""
    return {k: v for k, v in os.environ.items() if k.startswith('AF_')}

def validate_config(config: BaseSettings) -> list[str]:
    """
    Validate configuration and return list of issues
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation error messages
    """
    issues = []
    
    # Check required API keys if not in test mode
    if not config.test_mode and not config.mock_llm_enabled:
        if not any([config.openai_api_key, config.anthropic_api_key, config.google_api_key]):
            issues.append("At least one LLM provider API key must be configured")
    
    # Check database URL format
    if config.database_url and not config.database_url.startswith(('sqlite://', 'postgresql://', 'mysql://')):
        issues.append(f"Invalid database URL format: {config.database_url}")
    
    # Check port ranges
    if not (1024 <= config.port <= 65535):
        issues.append(f"Port must be between 1024-65535, got: {config.port}")
    
    # Check Redis configuration
    if config.redis_url and not config.redis_url.startswith('redis://'):
        issues.append(f"Invalid Redis URL format: {config.redis_url}")
    
    return issues

# Convenience functions for common configurations
def get_swarm_config() -> SwarmConfig:
    """Get swarm service configuration"""
    return get_config(SwarmConfig)

def get_orchestrator_config() -> OrchestratorConfig:
    """Get orchestrator service configuration"""
    return get_config(OrchestratorConfig)

def get_memory_config() -> MemoryConfig:
    """Get memory service configuration"""
    return get_config(MemoryConfig)
