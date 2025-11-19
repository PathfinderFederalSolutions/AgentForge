"""
Application settings and environment management for AgentForge
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

from .config import BaseConfig, get_config
from .logging import get_logger

logger = get_logger("settings")

@dataclass
class EnvironmentInfo:
    """Information about the current environment"""
    name: str
    is_production: bool
    is_development: bool
    is_testing: bool
    config_sources: List[str] = field(default_factory=list)
    features_enabled: Dict[str, bool] = field(default_factory=dict)

class Settings:
    """Global settings manager for AgentForge"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.environment = self._detect_environment()
        self.config = get_config()
        self.feature_flags = {}
        self.runtime_settings = {}
        
        self._load_feature_flags()
        self._load_runtime_settings()
        self._initialized = True
        
        logger.info(f"Settings initialized for environment: {self.environment.name}")
    
    def _detect_environment(self) -> EnvironmentInfo:
        """Detect current environment"""
        env_name = os.getenv("AF_ENVIRONMENT", os.getenv("ENVIRONMENT", "development")).lower()
        
        is_production = env_name in ["production", "prod"]
        is_development = env_name in ["development", "dev", "local"]
        is_testing = env_name in ["testing", "test"]
        
        # Detect config sources
        config_sources = []
        if Path(".env").exists():
            config_sources.append(".env")
        if Path("config.json").exists():
            config_sources.append("config.json")
        if os.getenv("AF_CONFIG_FILE"):
            config_sources.append(os.getenv("AF_CONFIG_FILE"))
        
        return EnvironmentInfo(
            name=env_name,
            is_production=is_production,
            is_development=is_development,
            is_testing=is_testing,
            config_sources=config_sources
        )
    
    def _load_feature_flags(self) -> None:
        """Load feature flags from environment and config"""
        # Default feature flags
        default_flags = {
            "neural_mesh_enabled": True,
            "quantum_scheduler_enabled": True,
            "universal_io_enabled": True,
            "advanced_fusion_enabled": True,
            "self_bootstrap_enabled": True,
            "security_orchestrator_enabled": True,
            "mega_swarm_enabled": True,
            "enhanced_logging_enabled": True,
            "database_analytics_enabled": True,
            "real_time_monitoring_enabled": True,
            "admin_dashboard_enabled": True,
            "websocket_support_enabled": True,
            "retry_handling_enabled": True,
            "conformal_prediction_enabled": True,
            "bayesian_fusion_enabled": True,
            "eo_ir_fusion_enabled": True,
            "roc_det_analysis_enabled": True
        }
        
        # Override with environment variables
        for flag_name in default_flags:
            env_var = f"AF_FEATURE_{flag_name.upper()}"
            env_value = os.getenv(env_var)
            if env_value is not None:
                default_flags[flag_name] = env_value.lower() in ("true", "1", "yes", "on")
        
        # Override with config file if available
        feature_config_file = os.getenv("AF_FEATURES_CONFIG", "features.json")
        if Path(feature_config_file).exists():
            try:
                with open(feature_config_file, 'r') as f:
                    file_flags = json.load(f)
                default_flags.update(file_flags)
                logger.info(f"Loaded feature flags from {feature_config_file}")
            except Exception as e:
                logger.warning(f"Failed to load feature flags from {feature_config_file}: {e}")
        
        self.feature_flags = default_flags
        logger.info(f"Feature flags loaded: {sum(self.feature_flags.values())}/{len(self.feature_flags)} enabled")
    
    def _load_runtime_settings(self) -> None:
        """Load runtime settings"""
        self.runtime_settings = {
            "startup_time": datetime.now(),
            "pid": os.getpid(),
            "python_version": os.sys.version,
            "platform": os.sys.platform,
            "working_directory": os.getcwd(),
            "environment_variables": {k: v for k, v in os.environ.items() if k.startswith("AF_")},
            "config_version": "2.0.0"
        }
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return self.feature_flags.get(feature_name, False)
    
    def enable_feature(self, feature_name: str) -> None:
        """Enable a feature flag"""
        self.feature_flags[feature_name] = True
        logger.info(f"Feature enabled: {feature_name}")
    
    def disable_feature(self, feature_name: str) -> None:
        """Disable a feature flag"""
        self.feature_flags[feature_name] = False
        logger.info(f"Feature disabled: {feature_name}")
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime information"""
        return {
            "environment": {
                "name": self.environment.name,
                "is_production": self.environment.is_production,
                "is_development": self.environment.is_development,
                "is_testing": self.environment.is_testing,
                "config_sources": self.environment.config_sources
            },
            "feature_flags": self.feature_flags,
            "runtime_settings": self.runtime_settings,
            "config_summary": {
                "service_name": self.config.service_name,
                "version": self.config.version,
                "host": self.config.host,
                "port": self.config.port,
                "log_level": self.config.log_level
            }
        }
    
    def get_enabled_features(self) -> List[str]:
        """Get list of enabled features"""
        return [name for name, enabled in self.feature_flags.items() if enabled]
    
    def get_disabled_features(self) -> List[str]:
        """Get list of disabled features"""
        return [name for name, enabled in self.feature_flags.items() if not enabled]
    
    def update_feature_flags(self, flags: Dict[str, bool]) -> None:
        """Update multiple feature flags"""
        for name, enabled in flags.items():
            if enabled:
                self.enable_feature(name)
            else:
                self.disable_feature(name)
    
    def save_feature_flags(self, file_path: str = "features.json") -> None:
        """Save current feature flags to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.feature_flags, f, indent=2)
            logger.info(f"Feature flags saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save feature flags: {e}")
    
    def get_security_settings(self) -> Dict[str, Any]:
        """Get security-related settings"""
        return {
            "api_key_required": self.config.api_key_required,
            "cors_origins": self.config.cors_origins,
            "rate_limit_enabled": self.config.rate_limit_enabled,
            "rate_limit_requests": self.config.rate_limit_requests,
            "rate_limit_window": self.config.rate_limit_window,
            "environment_is_secure": self.environment.is_production
        }
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance-related settings"""
        return {
            "max_agents": self.config.max_agents,
            "agent_timeout_seconds": self.config.agent_timeout_seconds,
            "task_queue_size": self.config.task_queue_size,
            "database_pool_size": self.config.database_pool_size,
            "workers": self.config.workers,
            "metrics_enabled": self.config.metrics_enabled
        }

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def is_feature_enabled(feature_name: str) -> bool:
    """Quick check if feature is enabled"""
    return get_settings().is_feature_enabled(feature_name)

def get_environment_name() -> str:
    """Get current environment name"""
    return get_settings().environment.name

def is_production() -> bool:
    """Check if running in production"""
    return get_settings().environment.is_production

def is_development() -> bool:
    """Check if running in development"""
    return get_settings().environment.is_development

def is_testing() -> bool:
    """Check if running in testing"""
    return get_settings().environment.is_testing
