"""
AF Common Settings - Bridge to existing configuration
"""

import os
from typing import Any, Dict

def get_settings() -> Dict[str, Any]:
    """Get system settings"""
    return {
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "max_agents": int(os.getenv("MAX_AGENTS", "1000")),
        "api_timeout": int(os.getenv("API_TIMEOUT", "30"))
    }

def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled"""
    return os.getenv(f"FEATURE_{feature.upper()}", "false").lower() == "true"

