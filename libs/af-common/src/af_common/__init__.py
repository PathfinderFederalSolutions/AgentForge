"""
AgentForge Common Library

Shared utilities, types, and functionality used across all AgentForge services.
"""

from .types import *
from .config import *
from .logging import *
from .metrics import *
from .utils import *

__version__ = "0.1.0"
__all__ = [
    # Types
    "Task", "AgentContract", "AgentSpec", "Provider",
    
    # Config
    "BaseConfig", "get_config",
    
    # Logging  
    "setup_logging", "get_logger",
    
    # Metrics
    "setup_metrics", "record_metric",
    
    # Utils
    "generate_id", "safe_json_loads", "ensure_dir"
]
