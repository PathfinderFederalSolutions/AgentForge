"""
AF Common Logging - Bridge to existing logging
"""

import logging
from typing import Dict, Any, Optional

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_logger(name: str):
    """Get logger instance"""
    return logging.getLogger(name)

def log_performance(operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
    """Log performance metrics"""
    logger = logging.getLogger("performance")
    logger.info(f"Operation {operation} took {duration:.3f}s", extra=metadata or {})

def log_agent_event(agent_id: str, event: str, details: Optional[Dict[str, Any]] = None):
    """Log agent events"""
    logger = logging.getLogger("agent-events")
    logger.info(f"Agent {agent_id}: {event}", extra=details or {})

