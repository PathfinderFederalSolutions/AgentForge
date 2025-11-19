"""
AF Common Tracing - Tracing utilities
"""

import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger("tracing")

def get_tracer(name: str):
    """Get tracer instance"""
    return logger

@contextmanager
def trace_operation(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Trace an operation"""
    start_time = time.time()
    logger.info(f"Starting operation: {operation}", extra=metadata or {})
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Completed operation: {operation} in {duration:.3f}s")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed operation: {operation} after {duration:.3f}s: {e}")
        raise

def trace_agent_operation(agent_id: str, operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Trace an agent operation"""
    return trace_operation(f"Agent-{agent_id}-{operation}", metadata)

