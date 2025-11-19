"""
AF Common Errors - Error handling utilities
"""

from typing import Dict, Any, Optional

class AgentForgeError(Exception):
    """Base AgentForge error"""
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code

class TaskExecutionError(AgentForgeError):
    """Task execution error"""
    def __init__(self, task_id: str, message: str):
        super().__init__(f"Task {task_id} failed: {message}")
        self.task_id = task_id

def create_error_context(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """Create error context for logging"""
    return {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context
    }

