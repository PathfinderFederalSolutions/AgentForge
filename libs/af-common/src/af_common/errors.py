"""
Enhanced error handling and exception management for AgentForge
"""
from __future__ import annotations

import traceback
import uuid
from typing import Any, Dict, List, Optional, Type
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .logging import get_logger

logger = get_logger("errors")

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    SYSTEM = "system"
    AGENT = "agent"
    TASK = "task"
    PROVIDER = "provider"
    MEMORY = "memory"
    NETWORK = "network"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    USER_INPUT = "user_input"

@dataclass
class ErrorContext:
    """Rich error context for debugging"""
    error_id: str
    timestamp: datetime
    service_name: str
    component: str
    operation: str
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    agent_name: Optional[str] = None
    correlation_id: Optional[str] = None
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}

class AgentForgeError(Exception):
    """Base exception for all AgentForge errors"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
        retry_after_seconds: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.severity = severity
        self.category = category
        self.context = context
        self.cause = cause
        self.recoverable = recoverable
        self.retry_after_seconds = retry_after_seconds
        self.created_at = datetime.now()
        
        # Log the error
        self._log_error()
    
    def _generate_error_code(self) -> str:
        """Generate unique error code"""
        return f"AF_{self.category.value.upper()}_{uuid.uuid4().hex[:8].upper()}"
    
    def _log_error(self) -> None:
        """Log the error with context"""
        log_data = {
            "error_code": self.error_code,
            "severity": self.severity.value,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "message": self.message
        }
        
        if self.context:
            log_data.update({
                "error_id": self.context.error_id,
                "service_name": self.context.service_name,
                "component": self.context.component,
                "operation": self.context.operation,
                "user_id": self.context.user_id,
                "task_id": self.context.task_id,
                "agent_name": self.context.agent_name
            })
        
        if self.cause:
            log_data["cause"] = str(self.cause)
        
        logger.error(self.message, extra=log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "retry_after_seconds": self.retry_after_seconds,
            "created_at": self.created_at.isoformat(),
            "context": {
                "error_id": self.context.error_id,
                "service_name": self.context.service_name,
                "component": self.context.component,
                "operation": self.context.operation
            } if self.context else None
        }

# Specific error types
class TaskExecutionError(AgentForgeError):
    """Task execution failed"""
    def __init__(self, message: str, task_id: str, agent_name: Optional[str] = None, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.TASK,
            **kwargs
        )
        self.task_id = task_id
        self.agent_name = agent_name

class AgentNotFoundError(AgentForgeError):
    """Agent not found or unavailable"""
    def __init__(self, message: str, agent_name: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AGENT,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.agent_name = agent_name

class ProviderError(AgentForgeError):
    """LLM Provider error"""
    def __init__(self, message: str, provider: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PROVIDER,
            **kwargs
        )
        self.provider = provider

class MemoryError(AgentForgeError):
    """Memory system error"""
    def __init__(self, message: str, memory_tier: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MEMORY,
            **kwargs
        )
        self.memory_tier = memory_tier

class SecurityError(AgentForgeError):
    """Security-related error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )

class ConfigurationError(AgentForgeError):
    """Configuration error"""
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.config_key = config_key

class NetworkError(AgentForgeError):
    """Network communication error"""
    def __init__(self, message: str, endpoint: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            recoverable=True,
            retry_after_seconds=5,
            **kwargs
        )
        self.endpoint = endpoint

# Error handling utilities
def create_error_context(
    service_name: str,
    component: str,
    operation: str,
    **additional_context: Any
) -> ErrorContext:
    """Create error context for exception handling"""
    return ErrorContext(
        error_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        service_name=service_name,
        component=component,
        operation=operation,
        additional_data=additional_context
    )

def handle_exception(
    exc: Exception,
    context: ErrorContext,
    reraise: bool = True
) -> AgentForgeError:
    """Handle and wrap exceptions with AgentForge context"""
    
    # If it's already an AgentForge error, just update context
    if isinstance(exc, AgentForgeError):
        if not exc.context:
            exc.context = context
        return exc
    
    # Wrap other exceptions
    wrapped_error = AgentForgeError(
        message=str(exc),
        category=ErrorCategory.SYSTEM,
        context=context,
        cause=exc
    )
    
    if reraise:
        raise wrapped_error
    
    return wrapped_error

def safe_execute(
    func: callable,
    context: ErrorContext,
    default_return: Any = None,
    log_errors: bool = True
) -> Any:
    """Safely execute function with error handling"""
    try:
        return func()
    except Exception as e:
        if log_errors:
            handle_exception(e, context, reraise=False)
        return default_return

# Decorator for automatic error handling
def with_error_handling(
    component: str,
    operation: Optional[str] = None,
    service_name: str = "agentforge"
):
    """Decorator for automatic error handling"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            context = create_error_context(
                service_name=service_name,
                component=component,
                operation=operation or func.__name__,
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handle_exception(e, context)
        
        return wrapper
    return decorator
