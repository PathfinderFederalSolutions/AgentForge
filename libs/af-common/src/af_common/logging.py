"""
Standardized logging configuration for AgentForge services
"""
import logging
import logging.handlers
import sys
import json
import traceback
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'service_name'):
            log_entry["service"] = record.service_name
            
        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = record.correlation_id
            
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
            
        if hasattr(record, 'task_id'):
            log_entry["task_id"] = record.task_id
            
        if hasattr(record, 'agent_name'):
            log_entry["agent_name"] = record.agent_name
            
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
            
        return json.dumps(log_entry)

class AgentForgeAdapter(logging.LoggerAdapter):
    """Logger adapter with AgentForge-specific context"""
    
    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
        """Add extra context to log records"""
        extra = kwargs.get('extra', {})
        
        # Add context from adapter
        for key, value in self.extra.items():
            extra[key] = value
            
        kwargs['extra'] = extra
        return msg, kwargs

def setup_logging(
    service_name: str,
    log_level: str = "INFO",
    log_format: str = "standard",  # standard, json
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True
) -> None:
    """
    Setup standardized logging for AgentForge services
    
    Args:
        service_name: Name of the service
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (standard, json)
        log_file: Path to log file (optional)
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup formatters
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handlers = []
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Log startup message
    logger = get_logger(service_name)
    logger.info(f"Logging initialized for {service_name} at level {log_level}")

def get_logger(
    name: str,
    service_name: Optional[str] = None,
    correlation_id: Optional[str] = None,
    **extra_context: Any
) -> AgentForgeAdapter:
    """
    Get a logger with AgentForge context
    
    Args:
        name: Logger name
        service_name: Service name for context
        correlation_id: Correlation ID for request tracing
        **extra_context: Additional context fields
        
    Returns:
        Logger adapter with context
    """
    logger = logging.getLogger(name)
    
    context = {}
    if service_name:
        context['service_name'] = service_name
    if correlation_id:
        context['correlation_id'] = correlation_id
        
    context.update(extra_context)
    
    return AgentForgeAdapter(logger, context)

def log_function_call(func_name: str, args: tuple = (), kwargs: Dict[str, Any] = None) -> None:
    """Log function call with arguments"""
    logger = get_logger("function_calls")
    kwargs = kwargs or {}
    
    logger.debug(
        f"Calling {func_name}",
        extra={
            "function": func_name,
            "args": str(args)[:200],  # Truncate long args
            "kwargs": str(kwargs)[:200]  # Truncate long kwargs
        }
    )

def log_performance(operation: str, duration_ms: float, **context: Any) -> None:
    """Log performance metrics"""
    logger = get_logger("performance")
    
    logger.info(
        f"{operation} completed in {duration_ms:.2f}ms",
        extra={
            "operation": operation,
            "duration_ms": duration_ms,
            **context
        }
    )

def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log error with context"""
    logger = get_logger("errors")
    context = context or {}
    
    logger.error(
        f"Error occurred: {str(error)}",
        exc_info=True,
        extra={
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        }
    )

def log_agent_event(
    agent_name: str,
    event_type: str,
    message: str,
    task_id: Optional[str] = None,
    **context: Any
) -> None:
    """Log agent-specific events"""
    logger = get_logger("agents")
    
    logger.info(
        message,
        extra={
            "agent_name": agent_name,
            "event_type": event_type,
            "task_id": task_id,
            **context
        }
    )

def log_task_event(
    task_id: str,
    event_type: str,
    message: str,
    agent_name: Optional[str] = None,
    **context: Any
) -> None:
    """Log task-specific events"""
    logger = get_logger("tasks")
    
    logger.info(
        message,
        extra={
            "task_id": task_id,
            "event_type": event_type,
            "agent_name": agent_name,
            **context
        }
    )

def log_system_event(
    event_type: str,
    message: str,
    component: Optional[str] = None,
    **context: Any
) -> None:
    """Log system-level events"""
    logger = get_logger("system")
    
    logger.info(
        message,
        extra={
            "event_type": event_type,
            "component": component,
            **context
        }
    )

# Context manager for correlation tracking
class CorrelationContext:
    """Context manager for correlation ID tracking"""
    
    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self._previous_id = None
        
    def __enter__(self):
        # Store previous correlation ID if any
        import threading
        local = getattr(threading.current_thread(), 'af_correlation_id', None)
        self._previous_id = local
        
        # Set new correlation ID
        threading.current_thread().af_correlation_id = self.correlation_id
        return self.correlation_id
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous correlation ID
        import threading
        if self._previous_id is not None:
            threading.current_thread().af_correlation_id = self._previous_id
        else:
            delattr(threading.current_thread(), 'af_correlation_id')

def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from thread context"""
    import threading
    return getattr(threading.current_thread(), 'af_correlation_id', None)
