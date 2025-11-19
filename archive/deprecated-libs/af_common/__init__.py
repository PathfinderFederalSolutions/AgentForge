"""
AF Common - Common utilities and types
Bridge to existing AgentForge components
"""

from .types import Task, AgentContract, TaskResult, AgentStatus, SystemMetric
from .logging import setup_logging, get_logger, log_performance, log_agent_event
from .settings import get_settings, is_feature_enabled
from .errors import AgentForgeError, TaskExecutionError, create_error_context
from .tracing import get_tracer, trace_operation, trace_agent_operation

__all__ = [
    'Task', 'AgentContract', 'TaskResult', 'AgentStatus', 'SystemMetric',
    'setup_logging', 'get_logger', 'log_performance', 'log_agent_event',
    'get_settings', 'is_feature_enabled',
    'AgentForgeError', 'TaskExecutionError', 'create_error_context',
    'get_tracer', 'trace_operation', 'trace_agent_operation'
]

