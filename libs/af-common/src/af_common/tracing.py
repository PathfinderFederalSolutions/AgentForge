"""
Distributed tracing and observability for AgentForge services
"""
from __future__ import annotations

import time
import uuid
import asyncio
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
from abc import ABC, abstractmethod

from .logging import get_logger

logger = get_logger("tracing")

@dataclass
class Span:
    """Distributed tracing span"""
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    service_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "ok"  # ok, error, timeout
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def finish(self, status: str = "ok") -> None:
        """Finish the span"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        
        logger.debug(f"Span finished: {self.operation_name} ({self.duration_ms:.2f}ms)")
    
    def add_tag(self, key: str, value: Any) -> None:
        """Add tag to span"""
        self.tags[key] = value
    
    def add_log(self, message: str, **fields: Any) -> None:
        """Add log entry to span"""
        log_entry = {
            "timestamp": time.time(),
            "message": message,
            **fields
        }
        self.logs.append(log_entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "tags": self.tags,
            "logs": self.logs
        }

class TraceContext:
    """Thread-local trace context"""
    
    def __init__(self):
        self._local = threading.local()
    
    def get_current_span(self) -> Optional[Span]:
        """Get current active span"""
        return getattr(self._local, 'current_span', None)
    
    def set_current_span(self, span: Optional[Span]) -> None:
        """Set current active span"""
        self._local.current_span = span
    
    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID"""
        span = self.get_current_span()
        return span.trace_id if span else None
    
    def get_span_id(self) -> Optional[str]:
        """Get current span ID"""
        span = self.get_current_span()
        return span.span_id if span else None

# Global trace context
_trace_context = TraceContext()

class Tracer:
    """Distributed tracer for AgentForge operations"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.spans: Dict[str, Span] = {}
        self.finished_spans: List[Span] = []
        self.max_finished_spans = 1000
    
    def start_span(
        self,
        operation_name: str,
        parent_span: Optional[Span] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span"""
        
        # Get parent from context if not provided
        if parent_span is None:
            parent_span = _trace_context.get_current_span()
        
        # Create new span
        span = Span(
            operation_name=operation_name,
            service_name=self.service_name,
            parent_span_id=parent_span.span_id if parent_span else None,
            trace_id=parent_span.trace_id if parent_span else uuid.uuid4().hex,
            tags=tags or {}
        )
        
        # Store span
        self.spans[span.span_id] = span
        
        # Set as current span
        _trace_context.set_current_span(span)
        
        logger.debug(f"Started span: {operation_name} (span_id: {span.span_id})")
        
        return span
    
    def finish_span(self, span: Span, status: str = "ok") -> None:
        """Finish a span"""
        span.finish(status)
        
        # Move to finished spans
        if span.span_id in self.spans:
            del self.spans[span.span_id]
        
        self.finished_spans.append(span)
        
        # Limit finished spans to prevent memory growth
        if len(self.finished_spans) > self.max_finished_spans:
            self.finished_spans.pop(0)
        
        # Clear current span if this was it
        if _trace_context.get_current_span() == span:
            # Set parent as current if available
            parent_span = self.get_span_by_id(span.parent_span_id) if span.parent_span_id else None
            _trace_context.set_current_span(parent_span)
    
    def get_span_by_id(self, span_id: str) -> Optional[Span]:
        """Get span by ID"""
        return self.spans.get(span_id)
    
    def get_active_spans(self) -> List[Span]:
        """Get all active spans"""
        return list(self.spans.values())
    
    def get_finished_spans(self, limit: Optional[int] = None) -> List[Span]:
        """Get finished spans"""
        if limit:
            return self.finished_spans[-limit:]
        return self.finished_spans.copy()
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary of a complete trace"""
        trace_spans = [s for s in self.finished_spans if s.trace_id == trace_id]
        
        if not trace_spans:
            return {"error": "Trace not found"}
        
        # Calculate trace statistics
        total_duration = sum(s.duration_ms for s in trace_spans if s.duration_ms)
        error_count = sum(1 for s in trace_spans if s.status == "error")
        
        return {
            "trace_id": trace_id,
            "span_count": len(trace_spans),
            "total_duration_ms": total_duration,
            "error_count": error_count,
            "success_rate": (len(trace_spans) - error_count) / len(trace_spans),
            "spans": [s.to_dict() for s in trace_spans]
        }

# Global tracer instance
_tracer: Optional[Tracer] = None

def get_tracer(service_name: str = "agentforge") -> Tracer:
    """Get global tracer instance"""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(service_name)
    return _tracer

def get_current_span() -> Optional[Span]:
    """Get current active span"""
    return _trace_context.get_current_span()

def get_current_trace_id() -> Optional[str]:
    """Get current trace ID"""
    return _trace_context.get_trace_id()

# Context managers for tracing
@contextmanager
def trace_operation(
    operation_name: str,
    service_name: str = "agentforge",
    tags: Optional[Dict[str, Any]] = None
):
    """Context manager for tracing operations"""
    tracer = get_tracer(service_name)
    span = tracer.start_span(operation_name, tags=tags)
    
    try:
        yield span
        tracer.finish_span(span, "ok")
    except Exception as e:
        span.add_log(f"Error: {str(e)}")
        span.add_tag("error", True)
        span.add_tag("error_type", type(e).__name__)
        tracer.finish_span(span, "error")
        raise

@asynccontextmanager
async def trace_async_operation(
    operation_name: str,
    service_name: str = "agentforge", 
    tags: Optional[Dict[str, Any]] = None
):
    """Async context manager for tracing operations"""
    tracer = get_tracer(service_name)
    span = tracer.start_span(operation_name, tags=tags)
    
    try:
        yield span
        tracer.finish_span(span, "ok")
    except Exception as e:
        span.add_log(f"Error: {str(e)}")
        span.add_tag("error", True)
        span.add_tag("error_type", type(e).__name__)
        tracer.finish_span(span, "error")
        raise

# Decorators for automatic tracing
def trace_function(
    operation_name: Optional[str] = None,
    service_name: str = "agentforge",
    tags: Optional[Dict[str, Any]] = None
):
    """Decorator for automatic function tracing"""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with trace_async_operation(op_name, service_name, tags) as span:
                    span.add_tag("function", func.__name__)
                    span.add_tag("module", func.__module__)
                    result = await func(*args, **kwargs)
                    return result
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with trace_operation(op_name, service_name, tags) as span:
                    span.add_tag("function", func.__name__)
                    span.add_tag("module", func.__module__)
                    result = func(*args, **kwargs)
                    return result
            return sync_wrapper
    
    return decorator

def trace_agent_operation(agent_name: str, operation: str):
    """Decorator for tracing agent operations"""
    return trace_function(
        operation_name=f"agent.{operation}",
        tags={"agent_name": agent_name, "operation_type": operation}
    )

def trace_llm_call(provider: str, model: str):
    """Decorator for tracing LLM calls"""
    return trace_function(
        operation_name=f"llm.{provider}",
        tags={"provider": provider, "model": model, "operation_type": "llm_call"}
    )

def trace_swarm_coordination(swarm_size: int, coordination_type: str):
    """Decorator for tracing swarm coordination"""
    return trace_function(
        operation_name=f"swarm.{coordination_type}",
        tags={"swarm_size": swarm_size, "coordination_type": coordination_type}
    )

# Utility functions
def create_child_span(
    operation_name: str,
    parent_span: Optional[Span] = None,
    tags: Optional[Dict[str, Any]] = None
) -> Span:
    """Create a child span"""
    tracer = get_tracer()
    return tracer.start_span(operation_name, parent_span, tags)

def get_trace_summary(trace_id: str) -> Dict[str, Any]:
    """Get trace summary by ID"""
    tracer = get_tracer()
    return tracer.get_trace_summary(trace_id)

def get_active_traces() -> List[Dict[str, Any]]:
    """Get all active traces"""
    tracer = get_tracer()
    active_spans = tracer.get_active_spans()
    
    # Group by trace_id
    traces = {}
    for span in active_spans:
        if span.trace_id not in traces:
            traces[span.trace_id] = []
        traces[span.trace_id].append(span.to_dict())
    
    return [{"trace_id": tid, "spans": spans} for tid, spans in traces.items()]
