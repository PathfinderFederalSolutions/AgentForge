"""
Shared type definitions for AgentForge
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime
from pydantic import BaseModel, Field
from dataclasses import dataclass

# Core task and agent types
class Task(BaseModel):
    """Unified task definition"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    capabilities_required: List[str] = Field(default_factory=list)
    priority: int = Field(default=5, ge=0, le=10)  # 0 = highest priority
    created_at: datetime = Field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    preferred_provider: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AgentContract(BaseModel):
    """Agent contract/specification"""
    name: str
    capabilities: List[str] = Field(default_factory=list)
    memory_scopes: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    budget: int = Field(default=1000)
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

class AgentSpec(BaseModel):
    """Agent specification for factory creation"""
    name: str
    capabilities: List[str] = Field(default_factory=list)
    llm: str = "gpt-4"
    tools: List[str] = Field(default_factory=list)
    policy: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class Provider(BaseModel):
    """LLM Provider specification"""
    key: str
    model: str
    capabilities: Set[str] = Field(default_factory=set)
    cost_per_token: float = 0.0
    rate_limit_rpm: int = 1000
    available: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

@dataclass
class MemoryScope:
    """Memory scope definition"""
    name: str
    ttl_seconds: Optional[int] = None
    max_entries: Optional[int] = None
    embedding_enabled: bool = True

# Result and response types
class TaskResult(BaseModel):
    """Task execution result"""
    task_id: str
    agent_name: str
    result: Any
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    provider_used: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentStatus(BaseModel):
    """Agent status information"""
    name: str
    status: str = "idle"  # idle, busy, error, offline
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    uptime_seconds: float = 0.0
    last_heartbeat: datetime = Field(default_factory=datetime.now)
    capabilities: List[str] = Field(default_factory=list)
    memory_usage_mb: float = 0.0

# Configuration types
class ServiceConfig(BaseModel):
    """Base service configuration"""
    name: str
    version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    metrics_enabled: bool = True
    health_check_interval: int = 30

# Event and messaging types  
class AgentEvent(BaseModel):
    """Agent lifecycle event"""
    event_type: str  # created, started, stopped, task_assigned, task_completed, error
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None

class SystemMetric(BaseModel):
    """System metric data point"""
    name: str
    value: float
    unit: str = "count"
    labels: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

# Error types
class AgentForgeError(Exception):
    """Base exception for AgentForge"""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class TaskExecutionError(AgentForgeError):
    """Task execution failed"""
    pass

class AgentNotFoundError(AgentForgeError):
    """Agent not found"""
    pass

class ProviderError(AgentForgeError):
    """LLM Provider error"""
    pass

class MemoryError(AgentForgeError):
    """Memory system error"""
    pass

# Utility functions for types
def generate_task_id() -> str:
    """Generate unique task ID"""
    return f"task_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

def generate_agent_id(name: str) -> str:
    """Generate unique agent ID"""
    return f"{name}_{uuid.uuid4().hex[:8]}"

def validate_capability(capability: str) -> bool:
    """Validate capability string format"""
    return isinstance(capability, str) and len(capability) > 0 and capability.replace("_", "").replace("-", "").isalnum()

# Type aliases for backward compatibility
TaskId = str
AgentId = str
Capability = str
Scope = str
