"""
Event schema definitions for AgentForge
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class EventType(Enum):
    """Types of events in AgentForge"""
    AGENT_CREATED = "agent_created"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_ERROR = "agent_error"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    SWARM_DEPLOYED = "swarm_deployed"
    SWARM_SCALED = "swarm_scaled"
    NEURAL_MESH_SYNC = "neural_mesh_sync"
    QUANTUM_SCHEDULE = "quantum_schedule"
    FUSION_COMPLETE = "fusion_complete"
    SECURITY_ALERT = "security_alert"
    SYSTEM_HEALTH = "system_health"
    CONFIG_CHANGED = "config_changed"
    USER_ACTION = "user_action"

class EventSeverity(Enum):
    """Event severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class BaseEvent(BaseModel):
    """Base event schema"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    source_service: str
    source_component: str
    severity: EventSeverity = EventSeverity.INFO
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AgentEvent(BaseEvent):
    """Agent-specific events"""
    agent_id: str
    agent_name: str
    agent_type: str
    agent_status: Optional[str] = None
    performance_data: Dict[str, Any] = Field(default_factory=dict)
    error_details: Optional[Dict[str, Any]] = None

class TaskEvent(BaseEvent):
    """Task-specific events"""
    task_id: str
    task_description: str
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    execution_time_ms: Optional[float] = None
    result_data: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None

class SwarmEvent(BaseEvent):
    """Swarm coordination events"""
    swarm_id: str
    swarm_name: str
    agents_count: int
    coordination_strategy: str
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    scaling_action: Optional[str] = None  # scale_up, scale_down, rebalance

class SystemEvent(BaseEvent):
    """System-level events"""
    component_name: str
    system_metrics: Dict[str, Any] = Field(default_factory=dict)
    health_status: str = "healthy"
    alert_level: Optional[str] = None
    action_required: bool = False

class UserEvent(BaseEvent):
    """User interaction events"""
    user_action: str
    endpoint: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None

class SecurityEvent(BaseEvent):
    """Security-related events"""
    security_level: str = "medium"  # low, medium, high, critical
    threat_type: Optional[str] = None
    source_ip: Optional[str] = None
    affected_resources: List[str] = Field(default_factory=list)
    mitigation_actions: List[str] = Field(default_factory=list)
    compliance_impact: Optional[str] = None

class FusionEvent(BaseEvent):
    """Data fusion events"""
    fusion_id: str
    fusion_method: str
    input_modalities: List[str] = Field(default_factory=list)
    output_format: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_time_ms: float
    data_sources_count: int

class NeuralMeshEvent(BaseEvent):
    """Neural mesh memory events"""
    memory_tier: str  # L1, L2, L3, L4
    operation: str  # store, retrieve, sync, prune
    memory_key: Optional[str] = None
    data_size_bytes: Optional[int] = None
    embedding_dimensions: Optional[int] = None
    similarity_score: Optional[float] = None

class QuantumEvent(BaseEvent):
    """Quantum scheduler events"""
    quantum_task_id: str
    agent_count: int
    coherence_level: str
    superposition_states: int
    entanglement_pairs: int
    execution_strategy: str
    quantum_metrics: Dict[str, Any] = Field(default_factory=dict)

# Event factory functions
def create_agent_event(
    event_type: EventType,
    agent_id: str,
    agent_name: str,
    agent_type: str,
    **kwargs
) -> AgentEvent:
    """Create agent event"""
    return AgentEvent(
        event_type=event_type,
        agent_id=agent_id,
        agent_name=agent_name,
        agent_type=agent_type,
        source_service="agent_system",
        source_component="agent_manager",
        **kwargs
    )

def create_task_event(
    event_type: EventType,
    task_id: str,
    task_description: str,
    **kwargs
) -> TaskEvent:
    """Create task event"""
    return TaskEvent(
        event_type=event_type,
        task_id=task_id,
        task_description=task_description,
        source_service="task_system",
        source_component="task_manager",
        **kwargs
    )

def create_swarm_event(
    event_type: EventType,
    swarm_id: str,
    swarm_name: str,
    agents_count: int,
    **kwargs
) -> SwarmEvent:
    """Create swarm event"""
    return SwarmEvent(
        event_type=event_type,
        swarm_id=swarm_id,
        swarm_name=swarm_name,
        agents_count=agents_count,
        source_service="swarm_system",
        source_component="swarm_coordinator",
        **kwargs
    )

def create_system_event(
    event_type: EventType,
    component_name: str,
    **kwargs
) -> SystemEvent:
    """Create system event"""
    return SystemEvent(
        event_type=event_type,
        component_name=component_name,
        source_service="system",
        source_component=component_name,
        **kwargs
    )

def create_security_event(
    event_type: EventType,
    security_level: str = "medium",
    **kwargs
) -> SecurityEvent:
    """Create security event"""
    return SecurityEvent(
        event_type=event_type,
        security_level=security_level,
        source_service="security_system",
        source_component="security_orchestrator",
        severity=EventSeverity.WARNING if security_level in ["high", "critical"] else EventSeverity.INFO,
        **kwargs
    )

def create_fusion_event(
    fusion_id: str,
    fusion_method: str,
    confidence_score: float,
    processing_time_ms: float,
    **kwargs
) -> FusionEvent:
    """Create fusion event"""
    return FusionEvent(
        event_type=EventType.FUSION_COMPLETE,
        fusion_id=fusion_id,
        fusion_method=fusion_method,
        confidence_score=confidence_score,
        processing_time_ms=processing_time_ms,
        source_service="fusion_system",
        source_component="data_fusion_engine",
        **kwargs
    )
