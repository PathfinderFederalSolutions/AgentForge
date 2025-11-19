"""
Agent schema definitions for AgentForge
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class AgentType(Enum):
    """Types of agents in the AgentForge system"""
    NEURAL_MESH = "neural_mesh"
    QUANTUM_SCHEDULER = "quantum_scheduler"
    UNIVERSAL_IO = "universal_io"
    DATA_PROCESSOR = "data_processor"
    CODE_ANALYZER = "code_analyzer"
    SECURITY_MONITOR = "security_monitor"
    FUSION_PROCESSOR = "fusion_processor"
    LIFECYCLE_MANAGER = "lifecycle_manager"
    SELF_IMPROVEMENT = "self_improvement"
    GENERAL_INTELLIGENCE = "general_intelligence"

class AgentStatus(Enum):
    """Agent status states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATING = "terminating"
    TERMINATED = "terminated"

class CapabilityLevel(Enum):
    """Agent capability proficiency levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    SME = "sme"  # Subject Matter Expert

class AgentCapability(BaseModel):
    """Individual agent capability definition"""
    name: str
    level: CapabilityLevel = CapabilityLevel.INTERMEDIATE
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    last_used: Optional[datetime] = None
    success_rate: float = Field(ge=0.0, le=1.0, default=0.9)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentResource(BaseModel):
    """Agent resource allocation and usage"""
    cpu_cores: float = Field(ge=0.0, default=1.0)
    memory_mb: int = Field(ge=0, default=512)
    gpu_memory_mb: int = Field(ge=0, default=0)
    storage_mb: int = Field(ge=0, default=1024)
    network_bandwidth_mbps: float = Field(ge=0.0, default=100.0)
    
    # Usage tracking
    cpu_usage_percent: float = Field(ge=0.0, le=100.0, default=0.0)
    memory_usage_mb: int = Field(ge=0, default=0)
    gpu_usage_percent: float = Field(ge=0.0, le=100.0, default=0.0)

class AgentConfiguration(BaseModel):
    """Agent configuration and parameters"""
    llm_provider: Optional[str] = "openai"
    llm_model: Optional[str] = "gpt-4o"
    max_tokens: int = Field(ge=1, default=4096)
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    timeout_seconds: int = Field(ge=1, default=300)
    retry_attempts: int = Field(ge=0, default=3)
    memory_scope: str = "agent_local"
    tools_enabled: List[str] = Field(default_factory=list)
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)

class AgentMetrics(BaseModel):
    """Agent performance metrics"""
    tasks_completed: int = Field(ge=0, default=0)
    tasks_failed: int = Field(ge=0, default=0)
    average_task_time_ms: float = Field(ge=0.0, default=0.0)
    success_rate: float = Field(ge=0.0, le=1.0, default=1.0)
    uptime_seconds: float = Field(ge=0.0, default=0.0)
    last_activity: Optional[datetime] = None
    tokens_processed: int = Field(ge=0, default=0)
    errors_encountered: int = Field(ge=0, default=0)

class AgentSchema(BaseModel):
    """Complete agent schema definition"""
    
    # Identity
    agent_id: str = Field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
    name: str
    agent_type: AgentType
    version: str = "2.0.0"
    
    # Status and lifecycle
    status: AgentStatus = AgentStatus.INITIALIZING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    
    # Capabilities and configuration
    capabilities: List[AgentCapability] = Field(default_factory=list)
    configuration: AgentConfiguration = Field(default_factory=AgentConfiguration)
    resources: AgentResource = Field(default_factory=AgentResource)
    
    # Performance and metrics
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)
    
    # Deployment and location
    deployment_location: Optional[str] = None
    cluster_id: Optional[str] = None
    node_id: Optional[str] = None
    
    # Relationships
    parent_agent_id: Optional[str] = None
    child_agent_ids: List[str] = Field(default_factory=list)
    coordination_group: Optional[str] = None
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('name')
    def validate_name(cls, v):
        """Validate agent name"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Agent name cannot be empty")
        if len(v) > 100:
            raise ValueError("Agent name too long (max 100 characters)")
        return v.strip()
    
    @validator('capabilities')
    def validate_capabilities(cls, v):
        """Validate capabilities list"""
        if not v:
            raise ValueError("Agent must have at least one capability")
        return v
    
    def add_capability(self, name: str, level: CapabilityLevel = CapabilityLevel.INTERMEDIATE) -> None:
        """Add capability to agent"""
        capability = AgentCapability(name=name, level=level)
        self.capabilities.append(capability)
    
    def remove_capability(self, name: str) -> bool:
        """Remove capability from agent"""
        for i, cap in enumerate(self.capabilities):
            if cap.name == name:
                del self.capabilities[i]
                return True
        return False
    
    def has_capability(self, name: str, min_level: CapabilityLevel = CapabilityLevel.BASIC) -> bool:
        """Check if agent has capability at minimum level"""
        for cap in self.capabilities:
            if cap.name == name:
                level_order = list(CapabilityLevel)
                return level_order.index(cap.level) >= level_order.index(min_level)
        return False
    
    def update_status(self, new_status: AgentStatus) -> None:
        """Update agent status"""
        old_status = self.status
        self.status = new_status
        
        if new_status == AgentStatus.ACTIVE and old_status != AgentStatus.ACTIVE:
            self.started_at = datetime.now()
        
        self.last_heartbeat = datetime.now()
    
    def record_task_completion(self, success: bool, duration_ms: float) -> None:
        """Record task completion metrics"""
        if success:
            self.metrics.tasks_completed += 1
        else:
            self.metrics.tasks_failed += 1
        
        # Update average task time
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        if total_tasks > 1:
            current_avg = self.metrics.average_task_time_ms
            self.metrics.average_task_time_ms = (
                (current_avg * (total_tasks - 1) + duration_ms) / total_tasks
            )
        else:
            self.metrics.average_task_time_ms = duration_ms
        
        # Update success rate
        self.metrics.success_rate = self.metrics.tasks_completed / total_tasks
        self.metrics.last_activity = datetime.now()
    
    def get_capability_summary(self) -> Dict[str, Any]:
        """Get summary of agent capabilities"""
        return {
            "total_capabilities": len(self.capabilities),
            "expert_capabilities": len([c for c in self.capabilities if c.level == CapabilityLevel.EXPERT]),
            "sme_capabilities": len([c for c in self.capabilities if c.level == CapabilityLevel.SME]),
            "average_confidence": sum(c.confidence for c in self.capabilities) / len(self.capabilities) if self.capabilities else 0,
            "capability_names": [c.name for c in self.capabilities]
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get agent performance summary"""
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        uptime_hours = self.metrics.uptime_seconds / 3600 if self.metrics.uptime_seconds > 0 else 0
        
        return {
            "total_tasks": total_tasks,
            "success_rate": self.metrics.success_rate,
            "average_task_time_ms": self.metrics.average_task_time_ms,
            "uptime_hours": uptime_hours,
            "tasks_per_hour": total_tasks / uptime_hours if uptime_hours > 0 else 0,
            "error_rate": self.metrics.errors_encountered / total_tasks if total_tasks > 0 else 0
        }

class AgentSwarmSchema(BaseModel):
    """Schema for agent swarm definition"""
    swarm_id: str = Field(default_factory=lambda: f"swarm_{uuid.uuid4().hex[:8]}")
    name: str
    description: Optional[str] = None
    
    # Swarm composition
    agents: List[AgentSchema] = Field(default_factory=list)
    target_size: int = Field(ge=1, default=5)
    max_size: int = Field(ge=1, default=100)
    min_size: int = Field(ge=1, default=1)
    
    # Coordination
    coordination_strategy: str = "neural_mesh"
    communication_pattern: str = "mesh"  # mesh, hierarchical, star
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "initializing"
    
    # Performance
    collective_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    def add_agent(self, agent: AgentSchema) -> None:
        """Add agent to swarm"""
        if len(self.agents) >= self.max_size:
            raise ValueError(f"Swarm at maximum size ({self.max_size})")
        
        agent.coordination_group = self.swarm_id
        self.agents.append(agent)
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from swarm"""
        for i, agent in enumerate(self.agents):
            if agent.agent_id == agent_id:
                if len(self.agents) <= self.min_size:
                    raise ValueError(f"Swarm at minimum size ({self.min_size})")
                del self.agents[i]
                return True
        return False
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[AgentSchema]:
        """Get agents of specific type"""
        return [agent for agent in self.agents if agent.agent_type == agent_type]
    
    def get_agents_by_capability(self, capability_name: str) -> List[AgentSchema]:
        """Get agents with specific capability"""
        return [agent for agent in self.agents if agent.has_capability(capability_name)]
    
    def calculate_swarm_metrics(self) -> Dict[str, Any]:
        """Calculate collective swarm metrics"""
        if not self.agents:
            return {}
        
        total_tasks = sum(agent.metrics.tasks_completed + agent.metrics.tasks_failed for agent in self.agents)
        total_success = sum(agent.metrics.tasks_completed for agent in self.agents)
        avg_task_time = sum(agent.metrics.average_task_time_ms for agent in self.agents) / len(self.agents)
        
        return {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents if a.status == AgentStatus.ACTIVE]),
            "total_tasks_processed": total_tasks,
            "collective_success_rate": total_success / total_tasks if total_tasks > 0 else 0,
            "average_task_time_ms": avg_task_time,
            "unique_capabilities": len(set(cap.name for agent in self.agents for cap in agent.capabilities)),
            "expert_agents": len([a for a in self.agents for cap in a.capabilities if cap.level == CapabilityLevel.EXPERT])
        }
