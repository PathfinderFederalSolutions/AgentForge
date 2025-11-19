"""
NATS subject management and routing for AgentForge
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

class SubjectType(Enum):
    """Types of NATS subjects"""
    COMMAND = "command"
    EVENT = "event"
    REQUEST = "request"
    RESPONSE = "response"
    STREAM = "stream"
    HEARTBEAT = "heartbeat"

@dataclass
class SubjectPattern:
    """NATS subject pattern definition"""
    pattern: str
    subject_type: SubjectType
    description: str
    required_permissions: List[str] = None
    
    def __post_init__(self):
        if self.required_permissions is None:
            self.required_permissions = []

class SubjectRegistry:
    """Registry for NATS subject patterns and routing"""
    
    def __init__(self):
        self.patterns: Dict[str, SubjectPattern] = {}
        self.wildcards: Dict[str, SubjectPattern] = {}
        self._init_default_patterns()
    
    def _init_default_patterns(self):
        """Initialize default AgentForge subject patterns"""
        default_patterns = [
            # Agent coordination
            SubjectPattern(
                "agent.*.command",
                SubjectType.COMMAND,
                "Commands to specific agents",
                ["agent.command"]
            ),
            SubjectPattern(
                "agent.*.event.*",
                SubjectType.EVENT,
                "Agent lifecycle events",
                ["agent.event"]
            ),
            SubjectPattern(
                "agent.*.heartbeat",
                SubjectType.HEARTBEAT,
                "Agent heartbeat messages",
                ["agent.heartbeat"]
            ),
            
            # Swarm coordination
            SubjectPattern(
                "swarm.coordination.*",
                SubjectType.COMMAND,
                "Swarm coordination commands",
                ["swarm.coordinate"]
            ),
            SubjectPattern(
                "swarm.status.*",
                SubjectType.EVENT,
                "Swarm status events",
                ["swarm.status"]
            ),
            SubjectPattern(
                "swarm.metrics.*",
                SubjectType.STREAM,
                "Swarm performance metrics",
                ["swarm.metrics"]
            ),
            
            # Task management
            SubjectPattern(
                "task.*.assign",
                SubjectType.COMMAND,
                "Task assignment commands",
                ["task.assign"]
            ),
            SubjectPattern(
                "task.*.result",
                SubjectType.RESPONSE,
                "Task execution results",
                ["task.result"]
            ),
            SubjectPattern(
                "task.*.status",
                SubjectType.EVENT,
                "Task status updates",
                ["task.status"]
            ),
            
            # Neural mesh
            SubjectPattern(
                "neural.mesh.memory.*",
                SubjectType.STREAM,
                "Neural mesh memory operations",
                ["neural.memory"]
            ),
            SubjectPattern(
                "neural.mesh.sync.*",
                SubjectType.EVENT,
                "Neural mesh synchronization",
                ["neural.sync"]
            ),
            
            # Quantum scheduler
            SubjectPattern(
                "quantum.schedule.*",
                SubjectType.COMMAND,
                "Quantum scheduling commands",
                ["quantum.schedule"]
            ),
            SubjectPattern(
                "quantum.status.*",
                SubjectType.EVENT,
                "Quantum scheduler status",
                ["quantum.status"]
            ),
            
            # Universal I/O
            SubjectPattern(
                "io.input.*",
                SubjectType.STREAM,
                "Universal input processing",
                ["io.input"]
            ),
            SubjectPattern(
                "io.output.*",
                SubjectType.STREAM,
                "Universal output generation",
                ["io.output"]
            ),
            
            # Security and audit
            SubjectPattern(
                "security.audit.*",
                SubjectType.EVENT,
                "Security audit events",
                ["security.audit"]
            ),
            SubjectPattern(
                "security.alert.*",
                SubjectType.EVENT,
                "Security alerts",
                ["security.alert"]
            ),
            
            # System management
            SubjectPattern(
                "system.health.*",
                SubjectType.HEARTBEAT,
                "System health checks",
                ["system.health"]
            ),
            SubjectPattern(
                "system.config.*",
                SubjectType.EVENT,
                "Configuration changes",
                ["system.config"]
            ),
            
            # Data fusion
            SubjectPattern(
                "fusion.sensor.*",
                SubjectType.STREAM,
                "Sensor data fusion",
                ["fusion.sensor"]
            ),
            SubjectPattern(
                "fusion.result.*",
                SubjectType.RESPONSE,
                "Fusion results",
                ["fusion.result"]
            )
        ]
        
        for pattern in default_patterns:
            self.register_pattern(pattern)
    
    def register_pattern(self, pattern: SubjectPattern) -> None:
        """Register a subject pattern"""
        if "*" in pattern.pattern:
            self.wildcards[pattern.pattern] = pattern
        else:
            self.patterns[pattern.pattern] = pattern
        
        logger.debug(f"Registered subject pattern: {pattern.pattern}")
    
    def get_pattern_for_subject(self, subject: str) -> Optional[SubjectPattern]:
        """Get pattern that matches a subject"""
        # Check exact matches first
        if subject in self.patterns:
            return self.patterns[subject]
        
        # Check wildcard patterns
        for pattern_str, pattern in self.wildcards.items():
            if self._matches_pattern(subject, pattern_str):
                return pattern
        
        return None
    
    def _matches_pattern(self, subject: str, pattern: str) -> bool:
        """Check if subject matches wildcard pattern"""
        # Convert NATS wildcard pattern to regex
        regex_pattern = pattern.replace("*", "[^.]+").replace(">", ".*")
        regex_pattern = f"^{regex_pattern}$"
        
        return bool(re.match(regex_pattern, subject))
    
    def validate_subject(self, subject: str) -> bool:
        """Validate subject format"""
        if not subject:
            return False
        
        # Basic NATS subject validation
        if subject.startswith(".") or subject.endswith("."):
            return False
        
        if ".." in subject:
            return False
        
        # Check for valid characters
        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
        if not all(c in valid_chars for c in subject):
            return False
        
        return True
    
    def suggest_subject(
        self,
        service: str,
        operation: str,
        subject_type: SubjectType,
        target: Optional[str] = None
    ) -> str:
        """Suggest subject name based on conventions"""
        parts = [service]
        
        if target:
            parts.append(target)
        
        if subject_type == SubjectType.COMMAND:
            parts.append("command")
        elif subject_type == SubjectType.EVENT:
            parts.append("event")
        elif subject_type == SubjectType.REQUEST:
            parts.append("request")
        elif subject_type == SubjectType.RESPONSE:
            parts.append("response")
        elif subject_type == SubjectType.STREAM:
            parts.append("stream")
        elif subject_type == SubjectType.HEARTBEAT:
            parts.append("heartbeat")
        
        parts.append(operation)
        
        return ".".join(parts)
    
    def get_all_patterns(self) -> List[SubjectPattern]:
        """Get all registered patterns"""
        return list(self.patterns.values()) + list(self.wildcards.values())
    
    def get_subjects_by_type(self, subject_type: SubjectType) -> List[str]:
        """Get all subjects of a specific type"""
        subjects = []
        
        for pattern in self.get_all_patterns():
            if pattern.subject_type == subject_type:
                subjects.append(pattern.pattern)
        
        return subjects

# Global subject registry
_subject_registry: Optional[SubjectRegistry] = None

def get_subject_registry() -> SubjectRegistry:
    """Get global subject registry"""
    global _subject_registry
    if _subject_registry is None:
        _subject_registry = SubjectRegistry()
    return _subject_registry

def validate_subject(subject: str) -> bool:
    """Validate NATS subject"""
    return get_subject_registry().validate_subject(subject)

def suggest_subject(
    service: str,
    operation: str,
    subject_type: SubjectType,
    target: Optional[str] = None
) -> str:
    """Suggest NATS subject name"""
    return get_subject_registry().suggest_subject(service, operation, subject_type, target)

def get_pattern_for_subject(subject: str) -> Optional[SubjectPattern]:
    """Get pattern that matches subject"""
    return get_subject_registry().get_pattern_for_subject(subject)

# Predefined subjects for common operations
class AgentForgeSubjects:
    """Predefined subjects for AgentForge operations"""
    
    # Agent management
    AGENT_DEPLOY = "agent.deploy.command"
    AGENT_TERMINATE = "agent.terminate.command"
    AGENT_STATUS = "agent.status.event"
    AGENT_HEARTBEAT = "agent.heartbeat"
    
    # Task management
    TASK_ASSIGN = "task.assign.command"
    TASK_RESULT = "task.result.response"
    TASK_STATUS = "task.status.event"
    
    # Swarm coordination
    SWARM_COORDINATE = "swarm.coordination.command"
    SWARM_STATUS = "swarm.status.event"
    SWARM_METRICS = "swarm.metrics.stream"
    
    # Neural mesh
    NEURAL_MEMORY = "neural.mesh.memory.stream"
    NEURAL_SYNC = "neural.mesh.sync.event"
    
    # System events
    SYSTEM_HEALTH = "system.health.heartbeat"
    SYSTEM_CONFIG = "system.config.event"
    SYSTEM_ALERT = "system.alert.event"
    
    # Data fusion
    FUSION_SENSOR = "fusion.sensor.stream"
    FUSION_RESULT = "fusion.result.response"
    
    @classmethod
    def get_all_subjects(cls) -> List[str]:
        """Get all predefined subjects"""
        return [
            value for name, value in cls.__dict__.items()
            if isinstance(value, str) and not name.startswith('_')
        ]
