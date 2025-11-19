"""
Unified Orchestrator - Main Export Module
Provides all orchestration capabilities through a single interface
"""

# Core orchestration
from .core.quantum_orchestrator import (
    UnifiedQuantumOrchestrator,
    QuantumAgent, 
    UnifiedTask,
    QuantumScheduler,
    TaskPriority,
    SecurityLevel,
    ExecutionState
)

# Agent lifecycle management
from .core.agent_lifecycle_manager import (
    AgentLifecycleManager,
    AgentState,
    LifecyclePolicy,
    SpawningStrategy,
    AgentLifecycleConfig
)

# Self-improvement system
from .core.self_improvement_system import (
    SelfImprovementSystem,
    SystemWeakness,
    ImprovementPlan,
    ImprovementType
)

# Quantum mathematical foundations
from .quantum.mathematical_foundations import (
    QuantumStateVector,
    UnitaryTransformation,
    QuantumMeasurement,
    EntanglementMatrix,
    QuantumCoherenceTracker
)

# Distributed systems
from .distributed.consensus_manager import (
    DistributedConsensusManager,
    ConsensusState,
    NodeRole
)

# Security framework
from .security.defense_framework import (
    DefenseSecurityFramework,
    SecurityLevel as DefenseSecurityLevel,
    SecurityEvent
)

# Monitoring and telemetry
from .monitoring.comprehensive_telemetry import (
    ComprehensiveTelemetrySystem,
    MetricType,
    TelemetryConfig
)

# Performance optimization
from .scalability.performance_optimizer import (
    PerformanceOptimizer,
    OptimizationStrategy,
    PerformanceMetrics
)

# Legacy bridge for backward compatibility
from .integrations.legacy_bridge import LegacyBridge

__all__ = [
    # Core orchestration
    'UnifiedQuantumOrchestrator',
    'QuantumAgent',
    'UnifiedTask', 
    'QuantumScheduler',
    'TaskPriority',
    'SecurityLevel',
    'ExecutionState',
    
    # Agent lifecycle
    'AgentLifecycleManager',
    'AgentState',
    'LifecyclePolicy',
    'SpawningStrategy',
    'AgentLifecycleConfig',
    
    # Self-improvement
    'SelfImprovementSystem',
    'SystemWeakness',
    'ImprovementPlan',
    'ImprovementType',
    
    # Quantum foundations
    'QuantumStateVector',
    'UnitaryTransformation',
    'QuantumMeasurement',
    'EntanglementMatrix',
    'QuantumCoherenceTracker',
    
    # Distributed systems
    'DistributedConsensusManager',
    'ConsensusState',
    'NodeRole',
    
    # Security
    'DefenseSecurityFramework',
    'DefenseSecurityLevel',
    'SecurityEvent',
    
    # Monitoring
    'ComprehensiveTelemetrySystem',
    'MetricType',
    'TelemetryConfig',
    
    # Performance
    'PerformanceOptimizer',
    'OptimizationStrategy',
    'PerformanceMetrics',
    
    # Legacy support
    'LegacyBridge'
]