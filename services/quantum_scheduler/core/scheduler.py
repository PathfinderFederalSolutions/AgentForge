"""
Quantum Scheduler Core - Bridge to Unified Orchestrator
"""

from ...unified_orchestrator.core.quantum_orchestrator import (
    UnifiedQuantumOrchestrator as QuantumScheduler,
    UnifiedTask as QuantumTask,
    TaskPriority,
    QuantumAgent,
    ExecutionPath
)

# Additional aliases for backward compatibility
ExecutionPath = TaskPriority  # Simple mapping

__all__ = [
    'QuantumScheduler',
    'QuantumTask',
    'TaskPriority',
    'QuantumAgent',
    'ExecutionPath'
]

