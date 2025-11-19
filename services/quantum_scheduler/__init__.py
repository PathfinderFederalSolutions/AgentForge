"""
Quantum Scheduler - Bridge to Unified Orchestrator
This module provides backward compatibility by bridging to the unified orchestrator
"""

# Import from unified orchestrator
from ..unified_orchestrator.core.quantum_orchestrator import (
    UnifiedQuantumOrchestrator as QuantumScheduler,
    UnifiedTask as QuantumTask,
    TaskPriority,
    QuantumAgent
)

__all__ = [
    'QuantumScheduler',
    'QuantumTask', 
    'TaskPriority',
    'QuantumAgent'
]

