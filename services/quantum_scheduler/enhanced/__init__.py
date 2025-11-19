"""
Enhanced Quantum Scheduler - Bridge to Unified Orchestrator
"""

from ...unified_orchestrator.core.quantum_orchestrator import (
    UnifiedQuantumOrchestrator as MillionScaleQuantumScheduler,
    UnifiedTask as MillionScaleTask,
    TaskPriority as QuantumCoherenceLevel,
    QuantumAgent
)

# Additional classes for compatibility
class QuantumClusterState:
    """Mock quantum cluster state for compatibility"""
    def __init__(self):
        self.coherence = 1.0
        self.entanglement = 0.8

def get_million_scale_scheduler():
    """Factory function for million scale scheduler"""
    return MillionScaleQuantumScheduler("million-scale-node")

__all__ = [
    'MillionScaleQuantumScheduler',
    'MillionScaleTask',
    'QuantumCoherenceLevel',
    'QuantumAgent',
    'QuantumClusterState',
    'get_million_scale_scheduler'
]

