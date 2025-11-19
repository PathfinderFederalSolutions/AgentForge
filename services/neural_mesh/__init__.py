"""
Neural Mesh System - Main Export Module
Provides all neural mesh capabilities through a single interface
"""

# Core memory system
from .core.enhanced_memory import EnhancedNeuralMesh
from .core.distributed_memory import DistributedMemory
from .core.memory_types import MemoryTier
from .core.distributed_memory_store import MemoryType
from .core.contextual_memory_system import ContextualMemorySystem

# Intelligence and emergence
from .intelligence.emergence import EmergentIntelligence, EmergenceMetrics
from .intelligence.collective_reasoning import CollectiveReasoningEngine

# Security and compliance
from .security.security_manager import SecurityManager, AuditLogger
from .security.encryption_manager import EncryptionManager

# Performance and optimization
from .performance.performance_optimizer import PerformanceOptimizer
from .performance.caching_layer import CachingLayer

# Cross-datacenter replication
from .core.cross_datacenter_replication import CrossDatacenterReplication

# Memory versioning
from .core.memory_versioning_system import MemoryVersioningSystem

__all__ = [
    # Core memory
    'EnhancedNeuralMesh',
    'DistributedMemory',
    'MemoryType',
    'MemoryTier',
    'ContextualMemorySystem',
    
    # Intelligence
    'EmergentIntelligence',
    'EmergenceMetrics',
    'CollectiveReasoningEngine',
    
    # Security
    'SecurityManager',
    'AuditLogger',
    'EncryptionManager',
    
    # Performance
    'PerformanceOptimizer',
    'CachingLayer',
    
    # Replication and versioning
    'CrossDatacenterReplication',
    'MemoryVersioningSystem'
]