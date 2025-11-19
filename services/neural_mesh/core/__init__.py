"""
Neural Mesh Core Components
"""
# Import consolidated types
from .memory_types import (
    MemoryItem, Query, Knowledge, Pattern, Interaction,
    MemoryTier, PatternType, PatternStrength, SecurityLevel,
    VectorStoreType, OrganizationConfig, GlobalKnowledgeSource,
    NeuralMeshLayer
)

# Import enhanced components
from .enhanced_memory import EnhancedNeuralMesh
from .l3_l4_memory import L3OrganizationMemory, L4GlobalMemory
from .distributed_memory import DistributedL1Memory, DistributedL2Memory
from .consensus_manager import ConsensusManager
from .performance_manager import PerformanceManager
from .redis_cluster_manager import RedisClusterManager

__all__ = [
    # Core types
    'MemoryItem',
    'Query', 
    'Knowledge',
    'Pattern',
    'Interaction',
    'MemoryTier',
    'PatternType',
    'PatternStrength',
    'SecurityLevel',
    'VectorStoreType',
    'OrganizationConfig',
    'GlobalKnowledgeSource',
    'NeuralMeshLayer',
    
    # Enhanced components
    'EnhancedNeuralMesh',
    'L3OrganizationMemory',
    'L4GlobalMemory',
    'DistributedL1Memory',
    'DistributedL2Memory',
    'ConsensusManager',
    'PerformanceManager',
    'RedisClusterManager'
]

