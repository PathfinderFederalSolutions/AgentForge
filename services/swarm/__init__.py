"""
Unified Swarm System - Main Export Module
Provides all swarm capabilities through a single interface
"""

# Core swarm system
from .unified_swarm_system import UnifiedSwarmSystem
from .core.unified_agent import UnifiedAgent, AgentType, AgentState
from .workers.unified_worker import UnifiedWorker

# Fusion capabilities
from .fusion.production_fusion_system import (
    ProductionFusionSystem,
    FusionQualityLevel,
    IntelligenceDomain
)
from .fusion.advanced_bayesian import AdvancedBayesianFusion
from .fusion.secure_evidence_chain import SecureEvidenceChain, IntegrityStatus
from .fusion.neural_mesh_integration import NeuralMeshIntegrator as NeuralMeshFusionBridge

# Capabilities system
from .capabilities.unified_capabilities import UnifiedCapabilityRegistry
from .capability_registry import CapabilityRegistry

# Memory and coordination
from .memory.mesh import MemoryMesh
from .coordination.enhanced_mega_coordinator import EnhancedMegaSwarmCoordinator as EnhancedMegaCoordinator

# Integration bridge
from .integration.unified_integration_bridge import UnifiedIntegrationBridge

# Types and contracts
from .forge_types import Task, AgentContract, TaskResult

__all__ = [
    # Core system
    'UnifiedSwarmSystem',
    'UnifiedAgent',
    'AgentType',
    'AgentState',
    'UnifiedWorker',
    
    # Fusion system
    'ProductionFusionSystem',
    'FusionQualityLevel',
    'IntelligenceDomain',
    'AdvancedBayesianFusion',
    'SecureEvidenceChain',
    'IntegrityStatus',
    'NeuralMeshFusionBridge',
    
    # Capabilities
    'UnifiedCapabilityRegistry',
    'CapabilityRegistry',
    
    # Memory and coordination
    'MemoryMesh',
    'EnhancedMegaCoordinator',
    
    # Integration
    'UnifiedIntegrationBridge',
    
    # Types
    'Task',
    'AgentContract', 
    'TaskResult'
]