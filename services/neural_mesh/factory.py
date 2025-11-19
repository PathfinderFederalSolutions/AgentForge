"""
Neural Mesh Factory - DEPRECATED
This file is deprecated. Use production_neural_mesh.py for new implementations.
Provides backward compatibility for existing code.
"""
from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Optional

# Import from new production system
from .production_neural_mesh import (
    create_development_neural_mesh as _create_development_neural_mesh,
    create_production_neural_mesh as _create_production_neural_mesh,
    create_defense_neural_mesh as _create_defense_neural_mesh,
    create_scif_neural_mesh as _create_scif_neural_mesh,
    create_agi_memory_bridge as _create_agi_memory_bridge,
    ProductionNeuralMesh
)
from .config.production_config import get_production_config, NeuralMeshProductionConfig
from .core.enhanced_memory import EnhancedNeuralMesh
from .integration.ai_memory_bridge import AGIMemoryBridge, MemoryConfiguration

log = logging.getLogger("neural-mesh-factory-deprecated")

def _deprecation_warning(func_name: str):
    """Issue deprecation warning"""
    warnings.warn(
        f"{func_name} is deprecated. Use production_neural_mesh.py functions instead.",
        DeprecationWarning,
        stacklevel=3
    )

async def create_neural_mesh(
    agent_id: Optional[str] = None,
    swarm_id: Optional[str] = None,
    config: Optional[NeuralMeshProductionConfig] = None
) -> EnhancedNeuralMesh:
    """
    DEPRECATED: Factory function to create a production-ready neural mesh instance
    Use production_neural_mesh.py functions instead.
    
    Args:
        agent_id: Unique agent identifier
        swarm_id: Swarm identifier for L2 memory
        config: Production configuration (auto-detected if None)
    
    Returns:
        Fully initialized EnhancedNeuralMesh instance
    """
    _deprecation_warning("create_neural_mesh")
    # Get configuration
    if config is None:
        config = get_production_config()
    
    # Override IDs if provided
    if agent_id:
        config.agent_id = agent_id
    if swarm_id:
        config.swarm_id = swarm_id
    
    # Validate configuration
    validation_issues = config.validate()
    if validation_issues:
        log.warning(f"Configuration issues: {validation_issues}")
    
    # Create neural mesh
    mesh = EnhancedNeuralMesh(
        agent_id=config.agent_id,
        swarm_id=config.swarm_id,
        redis_url=config.redis_url,
        org_config=config.get_organization_config(),
        postgres_url=config.postgres_url,
        global_sources=config.get_global_knowledge_sources()
    )
    
    # Initialize L3 and L4 if enabled
    if mesh.l3_memory and config.enable_l3_memory:
        try:
            await mesh.l3_memory.initialize()
            log.info("L3 organizational memory initialized")
        except Exception as e:
            log.error(f"L3 memory initialization failed: {e}")
    
    if mesh.l4_memory and config.enable_l4_memory:
        try:
            await mesh.l4_memory.initialize()
            log.info("L4 global memory initialized")
        except Exception as e:
            log.error(f"L4 memory initialization failed: {e}")
    
    log.info(f"Neural mesh created for agent {config.agent_id} with {mesh._count_active_layers()} active layers")
    return mesh

async def create_agi_memory_bridge(
    agent_id: Optional[str] = None,
    swarm_id: Optional[str] = None,
    org_id: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> AGIMemoryBridge:
    """
    DEPRECATED: Factory function to create AGI memory bridge with production configuration
    Use production_neural_mesh.create_agi_memory_bridge instead.
    
    Args:
        agent_id: Unique agent identifier
        swarm_id: Swarm identifier
        org_id: Organization identifier for L3 memory
        tenant_id: Tenant identifier for L3 memory
    
    Returns:
        Fully initialized AGIMemoryBridge instance
    """
    _deprecation_warning("create_agi_memory_bridge")
    # Get base configuration
    base_config = get_production_config()
    
    # Override parameters
    if agent_id:
        base_config.agent_id = agent_id
    if swarm_id:
        base_config.swarm_id = swarm_id
    if org_id:
        base_config.org_id = org_id
        base_config.enable_l3_memory = True
    if tenant_id:
        base_config.tenant_id = tenant_id
        base_config.enable_l3_memory = True
    
    # Create memory configuration
    memory_config = MemoryConfiguration(
        agent_id=base_config.agent_id,
        swarm_id=base_config.swarm_id,
        redis_url=base_config.redis_url,
        org_config=base_config.get_organization_config(),
        postgres_url=base_config.postgres_url,
        vector_store_type=base_config.vector_store_type,
        vector_store_config=base_config.get_vector_store_config(),
        global_sources=base_config.get_global_knowledge_sources()
    )
    
    # Create and initialize bridge
    bridge = AGIMemoryBridge(memory_config)
    await bridge.initialize()
    
    log.info(f"AGI memory bridge created for agent {base_config.agent_id}")
    return bridge

# Convenience functions for specific use cases
async def create_development_mesh(agent_id: str = "dev_agent") -> EnhancedNeuralMesh:
    """DEPRECATED: Create lightweight mesh for development"""
    _deprecation_warning("create_development_mesh")
    from .config.production_config import ProductionConfigs
    config = ProductionConfigs.development_config()
    config.agent_id = agent_id
    return await create_neural_mesh(config=config)

async def create_production_mesh(agent_id: str, org_id: str, tenant_id: str) -> EnhancedNeuralMesh:
    """DEPRECATED: Create full production mesh with all tiers"""
    _deprecation_warning("create_production_mesh")
    from .config.production_config import ProductionConfigs
    config = ProductionConfigs.enterprise_production_config()
    config.agent_id = agent_id
    config.org_id = org_id
    config.tenant_id = tenant_id
    return await create_neural_mesh(config=config)

async def create_defense_mesh(agent_id: str, org_id: str, tenant_id: str) -> EnhancedNeuralMesh:
    """DEPRECATED: Create defense/GovCloud mesh with compliance"""
    _deprecation_warning("create_defense_mesh")
    from .config.production_config import ProductionConfigs
    config = ProductionConfigs.defense_govcloud_config()
    config.agent_id = agent_id
    config.org_id = org_id
    config.tenant_id = tenant_id
    return await create_neural_mesh(config=config)

async def create_scif_mesh(agent_id: str, org_id: str, tenant_id: str) -> EnhancedNeuralMesh:
    """DEPRECATED: Create air-gapped SCIF mesh"""
    _deprecation_warning("create_scif_mesh")
    from .config.production_config import ProductionConfigs
    config = ProductionConfigs.scif_air_gapped_config()
    config.agent_id = agent_id
    config.org_id = org_id
    config.tenant_id = tenant_id
    return await create_neural_mesh(config=config)
