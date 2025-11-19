"""
AGI Memory Bridge - Integration between Neural Mesh and AGI Engine
Provides seamless integration between the 4-tier memory system and AGI processing
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Import neural mesh components
from ..core.enhanced_memory import EnhancedNeuralMesh
from ..core.memory_types import MemoryItem, Query, Knowledge
from ..core.l3_l4_memory import OrganizationConfig, GlobalKnowledgeSource, VectorStoreType

# Import AGI components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from services.universal_io.agi_integration import UniversalAGIEngine, AGIRequest, AGIResponse
except ImportError:
    UniversalAGIEngine = None
    AGIRequest = None
    AGIResponse = None

log = logging.getLogger("agi-memory-bridge")

@dataclass
class MemoryConfiguration:
    """Complete memory system configuration"""
    # Agent configuration
    agent_id: str
    swarm_id: str = "default"
    
    # L2 configuration
    redis_url: Optional[str] = None
    
    # L3 configuration
    org_config: Optional[OrganizationConfig] = None
    postgres_url: Optional[str] = None
    vector_store_type: VectorStoreType = VectorStoreType.MEMORY
    vector_store_config: Optional[Dict[str, Any]] = None
    
    # L4 configuration
    global_sources: Optional[List[GlobalKnowledgeSource]] = None
    
    # Performance configuration
    enable_emergent_intelligence: bool = True
    enable_cross_tier_sync: bool = True
    max_memory_items_per_tier: Dict[str, int] = None
    
    def __post_init__(self):
        if self.max_memory_items_per_tier is None:
            self.max_memory_items_per_tier = {
                "l1": 1000,
                "l2": 10000,
                "l3": 100000,
                "l4": 1000000
            }

class AGIMemoryBridge:
    """Bridge between AGI Engine and Neural Mesh Memory System"""
    
    def __init__(self, config: MemoryConfiguration):
        self.config = config
        self.neural_mesh: Optional[EnhancedNeuralMesh] = None
        self.agi_engine: Optional[UniversalAGIEngine] = None
        
        # Performance tracking
        self.memory_operations = 0
        self.successful_operations = 0
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize the complete memory system"""
        log.info(f"Initializing AGI Memory Bridge for agent {self.config.agent_id}")
        
        # Initialize Neural Mesh with all tiers
        self.neural_mesh = EnhancedNeuralMesh(
            agent_id=self.config.agent_id,
            swarm_id=self.config.swarm_id,
            redis_url=self.config.redis_url,
            org_config=self.config.org_config,
            postgres_url=self.config.postgres_url,
            global_sources=self.config.global_sources
        )
        
        # Initialize L3 and L4 if configured
        if self.neural_mesh.l3_memory:
            await self.neural_mesh.l3_memory.initialize()
            log.info("L3 organizational memory initialized")
        
        if self.neural_mesh.l4_memory:
            await self.neural_mesh.l4_memory.initialize()
            log.info("L4 global memory initialized")
        
        # Initialize AGI Engine if available
        if UniversalAGIEngine:
            try:
                self.agi_engine = UniversalAGIEngine()
                log.info("AGI Engine integrated with memory bridge")
            except Exception as e:
                log.warning(f"AGI Engine not available: {e}")
        
        log.info("AGI Memory Bridge initialization completed")
    
    async def store_agi_context(self, request: Any, response: Any, context: Dict[str, Any] = None) -> bool:
        """Store AGI request/response context in appropriate memory tiers"""
        if not self.neural_mesh:
            return False
        
        try:
            context = context or {}
            
            # Determine storage tier based on context importance
            importance = context.get("importance", "normal")  # low, normal, high, critical
            
            storage_kwargs = {
                "context": {
                    **context,
                    "request_type": getattr(request, 'request_type', 'unknown'),
                    "processing_time": getattr(response, 'processing_time', 0),
                    "agent_id": self.config.agent_id
                },
                "metadata": {
                    "content_type": "agi_context",
                    "importance": importance,
                    "timestamp": time.time()
                }
            }
            
            # Store based on importance
            if importance == "critical":
                storage_kwargs["global"] = True
                storage_kwargs["organizational"] = True
            elif importance == "high":
                storage_kwargs["organizational"] = True
            
            # Store request context
            request_key = f"agi_request:{getattr(request, 'request_id', 'unknown')}"
            request_success = await self.neural_mesh.store(
                request_key,
                self._serialize_agi_object(request),
                **storage_kwargs
            )
            
            # Store response context
            response_key = f"agi_response:{getattr(request, 'request_id', 'unknown')}"
            response_success = await self.neural_mesh.store(
                response_key,
                self._serialize_agi_object(response),
                **storage_kwargs
            )
            
            self.memory_operations += 2
            if request_success and response_success:
                self.successful_operations += 2
            
            return request_success and response_success
            
        except Exception as e:
            log.error(f"Failed to store AGI context: {e}")
            return False
    
    async def retrieve_agi_context(self, query: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve AGI context from memory system"""
        if not self.neural_mesh:
            return []
        
        try:
            # Search with AGI-specific filters
            search_kwargs = {
                "top_k": 10,
                "min_score": 0.6,
                "context": context or {},
                "filters": {"content_type": "agi_context"}
            }
            
            results = await self.neural_mesh.retrieve(query, **search_kwargs)
            
            # Convert to AGI-friendly format
            agi_contexts = []
            for item in results:
                agi_context = {
                    "key": item.key,
                    "content": item.value,
                    "relevance_score": item.metadata.get("relevance_score", 0),
                    "tier": item.tier.value,
                    "timestamp": item.timestamp,
                    "context": item.context
                }
                agi_contexts.append(agi_context)
            
            return agi_contexts
            
        except Exception as e:
            log.error(f"Failed to retrieve AGI context: {e}")
            return []
    
    async def propagate_agi_knowledge(self, knowledge_type: str, content: Any, 
                                   target_scope: str = "swarm") -> bool:
        """Propagate AGI-generated knowledge through the memory hierarchy"""
        if not self.neural_mesh:
            return False
        
        try:
            knowledge = Knowledge(
                knowledge_id=f"agi_knowledge:{knowledge_type}:{int(time.time())}",
                content=str(content),
                knowledge_type=knowledge_type,
                source_patterns=[],
                confidence=0.9,
                applicable_contexts=[target_scope],
                metadata={
                    "generated_by": "agi_engine",
                    "propagation_time": time.time(),
                    "source": "agi_engine",
                    "target_scope": target_scope,
                    "agent_id": self.config.agent_id
                }
            )
            
            return await self.neural_mesh.propagate_knowledge(knowledge)
            
        except Exception as e:
            log.error(f"Failed to propagate AGI knowledge: {e}")
            return False
    
    async def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory patterns for AGI optimization"""
        if not self.neural_mesh:
            return {}
        
        try:
            # Get emergent intelligence analysis
            emergent_knowledge = await self.neural_mesh.analyze_emergence()
            
            # Get comprehensive stats
            stats = await self.neural_mesh.get_comprehensive_stats()
            
            # Calculate performance metrics
            uptime = time.time() - self.start_time
            success_rate = (self.successful_operations / max(1, self.memory_operations)) * 100
            
            return {
                "emergent_patterns": [k.to_dict() for k in emergent_knowledge],
                "memory_stats": stats,
                "performance": {
                    "uptime_seconds": uptime,
                    "total_operations": self.memory_operations,
                    "successful_operations": self.successful_operations,
                    "success_rate_percent": success_rate,
                    "operations_per_second": self.memory_operations / max(1, uptime)
                },
                "recommendations": self._generate_optimization_recommendations(stats)
            }
            
        except Exception as e:
            log.error(f"Failed to analyze memory patterns: {e}")
            return {"error": str(e)}
    
    def _generate_optimization_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        summary = stats.get("summary", {})
        total_items = summary.get("total_items", 0)
        active_tiers = summary.get("active_tiers", 0)
        
        # Tier utilization recommendations
        if active_tiers < 3:
            recommendations.append("Consider enabling L3 organizational memory for better knowledge persistence")
        
        if not summary.get("l4_available", False):
            recommendations.append("Consider enabling L4 global memory for external knowledge integration")
        
        # Capacity recommendations
        if total_items > 50000:
            recommendations.append("High memory utilization detected - consider implementing automatic archiving")
        
        # Performance recommendations
        if self.memory_operations > 0:
            ops_per_sec = self.memory_operations / max(1, time.time() - self.start_time)
            if ops_per_sec > 1000:
                recommendations.append("High memory operation rate - consider implementing caching optimizations")
        
        if not recommendations:
            recommendations.append("Memory system is operating optimally")
        
        return recommendations
    
    def _serialize_agi_object(self, obj: Any) -> Dict[str, Any]:
        """Serialize AGI objects for storage"""
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return {"content": str(obj)}
    
    async def get_memory_health(self) -> Dict[str, Any]:
        """Get comprehensive memory system health status"""
        if not self.neural_mesh:
            return {"status": "not_initialized"}
        
        try:
            # Get stats from all tiers
            stats = await self.neural_mesh.get_comprehensive_stats()
            
            # Determine overall health
            health_status = "healthy"
            issues = []
            
            # Check tier availability
            expected_tiers = ["L1"]
            if self.config.redis_url:
                expected_tiers.append("L2")
            if self.config.org_config:
                expected_tiers.append("L3")
            if self.config.global_sources:
                expected_tiers.append("L4")
            
            available_tiers = list(stats["tiers"].keys())
            missing_tiers = set(expected_tiers) - set(available_tiers)
            
            if missing_tiers:
                health_status = "degraded"
                issues.append(f"Missing memory tiers: {', '.join(missing_tiers)}")
            
            # Check performance
            success_rate = (self.successful_operations / max(1, self.memory_operations)) * 100
            if success_rate < 95:
                health_status = "degraded"
                issues.append(f"Low success rate: {success_rate:.1f}%")
            
            return {
                "status": health_status,
                "issues": issues,
                "tiers_available": available_tiers,
                "tiers_expected": expected_tiers,
                "performance": {
                    "success_rate": success_rate,
                    "total_operations": self.memory_operations,
                    "uptime_seconds": time.time() - self.start_time
                },
                "stats": stats
            }
            
        except Exception as e:
            log.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

# Factory function for easy initialization
async def create_agi_memory_system(
    agent_id: str,
    swarm_id: str = "default",
    org_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    redis_url: Optional[str] = None,
    postgres_url: Optional[str] = None
) -> AGIMemoryBridge:
    """Factory function to create a complete AGI memory system"""
    
    # Create organization config if org_id provided
    org_config = None
    if org_id and tenant_id:
        org_config = OrganizationConfig(
            org_id=org_id,
            tenant_id=tenant_id,
            security_level="standard",
            retention_days=365
        )
    
    # Create sample global knowledge sources
    global_sources = [
        GlobalKnowledgeSource(
            source_id="wikipedia",
            source_type="api",
            endpoint="https://en.wikipedia.org/api/rest_v1/",
            credentials={},
            refresh_interval=86400,  # Daily refresh
            enabled=False  # Disabled by default
        )
    ]
    
    # Create configuration
    config = MemoryConfiguration(
        agent_id=agent_id,
        swarm_id=swarm_id,
        redis_url=redis_url,
        org_config=org_config,
        postgres_url=postgres_url,
        vector_store_type=VectorStoreType.MEMORY,  # Default to memory for now
        global_sources=global_sources
    )
    
    # Create and initialize bridge
    bridge = AGIMemoryBridge(config)
    await bridge.initialize()
    
    return bridge

