"""
Neural Mesh Core Types - Consolidated Data Types and Base Classes
Provides all essential types used across the neural mesh system
"""
from __future__ import annotations

import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum

class MemoryTier(Enum):
    """Memory hierarchy tiers"""
    L1_AGENT = "l1_agent"           # Local agent working memory
    L2_SWARM = "l2_swarm"           # Distributed cluster memory
    L3_ORGANIZATION = "l3_org"      # Persistent organizational knowledge
    L4_GLOBAL = "l4_global"         # Federated external knowledge

class VectorStoreType(Enum):
    """Supported vector store backends"""
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    PGVECTOR = "pgvector"
    MEMORY = "memory"  # In-memory fallback

class SecurityLevel(Enum):
    """Security classification levels"""
    UNCLASSIFIED = "unclassified"
    CUI = "cui"  # Controlled Unclassified Information
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

class PatternType(Enum):
    """Types of patterns that can be detected"""
    MEMORY_ACCESS = "memory_access"
    AGENT_COMMUNICATION = "agent_communication"
    TASK_EXECUTION = "task_execution"
    KNOWLEDGE_PROPAGATION = "knowledge_propagation"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    COLLABORATION = "collaboration"
    LEARNING = "learning"
    ANOMALY = "anomaly"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"

class PatternStrength(Enum):
    """Strength levels for detected patterns"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CRITICAL = "critical"

@dataclass
class MemoryItem:
    """Enhanced memory item with rich metadata"""
    key: str
    value: Any
    embedding: Optional[np.ndarray] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    tier: MemoryTier = MemoryTier.L1_AGENT
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "key": self.key,
            "value": self.value,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "context": self.context,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "tier": self.tier.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryItem:
        """Deserialize from dictionary"""
        embedding = None
        if data.get("embedding"):
            embedding = np.array(data["embedding"])
            
        return cls(
            key=data["key"],
            value=data["value"],
            embedding=embedding,
            context=data.get("context", {}),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", time.time()),
            tier=MemoryTier(data.get("tier", MemoryTier.L1_AGENT.value))
        )

@dataclass
class Query:
    """Enhanced query with context and filters"""
    text: str
    embedding: Optional[np.ndarray] = None
    context: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 5
    min_score: float = 0.7
    tiers: Set[MemoryTier] = field(default_factory=lambda: {MemoryTier.L1_AGENT})

@dataclass
class Knowledge:
    """Synthesized knowledge from patterns"""
    knowledge_id: str
    content: str
    knowledge_type: str
    source_patterns: List[str]
    confidence: float
    applicable_contexts: List[str]
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "knowledge_id": self.knowledge_id,
            "content": self.content,
            "knowledge_type": self.knowledge_type,
            "source_patterns": self.source_patterns,
            "confidence": self.confidence,
            "applicable_contexts": self.applicable_contexts,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Knowledge:
        """Deserialize from dictionary"""
        embedding = None
        if data.get("embedding"):
            embedding = np.array(data["embedding"])
            
        return cls(
            knowledge_id=data["knowledge_id"],
            content=data["content"],
            knowledge_type=data["knowledge_type"],
            source_patterns=data["source_patterns"],
            confidence=data["confidence"],
            applicable_contexts=data["applicable_contexts"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {}),
            embedding=embedding
        )

@dataclass
class Pattern:
    """Detected pattern in agent interactions"""
    pattern_id: str
    pattern_type: PatternType
    strength: PatternStrength
    confidence: float
    agents_involved: Set[str]
    interactions: List[str]  # interaction IDs
    discovered_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "agents_involved": list(self.agents_involved),
            "interactions": self.interactions,
            "discovered_at": self.discovered_at,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Pattern:
        """Deserialize from dictionary"""
        embedding = None
        if data.get("embedding"):
            embedding = np.array(data["embedding"])
            
        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=PatternType(data["pattern_type"]),
            strength=PatternStrength(data["strength"]),
            confidence=data["confidence"],
            agents_involved=set(data["agents_involved"]),
            interactions=data["interactions"],
            discovered_at=data["discovered_at"],
            metadata=data.get("metadata", {}),
            embedding=embedding
        )

@dataclass
class Interaction:
    """Individual agent interaction record"""
    interaction_id: str
    agent_id: str
    interaction_type: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "interaction_id": self.interaction_id,
            "agent_id": self.agent_id,
            "interaction_type": self.interaction_type,
            "timestamp": self.timestamp,
            "data": self.data,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Interaction:
        """Deserialize from dictionary"""
        embedding = None
        if data.get("embedding"):
            embedding = np.array(data["embedding"])
            
        return cls(
            interaction_id=data["interaction_id"],
            agent_id=data["agent_id"],
            interaction_type=data["interaction_type"],
            timestamp=data["timestamp"],
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
            embedding=embedding
        )

@dataclass
class OrganizationConfig:
    """Configuration for organization-level memory"""
    org_id: str
    tenant_id: str
    security_level: str = "standard"  # standard, confidential, secret, top_secret
    retention_days: int = 365
    encryption_enabled: bool = True
    compliance_frameworks: List[str] = None
    
    def __post_init__(self):
        if self.compliance_frameworks is None:
            self.compliance_frameworks = []

@dataclass
class GlobalKnowledgeSource:
    """External knowledge source configuration"""
    source_id: str
    source_type: str  # api, database, file_system, etc.
    endpoint: str
    credentials: Dict[str, Any]
    refresh_interval: int = 3600  # 1 hour default
    enabled: bool = True

class NeuralMeshLayer(ABC):
    """Abstract base class for memory hierarchy layers"""
    
    @abstractmethod
    async def store(self, item: MemoryItem) -> bool:
        """Store memory item in this layer"""
        pass
    
    @abstractmethod
    async def retrieve(self, query: Query) -> List[MemoryItem]:
        """Retrieve memory items matching query"""
        pass
    
    @abstractmethod
    async def propagate(self, knowledge: Knowledge) -> bool:
        """Propagate knowledge to relevant agents"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics"""
        pass
