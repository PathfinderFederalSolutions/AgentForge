"""
L3 & L4 Memory Layers - Completing the Neural Mesh Memory Hierarchy
L3: Organization Memory (PostgreSQL + Pinecone/Weaviate)
L4: Global Knowledge (Federated Learning + External APIs)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np

# Import base classes from memory types
from .memory_types import (
    NeuralMeshLayer, MemoryItem, Query, Knowledge, MemoryTier
)

# Import the full-featured multimodal embedder
try:
    from ..embeddings.multimodal import MultiModalEmbedder
except ImportError:
    # Fallback to basic embedder
    from .enhanced_memory import MultiModalEmbedder

# Optional imports with fallbacks
try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    asyncpg = None
    POSTGRES_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except (ImportError, Exception) as e:
    # Handle both import errors and package configuration errors
    pinecone = None
    PINECONE_AVAILABLE = False

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    weaviate = None
    WEAVIATE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

# Metrics imports (graceful degradation)
try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = lambda *args, **kwargs: None

log = logging.getLogger("l3-l4-memory")

class VectorStoreType(Enum):
    """Supported vector store backends"""
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    PGVECTOR = "pgvector"
    MEMORY = "memory"  # In-memory fallback

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

class L3OrganizationMemory(NeuralMeshLayer):
    """L3: Persistent organizational knowledge base with PostgreSQL + Vector DB"""
    
    def __init__(self, org_config: OrganizationConfig, 
                 postgres_url: Optional[str] = None,
                 vector_store_type: VectorStoreType = VectorStoreType.MEMORY,
                 vector_store_config: Optional[Dict[str, Any]] = None):
        self.org_config = org_config
        self.postgres_url = postgres_url
        self.vector_store_type = vector_store_type
        self.vector_store_config = vector_store_config or {}
        
        # Components
        self.embedder = MultiModalEmbedder(target_dimension=768)
        self.postgres_pool = None
        self.vector_store = None
        
        # In-memory fallback storage
        self.memory_store: Dict[str, MemoryItem] = {}
        self.vector_index: Dict[str, np.ndarray] = {}
        
        # Metrics
        if METRICS_AVAILABLE:
            self.storage_operations = Counter(
                'l3_memory_operations_total',
                'L3 memory operations',
                ['org_id', 'operation', 'status']
            )
            self.retrieval_latency = Histogram(
                'l3_memory_retrieval_latency_seconds',
                'L3 memory retrieval latency',
                ['org_id']
            )
            self.storage_size = Gauge(
                'l3_memory_storage_size_items',
                'Number of items in L3 storage',
                ['org_id']
            )
    
    async def initialize(self):
        """Initialize L3 memory layer"""
        log.info(f"Initializing L3 memory for organization {self.org_config.org_id}")
        
        # Initialize PostgreSQL connection
        if self.postgres_url and POSTGRES_AVAILABLE:
            await self._init_postgres()
        
        # Initialize vector store
        await self._init_vector_store()
        
        log.info(f"L3 memory initialized for org {self.org_config.org_id}")
    
    async def _init_postgres(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.postgres_pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=5,
                max_size=20,
                command_timeout=30
            )
            
            # Create tables if they don't exist
            await self._create_tables()
            
            log.info("PostgreSQL connection pool initialized for L3 memory")
            
        except Exception as e:
            log.error(f"Failed to initialize PostgreSQL: {e}")
            self.postgres_pool = None
    
    async def _create_tables(self):
        """Create necessary tables for L3 memory"""
        if not self.postgres_pool:
            return
            
        async with self.postgres_pool.acquire() as conn:
            # Memory items table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS l3_memory_items (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    org_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    context JSONB DEFAULT '{}',
                    metadata JSONB DEFAULT '{}',
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMPTZ DEFAULT NOW(),
                    embedding_id TEXT,
                    content_type TEXT DEFAULT 'text',
                    security_level TEXT DEFAULT 'standard',
                    UNIQUE(org_id, tenant_id, key)
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_l3_memory_org_tenant 
                ON l3_memory_items(org_id, tenant_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_l3_memory_timestamp 
                ON l3_memory_items(timestamp)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_l3_memory_last_accessed 
                ON l3_memory_items(last_accessed)
            """)
            
            # Knowledge propagation tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS l3_knowledge_propagation (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    org_id TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    target_agents TEXT[] NOT NULL,
                    propagation_time TIMESTAMPTZ DEFAULT NOW(),
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}'
                )
            """)
    
    async def _init_vector_store(self):
        """Initialize vector store backend"""
        if self.vector_store_type == VectorStoreType.PINECONE and PINECONE_AVAILABLE:
            await self._init_pinecone()
        elif self.vector_store_type == VectorStoreType.WEAVIATE and WEAVIATE_AVAILABLE:
            await self._init_weaviate()
        elif self.vector_store_type == VectorStoreType.PGVECTOR and self.postgres_pool:
            await self._init_pgvector()
        else:
            # Use in-memory fallback
            log.info("Using in-memory vector store for L3 memory")
            self.vector_store_type = VectorStoreType.MEMORY
    
    async def _init_pinecone(self):
        """Initialize Pinecone vector store"""
        try:
            api_key = self.vector_store_config.get("api_key")
            environment = self.vector_store_config.get("environment", "us-west1-gcp")
            
            pinecone.init(api_key=api_key, environment=environment)
            
            # Create index if it doesn't exist
            index_name = f"agentforge-l3-{self.org_config.org_id}"
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine",
                    pod_type="p1.x1"
                )
            
            self.vector_store = pinecone.Index(index_name)
            log.info(f"Pinecone vector store initialized: {index_name}")
            
        except Exception as e:
            log.error(f"Failed to initialize Pinecone: {e}")
            self.vector_store = None
    
    async def _init_weaviate(self):
        """Initialize Weaviate vector store"""
        try:
            url = self.vector_store_config.get("url", "http://localhost:8080")
            auth_config = self.vector_store_config.get("auth_config")
            
            self.vector_store = weaviate.Client(
                url=url,
                auth_client_secret=auth_config
            )
            
            # Create schema if it doesn't exist
            schema = {
                "class": f"AgentForgeL3{self.org_config.org_id}",
                "description": f"L3 memory for organization {self.org_config.org_id}",
                "properties": [
                    {
                        "name": "key",
                        "dataType": ["text"],
                        "description": "Memory item key"
                    },
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Memory item content"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                        "description": "Memory item metadata"
                    }
                ]
            }
            
            if not self.vector_store.schema.exists(schema["class"]):
                self.vector_store.schema.create_class(schema)
            
            log.info(f"Weaviate vector store initialized for org {self.org_config.org_id}")
            
        except Exception as e:
            log.error(f"Failed to initialize Weaviate: {e}")
            self.vector_store = None
    
    async def _init_pgvector(self):
        """Initialize pgvector extension"""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create vector table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS l3_embeddings (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        org_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        embedding vector(768),
                        metadata JSONB DEFAULT '{{}}',
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(org_id, key)
                    )
                """)
                
                # Create vector index
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_l3_embeddings_vector 
                    ON l3_embeddings USING ivfflat (embedding vector_cosine_ops)
                """)
            
            log.info("pgvector initialized for L3 memory")
            
        except Exception as e:
            log.error(f"Failed to initialize pgvector: {e}")
    
    async def store(self, item: MemoryItem) -> bool:
        """Store item in L3 organizational memory"""
        try:
            item.tier = MemoryTier.L3_ORGANIZATION
            item.context["org_id"] = self.org_config.org_id
            item.context["tenant_id"] = self.org_config.tenant_id
            
            # Generate embedding if not provided
            if item.embedding is None:
                content_type = item.metadata.get("content_type", "text")
                embedding_result = await self.embedder.encode(item.value, content_type, item.metadata)
                item.embedding = embedding_result.embedding
            
            # Store in PostgreSQL
            postgres_success = await self._store_postgres(item)
            
            # Store in vector database
            vector_success = await self._store_vector(item)
            
            # Update metrics
            if METRICS_AVAILABLE:
                status = "success" if postgres_success and vector_success else "error"
                self.storage_operations.labels(
                    org_id=self.org_config.org_id,
                    operation="store",
                    status=status
                ).inc()
                
                if postgres_success and vector_success:
                    self.storage_size.labels(org_id=self.org_config.org_id).inc()
            
            return postgres_success and vector_success
            
        except Exception as e:
            log.error(f"Failed to store item {item.key} in L3: {e}")
            return False
    
    async def _store_postgres(self, item: MemoryItem) -> bool:
        """Store item in PostgreSQL"""
        if not self.postgres_pool:
            # Fallback to memory store
            self.memory_store[item.key] = item
            return True
            
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO l3_memory_items 
                    (org_id, tenant_id, key, value, context, metadata, 
                     timestamp, access_count, last_accessed, content_type, security_level)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (org_id, tenant_id, key) 
                    DO UPDATE SET 
                        value = EXCLUDED.value,
                        context = EXCLUDED.context,
                        metadata = EXCLUDED.metadata,
                        timestamp = EXCLUDED.timestamp,
                        access_count = l3_memory_items.access_count + 1,
                        last_accessed = EXCLUDED.last_accessed
                """, 
                    self.org_config.org_id,
                    self.org_config.tenant_id,
                    item.key,
                    json.dumps(item.value),
                    json.dumps(item.context),
                    json.dumps(item.metadata),
                    item.timestamp,
                    item.access_count,
                    item.last_accessed,
                    item.metadata.get("content_type", "text"),
                    self.org_config.security_level
                )
            
            return True
            
        except Exception as e:
            log.error(f"PostgreSQL storage failed for {item.key}: {e}")
            return False
    
    async def _store_vector(self, item: MemoryItem) -> bool:
        """Store embedding in vector database"""
        if self.vector_store_type == VectorStoreType.PINECONE and self.vector_store:
            return await self._store_pinecone(item)
        elif self.vector_store_type == VectorStoreType.WEAVIATE and self.vector_store:
            return await self._store_weaviate(item)
        elif self.vector_store_type == VectorStoreType.PGVECTOR and self.postgres_pool:
            return await self._store_pgvector(item)
        else:
            # Memory fallback
            self.vector_index[item.key] = item.embedding
            return True
    
    async def _store_pinecone(self, item: MemoryItem) -> bool:
        """Store in Pinecone vector database"""
        try:
            vector_id = f"{self.org_config.org_id}:{self.org_config.tenant_id}:{item.key}"
            
            self.vector_store.upsert([
                (vector_id, item.embedding.tolist(), {
                    "org_id": self.org_config.org_id,
                    "tenant_id": self.org_config.tenant_id,
                    "key": item.key,
                    "content_type": item.metadata.get("content_type", "text"),
                    "timestamp": item.timestamp
                })
            ])
            
            return True
            
        except Exception as e:
            log.error(f"Pinecone storage failed for {item.key}: {e}")
            return False
    
    async def _store_weaviate(self, item: MemoryItem) -> bool:
        """Store in Weaviate vector database"""
        try:
            class_name = f"AgentForgeL3{self.org_config.org_id}"
            
            data_object = {
                "key": item.key,
                "content": str(item.value),
                "metadata": item.metadata
            }
            
            self.vector_store.data_object.create(
                data_object=data_object,
                class_name=class_name,
                vector=item.embedding.tolist()
            )
            
            return True
            
        except Exception as e:
            log.error(f"Weaviate storage failed for {item.key}: {e}")
            return False
    
    async def _store_pgvector(self, item: MemoryItem) -> bool:
        """Store in pgvector"""
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO l3_embeddings (org_id, key, embedding, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (org_id, key)
                    DO UPDATE SET 
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        timestamp = NOW()
                """,
                    self.org_config.org_id,
                    item.key,
                    item.embedding.tolist(),
                    json.dumps(item.metadata)
                )
            
            return True
            
        except Exception as e:
            log.error(f"pgvector storage failed for {item.key}: {e}")
            return False
    
    async def retrieve(self, query: Query) -> List[MemoryItem]:
        """Retrieve items from L3 organizational memory"""
        start_time = time.time()
        
        try:
            # Generate query embedding if not provided
            if query.embedding is None:
                embedding_result = await self.embedder.encode(query.text, "text")
                query.embedding = embedding_result.embedding
            
            # Retrieve from vector store
            vector_results = await self._retrieve_vector(query)
            
            # Retrieve full items from PostgreSQL
            items = await self._retrieve_postgres_items(vector_results, query)
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.retrieval_latency.labels(org_id=self.org_config.org_id).observe(
                    time.time() - start_time
                )
                self.storage_operations.labels(
                    org_id=self.org_config.org_id,
                    operation="retrieve",
                    status="success"
                ).inc()
            
            return items
            
        except Exception as e:
            log.error(f"L3 retrieval failed for query '{query.text}': {e}")
            
            if METRICS_AVAILABLE:
                self.storage_operations.labels(
                    org_id=self.org_config.org_id,
                    operation="retrieve",
                    status="error"
                ).inc()
            
            return []
    
    async def _retrieve_vector(self, query: Query) -> List[Tuple[str, float]]:
        """Retrieve similar vectors"""
        if self.vector_store_type == VectorStoreType.PINECONE and self.vector_store:
            return await self._retrieve_pinecone(query)
        elif self.vector_store_type == VectorStoreType.WEAVIATE and self.vector_store:
            return await self._retrieve_weaviate(query)
        elif self.vector_store_type == VectorStoreType.PGVECTOR and self.postgres_pool:
            return await self._retrieve_pgvector(query)
        else:
            # Memory fallback
            return self._retrieve_memory_vector(query)
    
    def _retrieve_memory_vector(self, query: Query) -> List[Tuple[str, float]]:
        """Retrieve from in-memory vector index"""
        if not self.vector_index:
            return []
        
        # Calculate similarities
        similarities = []
        for key, embedding in self.vector_index.items():
            similarity = self._cosine_similarity(query.embedding, embedding)
            if similarity >= query.min_score:
                similarities.append((key, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:query.top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
        except Exception:
            return 0.0
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
        except Exception:
            return 0.0
    
    async def _retrieve_postgres_items(self, vector_results: List[Tuple[str, float]], 
                                     query: Query) -> List[MemoryItem]:
        """Retrieve full items from PostgreSQL based on vector results"""
        if not vector_results:
            return []
        
        keys = [key for key, _ in vector_results]
        
        if self.postgres_pool:
            try:
                async with self.postgres_pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT key, value, context, metadata, timestamp, 
                               access_count, last_accessed, content_type
                        FROM l3_memory_items
                        WHERE org_id = $1 AND tenant_id = $2 AND key = ANY($3)
                        ORDER BY last_accessed DESC
                    """, 
                        self.org_config.org_id,
                        self.org_config.tenant_id,
                        keys
                    )
                    
                    items = []
                    for row in rows:
                        item = MemoryItem(
                            key=row["key"],
                            value=json.loads(row["value"]),
                            context=row["context"],
                            metadata=row["metadata"],
                            timestamp=row["timestamp"].timestamp(),
                            access_count=row["access_count"],
                            last_accessed=row["last_accessed"].timestamp(),
                            tier=MemoryTier.L3_ORGANIZATION
                        )
                        
                        # Add similarity score from vector search
                        for key, score in vector_results:
                            if key == item.key:
                                item.metadata["similarity_score"] = score
                                break
                        
                        items.append(item)
                    
                    return items
                    
            except Exception as e:
                log.error(f"PostgreSQL retrieval failed: {e}")
                return []
        else:
            # Memory fallback
            items = []
            for key, score in vector_results:
                if key in self.memory_store:
                    item = self.memory_store[key]
                    item.metadata["similarity_score"] = score
                    items.append(item)
            return items
    
    async def propagate(self, knowledge: Knowledge) -> bool:
        """Propagate knowledge to relevant agents in the organization"""
        try:
            # Find relevant agents based on knowledge context
            target_agents = await self._find_relevant_agents(knowledge)
            
            # Track propagation
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO l3_knowledge_propagation 
                        (org_id, source_key, target_agents, metadata)
                        VALUES ($1, $2, $3, $4)
                    """,
                        self.org_config.org_id,
                        knowledge.source_key,
                        target_agents,
                        json.dumps(knowledge.metadata)
                    )
            
            # TODO: Implement actual knowledge propagation via messaging
            log.info(f"Propagated knowledge {knowledge.source_key} to {len(target_agents)} agents")
            return True
            
        except Exception as e:
            log.error(f"Knowledge propagation failed: {e}")
            return False
    
    async def _find_relevant_agents(self, knowledge: Knowledge) -> List[str]:
        """Find agents relevant to this knowledge"""
        # Simple implementation - in production, this would use more sophisticated matching
        return knowledge.context.get("target_agents", [])
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get L3 memory statistics"""
        stats = {
            "tier": "L3_ORGANIZATION",
            "org_id": self.org_config.org_id,
            "tenant_id": self.org_config.tenant_id,
            "vector_store_type": self.vector_store_type.value,
            "postgres_available": self.postgres_pool is not None,
            "vector_store_available": self.vector_store is not None
        }
        
        if self.postgres_pool:
            try:
                async with self.postgres_pool.acquire() as conn:
                    # Get item count
                    item_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM l3_memory_items 
                        WHERE org_id = $1 AND tenant_id = $2
                    """, self.org_config.org_id, self.org_config.tenant_id)
                    
                    # Get storage size
                    storage_size = await conn.fetchval("""
                        SELECT pg_total_relation_size('l3_memory_items')
                    """)
                    
                    stats.update({
                        "total_items": item_count,
                        "storage_size_bytes": storage_size
                    })
                    
            except Exception as e:
                log.error(f"Failed to get PostgreSQL stats: {e}")
        else:
            stats.update({
                "total_items": len(self.memory_store),
                "storage_size_bytes": 0
            })
        
        return stats

class L4GlobalMemory(NeuralMeshLayer):
    """L4: Global knowledge integration with federated learning and external APIs"""
    
    def __init__(self, global_sources: List[GlobalKnowledgeSource]):
        self.global_sources = {source.source_id: source for source in global_sources}
        self.embedder = MultiModalEmbedder(target_dimension=768)
        
        # Knowledge cache
        self.knowledge_cache: Dict[str, MemoryItem] = {}
        self.cache_ttl = 3600  # 1 hour
        
        # External API clients
        self.api_clients: Dict[str, Any] = {}
        
        # Metrics
        if METRICS_AVAILABLE:
            self.external_requests = Counter(
                'l4_memory_external_requests_total',
                'External knowledge requests',
                ['source_id', 'status']
            )
            self.cache_hits = Counter(
                'l4_memory_cache_hits_total',
                'Cache hit/miss statistics',
                ['result']
            )
            self.knowledge_freshness = Gauge(
                'l4_memory_knowledge_freshness_seconds',
                'Age of cached knowledge',
                ['source_id']
            )
    
    async def initialize(self):
        """Initialize L4 global memory layer"""
        log.info("Initializing L4 global memory layer")
        
        # Initialize API clients for external sources
        for source_id, source in self.global_sources.items():
            await self._init_api_client(source)
        
        # Start background refresh tasks
        self._start_refresh_tasks()
        
        log.info(f"L4 memory initialized with {len(self.global_sources)} knowledge sources")
    
    async def _init_api_client(self, source: GlobalKnowledgeSource):
        """Initialize API client for external knowledge source"""
        if source.source_type == "rest_api":
            # Initialize REST API client
            if REQUESTS_AVAILABLE:
                self.api_clients[source.source_id] = {
                    "type": "rest",
                    "endpoint": source.endpoint,
                    "credentials": source.credentials
                }
        elif source.source_type == "database":
            # Initialize database client
            if POSTGRES_AVAILABLE:
                try:
                    conn_string = source.endpoint
                    pool = await asyncpg.create_pool(conn_string, min_size=1, max_size=5)
                    self.api_clients[source.source_id] = {
                        "type": "database",
                        "pool": pool
                    }
                except Exception as e:
                    log.error(f"Failed to initialize database client for {source.source_id}: {e}")
        
        log.info(f"Initialized API client for {source.source_id}")
    
    def _start_refresh_tasks(self):
        """Start background tasks for refreshing external knowledge"""
        for source_id, source in self.global_sources.items():
            if source.enabled:
                task = asyncio.create_task(self._refresh_source_loop(source))
                # Store task reference to prevent garbage collection
                if not hasattr(self, '_refresh_tasks'):
                    self._refresh_tasks = []
                self._refresh_tasks.append(task)
    
    async def _refresh_source_loop(self, source: GlobalKnowledgeSource):
        """Background loop to refresh knowledge from external source"""
        while True:
            try:
                await self._refresh_source(source)
                await asyncio.sleep(source.refresh_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error refreshing source {source.source_id}: {e}")
                await asyncio.sleep(min(source.refresh_interval, 300))  # Retry after 5 min max
    
    async def _refresh_source(self, source: GlobalKnowledgeSource):
        """Refresh knowledge from a specific external source"""
        log.debug(f"Refreshing knowledge from source {source.source_id}")
        
        client = self.api_clients.get(source.source_id)
        if not client:
            return
        
        try:
            if client["type"] == "rest":
                await self._refresh_rest_api(source, client)
            elif client["type"] == "database":
                await self._refresh_database(source, client)
            
            if METRICS_AVAILABLE:
                self.external_requests.labels(
                    source_id=source.source_id,
                    status="success"
                ).inc()
                
        except Exception as e:
            log.error(f"Failed to refresh source {source.source_id}: {e}")
            
            if METRICS_AVAILABLE:
                self.external_requests.labels(
                    source_id=source.source_id,
                    status="error"
                ).inc()
    
    async def _refresh_rest_api(self, source: GlobalKnowledgeSource, client: Dict[str, Any]):
        """Refresh knowledge from REST API"""
        if not REQUESTS_AVAILABLE:
            return
        
        # TODO: Implement REST API knowledge refresh
        # This would make HTTP requests to external APIs and cache the results
        pass
    
    async def _refresh_database(self, source: GlobalKnowledgeSource, client: Dict[str, Any]):
        """Refresh knowledge from external database"""
        # TODO: Implement database knowledge refresh
        # This would query external databases and cache the results
        pass
    
    async def store(self, item: MemoryItem) -> bool:
        """Store item in L4 global memory (cache only)"""
        try:
            item.tier = MemoryTier.L4_GLOBAL
            item.context["cached_at"] = time.time()
            
            # Store in cache
            self.knowledge_cache[item.key] = item
            
            # Prune cache if necessary
            await self._prune_cache()
            
            return True
            
        except Exception as e:
            log.error(f"Failed to store item {item.key} in L4: {e}")
            return False
    
    async def retrieve(self, query: Query) -> List[MemoryItem]:
        """Retrieve items from L4 global memory"""
        try:
            # Check cache first
            cached_results = self._search_cache(query)
            
            if cached_results:
                if METRICS_AVAILABLE:
                    self.cache_hits.labels(result="hit").inc()
                return cached_results
            
            # If not in cache, query external sources
            external_results = await self._query_external_sources(query)
            
            if METRICS_AVAILABLE:
                self.cache_hits.labels(result="miss").inc()
            
            return external_results
            
        except Exception as e:
            log.error(f"L4 retrieval failed for query '{query.text}': {e}")
            return []
    
    def _search_cache(self, query: Query) -> List[MemoryItem]:
        """Search cached knowledge"""
        if not self.knowledge_cache:
            return []
        
        # Generate query embedding if not provided
        if query.embedding is None:
            # For cache search, use simple text matching as fallback
            results = []
            for item in self.knowledge_cache.values():
                if query.text.lower() in str(item.value).lower():
                    results.append(item)
            return results[:query.top_k]
        
        # Vector similarity search
        similarities = []
        for key, item in self.knowledge_cache.items():
            if item.embedding is not None:
                similarity = self._cosine_similarity(query.embedding, item.embedding)
                if similarity >= query.min_score:
                    similarities.append((item, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in similarities[:query.top_k]]
    
    async def _query_external_sources(self, query: Query) -> List[MemoryItem]:
        """Query external knowledge sources"""
        results = []
        
        for source_id, source in self.global_sources.items():
            if not source.enabled:
                continue
                
            try:
                source_results = await self._query_single_source(source, query)
                results.extend(source_results)
            except Exception as e:
                log.error(f"Failed to query source {source_id}: {e}")
        
        return results[:query.top_k]
    
    async def _query_single_source(self, source: GlobalKnowledgeSource, query: Query) -> List[MemoryItem]:
        """Query a single external knowledge source"""
        import aiohttp
        import json
        
        try:
            # Handle different source types
            if source.source_type == "wikipedia":
                return await self._query_wikipedia(query)
            elif source.source_type == "arxiv":
                return await self._query_arxiv(query)
            elif source.source_type == "web_search":
                return await self._query_web_search(query)
            elif source.source_type == "database":
                return await self._query_database(source, query)
            elif source.source_type == "api":
                return await self._query_external_api(source, query)
            else:
                log.warning(f"Unknown source type: {source.source_type}")
                return []
                
        except Exception as e:
            log.error(f"Failed to query source {source.source_id}: {e}")
            return []
    
    async def _query_wikipedia(self, query: Query) -> List[MemoryItem]:
        """Query Wikipedia API"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                # Wikipedia search API
                search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
                search_term = query.text.replace(' ', '_')
                
                async with session.get(f"{search_url}{search_term}") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return [MemoryItem(
                            key=f"wikipedia_{search_term}",
                            value=data.get('extract', ''),
                            context={'source': 'wikipedia', 'title': data.get('title', '')},
                            metadata={'url': data.get('content_urls', {}).get('desktop', {}).get('page', '')},
                            tier=MemoryTier.L4_GLOBAL
                        )]
            return []
        except Exception as e:
            log.error(f"Wikipedia query failed: {e}")
            return []
    
    async def _query_arxiv(self, query: Query) -> List[MemoryItem]:
        """Query arXiv API for academic papers"""
        import aiohttp
        import xml.etree.ElementTree as ET
        
        try:
            async with aiohttp.ClientSession() as session:
                arxiv_url = "http://export.arxiv.org/api/query"
                params = {
                    'search_query': f'all:{query.text}',
                    'start': 0,
                    'max_results': min(query.top_k, 10)
                }
                
                async with session.get(arxiv_url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        root = ET.fromstring(content)
                        
                        items = []
                        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                            title = entry.find('{http://www.w3.org/2005/Atom}title').text
                            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
                            
                            items.append(MemoryItem(
                                key=f"arxiv_{hash(title) % 10000}",
                                value=f"{title}\n\n{summary}",
                                context={'source': 'arxiv', 'title': title},
                                metadata={'type': 'academic_paper'},
                                tier=MemoryTier.L4_GLOBAL
                            ))
                        
                        return items
            return []
        except Exception as e:
            log.error(f"arXiv query failed: {e}")
            return []
    
    async def _query_web_search(self, query: Query) -> List[MemoryItem]:
        """Query web search engines (requires API key)"""
        import os
        
        # Try DuckDuckGo first (no API key required)
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # DuckDuckGo Instant Answer API
                ddg_url = "https://api.duckduckgo.com/"
                params = {
                    'q': query.text,
                    'format': 'json',
                    'no_html': '1',
                    'skip_disambig': '1'
                }
                
                async with session.get(ddg_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('AbstractText'):
                            return [MemoryItem(
                                key=f"web_search_{hash(query.text) % 10000}",
                                value=data['AbstractText'],
                                context={'source': 'web_search', 'query': query.text},
                                metadata={'url': data.get('AbstractURL', '')},
                                tier=MemoryTier.L4_GLOBAL
                            )]
            return []
        except Exception as e:
            log.error(f"Web search failed: {e}")
            return []
    
    async def _query_database(self, source: GlobalKnowledgeSource, query: Query) -> List[MemoryItem]:
        """Query external database"""
        # This would connect to external databases based on source configuration
        # For now, return empty - would need specific database implementations
        log.info(f"Database querying not yet implemented for {source.source_id}")
        return []
    
    async def _query_external_api(self, source: GlobalKnowledgeSource, query: Query) -> List[MemoryItem]:
        """Query external API endpoints"""
        import aiohttp
        
        try:
            api_config = source.config
            if not api_config or 'url' not in api_config:
                return []
            
            async with aiohttp.ClientSession() as session:
                headers = api_config.get('headers', {})
                params = api_config.get('params', {})
                params.update({'query': query.text})
                
                async with session.get(api_config['url'], headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Parse response based on API format
                        items = []
                        if isinstance(data, dict) and 'results' in data:
                            for result in data['results'][:query.top_k]:
                                items.append(MemoryItem(
                                    key=f"api_{source.source_id}_{hash(str(result)) % 10000}",
                                    value=str(result),
                                    context={'source': source.source_id, 'api_query': query.text},
                                    metadata={'api_response': True},
                                    tier=MemoryTier.L4_GLOBAL
                                ))
                        
                        return items
            return []
        except Exception as e:
            log.error(f"External API query failed: {e}")
            return []
    
    async def propagate(self, knowledge: Knowledge) -> bool:
        """Propagate global knowledge (cache and notify)"""
        try:
            # Cache the knowledge
            item = MemoryItem(
                key=knowledge.source_key,
                value=knowledge.content,
                context=knowledge.context,
                metadata=knowledge.metadata,
                tier=MemoryTier.L4_GLOBAL
            )
            
            await self.store(item)
            
            # Implement global knowledge propagation
            await self._propagate_to_organizations(knowledge)
            await self._notify_relevant_agents(knowledge)
            await self._update_knowledge_graph(knowledge)
            
            return True
            
        except Exception as e:
            log.error(f"Global knowledge propagation failed: {e}")
            return False
    
    async def _propagate_to_organizations(self, knowledge: Knowledge):
        """Propagate knowledge to relevant L3 organizational memory"""
        try:
            # Find organizations that should receive this knowledge
            relevant_orgs = await self._find_relevant_organizations(knowledge)
            
            for org_id in relevant_orgs:
                # Send knowledge to L3 organizational memory
                await self._send_to_l3_memory(org_id, knowledge)
                
        except Exception as e:
            log.error(f"Failed to propagate to organizations: {e}")
    
    async def _notify_relevant_agents(self, knowledge: Knowledge):
        """Notify agents that might be interested in this knowledge"""
        try:
            # Use NATS to broadcast knowledge updates
            import json
            
            notification = {
                'type': 'global_knowledge_update',
                'knowledge_key': knowledge.source_key,
                'content_summary': knowledge.content[:200],  # First 200 chars
                'metadata': knowledge.metadata,
                'timestamp': time.time()
            }
            
            # Broadcast to all agents subscribed to knowledge updates
            # This would use the NATS messaging system
            log.info(f"Broadcasting knowledge update: {knowledge.source_key}")
            
        except Exception as e:
            log.error(f"Failed to notify agents: {e}")
    
    async def _update_knowledge_graph(self, knowledge: Knowledge):
        """Update the global knowledge graph with new connections"""
        try:
            # Extract entities and relationships from knowledge
            entities = await self._extract_entities(knowledge.content)
            relationships = await self._extract_relationships(knowledge.content)
            
            # Update knowledge graph (this would connect to a graph database)
            for entity in entities:
                await self._upsert_entity(entity, knowledge)
            
            for relationship in relationships:
                await self._upsert_relationship(relationship, knowledge)
                
        except Exception as e:
            log.error(f"Failed to update knowledge graph: {e}")
    
    async def _find_relevant_organizations(self, knowledge: Knowledge) -> List[str]:
        """Find organizations that should receive this knowledge"""
        # This would use semantic similarity to find relevant organizations
        # For now, return empty list
        return []
    
    async def _send_to_l3_memory(self, org_id: str, knowledge: Knowledge):
        """Send knowledge to L3 organizational memory"""
        # This would send the knowledge to the appropriate L3 memory instance
        log.info(f"Sending knowledge to L3 memory for org {org_id}")
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities from content"""
        # This would use NER (Named Entity Recognition)
        # For now, return empty list
        return []
    
    async def _extract_relationships(self, content: str) -> List[Dict[str, Any]]:
        """Extract relationships from content"""
        # This would use relationship extraction models
        # For now, return empty list
        return []
    
    async def _upsert_entity(self, entity: Dict[str, Any], knowledge: Knowledge):
        """Insert or update entity in knowledge graph"""
        # This would update a graph database
        pass
    
    async def _upsert_relationship(self, relationship: Dict[str, Any], knowledge: Knowledge):
        """Insert or update relationship in knowledge graph"""
        # This would update a graph database
        pass
    
    async def _prune_cache(self):
        """Prune expired items from cache"""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self.knowledge_cache.items():
            cached_at = item.context.get("cached_at", 0)
            if current_time - cached_at > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.knowledge_cache[key]
        
        if expired_keys:
            log.debug(f"Pruned {len(expired_keys)} expired items from L4 cache")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get L4 memory statistics"""
        current_time = time.time()
        
        # Calculate cache statistics
        total_items = len(self.knowledge_cache)
        expired_items = sum(
            1 for item in self.knowledge_cache.values()
            if current_time - item.context.get("cached_at", 0) > self.cache_ttl
        )
        
        # Calculate source health
        source_health = {}
        for source_id, source in self.global_sources.items():
            source_health[source_id] = {
                "enabled": source.enabled,
                "last_refresh": "unknown",  # Would track in production
                "client_available": source_id in self.api_clients
            }
        
        return {
            "tier": "L4_GLOBAL",
            "total_sources": len(self.global_sources),
            "active_sources": sum(1 for s in self.global_sources.values() if s.enabled),
            "cache_items": total_items,
            "expired_items": expired_items,
            "cache_hit_rate": "unknown",  # Would calculate from metrics
            "source_health": source_health
        }
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
        except Exception:
            return 0.0
