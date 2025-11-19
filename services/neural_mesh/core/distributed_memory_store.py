#!/usr/bin/env python3
"""
Distributed Memory Store for Neural Mesh
Redis Cluster and PostgreSQL TimescaleDB integration for scalable memory architecture
"""

import asyncio
import json
import time
import logging
import hashlib
import pickle
import zlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import uuid

# Database imports
try:
    import redis.asyncio as redis
    from redis.asyncio.cluster import RedisCluster
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import asyncpg
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    import pymemcache
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False

log = logging.getLogger("distributed-memory-store")

class MemoryTier(Enum):
    """Memory tier levels"""
    L1 = "L1"  # Working memory (Redis, high-speed)
    L2 = "L2"  # Short-term memory (Redis, medium TTL)
    L3 = "L3"  # Long-term memory (PostgreSQL, persistent)
    L4 = "L4"  # Archive memory (PostgreSQL + compression)

class MemoryType(Enum):
    """Types of memory content"""
    CONVERSATIONAL = "conversational"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    CONTEXTUAL = "contextual"

class ConsistencyLevel(Enum):
    """Memory consistency levels"""
    EVENTUAL = "eventual"      # Best performance, eventual consistency
    STRONG = "strong"          # Strong consistency, higher latency
    CAUSAL = "causal"          # Causal consistency
    SESSION = "session"        # Session consistency

@dataclass
class MemoryEntry:
    """Individual memory entry"""
    memory_id: str
    agent_id: str
    memory_type: MemoryType
    memory_tier: MemoryTier
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    ttl: Optional[int] = None  # TTL in seconds
    compressed: bool = False
    checksum: str = ""
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for integrity validation"""
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

@dataclass
class MemoryPartition:
    """Memory partition configuration"""
    partition_id: str
    partition_type: str  # "agent", "conversation", "topic"
    partition_key: str
    redis_db: int
    postgresql_schema: str
    replication_factor: int = 3
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL

@dataclass
class SyncOperation:
    """Memory synchronization operation"""
    operation_id: str
    operation_type: str  # "create", "update", "delete", "merge"
    memory_id: str
    agent_id: str
    data: Dict[str, Any]
    vector_clock: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    applied: bool = False

class DistributedMemoryStore:
    """Distributed memory store with Redis Cluster and PostgreSQL TimescaleDB"""
    
    def __init__(self):
        # Redis cluster configuration
        self.redis_cluster = None
        self.redis_pools = {}
        
        # PostgreSQL configuration
        self.pg_pool = None
        self.timescale_enabled = False
        
        # Memcached fallback
        self.memcached_client = None
        
        # Memory partitions
        self.partitions: Dict[str, MemoryPartition] = {}
        
        # Synchronization
        self.sync_queue = asyncio.Queue()
        self.vector_clocks: Dict[str, Dict[str, int]] = {}
        self.pending_operations: Dict[str, SyncOperation] = {}
        
        # Configuration
        self.compression_threshold = 1024  # Compress entries > 1KB
        self.default_ttl = {
            MemoryTier.L1: 300,     # 5 minutes
            MemoryTier.L2: 3600,    # 1 hour
            MemoryTier.L3: None,    # Persistent
            MemoryTier.L4: None     # Persistent
        }
        
        # Initialize (defer async initialization)
        self._initialized = False
    
    async def _initialize_async(self):
        """Initialize distributed memory store"""
        if self._initialized:
            return
            
        try:
            await self._initialize_redis_cluster()
            await self._initialize_postgresql()
            await self._initialize_partitions()
            await self._start_sync_workers()
            
            self._initialized = True
            log.info("✅ Distributed memory store initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize distributed memory store: {e}")
    
    async def ensure_initialized(self):
        """Ensure the store is initialized"""
        if not self._initialized:
            await self._initialize_async()
    
    async def _initialize_redis_cluster(self):
        """Initialize Redis cluster for high-speed memory tiers"""
        
        if not REDIS_AVAILABLE:
            log.warning("Redis not available, using fallback storage")
            return
        
        try:
            # Redis cluster configuration
            redis_nodes = os.getenv("REDIS_CLUSTER_NODES", "localhost:7000,localhost:7001,localhost:7002").split(",")
            
            if len(redis_nodes) > 1:
                # Use Redis Cluster
                startup_nodes = [{"host": node.split(":")[0], "port": int(node.split(":")[1])} for node in redis_nodes]
                self.redis_cluster = RedisCluster(
                    startup_nodes=startup_nodes,
                    decode_responses=True,
                    skip_full_coverage_check=True,
                    health_check_interval=30
                )
                
                # Test cluster connection
                await self.redis_cluster.ping()
                log.info(f"✅ Redis cluster initialized with {len(redis_nodes)} nodes")
            else:
                # Single Redis instance
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                self.redis_cluster = redis.from_url(redis_url, decode_responses=True)
                await self.redis_cluster.ping()
                log.info("✅ Single Redis instance initialized")
                
        except Exception as e:
            log.error(f"Failed to initialize Redis: {e}")
            # Try Memcached fallback
            await self._initialize_memcached_fallback()
    
    async def _initialize_memcached_fallback(self):
        """Initialize Memcached as fallback"""
        
        if not MEMCACHED_AVAILABLE:
            log.warning("Memcached not available, using in-memory fallback")
            return
        
        try:
            memcached_servers = os.getenv("MEMCACHED_SERVERS", "localhost:11211").split(",")
            self.memcached_client = pymemcache.Client(
                (memcached_servers[0].split(":")[0], int(memcached_servers[0].split(":")[1])),
                serializer=pymemcache.serde.json_serde
            )
            log.info("✅ Memcached fallback initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize Memcached: {e}")
    
    async def _initialize_postgresql(self):
        """Initialize PostgreSQL with TimescaleDB for persistent memory"""
        
        if not POSTGRESQL_AVAILABLE:
            log.warning("PostgreSQL not available, using file-based fallback")
            return
        
        try:
            database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/agentforge")
            
            # Create connection pool
            self.pg_pool = await asyncpg.create_pool(
                database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Initialize database schema
            await self._initialize_memory_schema()
            
            # Check for TimescaleDB extension
            await self._check_timescaledb()
            
            log.info("✅ PostgreSQL memory store initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize PostgreSQL: {e}")
    
    async def _initialize_memory_schema(self):
        """Initialize memory database schema"""
        
        schema_sql = """
        -- Memory entries table
        CREATE TABLE IF NOT EXISTS memory_entries (
            memory_id VARCHAR(64) PRIMARY KEY,
            agent_id VARCHAR(64) NOT NULL,
            memory_type VARCHAR(32) NOT NULL,
            memory_tier VARCHAR(8) NOT NULL,
            content JSONB NOT NULL,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            version INTEGER DEFAULT 1,
            ttl_expires_at TIMESTAMPTZ,
            compressed BOOLEAN DEFAULT FALSE,
            checksum VARCHAR(64) NOT NULL,
            partition_key VARCHAR(128)
        );
        
        -- Memory synchronization log
        CREATE TABLE IF NOT EXISTS memory_sync_log (
            sync_id VARCHAR(64) PRIMARY KEY,
            operation_type VARCHAR(32) NOT NULL,
            memory_id VARCHAR(64) NOT NULL,
            agent_id VARCHAR(64) NOT NULL,
            vector_clock JSONB NOT NULL,
            sync_data JSONB NOT NULL,
            applied BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            applied_at TIMESTAMPTZ
        );
        
        -- Memory access patterns (for optimization)
        CREATE TABLE IF NOT EXISTS memory_access_patterns (
            access_id VARCHAR(64) PRIMARY KEY,
            memory_id VARCHAR(64) NOT NULL,
            agent_id VARCHAR(64) NOT NULL,
            access_type VARCHAR(32) NOT NULL,
            access_frequency INTEGER DEFAULT 1,
            last_accessed TIMESTAMPTZ DEFAULT NOW(),
            access_context JSONB DEFAULT '{}'
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_memory_entries_agent_id ON memory_entries(agent_id);
        CREATE INDEX IF NOT EXISTS idx_memory_entries_type ON memory_entries(memory_type);
        CREATE INDEX IF NOT EXISTS idx_memory_entries_tier ON memory_entries(memory_tier);
        CREATE INDEX IF NOT EXISTS idx_memory_entries_created_at ON memory_entries(created_at);
        CREATE INDEX IF NOT EXISTS idx_memory_entries_partition ON memory_entries(partition_key);
        
        CREATE INDEX IF NOT EXISTS idx_sync_log_memory_id ON memory_sync_log(memory_id);
        CREATE INDEX IF NOT EXISTS idx_sync_log_agent_id ON memory_sync_log(agent_id);
        CREATE INDEX IF NOT EXISTS idx_sync_log_created_at ON memory_sync_log(created_at);
        
        CREATE INDEX IF NOT EXISTS idx_access_patterns_memory_id ON memory_access_patterns(memory_id);
        CREATE INDEX IF NOT EXISTS idx_access_patterns_agent_id ON memory_access_patterns(agent_id);
        """
        
        async with self.pg_pool.acquire() as conn:
            await conn.execute(schema_sql)
    
    async def _check_timescaledb(self):
        """Check and enable TimescaleDB extension"""
        
        try:
            async with self.pg_pool.acquire() as conn:
                # Check if TimescaleDB is available
                result = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
                )
                
                if result:
                    self.timescale_enabled = True
                    
                    # Convert memory tables to hypertables
                    await conn.execute(
                        "SELECT create_hypertable('memory_entries', 'created_at', if_not_exists => TRUE)"
                    )
                    await conn.execute(
                        "SELECT create_hypertable('memory_sync_log', 'created_at', if_not_exists => TRUE)"
                    )
                    await conn.execute(
                        "SELECT create_hypertable('memory_access_patterns', 'last_accessed', if_not_exists => TRUE)"
                    )
                    
                    log.info("✅ TimescaleDB enabled for memory tables")
                else:
                    log.info("TimescaleDB not available, using standard PostgreSQL")
                    
        except Exception as e:
            log.warning(f"TimescaleDB setup failed: {e}")
    
    async def _initialize_partitions(self):
        """Initialize memory partitions"""
        
        # Default partitions
        default_partitions = [
            MemoryPartition(
                partition_id="agents",
                partition_type="agent",
                partition_key="agent_id",
                redis_db=0,
                postgresql_schema="agent_memory"
            ),
            MemoryPartition(
                partition_id="conversations",
                partition_type="conversation",
                partition_key="conversation_id",
                redis_db=1,
                postgresql_schema="conversation_memory"
            ),
            MemoryPartition(
                partition_id="topics",
                partition_type="topic",
                partition_key="topic_hash",
                redis_db=2,
                postgresql_schema="topic_memory"
            ),
            MemoryPartition(
                partition_id="global",
                partition_type="global",
                partition_key="global",
                redis_db=3,
                postgresql_schema="global_memory"
            )
        ]
        
        for partition in default_partitions:
            self.partitions[partition.partition_id] = partition
            
            # Create PostgreSQL schema if needed
            if self.pg_pool:
                try:
                    async with self.pg_pool.acquire() as conn:
                        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {partition.postgresql_schema}")
                except Exception as e:
                    log.warning(f"Failed to create schema {partition.postgresql_schema}: {e}")
        
        log.info(f"Initialized {len(self.partitions)} memory partitions")
    
    async def _start_sync_workers(self):
        """Start synchronization worker tasks"""
        
        # Start sync workers
        asyncio.create_task(self._sync_worker())
        asyncio.create_task(self._cleanup_worker())
        asyncio.create_task(self._replication_worker())
        
        log.info("✅ Memory synchronization workers started")
    
    async def store_memory(
        self,
        agent_id: str,
        memory_type: MemoryType,
        memory_tier: MemoryTier,
        content: Dict[str, Any],
        metadata: Dict[str, Any] = None,
        ttl: Optional[int] = None,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    ) -> str:
        """Store memory entry in distributed store"""
        
        try:
            # Generate memory ID
            memory_id = self._generate_memory_id(agent_id, memory_type, content)
            
            # Create memory entry
            entry = MemoryEntry(
                memory_id=memory_id,
                agent_id=agent_id,
                memory_type=memory_type,
                memory_tier=memory_tier,
                content=content,
                metadata=metadata or {},
                ttl=ttl or self.default_ttl.get(memory_tier)
            )
            
            # Determine partition
            partition = self._get_partition_for_entry(entry)
            
            # Compress if needed
            if len(json.dumps(content)) > self.compression_threshold:
                entry.content = self._compress_content(content)
                entry.compressed = True
            
            # Store based on memory tier
            if memory_tier in [MemoryTier.L1, MemoryTier.L2]:
                # Store in Redis for fast access
                await self._store_in_redis(entry, partition, consistency_level)
            
            if memory_tier in [MemoryTier.L3, MemoryTier.L4]:
                # Store in PostgreSQL for persistence
                await self._store_in_postgresql(entry, partition)
            
            # Create sync operation
            sync_op = SyncOperation(
                operation_id=str(uuid.uuid4()),
                operation_type="create",
                memory_id=memory_id,
                agent_id=agent_id,
                data=asdict(entry),
                vector_clock=self._get_vector_clock(agent_id)
            )
            
            # Queue for synchronization
            await self.sync_queue.put(sync_op)
            
            log.debug(f"Stored memory {memory_id} for agent {agent_id}")
            return memory_id
            
        except Exception as e:
            log.error(f"Error storing memory: {e}")
            raise
    
    async def retrieve_memory(
        self,
        memory_id: str,
        agent_id: Optional[str] = None,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    ) -> Optional[MemoryEntry]:
        """Retrieve memory entry from distributed store"""
        
        try:
            # Try Redis first (L1/L2 tiers)
            if self.redis_cluster:
                entry = await self._retrieve_from_redis(memory_id)
                if entry:
                    await self._record_access(memory_id, agent_id, "redis_hit")
                    return entry
            
            # Try PostgreSQL (L3/L4 tiers)
            if self.pg_pool:
                entry = await self._retrieve_from_postgresql(memory_id)
                if entry:
                    await self._record_access(memory_id, agent_id, "postgresql_hit")
                    
                    # Promote to Redis if frequently accessed
                    if await self._should_promote_to_redis(memory_id):
                        await self._promote_to_redis(entry)
                    
                    return entry
            
            log.debug(f"Memory {memory_id} not found")
            return None
            
        except Exception as e:
            log.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    async def update_memory(
        self,
        memory_id: str,
        agent_id: str,
        updates: Dict[str, Any],
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    ) -> bool:
        """Update existing memory entry"""
        
        try:
            # Retrieve current entry
            current_entry = await self.retrieve_memory(memory_id, agent_id)
            if not current_entry:
                raise ValueError(f"Memory {memory_id} not found")
            
            # Create updated entry
            updated_content = {**current_entry.content, **updates}
            updated_entry = MemoryEntry(
                memory_id=memory_id,
                agent_id=agent_id,
                memory_type=current_entry.memory_type,
                memory_tier=current_entry.memory_tier,
                content=updated_content,
                metadata=current_entry.metadata,
                created_at=current_entry.created_at,
                updated_at=time.time(),
                version=current_entry.version + 1,
                ttl=current_entry.ttl,
                compressed=current_entry.compressed
            )
            
            # Store updated entry
            partition = self._get_partition_for_entry(updated_entry)
            
            if updated_entry.memory_tier in [MemoryTier.L1, MemoryTier.L2]:
                await self._store_in_redis(updated_entry, partition, consistency_level)
            
            if updated_entry.memory_tier in [MemoryTier.L3, MemoryTier.L4]:
                await self._store_in_postgresql(updated_entry, partition)
            
            # Create sync operation
            sync_op = SyncOperation(
                operation_id=str(uuid.uuid4()),
                operation_type="update",
                memory_id=memory_id,
                agent_id=agent_id,
                data=asdict(updated_entry),
                vector_clock=self._increment_vector_clock(agent_id)
            )
            
            await self.sync_queue.put(sync_op)
            
            log.debug(f"Updated memory {memory_id}")
            return True
            
        except Exception as e:
            log.error(f"Error updating memory {memory_id}: {e}")
            return False
    
    async def search_memories(
        self,
        agent_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        memory_tier: Optional[MemoryTier] = None,
        query: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[MemoryEntry]:
        """Search memories with various filters"""
        
        try:
            memories = []
            
            # Search in PostgreSQL for comprehensive results
            if self.pg_pool:
                memories.extend(await self._search_postgresql(
                    agent_id, memory_type, memory_tier, query, limit, offset
                ))
            
            # Search in Redis for recent memories
            if self.redis_cluster and len(memories) < limit:
                redis_memories = await self._search_redis(
                    agent_id, memory_type, memory_tier, limit - len(memories)
                )
                memories.extend(redis_memories)
            
            # Sort by relevance and recency
            memories.sort(key=lambda m: (m.updated_at, m.created_at), reverse=True)
            
            return memories[:limit]
            
        except Exception as e:
            log.error(f"Error searching memories: {e}")
            return []
    
    async def _store_in_redis(
        self,
        entry: MemoryEntry,
        partition: MemoryPartition,
        consistency_level: ConsistencyLevel
    ):
        """Store memory entry in Redis"""
        
        if not self.redis_cluster:
            return
        
        try:
            # Serialize entry
            entry_data = json.dumps(asdict(entry))
            
            # Store in Redis
            key = f"memory:{entry.memory_tier.value}:{entry.memory_id}"
            
            if entry.ttl:
                await self.redis_cluster.setex(key, entry.ttl, entry_data)
            else:
                await self.redis_cluster.set(key, entry_data)
            
            # Store in partition-specific key for efficient queries
            partition_key = f"partition:{partition.partition_id}:{entry.agent_id}"
            await self.redis_cluster.sadd(partition_key, entry.memory_id)
            
            # Set partition TTL if entry has TTL
            if entry.ttl:
                await self.redis_cluster.expire(partition_key, entry.ttl)
            
        except Exception as e:
            log.error(f"Error storing in Redis: {e}")
            raise
    
    async def _store_in_postgresql(
        self,
        entry: MemoryEntry,
        partition: MemoryPartition
    ):
        """Store memory entry in PostgreSQL"""
        
        if not self.pg_pool:
            return
        
        try:
            # Calculate TTL expiration
            ttl_expires_at = None
            if entry.ttl:
                ttl_expires_at = datetime.fromtimestamp(entry.created_at + entry.ttl)
            
            # Insert or update
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO memory_entries (
                        memory_id, agent_id, memory_type, memory_tier,
                        content, metadata, created_at, updated_at,
                        version, ttl_expires_at, compressed, checksum, partition_key
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (memory_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at,
                        version = EXCLUDED.version,
                        ttl_expires_at = EXCLUDED.ttl_expires_at,
                        compressed = EXCLUDED.compressed,
                        checksum = EXCLUDED.checksum
                """, 
                    entry.memory_id,
                    entry.agent_id,
                    entry.memory_type.value,
                    entry.memory_tier.value,
                    json.dumps(entry.content),
                    json.dumps(entry.metadata),
                    datetime.fromtimestamp(entry.created_at),
                    datetime.fromtimestamp(entry.updated_at),
                    entry.version,
                    ttl_expires_at,
                    entry.compressed,
                    entry.checksum,
                    partition.partition_key
                )
                
        except Exception as e:
            log.error(f"Error storing in PostgreSQL: {e}")
            raise
    
    async def _retrieve_from_redis(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve memory from Redis"""
        
        if not self.redis_cluster:
            return None
        
        try:
            # Try different tier keys
            for tier in MemoryTier:
                key = f"memory:{tier.value}:{memory_id}"
                data = await self.redis_cluster.get(key)
                
                if data:
                    entry_dict = json.loads(data)
                    
                    # Decompress if needed
                    if entry_dict.get("compressed", False):
                        entry_dict["content"] = self._decompress_content(entry_dict["content"])
                        entry_dict["compressed"] = False
                    
                    # Convert back to MemoryEntry
                    entry_dict["memory_type"] = MemoryType(entry_dict["memory_type"])
                    entry_dict["memory_tier"] = MemoryTier(entry_dict["memory_tier"])
                    
                    return MemoryEntry(**entry_dict)
            
            return None
            
        except Exception as e:
            log.error(f"Error retrieving from Redis: {e}")
            return None
    
    async def _retrieve_from_postgresql(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve memory from PostgreSQL"""
        
        if not self.pg_pool:
            return None
        
        try:
            async with self.pg_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT memory_id, agent_id, memory_type, memory_tier,
                           content, metadata, created_at, updated_at,
                           version, ttl_expires_at, compressed, checksum
                    FROM memory_entries
                    WHERE memory_id = $1
                    AND (ttl_expires_at IS NULL OR ttl_expires_at > NOW())
                """, memory_id)
                
                if row:
                    content = json.loads(row["content"])
                    
                    # Decompress if needed
                    if row["compressed"]:
                        content = self._decompress_content(content)
                    
                    return MemoryEntry(
                        memory_id=row["memory_id"],
                        agent_id=row["agent_id"],
                        memory_type=MemoryType(row["memory_type"]),
                        memory_tier=MemoryTier(row["memory_tier"]),
                        content=content,
                        metadata=json.loads(row["metadata"]),
                        created_at=row["created_at"].timestamp(),
                        updated_at=row["updated_at"].timestamp(),
                        version=row["version"],
                        compressed=False,  # Already decompressed
                        checksum=row["checksum"]
                    )
            
            return None
            
        except Exception as e:
            log.error(f"Error retrieving from PostgreSQL: {e}")
            return None
    
    def _compress_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Compress content for storage"""
        
        try:
            content_bytes = json.dumps(content).encode('utf-8')
            compressed_bytes = zlib.compress(content_bytes)
            
            return {
                "compressed_data": compressed_bytes.hex(),
                "original_size": len(content_bytes),
                "compressed_size": len(compressed_bytes),
                "compression_ratio": len(compressed_bytes) / len(content_bytes)
            }
            
        except Exception as e:
            log.error(f"Error compressing content: {e}")
            return content
    
    def _decompress_content(self, compressed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress content from storage"""
        
        try:
            if "compressed_data" in compressed_content:
                compressed_bytes = bytes.fromhex(compressed_content["compressed_data"])
                decompressed_bytes = zlib.decompress(compressed_bytes)
                return json.loads(decompressed_bytes.decode('utf-8'))
            else:
                return compressed_content
                
        except Exception as e:
            log.error(f"Error decompressing content: {e}")
            return compressed_content
    
    def _get_partition_for_entry(self, entry: MemoryEntry) -> MemoryPartition:
        """Get appropriate partition for memory entry"""
        
        # Determine partition based on content and metadata
        if "conversation_id" in entry.metadata:
            return self.partitions["conversations"]
        elif entry.memory_type == MemoryType.SEMANTIC:
            return self.partitions["topics"]
        elif entry.agent_id:
            return self.partitions["agents"]
        else:
            return self.partitions["global"]
    
    def _generate_memory_id(
        self,
        agent_id: str,
        memory_type: MemoryType,
        content: Dict[str, Any]
    ) -> str:
        """Generate unique memory ID"""
        
        # Create deterministic ID based on content
        content_str = json.dumps(content, sort_keys=True)
        hash_input = f"{agent_id}:{memory_type.value}:{content_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]
    
    def _get_vector_clock(self, agent_id: str) -> Dict[str, int]:
        """Get current vector clock for agent"""
        
        if agent_id not in self.vector_clocks:
            self.vector_clocks[agent_id] = {}
        
        return self.vector_clocks[agent_id].copy()
    
    def _increment_vector_clock(self, agent_id: str) -> Dict[str, int]:
        """Increment vector clock for agent"""
        
        if agent_id not in self.vector_clocks:
            self.vector_clocks[agent_id] = {}
        
        self.vector_clocks[agent_id][agent_id] = self.vector_clocks[agent_id].get(agent_id, 0) + 1
        return self.vector_clocks[agent_id].copy()
    
    async def _sync_worker(self):
        """Worker for processing synchronization operations"""
        
        while True:
            try:
                # Get sync operation from queue
                sync_op = await self.sync_queue.get()
                
                # Process synchronization
                await self._process_sync_operation(sync_op)
                
                # Mark as done
                self.sync_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in sync worker: {e}")
                await asyncio.sleep(1)
    
    async def _process_sync_operation(self, sync_op: SyncOperation):
        """Process individual sync operation"""
        
        try:
            # Check for conflicts using vector clocks
            conflict = await self._detect_conflict(sync_op)
            
            if conflict:
                # Resolve conflict
                resolved_op = await self._resolve_conflict(sync_op, conflict)
                if resolved_op:
                    sync_op = resolved_op
                else:
                    log.warning(f"Could not resolve conflict for {sync_op.memory_id}")
                    return
            
            # Apply operation
            if sync_op.operation_type == "create":
                await self._apply_create_operation(sync_op)
            elif sync_op.operation_type == "update":
                await self._apply_update_operation(sync_op)
            elif sync_op.operation_type == "delete":
                await self._apply_delete_operation(sync_op)
            elif sync_op.operation_type == "merge":
                await self._apply_merge_operation(sync_op)
            
            # Mark as applied
            sync_op.applied = True
            
            # Log sync operation
            await self._log_sync_operation(sync_op)
            
        except Exception as e:
            log.error(f"Error processing sync operation {sync_op.operation_id}: {e}")
    
    async def _cleanup_worker(self):
        """Worker for cleaning up expired memories"""
        
        while True:
            try:
                # Clean up Redis expired keys
                if self.redis_cluster:
                    await self._cleanup_redis_expired()
                
                # Clean up PostgreSQL expired entries
                if self.pg_pool:
                    await self._cleanup_postgresql_expired()
                
                # Sleep for cleanup interval
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in cleanup worker: {e}")
                await asyncio.sleep(60)
    
    async def _replication_worker(self):
        """Worker for cross-datacenter replication"""
        
        while True:
            try:
                # Check replication health
                await self._check_replication_health()
                
                # Perform replication sync
                await self._perform_replication_sync()
                
                # Sleep for replication interval
                await asyncio.sleep(60)  # 1 minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in replication worker: {e}")
                await asyncio.sleep(30)
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        
        stats = {
            "timestamp": time.time(),
            "redis_stats": {},
            "postgresql_stats": {},
            "partition_stats": {},
            "sync_stats": {},
            "performance_stats": {}
        }
        
        # Redis statistics
        if self.redis_cluster:
            try:
                redis_info = await self.redis_cluster.info()
                stats["redis_stats"] = {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory": redis_info.get("used_memory_human", "0B"),
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                    "ops_per_sec": redis_info.get("instantaneous_ops_per_sec", 0)
                }
            except Exception as e:
                log.error(f"Error getting Redis stats: {e}")
        
        # PostgreSQL statistics
        if self.pg_pool:
            try:
                async with self.pg_pool.acquire() as conn:
                    # Count memories by tier
                    tier_counts = await conn.fetch("""
                        SELECT memory_tier, COUNT(*) as count
                        FROM memory_entries
                        WHERE ttl_expires_at IS NULL OR ttl_expires_at > NOW()
                        GROUP BY memory_tier
                    """)
                    
                    stats["postgresql_stats"] = {
                        "tier_counts": {row["memory_tier"]: row["count"] for row in tier_counts},
                        "total_memories": sum(row["count"] for row in tier_counts)
                    }
                    
            except Exception as e:
                log.error(f"Error getting PostgreSQL stats: {e}")
        
        # Partition statistics
        for partition_id, partition in self.partitions.items():
            try:
                if self.redis_cluster:
                    partition_key = f"partition:{partition_id}:*"
                    keys = await self.redis_cluster.keys(partition_key)
                    stats["partition_stats"][partition_id] = len(keys)
            except Exception as e:
                log.error(f"Error getting partition stats: {e}")
        
        # Synchronization statistics
        stats["sync_stats"] = {
            "pending_operations": self.sync_queue.qsize(),
            "vector_clocks": len(self.vector_clocks),
            "pending_sync_ops": len(self.pending_operations)
        }
        
        return stats
    
    async def create_memory_snapshot(
        self,
        snapshot_name: str,
        agent_ids: List[str] = None,
        memory_tiers: List[MemoryTier] = None
    ) -> str:
        """Create memory snapshot for backup/rollback"""
        
        try:
            snapshot_id = f"snapshot_{snapshot_name}_{int(time.time())}"
            
            # Query memories to include in snapshot
            memories = await self.search_memories(
                agent_id=None,  # All agents if agent_ids not specified
                memory_tier=None,  # All tiers if not specified
                limit=10000  # Large limit for snapshot
            )
            
            # Filter by criteria
            if agent_ids:
                memories = [m for m in memories if m.agent_id in agent_ids]
            
            if memory_tiers:
                memories = [m for m in memories if m.memory_tier in memory_tiers]
            
            # Create snapshot data
            snapshot_data = {
                "snapshot_id": snapshot_id,
                "snapshot_name": snapshot_name,
                "created_at": time.time(),
                "agent_ids": agent_ids,
                "memory_tiers": [t.value for t in memory_tiers] if memory_tiers else None,
                "memory_count": len(memories),
                "memories": [asdict(memory) for memory in memories]
            }
            
            # Store snapshot in PostgreSQL
            if self.pg_pool:
                async with self.pg_pool.acquire() as conn:
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS memory_snapshots (
                            snapshot_id VARCHAR(128) PRIMARY KEY,
                            snapshot_name VARCHAR(256),
                            snapshot_data JSONB,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    
                    await conn.execute("""
                        INSERT INTO memory_snapshots (snapshot_id, snapshot_name, snapshot_data)
                        VALUES ($1, $2, $3)
                    """, snapshot_id, snapshot_name, json.dumps(snapshot_data))
            
            log.info(f"Created memory snapshot {snapshot_id} with {len(memories)} memories")
            return snapshot_id
            
        except Exception as e:
            log.error(f"Error creating memory snapshot: {e}")
            raise

# Global instance
distributed_memory_store = DistributedMemoryStore()
