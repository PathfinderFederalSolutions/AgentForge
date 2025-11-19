"""
Distributed Memory System - Scalable L1/L2 Memory for Million-Agent Deployments
Addresses scalability bottlenecks with proper distributed data structures
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import hashlib
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List
from enum import Enum
import numpy as np
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

# Import base classes
from .enhanced_memory import MemoryItem, Query, Knowledge, MemoryTier

# Optional imports with fallbacks
try:
    import redis.asyncio as redis
    from redis.asyncio.cluster import RedisCluster
    from redis.exceptions import RedisClusterException, RedisError
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    RedisCluster = None
    RedisClusterException = None
    RedisError = None
    REDIS_AVAILABLE = False


# Metrics imports
try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = lambda *args, **kwargs: None

log = logging.getLogger("distributed-memory")

class PartitionStrategy(Enum):
    """Partitioning strategies for distributed memory"""
    CONSISTENT_HASH = "consistent_hash"
    RANGE_BASED = "range_based"
    AGENT_BASED = "agent_based"
    CONTENT_BASED = "content_based"

@dataclass
class MemoryPartition:
    """Represents a memory partition"""
    partition_id: str
    start_hash: int
    end_hash: int
    nodes: List[str]
    primary_node: str
    replica_count: int = 2
    
class ConsistentHashRing:
    """Consistent hashing implementation for distributed partitioning"""
    
    def __init__(self, nodes: List[str], virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []
        self.nodes = set()
        
        for node in nodes:
            self.add_node(node)
    
    def add_node(self, node: str):
        """Add a node to the hash ring"""
        self.nodes.add(node)
        for i in range(self.virtual_nodes):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
        
        self.sorted_keys = sorted(self.ring.keys())
        log.info(f"Added node {node} to consistent hash ring")
    
    def remove_node(self, node: str):
        """Remove a node from the hash ring"""
        if node not in self.nodes:
            return
            
        self.nodes.remove(node)
        for i in range(self.virtual_nodes):
            key = self._hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]
        
        self.sorted_keys = sorted(self.ring.keys())
        log.info(f"Removed node {node} from consistent hash ring")
    
    def get_node(self, key: str) -> str:
        """Get the node responsible for a key"""
        if not self.ring:
            raise ValueError("No nodes in hash ring")
        
        hash_key = self._hash(key)
        
        # Find the first node clockwise from the hash
        for ring_key in self.sorted_keys:
            if hash_key <= ring_key:
                return self.ring[ring_key]
        
        # Wrap around to the first node
        return self.ring[self.sorted_keys[0]]
    
    def get_nodes(self, key: str, count: int) -> List[str]:
        """Get multiple nodes for replication"""
        if not self.ring or count <= 0:
            return []
        
        hash_key = self._hash(key)
        nodes = []
        
        # Find starting position
        start_idx = 0
        for i, ring_key in enumerate(self.sorted_keys):
            if hash_key <= ring_key:
                start_idx = i
                break
        
        # Collect unique nodes
        seen_nodes = set()
        for i in range(len(self.sorted_keys)):
            idx = (start_idx + i) % len(self.sorted_keys)
            node = self.ring[self.sorted_keys[idx]]
            
            if node not in seen_nodes:
                nodes.append(node)
                seen_nodes.add(node)
                
                if len(nodes) >= count:
                    break
        
        return nodes
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing"""
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)

class DistributedL1Memory:
    """Distributed L1 Memory with consistent hashing and intelligent caching"""
    
    def __init__(self, agent_id: str, cluster_nodes: List[str], max_local_items: int = 10000):
        self.agent_id = agent_id
        self.cluster_nodes = cluster_nodes
        self.max_local_items = max_local_items
        
        # Local cache for hot data
        self.local_cache: Dict[str, MemoryItem] = {}
        self.access_frequency: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, float] = {}
        
        # Distributed components
        self.hash_ring = ConsistentHashRing(cluster_nodes)
        self.redis_clients: Dict[str, redis.Redis] = {}
        self.connection_pool = None
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.local_lock = threading.RLock()
        
        # Metrics
        if METRICS_AVAILABLE:
            self.cache_hits = Counter(
                'l1_distributed_cache_hits_total',
                'L1 distributed cache hits',
                ['agent_id', 'result']
            )
            self.memory_operations = Counter(
                'l1_distributed_operations_total',
                'L1 distributed memory operations',
                ['agent_id', 'operation', 'status']
            )
            self.latency_histogram = Histogram(
                'l1_distributed_latency_seconds',
                'L1 distributed operation latency',
                ['agent_id', 'operation']
            )
            self.local_cache_size = Gauge(
                'l1_distributed_local_cache_size',
                'Local cache size',
                ['agent_id']
            )
    
    async def initialize(self):
        """Initialize distributed memory system"""
        log.info(f"Initializing distributed L1 memory for agent {self.agent_id}")
        
        # Initialize Redis connections
        await self._init_redis_connections()
        
        # Start background maintenance tasks
        asyncio.create_task(self._cache_maintenance_loop())
        asyncio.create_task(self._health_check_loop())
        
        log.info(f"Distributed L1 memory initialized with {len(self.cluster_nodes)} nodes")
    
    async def _init_redis_connections(self):
        """Initialize Redis connections to cluster nodes"""
        for node in self.cluster_nodes:
            try:
                client = redis.from_url(f"redis://{node}")
                await client.ping()
                self.redis_clients[node] = client
                log.info(f"Connected to Redis node: {node}")
            except Exception as e:
                log.error(f"Failed to connect to Redis node {node}: {e}")
    
    async def store(self, item: MemoryItem) -> bool:
        """Store item with intelligent distribution and caching"""
        start_time = time.time()
        
        try:
            # Determine storage nodes using consistent hashing
            storage_nodes = self.hash_ring.get_nodes(item.key, 3)  # Primary + 2 replicas
            
            if not storage_nodes:
                log.error("No storage nodes available")
                return False
            
            # Update local cache if item is hot
            await self._maybe_cache_locally(item)
            
            # Store in distributed nodes
            success_count = 0
            tasks = []
            
            for i, node in enumerate(storage_nodes):
                if node in self.redis_clients:
                    is_primary = (i == 0)
                    task = self._store_in_node(node, item, is_primary)
                    tasks.append(task)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success_count = sum(1 for r in results if r is True)
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.memory_operations.labels(
                    agent_id=self.agent_id,
                    operation="store",
                    status="success" if success_count > 0 else "error"
                ).inc()
                
                self.latency_histogram.labels(
                    agent_id=self.agent_id,
                    operation="store"
                ).observe(time.time() - start_time)
            
            # Require at least one successful write
            return success_count > 0
            
        except Exception as e:
            log.error(f"Failed to store item {item.key}: {e}")
            
            if METRICS_AVAILABLE:
                self.memory_operations.labels(
                    agent_id=self.agent_id,
                    operation="store",
                    status="error"
                ).inc()
            
            return False
    
    async def _store_in_node(self, node: str, item: MemoryItem, is_primary: bool) -> bool:
        """Store item in a specific Redis node"""
        try:
            client = self.redis_clients.get(node)
            if not client:
                return False
            
            # Serialize item
            item_data = {
                **item.to_dict(),
                "stored_at": time.time(),
                "is_primary": is_primary,
                "agent_id": self.agent_id
            }
            
            # Compress if large
            serialized = json.dumps(item_data)
            if len(serialized) > 1024:  # Compress items > 1KB
                compressed = zlib.compress(serialized.encode())
                await client.set(
                    f"l1:{self.agent_id}:{item.key}",
                    compressed,
                    ex=3600  # 1 hour TTL
                )
                await client.set(
                    f"l1:{self.agent_id}:{item.key}:compressed",
                    "1",
                    ex=3600
                )
            else:
                await client.set(
                    f"l1:{self.agent_id}:{item.key}",
                    serialized,
                    ex=3600
                )
            
            # Store embedding separately for efficient search
            if item.embedding is not None:
                await client.set(
                    f"l1:emb:{self.agent_id}:{item.key}",
                    json.dumps(item.embedding.tolist()),
                    ex=3600
                )
            
            return True
            
        except Exception as e:
            log.error(f"Failed to store in node {node}: {e}")
            return False
    
    async def retrieve(self, query: Query) -> List[MemoryItem]:
        """Retrieve items with intelligent caching and distribution"""
        start_time = time.time()
        
        try:
            # Check local cache first
            local_results = self._search_local_cache(query)
            
            if local_results and len(local_results) >= query.top_k:
                if METRICS_AVAILABLE:
                    self.cache_hits.labels(
                        agent_id=self.agent_id,
                        result="hit"
                    ).inc()
                return local_results[:query.top_k]
            
            # Search distributed nodes
            distributed_results = await self._search_distributed(query)
            
            # Combine and rank results
            all_results = local_results + distributed_results
            final_results = self._rank_and_deduplicate(all_results, query)
            
            # Cache hot items locally
            for item in final_results[:5]:  # Cache top 5 results
                await self._maybe_cache_locally(item)
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.cache_hits.labels(
                    agent_id=self.agent_id,
                    result="miss" if not local_results else "partial"
                ).inc()
                
                self.latency_histogram.labels(
                    agent_id=self.agent_id,
                    operation="retrieve"
                ).observe(time.time() - start_time)
            
            return final_results[:query.top_k]
            
        except Exception as e:
            log.error(f"Failed to retrieve items for query '{query.text}': {e}")
            return []
    
    def _search_local_cache(self, query: Query) -> List[MemoryItem]:
        """Search local cache"""
        if not query.embedding:
            # Text-based search
            results = []
            query_lower = query.text.lower()
            
            with self.local_lock:
                for item in self.local_cache.values():
                    if query_lower in str(item.value).lower():
                        self.access_frequency[item.key] += 1
                        self.last_access[item.key] = time.time()
                        results.append(item)
            
            return results
        
        # Vector-based search
        results = []
        with self.local_lock:
            for item in self.local_cache.values():
                if item.embedding is not None:
                    similarity = self._cosine_similarity(query.embedding, item.embedding)
                    if similarity >= query.min_score:
                        self.access_frequency[item.key] += 1
                        self.last_access[item.key] = time.time()
                        results.append((similarity, item))
        
        # Sort by similarity
        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results]
    
    async def _search_distributed(self, query: Query) -> List[MemoryItem]:
        """Search across distributed nodes"""
        results = []
        
        # Search all available nodes in parallel
        tasks = []
        for node in self.redis_clients:
            task = self._search_node(node, query)
            tasks.append(task)
        
        if tasks:
            node_results = await asyncio.gather(*tasks, return_exceptions=True)
            for node_result in node_results:
                if isinstance(node_result, list):
                    results.extend(node_result)
        
        return results
    
    async def _search_node(self, node: str, query: Query) -> List[MemoryItem]:
        """Search a specific Redis node"""
        try:
            client = self.redis_clients.get(node)
            if not client:
                return []
            
            # Get all keys for this agent
            pattern = f"l1:{self.agent_id}:*"
            keys = await client.keys(pattern)
            
            results = []
            
            # Process keys in batches to avoid blocking
            batch_size = 50
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                
                # Get items in parallel
                pipe = client.pipeline()
                for key in batch_keys:
                    pipe.get(key)
                    pipe.get(f"{key}:compressed")
                
                batch_data = await pipe.execute()
                
                # Process batch results
                for j in range(0, len(batch_data), 2):
                    item_data = batch_data[j]
                    is_compressed = batch_data[j + 1]
                    
                    if item_data:
                        try:
                            if is_compressed:
                                decompressed = zlib.decompress(item_data)
                                item_dict = json.loads(decompressed.decode())
                            else:
                                item_dict = json.loads(item_data)
                            
                            item = MemoryItem.from_dict(item_dict)
                            
                            # Check if item matches query
                            if self._item_matches_query(item, query):
                                results.append(item)
                                
                        except Exception as e:
                            log.debug(f"Failed to deserialize item: {e}")
                            continue
            
            return results
            
        except Exception as e:
            log.error(f"Failed to search node {node}: {e}")
            return []
    
    def _item_matches_query(self, item: MemoryItem, query: Query) -> bool:
        """Check if item matches query criteria"""
        if query.embedding and item.embedding is not None:
            similarity = self._cosine_similarity(query.embedding, item.embedding)
            return similarity >= query.min_score
        else:
            # Text-based matching
            return query.text.lower() in str(item.value).lower()
    
    def _rank_and_deduplicate(self, results: List[MemoryItem], query: Query) -> List[MemoryItem]:
        """Rank and deduplicate results"""
        # Remove duplicates by key
        unique_results = {}
        for item in results:
            if item.key not in unique_results:
                unique_results[item.key] = item
            else:
                # Keep the more recent one
                existing = unique_results[item.key]
                if item.last_accessed > existing.last_accessed:
                    unique_results[item.key] = item
        
        # Calculate relevance scores and sort
        scored_results = []
        for item in unique_results.values():
            score = self._calculate_relevance_score(item, query)
            scored_results.append((score, item))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_results]
    
    def _calculate_relevance_score(self, item: MemoryItem, query: Query) -> float:
        """Calculate relevance score for ranking"""
        score = 0.0
        
        # Semantic similarity
        if query.embedding and item.embedding is not None:
            score += self._cosine_similarity(query.embedding, item.embedding) * 0.7
        
        # Recency bonus
        age = time.time() - item.last_accessed
        recency_score = max(0, 1 - age / 3600)  # Decay over 1 hour
        score += recency_score * 0.2
        
        # Frequency bonus
        frequency = self.access_frequency.get(item.key, 0)
        frequency_score = min(1.0, frequency / 10)  # Normalize to [0, 1]
        score += frequency_score * 0.1
        
        return score
    
    async def _maybe_cache_locally(self, item: MemoryItem):
        """Cache item locally if it's hot or important"""
        with self.local_lock:
            # Check if we should cache this item
            should_cache = (
                self.access_frequency.get(item.key, 0) >= 2 or  # Accessed multiple times
                item.metadata.get("importance", "normal") in ["high", "critical"] or
                len(self.local_cache) < self.max_local_items * 0.8  # Cache not full
            )
            
            if should_cache:
                self.local_cache[item.key] = item
                self.access_frequency[item.key] += 1
                self.last_access[item.key] = time.time()
                
                # Evict if cache is full
                if len(self.local_cache) > self.max_local_items:
                    self._evict_from_local_cache()
                
                if METRICS_AVAILABLE:
                    self.local_cache_size.labels(agent_id=self.agent_id).set(
                        len(self.local_cache)
                    )
    
    def _evict_from_local_cache(self):
        """Evict items from local cache using intelligent policy"""
        if len(self.local_cache) <= self.max_local_items:
            return
        
        # Calculate eviction scores (lower = more likely to evict)
        eviction_candidates = []
        
        for key, item in self.local_cache.items():
            frequency = self.access_frequency.get(key, 0)
            last_access_time = self.last_access.get(key, 0)
            age = time.time() - last_access_time
            importance = {"low": 0.1, "normal": 0.5, "high": 0.8, "critical": 1.0}.get(
                item.metadata.get("importance", "normal"), 0.5
            )
            
            # Lower score = more likely to evict
            eviction_score = (frequency * 0.4) + (1 / (age + 1) * 0.4) + (importance * 0.2)
            eviction_candidates.append((eviction_score, key))
        
        # Sort by eviction score and remove lowest scoring items
        eviction_candidates.sort()
        items_to_evict = len(self.local_cache) - self.max_local_items + 100  # Evict extra for hysteresis
        
        for _, key in eviction_candidates[:items_to_evict]:
            del self.local_cache[key]
            if key in self.access_frequency:
                del self.access_frequency[key]
            if key in self.last_access:
                del self.last_access[key]
        
        log.debug(f"Evicted {items_to_evict} items from local cache")
    
    async def _cache_maintenance_loop(self):
        """Background task for cache maintenance"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                with self.local_lock:
                    # Clean up old access records
                    current_time = time.time()
                    old_keys = [
                        key for key, last_access in self.last_access.items()
                        if current_time - last_access > 3600  # 1 hour
                    ]
                    
                    for key in old_keys:
                        if key not in self.local_cache:  # Only clean if not in cache
                            del self.last_access[key]
                            if key in self.access_frequency:
                                del self.access_frequency[key]
                
                log.debug(f"Cache maintenance: cleaned {len(old_keys)} old records")
                
            except Exception as e:
                log.error(f"Cache maintenance error: {e}")
    
    async def _health_check_loop(self):
        """Background task for health checking Redis nodes"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                failed_nodes = []
                for node, client in self.redis_clients.items():
                    try:
                        await client.ping()
                    except Exception as e:
                        log.warning(f"Health check failed for node {node}: {e}")
                        failed_nodes.append(node)
                
                # Remove failed nodes from hash ring
                for node in failed_nodes:
                    self.hash_ring.remove_node(node)
                    del self.redis_clients[node]
                
                # Try to reconnect to failed nodes
                for node in failed_nodes:
                    try:
                        client = redis.from_url(f"redis://{node}")
                        await client.ping()
                        self.redis_clients[node] = client
                        self.hash_ring.add_node(node)
                        log.info(f"Reconnected to Redis node: {node}")
                    except Exception:
                        pass  # Will try again next cycle
                        
            except Exception as e:
                log.error(f"Health check error: {e}")
    
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
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get distributed memory statistics"""
        with self.local_lock:
            local_cache_size = len(self.local_cache)
            access_frequency_size = len(self.access_frequency)
        
        return {
            "tier": "L1_DISTRIBUTED",
            "agent_id": self.agent_id,
            "cluster_nodes": len(self.cluster_nodes),
            "active_nodes": len(self.redis_clients),
            "local_cache_size": local_cache_size,
            "max_local_items": self.max_local_items,
            "cache_utilization": local_cache_size / self.max_local_items,
            "access_records": access_frequency_size,
            "hash_ring_nodes": len(self.hash_ring.nodes)
        }

class DistributedL2Memory:
    """Distributed L2 Memory with Redis Cluster support and failover"""
    
    def __init__(self, swarm_id: str, cluster_config: Dict[str, Any]):
        self.swarm_id = swarm_id
        self.cluster_config = cluster_config
        self.redis_cluster = None
        self.fallback_clients = {}
        
        # Metrics
        if METRICS_AVAILABLE:
            self.cluster_operations = Counter(
                'l2_cluster_operations_total',
                'L2 cluster operations',
                ['swarm_id', 'operation', 'status']
            )
            self.cluster_latency = Histogram(
                'l2_cluster_latency_seconds',
                'L2 cluster operation latency',
                ['swarm_id', 'operation']
            )
    
    async def initialize(self):
        """Initialize Redis Cluster connection"""
        log.info(f"Initializing distributed L2 memory for swarm {self.swarm_id}")
        
        try:
            # Try Redis Cluster first
            startup_nodes = [
                {"host": node.split(':')[0], "port": int(node.split(':')[1])}
                for node in self.cluster_config.get("nodes", [])
            ]
            
            if startup_nodes and RedisCluster:
                self.redis_cluster = RedisCluster(
                    startup_nodes=startup_nodes,
                    decode_responses=False,
                    skip_full_coverage_check=True,
                    max_connections_per_node=10
                )
                await self.redis_cluster.ping()
                log.info("Redis Cluster connection established")
            else:
                # Fallback to individual connections
                await self._init_fallback_clients()
                
        except Exception as e:
            log.warning(f"Redis Cluster initialization failed: {e}, using fallback")
            await self._init_fallback_clients()
    
    async def _init_fallback_clients(self):
        """Initialize individual Redis clients as fallback"""
        for node in self.cluster_config.get("nodes", []):
            try:
                client = redis.from_url(f"redis://{node}")
                await client.ping()
                self.fallback_clients[node] = client
                log.info(f"Connected to fallback Redis node: {node}")
            except Exception as e:
                log.error(f"Failed to connect to fallback node {node}: {e}")
    
    async def store(self, item: MemoryItem) -> bool:
        """Store item in distributed L2 memory"""
        start_time = time.time()
        
        try:
            item.tier = MemoryTier.L2_SWARM
            item.context["swarm_id"] = self.swarm_id
            
            key = f"l2:{self.swarm_id}:{item.key}"
            value = json.dumps(item.to_dict())
            
            success = False
            
            if self.redis_cluster:
                # Use Redis Cluster
                await self.redis_cluster.setex(key, 7200, value)  # 2 hour TTL
                
                # Store embedding separately
                if item.embedding is not None:
                    emb_key = f"l2:emb:{self.swarm_id}:{item.key}"
                    emb_data = json.dumps({
                        "embedding": item.embedding.tolist(),
                        "metadata": item.metadata
                    })
                    await self.redis_cluster.setex(emb_key, 7200, emb_data)
                
                success = True
                
            elif self.fallback_clients:
                # Use fallback clients with replication
                success_count = 0
                
                for client in list(self.fallback_clients.values())[:3]:  # Replicate to 3 nodes
                    try:
                        await client.setex(key, 7200, value)
                        
                        if item.embedding is not None:
                            emb_key = f"l2:emb:{self.swarm_id}:{item.key}"
                            emb_data = json.dumps({
                                "embedding": item.embedding.tolist(),
                                "metadata": item.metadata
                            })
                            await client.setex(emb_key, 7200, emb_data)
                        
                        success_count += 1
                        
                    except Exception as e:
                        log.debug(f"Failed to store in fallback client: {e}")
                
                success = success_count > 0
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.cluster_operations.labels(
                    swarm_id=self.swarm_id,
                    operation="store",
                    status="success" if success else "error"
                ).inc()
                
                self.cluster_latency.labels(
                    swarm_id=self.swarm_id,
                    operation="store"
                ).observe(time.time() - start_time)
            
            return success
            
        except Exception as e:
            log.error(f"Failed to store item {item.key} in L2: {e}")
            
            if METRICS_AVAILABLE:
                self.cluster_operations.labels(
                    swarm_id=self.swarm_id,
                    operation="store",
                    status="error"
                ).inc()
            
            return False
    
    async def retrieve(self, query: Query) -> List[MemoryItem]:
        """Retrieve items from distributed L2 memory"""
        start_time = time.time()
        
        try:
            results = []
            
            if self.redis_cluster:
                results = await self._retrieve_from_cluster(query)
            elif self.fallback_clients:
                results = await self._retrieve_from_fallback(query)
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.cluster_operations.labels(
                    swarm_id=self.swarm_id,
                    operation="retrieve",
                    status="success"
                ).inc()
                
                self.cluster_latency.labels(
                    swarm_id=self.swarm_id,
                    operation="retrieve"
                ).observe(time.time() - start_time)
            
            return results[:query.top_k]
            
        except Exception as e:
            log.error(f"Failed to retrieve from L2 memory: {e}")
            
            if METRICS_AVAILABLE:
                self.cluster_operations.labels(
                    swarm_id=self.swarm_id,
                    operation="retrieve",
                    status="error"
                ).inc()
            
            return []
    
    async def _retrieve_from_cluster(self, query: Query) -> List[MemoryItem]:
        """Retrieve from Redis Cluster"""
        # This would implement cluster-aware retrieval
        # For now, simplified implementation
        pattern = f"l2:{self.swarm_id}:*"
        
        try:
            # Get keys from all cluster nodes
            all_keys = []
            for node in self.redis_cluster.get_nodes():
                node_keys = await node.keys(pattern)
                all_keys.extend(node_keys)
            
            # Process keys in batches
            results = []
            batch_size = 100
            
            for i in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[i:i + batch_size]
                
                # Get items in parallel
                pipe = self.redis_cluster.pipeline()
                for key in batch_keys:
                    pipe.get(key)
                
                batch_data = await pipe.execute()
                
                for item_data in batch_data:
                    if item_data:
                        try:
                            item_dict = json.loads(item_data)
                            item = MemoryItem.from_dict(item_dict)
                            
                            if self._item_matches_query(item, query):
                                results.append(item)
                                
                        except Exception as e:
                            log.debug(f"Failed to deserialize L2 item: {e}")
                            continue
            
            return results
            
        except Exception as e:
            log.error(f"Cluster retrieval failed: {e}")
            return []
    
    async def _retrieve_from_fallback(self, query: Query) -> List[MemoryItem]:
        """Retrieve from fallback clients"""
        results = []
        
        # Search all fallback clients
        for client in self.fallback_clients.values():
            try:
                pattern = f"l2:{self.swarm_id}:*"
                keys = await client.keys(pattern)
                
                # Process in batches
                batch_size = 50
                for i in range(0, len(keys), batch_size):
                    batch_keys = keys[i:i + batch_size]
                    
                    pipe = client.pipeline()
                    for key in batch_keys:
                        pipe.get(key)
                    
                    batch_data = await pipe.execute()
                    
                    for item_data in batch_data:
                        if item_data:
                            try:
                                item_dict = json.loads(item_data)
                                item = MemoryItem.from_dict(item_dict)
                                
                                if self._item_matches_query(item, query):
                                    results.append(item)
                                    
                            except Exception as e:
                                log.debug(f"Failed to deserialize fallback item: {e}")
                                continue
                                
            except Exception as e:
                log.debug(f"Fallback client search failed: {e}")
                continue
        
        return results
    
    def _item_matches_query(self, item: MemoryItem, query: Query) -> bool:
        """Check if item matches query criteria"""
        if query.embedding and item.embedding is not None:
            similarity = self._cosine_similarity(query.embedding, item.embedding)
            return similarity >= query.min_score
        else:
            return query.text.lower() in str(item.value).lower()
    
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
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get L2 distributed memory statistics"""
        stats = {
            "tier": "L2_DISTRIBUTED",
            "swarm_id": self.swarm_id,
            "cluster_mode": self.redis_cluster is not None,
            "fallback_clients": len(self.fallback_clients)
        }
        
        if self.redis_cluster:
            try:
                cluster_info = await self.redis_cluster.cluster_info()
                stats.update({
                    "cluster_state": cluster_info.get("cluster_state", "unknown"),
                    "cluster_nodes": len(self.redis_cluster.get_nodes())
                })
            except Exception as e:
                stats["cluster_error"] = str(e)
        
        return stats


class DistributedMemory:
    """
    Unified Distributed Memory System combining L1 and L2 memory tiers
    Provides a single interface for distributed memory operations
    """
    
    def __init__(self):
        self.l1_memory = DistributedL1Memory()
        self.l2_memory = DistributedL2Memory()
        self._initialized = False
    
    async def initialize(self):
        """Initialize both memory tiers"""
        if self._initialized:
            return
            
        await self.l1_memory.initialize()
        await self.l2_memory.initialize()
        self._initialized = True
        log.info("✅ Unified Distributed Memory System initialized")
    
    async def store(self, key: str, data: Any, tier: MemoryTier = MemoryTier.L1_AGENT) -> bool:
        """Store data in the appropriate memory tier"""
        if not self._initialized:
            await self.initialize()
            
        if tier == MemoryTier.L1_AGENT:
            return await self.l1_memory.store(key, data)
        else:
            return await self.l2_memory.store(key, data)
    
    async def retrieve(self, key: str, tier: MemoryTier = None) -> Any:
        """Retrieve data from memory tiers (L1 first, then L2 if not specified)"""
        if not self._initialized:
            await self.initialize()
            
        if tier == MemoryTier.L1_AGENT:
            return await self.l1_memory.retrieve(key)
        elif tier == MemoryTier.L2_SWARM:
            return await self.l2_memory.retrieve(key)
        else:
            # Try L1 first, then L2
            result = await self.l1_memory.retrieve(key)
            if result is None:
                result = await self.l2_memory.retrieve(key)
            return result
    
    async def delete(self, key: str, tier: MemoryTier = None) -> bool:
        """Delete data from memory tiers"""
        if not self._initialized:
            await self.initialize()
            
        success = True
        if tier is None or tier == MemoryTier.L1_AGENT:
            success &= await self.l1_memory.delete(key)
        if tier is None or tier == MemoryTier.L2_SWARM:
            success &= await self.l2_memory.delete(key)
        return success
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from both memory tiers"""
        if not self._initialized:
            await self.initialize()
            
        l1_stats = await self.l1_memory.get_stats()
        l2_stats = await self.l2_memory.get_stats()
        
        return {
            "l1_memory": l1_stats,
            "l2_memory": l2_stats,
            "total_items": l1_stats.get("total_items", 0) + l2_stats.get("total_items", 0),
            "initialized": self._initialized
        }
