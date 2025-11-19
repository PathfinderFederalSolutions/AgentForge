"""
AWS ElastiCache Redis Cluster Manager
High-performance caching for million-scale agent operations
"""
import os
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError, ConnectionError
import pickle
from dataclasses import dataclass

log = logging.getLogger("cache-manager")

@dataclass
class CacheConfig:
    """Redis cache configuration from environment"""
    url: str
    cluster_enabled: bool
    cluster_nodes: List[str]
    password: str
    max_connections: int
    pool_size: int
    socket_keepalive: bool
    retry_on_timeout: bool
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        cluster_nodes = []
        if os.getenv("REDIS_CLUSTER_NODES"):
            cluster_nodes = os.getenv("REDIS_CLUSTER_NODES").split(",")
        
        return cls(
            url=os.getenv("REDIS_URL"),
            cluster_enabled=os.getenv("REDIS_CLUSTER_ENABLED", "false").lower() == "true",
            cluster_nodes=cluster_nodes,
            password=os.getenv("REDIS_PASSWORD"),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "1000")),
            pool_size=int(os.getenv("REDIS_POOL_SIZE", "200")),
            socket_keepalive=os.getenv("REDIS_SOCKET_KEEPALIVE", "true").lower() == "true",
            retry_on_timeout=os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
        )

class CacheManager:
    """Production Redis cache manager with cluster support"""
    
    def __init__(self):
        self.config = CacheConfig.from_env()
        self.redis_client: Optional[Redis] = None
        self._initialized = False
        
        # Cache statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_sets": 0,
            "cache_deletes": 0,
            "connection_errors": 0,
            "total_operations": 0,
            "avg_operation_time": 0.0
        }
        
        # Cache prefixes for different data types
        self.prefixes = {
            "agent": "agent:",
            "swarm": "swarm:",
            "neural_mesh": "nm:",
            "knowledge": "kg:",
            "session": "session:",
            "user": "user:",
            "llm_response": "llm:",
            "embedding": "emb:",
            "search": "search:"
        }
    
    async def initialize(self):
        """Initialize Redis connection with cluster support"""
        if self._initialized:
            return
        
        try:
            if self.config.cluster_enabled:
                # Redis Cluster mode
                from redis.asyncio.cluster import RedisCluster
                
                startup_nodes = [
                    {"host": node.split(":")[0], "port": int(node.split(":")[1])}
                    for node in self.config.cluster_nodes
                ]
                
                self.redis_client = RedisCluster(
                    startup_nodes=startup_nodes,
                    password=self.config.password,
                    decode_responses=False,
                    skip_full_coverage_check=True,
                    max_connections=self.config.max_connections,
                    retry_on_timeout=self.config.retry_on_timeout,
                    socket_keepalive=self.config.socket_keepalive
                )
            else:
                # Single Redis instance
                pool = ConnectionPool.from_url(
                    self.config.url,
                    password=self.config.password,
                    max_connections=self.config.max_connections,
                    retry_on_timeout=self.config.retry_on_timeout,
                    socket_keepalive=self.config.socket_keepalive
                )
                
                self.redis_client = Redis(
                    connection_pool=pool,
                    decode_responses=False
                )
            
            # Test connection
            await self.redis_client.ping()
            
            # Start health monitoring
            asyncio.create_task(self._monitor_health())
            
            self._initialized = True
            log.info(f"Cache manager initialized ({'cluster' if self.config.cluster_enabled else 'single'} mode)")
            
        except Exception as e:
            log.error(f"Failed to initialize cache manager: {e}")
            raise
    
    def _get_key(self, prefix: str, key: str) -> str:
        """Generate prefixed cache key"""
        return f"{self.prefixes.get(prefix, 'misc:')}{key}"
    
    async def _execute_with_stats(self, operation_name: str, coro):
        """Execute Redis operation with performance tracking"""
        start_time = time.time()
        
        try:
            result = await coro
            
            # Update statistics
            operation_time = time.time() - start_time
            self.stats["total_operations"] += 1
            
            # Update average operation time
            total_ops = self.stats["total_operations"]
            self.stats["avg_operation_time"] = (
                (self.stats["avg_operation_time"] * (total_ops - 1) + operation_time) / total_ops
            )
            
            return result
            
        except (RedisError, ConnectionError) as e:
            self.stats["connection_errors"] += 1
            log.error(f"Redis {operation_name} error: {e}")
            raise
    
    async def get(self, prefix: str, key: str, default=None) -> Any:
        """Get value from cache with deserialization"""
        cache_key = self._get_key(prefix, key)
        
        try:
            result = await self._execute_with_stats(
                "get",
                self.redis_client.get(cache_key)
            )
            
            if result is not None:
                self.stats["cache_hits"] += 1
                return pickle.loads(result)
            else:
                self.stats["cache_misses"] += 1
                return default
                
        except Exception as e:
            log.error(f"Cache get error for key {cache_key}: {e}")
            self.stats["cache_misses"] += 1
            return default
    
    async def set(self, prefix: str, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with serialization"""
        cache_key = self._get_key(prefix, key)
        
        try:
            serialized_value = pickle.dumps(value)
            
            if ttl:
                result = await self._execute_with_stats(
                    "setex",
                    self.redis_client.setex(cache_key, ttl, serialized_value)
                )
            else:
                result = await self._execute_with_stats(
                    "set",
                    self.redis_client.set(cache_key, serialized_value)
                )
            
            self.stats["cache_sets"] += 1
            return result
            
        except Exception as e:
            log.error(f"Cache set error for key {cache_key}: {e}")
            return False
    
    async def delete(self, prefix: str, key: str) -> bool:
        """Delete key from cache"""
        cache_key = self._get_key(prefix, key)
        
        try:
            result = await self._execute_with_stats(
                "delete",
                self.redis_client.delete(cache_key)
            )
            
            self.stats["cache_deletes"] += 1
            return bool(result)
            
        except Exception as e:
            log.error(f"Cache delete error for key {cache_key}: {e}")
            return False
    
    async def exists(self, prefix: str, key: str) -> bool:
        """Check if key exists in cache"""
        cache_key = self._get_key(prefix, key)
        
        try:
            result = await self._execute_with_stats(
                "exists",
                self.redis_client.exists(cache_key)
            )
            return bool(result)
            
        except Exception as e:
            log.error(f"Cache exists error for key {cache_key}: {e}")
            return False
    
    async def mget(self, prefix: str, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        cache_keys = [self._get_key(prefix, key) for key in keys]
        
        try:
            results = await self._execute_with_stats(
                "mget",
                self.redis_client.mget(cache_keys)
            )
            
            output = {}
            for i, result in enumerate(results):
                if result is not None:
                    self.stats["cache_hits"] += 1
                    output[keys[i]] = pickle.loads(result)
                else:
                    self.stats["cache_misses"] += 1
                    
            return output
            
        except Exception as e:
            log.error(f"Cache mget error: {e}")
            self.stats["cache_misses"] += len(keys)
            return {}
    
    async def mset(self, prefix: str, mapping: Dict[str, Any], ttl: int = None) -> bool:
        """Set multiple values in cache"""
        cache_mapping = {
            self._get_key(prefix, key): pickle.dumps(value)
            for key, value in mapping.items()
        }
        
        try:
            if ttl:
                # Use pipeline for multiple setex operations
                pipe = self.redis_client.pipeline()
                for cache_key, serialized_value in cache_mapping.items():
                    pipe.setex(cache_key, ttl, serialized_value)
                
                result = await self._execute_with_stats("pipeline", pipe.execute())
            else:
                result = await self._execute_with_stats(
                    "mset",
                    self.redis_client.mset(cache_mapping)
                )
            
            self.stats["cache_sets"] += len(mapping)
            return bool(result)
            
        except Exception as e:
            log.error(f"Cache mset error: {e}")
            return False
    
    async def increment(self, prefix: str, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        cache_key = self._get_key(prefix, key)
        
        try:
            result = await self._execute_with_stats(
                "incr",
                self.redis_client.incr(cache_key, amount)
            )
            return result
            
        except Exception as e:
            log.error(f"Cache increment error for key {cache_key}: {e}")
            return 0
    
    async def expire(self, prefix: str, key: str, ttl: int) -> bool:
        """Set expiration time for key"""
        cache_key = self._get_key(prefix, key)
        
        try:
            result = await self._execute_with_stats(
                "expire",
                self.redis_client.expire(cache_key, ttl)
            )
            return bool(result)
            
        except Exception as e:
            log.error(f"Cache expire error for key {cache_key}: {e}")
            return False
    
    async def keys_pattern(self, prefix: str, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        cache_pattern = self._get_key(prefix, pattern)
        
        try:
            keys = await self._execute_with_stats(
                "keys",
                self.redis_client.keys(cache_pattern)
            )
            
            # Remove prefix from returned keys
            prefix_len = len(self.prefixes.get(prefix, 'misc:'))
            return [key.decode('utf-8')[prefix_len:] for key in keys]
            
        except Exception as e:
            log.error(f"Cache keys pattern error: {e}")
            return []
    
    async def flush_prefix(self, prefix: str) -> bool:
        """Delete all keys with given prefix"""
        try:
            keys = await self.keys_pattern(prefix, "*")
            if keys:
                cache_keys = [self._get_key(prefix, key) for key in keys]
                result = await self._execute_with_stats(
                    "delete",
                    self.redis_client.delete(*cache_keys)
                )
                self.stats["cache_deletes"] += len(keys)
                return bool(result)
            return True
            
        except Exception as e:
            log.error(f"Cache flush prefix error: {e}")
            return False
    
    async def _monitor_health(self):
        """Monitor Redis connection health"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Ping Redis to check connectivity
                await self.redis_client.ping()
                
                # Get Redis info
                info = await self.redis_client.info()
                
                # Log basic stats
                log.debug(f"Redis connected clients: {info.get('connected_clients', 0)}")
                log.debug(f"Redis used memory: {info.get('used_memory_human', 'unknown')}")
                
            except Exception as e:
                log.error(f"Redis health check failed: {e}")
                self.stats["connection_errors"] += 1
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache manager statistics"""
        hit_rate = 0.0
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_requests > 0:
            hit_rate = self.stats["cache_hits"] / total_requests
        
        return {
            **self.stats,
            "cache_hit_rate": hit_rate,
            "total_cache_requests": total_requests
        }
    
    async def cleanup(self):
        """Cleanup Redis connections"""
        if self.redis_client:
            await self.redis_client.aclose()
        log.info("Cache manager cleaned up")

# Global cache manager instance
cache_manager = CacheManager()

async def get_cache_manager() -> CacheManager:
    """Get initialized cache manager"""
    if not cache_manager._initialized:
        await cache_manager.initialize()
    return cache_manager
