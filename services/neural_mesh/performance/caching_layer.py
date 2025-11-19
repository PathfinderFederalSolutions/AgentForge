"""
Caching Layer for Neural Mesh
Provides intelligent caching for improved performance
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import time
import hashlib
import json
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("caching-layer")

class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    size_bytes: int = 0

class CachingLayer:
    """
    Intelligent caching layer for neural mesh operations
    """
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = []  # For LRU
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self.stats["total_requests"] += 1
        
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        
        # Check TTL expiration
        if entry.ttl and time.time() - entry.created_at > entry.ttl:
            await self.delete(key)
            self.stats["misses"] += 1
            return None
        
        # Update access metadata
        entry.last_accessed = time.time()
        entry.access_count += 1
        
        # Update LRU order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        self.stats["hits"] += 1
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache"""
        try:
            # Calculate size estimate
            size_bytes = len(json.dumps(value, default=str).encode('utf-8'))
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # Check if we need to evict entries
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_entries(1)
            
            # Store entry
            self.cache[key] = entry
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return True
            
        except Exception as e:
            log.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return True
        return False
    
    async def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_order.clear()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    async def _evict_entries(self, count: int):
        """Evict entries based on strategy"""
        if not self.cache:
            return
        
        evicted = 0
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            keys_to_evict = self.access_order[:count]
            for key in keys_to_evict:
                await self.delete(key)
                evicted += 1
                
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].access_count
            )
            for key, _ in sorted_entries[:count]:
                await self.delete(key)
                evicted += 1
                
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first, then oldest
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.ttl and current_time - entry.created_at > entry.ttl
            ]
            
            for key in expired_keys[:count]:
                await self.delete(key)
                evicted += 1
            
            # If still need to evict more, evict oldest
            if evicted < count:
                remaining = count - evicted
                sorted_entries = sorted(
                    self.cache.items(),
                    key=lambda x: x[1].created_at
                )
                for key, _ in sorted_entries[:remaining]:
                    await self.delete(key)
                    evicted += 1
                    
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy based on access patterns
            current_time = time.time()
            
            # Score entries based on recency, frequency, and size
            scored_entries = []
            for key, entry in self.cache.items():
                recency_score = 1.0 / (current_time - entry.last_accessed + 1)
                frequency_score = entry.access_count / max(1, self.stats["total_requests"])
                size_penalty = entry.size_bytes / (1024 * 1024)  # MB penalty
                
                total_score = (recency_score + frequency_score) / (1 + size_penalty)
                scored_entries.append((key, total_score))
            
            # Evict lowest scoring entries
            scored_entries.sort(key=lambda x: x[1])
            for key, _ in scored_entries[:count]:
                await self.delete(key)
                evicted += 1
        
        self.stats["evictions"] += evicted
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0.0
        if self.stats["total_requests"] > 0:
            hit_rate = self.stats["hits"] / self.stats["total_requests"]
        
        total_size_bytes = sum(entry.size_bytes for entry in self.cache.values())
        
        return {
            "entries": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size,
            "hit_rate": hit_rate,
            "total_requests": self.stats["total_requests"],
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "total_size_mb": total_size_bytes / (1024 * 1024),
            "strategy": self.strategy.value
        }
    
    async def optimize_cache(self):
        """Optimize cache performance"""
        try:
            # Clean up expired entries
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.ttl and current_time - entry.created_at > entry.ttl
            ]
            
            for key in expired_keys:
                await self.delete(key)
            
            # Adjust strategy based on hit rate
            stats = await self.get_stats()
            hit_rate = stats["hit_rate"]
            
            if hit_rate < 0.5 and self.strategy != CacheStrategy.ADAPTIVE:
                log.info("Low hit rate detected, switching to adaptive strategy")
                self.strategy = CacheStrategy.ADAPTIVE
            
            log.info(f"Cache optimization complete. Hit rate: {hit_rate:.2%}")
            
        except Exception as e:
            log.error(f"Cache optimization error: {e}")
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create deterministic key from arguments
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items()) if kwargs else {}
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
