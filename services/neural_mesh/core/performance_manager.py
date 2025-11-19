"""
Performance Manager - Intelligent Caching, Error Handling, and Resource Management
Implements production-grade performance optimizations and reliability patterns
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, TypeVar, Generic
from enum import Enum
from collections import defaultdict, OrderedDict
import heapq
import gc
import psutil
import resource

# Import base classes
from .memory_types import MemoryItem, Knowledge

# Optional imports
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    aiofiles = None
    AIOFILES_AVAILABLE = False

# Metrics imports
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = Summary = lambda *args, **kwargs: None

log = logging.getLogger("performance-manager")

T = TypeVar('T')

class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    INTELLIGENCE_AWARE = "intelligence_aware"
    ADAPTIVE = "adaptive"

class Priority(Enum):
    """Priority levels for caching and processing"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata"""
    key: str
    value: T
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    priority: Priority = Priority.NORMAL
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """Get age in seconds"""
        return time.time() - self.created_at
    
    def get_idle_time(self) -> float:
        """Get idle time since last access"""
        return time.time() - self.last_accessed

class IntelligentCache(Generic[T]):
    """Intelligent cache with multiple eviction policies and priority-based retention"""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 100, 
                 policy: CachePolicy = CachePolicy.INTELLIGENCE_AWARE):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.policy = policy
        
        # Cache storage
        self.cache: Dict[str, CacheEntry[T]] = {}
        self.access_order = OrderedDict()  # For LRU
        self.frequency_heap = []  # For LFU
        self.insertion_order = OrderedDict()  # For FIFO
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_memory_bytes = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Intelligence-aware scoring
        self.intelligence_weights = {
            "recency": 0.3,
            "frequency": 0.3,
            "priority": 0.2,
            "size_efficiency": 0.1,
            "semantic_value": 0.1
        }
        
        # Adaptive policy parameters
        self.policy_performance = defaultdict(lambda: {"hits": 0, "total": 0})
        self.last_policy_evaluation = time.time()
        self.policy_evaluation_interval = 300  # 5 minutes
        
        # Metrics
        if METRICS_AVAILABLE:
            self.cache_hits = Counter('intelligent_cache_hits_total', 'Cache hits', ['cache_id'])
            self.cache_misses = Counter('intelligent_cache_misses_total', 'Cache misses', ['cache_id'])
            self.cache_evictions = Counter('intelligent_cache_evictions_total', 'Cache evictions', ['cache_id', 'reason'])
            self.cache_size_gauge = Gauge('intelligent_cache_size', 'Current cache size', ['cache_id'])
            self.cache_memory_gauge = Gauge('intelligent_cache_memory_bytes', 'Current cache memory usage', ['cache_id'])
    
    def get(self, key: str) -> Optional[T]:
        """Get item from cache"""
        with self.lock:
            entry = self.cache.get(key)
            if entry:
                entry.update_access()
                self._update_access_structures(key, entry)
                self.hits += 1
                
                if METRICS_AVAILABLE:
                    self.cache_hits.labels(cache_id=id(self)).inc()
                
                return entry.value
            else:
                self.misses += 1
                
                if METRICS_AVAILABLE:
                    self.cache_misses.labels(cache_id=id(self)).inc()
                
                return None
    
    def put(self, key: str, value: T, priority: Priority = Priority.NORMAL, 
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put item in cache"""
        with self.lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Check if we need to evict
            if key not in self.cache:
                if (len(self.cache) >= self.max_size or 
                    self.current_memory_bytes + size_bytes > self.max_memory_bytes):
                    
                    if not self._evict_items(size_bytes):
                        return False  # Could not make space
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                priority=priority,
                size_bytes=size_bytes,
                metadata=metadata or {}
            )
            
            # Update existing entry
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory_bytes -= old_entry.size_bytes
            
            # Store entry
            self.cache[key] = entry
            self.current_memory_bytes += size_bytes
            
            # Update tracking structures
            self._update_access_structures(key, entry)
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.cache_size_gauge.labels(cache_id=id(self)).set(len(self.cache))
                self.cache_memory_gauge.labels(cache_id=id(self)).set(self.current_memory_bytes)
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove item from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                del self.cache[key]
                self.current_memory_bytes -= entry.size_bytes
                
                # Clean up tracking structures
                self._remove_from_structures(key)
                
                # Update metrics
                if METRICS_AVAILABLE:
                    self.cache_size_gauge.labels(cache_id=id(self)).set(len(self.cache))
                    self.cache_memory_gauge.labels(cache_id=id(self)).set(self.current_memory_bytes)
                
                return True
            return False
    
    def _evict_items(self, needed_bytes: int) -> bool:
        """Evict items based on policy"""
        if self.policy == CachePolicy.ADAPTIVE:
            # Evaluate and potentially switch policy
            self._evaluate_adaptive_policy()
        
        evicted_bytes = 0
        evicted_count = 0
        
        while (len(self.cache) >= self.max_size or 
               self.current_memory_bytes + needed_bytes > self.max_memory_bytes):
            
            victim_key = self._select_eviction_victim()
            if not victim_key:
                break  # No more victims
            
            entry = self.cache[victim_key]
            evicted_bytes += entry.size_bytes
            evicted_count += 1
            
            # Remove victim
            del self.cache[victim_key]
            self.current_memory_bytes -= entry.size_bytes
            self._remove_from_structures(victim_key)
            
            # Update metrics
            if METRICS_AVAILABLE:
                self.cache_evictions.labels(
                    cache_id=id(self),
                    reason=self.policy.value
                ).inc()
            
            if evicted_bytes >= needed_bytes and len(self.cache) < self.max_size:
                break
        
        self.evictions += evicted_count
        log.debug(f"Evicted {evicted_count} items, {evicted_bytes} bytes")
        
        return evicted_bytes >= needed_bytes or len(self.cache) < self.max_size
    
    def _select_eviction_victim(self) -> Optional[str]:
        """Select victim for eviction based on policy"""
        if not self.cache:
            return None
        
        if self.policy == CachePolicy.LRU:
            return self._select_lru_victim()
        elif self.policy == CachePolicy.LFU:
            return self._select_lfu_victim()
        elif self.policy == CachePolicy.FIFO:
            return self._select_fifo_victim()
        elif self.policy == CachePolicy.INTELLIGENCE_AWARE:
            return self._select_intelligence_victim()
        elif self.policy == CachePolicy.ADAPTIVE:
            # Use current best policy
            current_policy = self._get_best_policy()
            temp_policy = self.policy
            self.policy = current_policy
            victim = self._select_eviction_victim()
            self.policy = temp_policy
            return victim
        else:
            return self._select_lru_victim()  # Default fallback
    
    def _select_lru_victim(self) -> Optional[str]:
        """Select LRU victim"""
        if not self.access_order:
            return None
        return next(iter(self.access_order))
    
    def _select_lfu_victim(self) -> Optional[str]:
        """Select LFU victim"""
        if not self.cache:
            return None
        
        # Find item with lowest access count
        min_access = float('inf')
        victim_key = None
        
        for key, entry in self.cache.items():
            if entry.access_count < min_access:
                min_access = entry.access_count
                victim_key = key
        
        return victim_key
    
    def _select_fifo_victim(self) -> Optional[str]:
        """Select FIFO victim"""
        if not self.insertion_order:
            return None
        return next(iter(self.insertion_order))
    
    def _select_intelligence_victim(self) -> Optional[str]:
        """Select victim using intelligence-aware scoring"""
        if not self.cache:
            return None
        
        min_score = float('inf')
        victim_key = None
        
        for key, entry in self.cache.items():
            score = self._calculate_intelligence_score(entry)
            if score < min_score:
                min_score = score
                victim_key = key
        
        return victim_key
    
    def _calculate_intelligence_score(self, entry: CacheEntry[T]) -> float:
        """Calculate intelligence-aware retention score"""
        current_time = time.time()
        
        # Recency score (higher = more recent)
        age = current_time - entry.last_accessed
        recency_score = 1.0 / (1.0 + age / 3600)  # Decay over 1 hour
        
        # Frequency score (higher = more frequent)
        frequency_score = min(1.0, entry.access_count / 10)
        
        # Priority score
        priority_score = entry.priority.value / 4.0
        
        # Size efficiency score (higher = more efficient)
        size_efficiency = 1.0 / (1.0 + entry.size_bytes / 1024)  # Prefer smaller items
        
        # Semantic value score (from metadata)
        semantic_value = entry.metadata.get("semantic_value", 0.5)
        
        # Combine scores
        total_score = (
            self.intelligence_weights["recency"] * recency_score +
            self.intelligence_weights["frequency"] * frequency_score +
            self.intelligence_weights["priority"] * priority_score +
            self.intelligence_weights["size_efficiency"] * size_efficiency +
            self.intelligence_weights["semantic_value"] * semantic_value
        )
        
        return total_score
    
    def _update_access_structures(self, key: str, entry: CacheEntry[T]):
        """Update tracking structures for different policies"""
        # LRU tracking
        if key in self.access_order:
            del self.access_order[key]
        self.access_order[key] = None
        
        # FIFO tracking
        if key not in self.insertion_order:
            self.insertion_order[key] = None
    
    def _remove_from_structures(self, key: str):
        """Remove key from all tracking structures"""
        if key in self.access_order:
            del self.access_order[key]
        if key in self.insertion_order:
            del self.insertion_order[key]
    
    def _estimate_size(self, value: T) -> int:
        """Estimate memory size of value"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, dict):
                return len(json.dumps(value))
            elif hasattr(value, '__sizeof__'):
                return value.__sizeof__()
            else:
                return 1024  # Default estimate
        except:
            return 1024  # Fallback
    
    def _evaluate_adaptive_policy(self):
        """Evaluate and adapt cache policy"""
        current_time = time.time()
        if current_time - self.last_policy_evaluation < self.policy_evaluation_interval:
            return
        
        self.last_policy_evaluation = current_time
        
        # Calculate hit rates for each policy
        best_policy = CachePolicy.LRU
        best_hit_rate = 0.0
        
        for policy_name, stats in self.policy_performance.items():
            if stats["total"] > 0:
                hit_rate = stats["hits"] / stats["total"]
                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    try:
                        best_policy = CachePolicy(policy_name)
                    except ValueError:
                        continue
        
        if best_policy != self.policy:
            log.info(f"Adaptive cache switching from {self.policy.value} to {best_policy.value}")
            self.policy = best_policy
    
    def _get_best_policy(self) -> CachePolicy:
        """Get currently best performing policy"""
        best_policy = CachePolicy.LRU
        best_hit_rate = 0.0
        
        for policy_name, stats in self.policy_performance.items():
            if stats["total"] > 0:
                hit_rate = stats["hits"] / stats["total"]
                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    try:
                        best_policy = CachePolicy(policy_name)
                    except ValueError:
                        continue
        
        return best_policy
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_bytes": self.current_memory_bytes,
                "max_memory_bytes": self.max_memory_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "policy": self.policy.value,
                "utilization": len(self.cache) / self.max_size,
                "memory_utilization": self.current_memory_bytes / self.max_memory_bytes
            }
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.insertion_order.clear()
            self.frequency_heap.clear()
            self.current_memory_bytes = 0
            
            if METRICS_AVAILABLE:
                self.cache_size_gauge.labels(cache_id=id(self)).set(0)
                self.cache_memory_gauge.labels(cache_id=id(self)).set(0)

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 10.0

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        
        # State
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.circuit_open_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Metrics
        if METRICS_AVAILABLE:
            self.circuit_breaker_state = Gauge(
                'circuit_breaker_state',
                'Circuit breaker state (0=closed, 1=open, 2=half_open)',
                ['circuit_name']
            )
            self.circuit_breaker_requests = Counter(
                'circuit_breaker_requests_total',
                'Circuit breaker requests',
                ['circuit_name', 'result']
            )
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            self.total_requests += 1
            
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    # Circuit still open
                    if METRICS_AVAILABLE:
                        self.circuit_breaker_requests.labels(
                            circuit_name=self.name,
                            result="circuit_open"
                        ).inc()
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
                else:
                    # Try to recover
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    
                    if METRICS_AVAILABLE:
                        self.circuit_breaker_state.labels(circuit_name=self.name).set(2)
        
        # Execute function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            else:
                result = func(*args, **kwargs)
            
            # Success
            await self._on_success()
            return result
            
        except Exception as e:
            # Failure
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful request"""
        with self.lock:
            self.successful_requests += 1
            self.failure_count = 0
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    # Recover to closed state
                    self.state = CircuitBreakerState.CLOSED
                    self.success_count = 0
                    log.info(f"Circuit breaker {self.name} recovered to CLOSED")
                    
                    if METRICS_AVAILABLE:
                        self.circuit_breaker_state.labels(circuit_name=self.name).set(0)
            
            if METRICS_AVAILABLE:
                self.circuit_breaker_requests.labels(
                    circuit_name=self.name,
                    result="success"
                ).inc()
    
    async def _on_failure(self):
        """Handle failed request"""
        with self.lock:
            self.failed_requests += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if (self.state == CircuitBreakerState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                # Open circuit
                self.state = CircuitBreakerState.OPEN
                self.circuit_open_count += 1
                log.warning(f"Circuit breaker {self.name} opened due to failures")
                
                if METRICS_AVAILABLE:
                    self.circuit_breaker_state.labels(circuit_name=self.name).set(1)
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Return to open state
                self.state = CircuitBreakerState.OPEN
                log.warning(f"Circuit breaker {self.name} returned to OPEN from HALF_OPEN")
                
                if METRICS_AVAILABLE:
                    self.circuit_breaker_state.labels(circuit_name=self.name).set(1)
            
            if METRICS_AVAILABLE:
                self.circuit_breaker_requests.labels(
                    circuit_name=self.name,
                    result="failure"
                ).inc()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self.lock:
            failure_rate = self.failed_requests / max(1, self.total_requests)
            
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "failure_rate": failure_rate,
                "circuit_open_count": self.circuit_open_count,
                "last_failure_time": self.last_failure_time
            }

class RetryPolicy:
    """Configurable retry policy with exponential backoff"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add jitter to prevent thundering herd
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay

class RetryManager:
    """Manages retry logic with different policies"""
    
    def __init__(self):
        self.retry_policies: Dict[str, RetryPolicy] = {}
        
        # Default policies
        self.retry_policies["default"] = RetryPolicy()
        self.retry_policies["aggressive"] = RetryPolicy(max_attempts=5, base_delay=0.5)
        self.retry_policies["conservative"] = RetryPolicy(max_attempts=2, base_delay=2.0)
        self.retry_policies["critical"] = RetryPolicy(max_attempts=10, base_delay=0.1, max_delay=30.0)
        
        # Statistics
        self.retry_stats = defaultdict(lambda: {"attempts": 0, "successes": 0, "failures": 0})
        
        # Metrics
        if METRICS_AVAILABLE:
            self.retry_attempts = Counter(
                'retry_manager_attempts_total',
                'Retry attempts',
                ['policy', 'result']
            )
    
    async def retry(self, func: Callable, policy_name: str = "default", 
                   *args, **kwargs):
        """Execute function with retry logic"""
        policy = self.retry_policies.get(policy_name, self.retry_policies["default"])
        
        last_exception = None
        
        for attempt in range(1, policy.max_attempts + 1):
            try:
                self.retry_stats[policy_name]["attempts"] += 1
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success
                self.retry_stats[policy_name]["successes"] += 1
                
                if METRICS_AVAILABLE:
                    self.retry_attempts.labels(
                        policy=policy_name,
                        result="success"
                    ).inc()
                
                if attempt > 1:
                    log.info(f"Function succeeded on attempt {attempt} with policy {policy_name}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < policy.max_attempts:
                    delay = policy.calculate_delay(attempt)
                    log.warning(f"Function failed on attempt {attempt}, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    self.retry_stats[policy_name]["failures"] += 1
                    
                    if METRICS_AVAILABLE:
                        self.retry_attempts.labels(
                            policy=policy_name,
                            result="failure"
                        ).inc()
                    
                    log.error(f"Function failed after {policy.max_attempts} attempts with policy {policy_name}: {e}")
        
        # All attempts failed
        raise last_exception
    
    def add_policy(self, name: str, policy: RetryPolicy):
        """Add custom retry policy"""
        self.retry_policies[name] = policy
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics"""
        return dict(self.retry_stats)

class ResourceManager:
    """Manages system resources and connection pools"""
    
    def __init__(self):
        # Connection pools
        self.connection_pools: Dict[str, Any] = {}
        self.pool_configs: Dict[str, Dict[str, Any]] = {}
        
        # Resource limits
        self.memory_limit_mb = 1024  # 1GB default
        self.connection_limit = 100
        self.file_descriptor_limit = 1000
        
        # Resource tracking
        self.active_connections = defaultdict(int)
        self.active_file_descriptors = 0
        
        # Background monitoring
        self.monitoring_task = None
        self.is_monitoring = False
        
        # Cleanup registry
        self.cleanup_callbacks: List[Callable] = []
        
        # Metrics
        if METRICS_AVAILABLE:
            self.memory_usage = Gauge('resource_manager_memory_usage_mb', 'Memory usage in MB')
            self.connection_count = Gauge('resource_manager_connections', 'Active connections', ['pool'])
            self.file_descriptor_count = Gauge('resource_manager_file_descriptors', 'Active file descriptors')
    
    async def initialize(self):
        """Initialize resource manager"""
        log.info("Initializing resource manager")
        
        # Set resource limits
        self._set_resource_limits()
        
        # Start monitoring
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._resource_monitor())
        
        log.info("Resource manager initialized")
    
    async def shutdown(self):
        """Shutdown resource manager"""
        log.info("Shutting down resource manager")
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                log.error(f"Cleanup callback failed: {e}")
        
        # Close connection pools
        for pool_name, pool in self.connection_pools.items():
            try:
                if hasattr(pool, 'close'):
                    if asyncio.iscoroutinefunction(pool.close):
                        await pool.close()
                    else:
                        pool.close()
            except Exception as e:
                log.error(f"Failed to close connection pool {pool_name}: {e}")
        
        log.info("Resource manager shutdown complete")
    
    def _set_resource_limits(self):
        """Set system resource limits"""
        try:
            # Set file descriptor limit
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft < self.file_descriptor_limit:
                new_limit = min(self.file_descriptor_limit, hard)
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
                log.info(f"Set file descriptor limit to {new_limit}")
        except Exception as e:
            log.warning(f"Failed to set resource limits: {e}")
    
    async def _resource_monitor(self):
        """Background resource monitoring"""
        while self.is_monitoring:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Monitor memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                
                if METRICS_AVAILABLE:
                    self.memory_usage.set(memory_mb)
                
                # Check memory limit
                if memory_mb > self.memory_limit_mb:
                    log.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.memory_limit_mb}MB")
                    
                    # Trigger garbage collection
                    gc.collect()
                    
                    # Notify about memory pressure
                    await self._handle_memory_pressure()
                
                # Monitor connections
                total_connections = sum(self.active_connections.values())
                if total_connections > self.connection_limit:
                    log.warning(f"Connection count {total_connections} exceeds limit {self.connection_limit}")
                
                # Monitor file descriptors
                try:
                    fd_count = len(psutil.Process().open_files())
                    self.active_file_descriptors = fd_count
                    
                    if METRICS_AVAILABLE:
                        self.file_descriptor_count.set(fd_count)
                    
                    if fd_count > self.file_descriptor_limit * 0.8:
                        log.warning(f"File descriptor usage {fd_count} approaching limit")
                except:
                    pass  # Not available on all platforms
                
            except Exception as e:
                log.error(f"Resource monitoring error: {e}")
    
    async def _handle_memory_pressure(self):
        """Handle memory pressure situations"""
        # Force garbage collection
        gc.collect()
        
        # Clear caches if available
        # This would integrate with cache managers
        
        # Log memory pressure event
        log.warning("Memory pressure detected, performed cleanup")
    
    def register_cleanup(self, callback: Callable):
        """Register cleanup callback"""
        self.cleanup_callbacks.append(callback)
    
    def create_connection_pool(self, pool_name: str, pool_factory: Callable, 
                             **pool_config) -> Any:
        """Create managed connection pool"""
        try:
            pool = pool_factory(**pool_config)
            self.connection_pools[pool_name] = pool
            self.pool_configs[pool_name] = pool_config
            
            log.info(f"Created connection pool: {pool_name}")
            return pool
            
        except Exception as e:
            log.error(f"Failed to create connection pool {pool_name}: {e}")
            raise
    
    def get_connection_pool(self, pool_name: str) -> Optional[Any]:
        """Get connection pool by name"""
        return self.connection_pools.get(pool_name)
    
    def track_connection(self, pool_name: str, increment: int = 1):
        """Track active connections"""
        self.active_connections[pool_name] += increment
        
        if METRICS_AVAILABLE:
            self.connection_count.labels(pool=pool_name).set(
                self.active_connections[pool_name]
            )
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent()
            
            return {
                "memory_usage_mb": memory_mb,
                "memory_limit_mb": self.memory_limit_mb,
                "memory_utilization": memory_mb / self.memory_limit_mb,
                "cpu_percent": cpu_percent,
                "active_connections": dict(self.active_connections),
                "total_connections": sum(self.active_connections.values()),
                "connection_limit": self.connection_limit,
                "file_descriptors": self.active_file_descriptors,
                "file_descriptor_limit": self.file_descriptor_limit,
                "connection_pools": list(self.connection_pools.keys())
            }
        except Exception as e:
            log.error(f"Failed to get resource stats: {e}")
            return {"error": str(e)}

class PerformanceManager:
    """Main performance manager coordinating all performance components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.memory_cache = IntelligentCache[MemoryItem](
            max_size=config.get("memory_cache_size", 10000),
            max_memory_mb=config.get("memory_cache_mb", 100),
            policy=CachePolicy(config.get("memory_cache_policy", "intelligence_aware"))
        )
        
        self.knowledge_cache = IntelligentCache[Knowledge](
            max_size=config.get("knowledge_cache_size", 5000),
            max_memory_mb=config.get("knowledge_cache_mb", 50),
            policy=CachePolicy(config.get("knowledge_cache_policy", "intelligence_aware"))
        )
        
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager()
        self.resource_manager = ResourceManager()
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
        
        # Performance monitoring
        self.performance_stats = {
            "operations": defaultdict(int),
            "latencies": defaultdict(list),
            "errors": defaultdict(int)
        }
        
        # Metrics
        if METRICS_AVAILABLE:
            self.performance_operations = Counter(
                'performance_manager_operations_total',
                'Performance manager operations',
                ['operation', 'status']
            )
            self.operation_latency = Histogram(
                'performance_manager_operation_latency_seconds',
                'Operation latency',
                ['operation']
            )
    
    async def initialize(self):
        """Initialize performance manager"""
        log.info("Initializing performance manager")
        
        # Initialize resource manager
        await self.resource_manager.initialize()
        
        # Create default circuit breakers
        self._create_default_circuit_breakers()
        
        # Start background tasks
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._cache_optimizer())
        ]
        
        log.info("Performance manager initialized")
    
    async def shutdown(self):
        """Shutdown performance manager"""
        log.info("Shutting down performance manager")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        # Shutdown resource manager
        await self.resource_manager.shutdown()
        
        log.info("Performance manager shutdown complete")
    
    def _create_default_circuit_breakers(self):
        """Create default circuit breakers"""
        # Database operations
        self.circuit_breakers["database"] = CircuitBreaker(
            "database",
            CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0)
        )
        
        # External API calls
        self.circuit_breakers["external_api"] = CircuitBreaker(
            "external_api",
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60.0)
        )
        
        # Memory operations
        self.circuit_breakers["memory"] = CircuitBreaker(
            "memory",
            CircuitBreakerConfig(failure_threshold=10, recovery_timeout=10.0)
        )
    
    async def _performance_monitor(self):
        """Background performance monitoring"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Collect performance metrics
                stats = self.get_comprehensive_stats()
                
                # Log performance summary
                log.info(f"Performance summary: "
                        f"Memory cache hit rate: {stats['memory_cache']['hit_rate']:.2f}, "
                        f"Knowledge cache hit rate: {stats['knowledge_cache']['hit_rate']:.2f}, "
                        f"Memory usage: {stats['resources']['memory_usage_mb']:.1f}MB")
                
            except Exception as e:
                log.error(f"Performance monitoring error: {e}")
    
    async def _cache_optimizer(self):
        """Background cache optimization"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
                # Optimize cache policies based on performance
                await self._optimize_cache_policies()
                
                # Clean up expired entries
                await self._cleanup_caches()
                
            except Exception as e:
                log.error(f"Cache optimization error: {e}")
    
    async def _optimize_cache_policies(self):
        """Optimize cache policies based on performance metrics"""
        # This would implement adaptive policy optimization
        # For now, just log current performance
        memory_stats = self.memory_cache.get_stats()
        knowledge_stats = self.knowledge_cache.get_stats()
        
        if memory_stats["hit_rate"] < 0.5:
            log.info("Memory cache hit rate low, consider policy adjustment")
        
        if knowledge_stats["hit_rate"] < 0.6:
            log.info("Knowledge cache hit rate low, consider policy adjustment")
    
    async def _cleanup_caches(self):
        """Clean up cache entries"""
        # Force garbage collection
        gc.collect()
        
        # Log cleanup
        log.debug("Performed cache cleanup")
    
    # Cache operations
    def cache_memory_item(self, key: str, item: MemoryItem, priority: Priority = Priority.NORMAL) -> bool:
        """Cache memory item"""
        metadata = {
            "semantic_value": 0.8 if item.embedding is not None else 0.5,
            "tier": item.tier.value,
            "importance": item.metadata.get("importance", "normal")
        }
        return self.memory_cache.put(key, item, priority, metadata)
    
    def get_cached_memory_item(self, key: str) -> Optional[MemoryItem]:
        """Get cached memory item"""
        return self.memory_cache.get(key)
    
    def cache_knowledge(self, key: str, knowledge: Knowledge, priority: Priority = Priority.HIGH) -> bool:
        """Cache knowledge item"""
        metadata = {
            "semantic_value": knowledge.confidence,
            "knowledge_type": knowledge.knowledge_type,
            "confidence": knowledge.confidence
        }
        return self.knowledge_cache.put(key, knowledge, priority, metadata)
    
    def get_cached_knowledge(self, key: str) -> Optional[Knowledge]:
        """Get cached knowledge"""
        return self.knowledge_cache.get(key)
    
    # Circuit breaker operations
    async def execute_with_circuit_breaker(self, circuit_name: str, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        circuit_breaker = self.circuit_breakers.get(circuit_name)
        if not circuit_breaker:
            # Create default circuit breaker
            circuit_breaker = CircuitBreaker(
                circuit_name,
                CircuitBreakerConfig()
            )
            self.circuit_breakers[circuit_name] = circuit_breaker
        
        return await circuit_breaker.call(func, *args, **kwargs)
    
    # Retry operations
    async def execute_with_retry(self, func: Callable, policy: str = "default", *args, **kwargs):
        """Execute function with retry logic"""
        return await self.retry_manager.retry(func, policy, *args, **kwargs)
    
    # Performance tracking
    async def track_operation(self, operation_name: str, func: Callable, *args, **kwargs):
        """Track operation performance"""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success
            latency = time.time() - start_time
            self.performance_stats["operations"][operation_name] += 1
            self.performance_stats["latencies"][operation_name].append(latency)
            
            # Keep only recent latencies
            if len(self.performance_stats["latencies"][operation_name]) > 1000:
                self.performance_stats["latencies"][operation_name] = \
                    self.performance_stats["latencies"][operation_name][-1000:]
            
            if METRICS_AVAILABLE:
                self.performance_operations.labels(
                    operation=operation_name,
                    status="success"
                ).inc()
                
                self.operation_latency.labels(operation=operation_name).observe(latency)
            
            return result
            
        except Exception as e:
            # Failure
            self.performance_stats["errors"][operation_name] += 1
            
            if METRICS_AVAILABLE:
                self.performance_operations.labels(
                    operation=operation_name,
                    status="error"
                ).inc()
            
            raise
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "memory_cache": self.memory_cache.get_stats(),
            "knowledge_cache": self.knowledge_cache.get_stats(),
            "circuit_breakers": {
                name: cb.get_stats() 
                for name, cb in self.circuit_breakers.items()
            },
            "retry_stats": self.retry_manager.get_stats(),
            "resources": self.resource_manager.get_resource_stats(),
            "performance": {
                "operations": dict(self.performance_stats["operations"]),
                "errors": dict(self.performance_stats["errors"]),
                "avg_latencies": {
                    op: sum(latencies) / len(latencies) if latencies else 0
                    for op, latencies in self.performance_stats["latencies"].items()
                }
            }
        }
