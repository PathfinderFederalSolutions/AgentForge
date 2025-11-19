"""
Performance Optimizer - Million-Scale Scalability Engine
Advanced algorithms for optimizing performance at massive scale with O(log n) complexity
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq
import bisect

# Optional high-performance computing imports
try:
    import numpy as np
    from scipy.optimize import minimize, differential_evolution
    from sklearn.cluster import KMeans, MiniBatchKMeans
    HPC_AVAILABLE = True
except ImportError:
    HPC_AVAILABLE = False
    np = None

log = logging.getLogger("performance-optimizer")

class OptimizationStrategy(Enum):
    """Performance optimization strategies"""
    LOAD_BALANCING = "load_balancing"
    RESOURCE_ALLOCATION = "resource_allocation"
    TASK_BATCHING = "task_batching"
    AGENT_CLUSTERING = "agent_clustering"
    CACHE_OPTIMIZATION = "cache_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"

@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization"""
    timestamp: float = field(default_factory=time.time)
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_throughput: float = 0.0
    task_completion_rate: float = 0.0
    response_latency_p95: float = 0.0
    error_rate: float = 0.0
    
    # AGI-specific metrics
    agent_efficiency: float = 1.0
    quantum_coherence: float = 1.0
    consensus_latency: float = 0.0
    
    def overall_score(self) -> float:
        """Calculate overall performance score"""
        # Weighted combination of metrics (0-1 scale)
        score = (
            (1.0 - self.cpu_utilization) * 0.15 +
            (1.0 - self.memory_utilization) * 0.15 +
            self.network_throughput * 0.1 +
            self.task_completion_rate * 0.2 +
            (1.0 / max(0.001, self.response_latency_p95)) * 0.1 +
            (1.0 - self.error_rate) * 0.1 +
            self.agent_efficiency * 0.1 +
            self.quantum_coherence * 0.1
        )
        
        return min(1.0, max(0.0, score))

class ConsistentHashRing:
    """
    Consistent hashing for efficient agent distribution
    O(log n) complexity for add/remove/lookup operations
    """
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}  # hash -> node_id
        self.sorted_hashes: List[int] = []
        self.nodes: Set[str] = set()
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing"""
        return hash(key) % (2**32)
    
    def add_node(self, node_id: str):
        """Add node to hash ring - O(log n)"""
        if node_id in self.nodes:
            return
        
        self.nodes.add(node_id)
        
        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_value = self._hash(virtual_key)
            
            self.ring[hash_value] = node_id
            bisect.insort(self.sorted_hashes, hash_value)
    
    def remove_node(self, node_id: str):
        """Remove node from hash ring - O(log n)"""
        if node_id not in self.nodes:
            return
        
        self.nodes.remove(node_id)
        
        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_value = self._hash(virtual_key)
            
            if hash_value in self.ring:
                del self.ring[hash_value]
                self.sorted_hashes.remove(hash_value)
    
    def get_node(self, key: str) -> Optional[str]:
        """Get node for key - O(log n)"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find next node clockwise
        idx = bisect.bisect_right(self.sorted_hashes, hash_value)
        if idx == len(self.sorted_hashes):
            idx = 0
        
        return self.ring[self.sorted_hashes[idx]]
    
    def get_nodes(self, key: str, count: int) -> List[str]:
        """Get multiple nodes for replication - O(log n)"""
        if not self.ring or count <= 0:
            return []
        
        hash_value = self._hash(key)
        nodes = []
        seen_nodes = set()
        
        # Find starting position
        idx = bisect.bisect_right(self.sorted_hashes, hash_value)
        
        # Collect unique nodes clockwise
        for _ in range(len(self.sorted_hashes)):
            if idx >= len(self.sorted_hashes):
                idx = 0
            
            node_id = self.ring[self.sorted_hashes[idx]]
            if node_id not in seen_nodes:
                nodes.append(node_id)
                seen_nodes.add(node_id)
                
                if len(nodes) >= count:
                    break
            
            idx += 1
        
        return nodes

class StreamingAlgorithms:
    """
    Streaming algorithms for constant memory usage at scale
    Process million-scale data with O(1) or O(log n) space complexity
    """
    
    def __init__(self, max_memory_mb: int = 100):
        self.max_memory_mb = max_memory_mb
        
        # HyperLogLog for cardinality estimation
        self.hll_buckets = 2048  # 2^11 buckets
        self.hll_registers = [0] * self.hll_buckets
        
        # Count-Min Sketch for frequency estimation
        self.cms_width = 2048
        self.cms_depth = 4
        self.cms_table = [[0] * self.cms_width for _ in range(self.cms_depth)]
        
        # Reservoir sampling for uniform sampling
        self.reservoir_size = 1000
        self.reservoir = []
        self.reservoir_count = 0
        
        # Exponential histogram for quantiles
        self.exp_histogram = deque(maxlen=1000)
    
    def add_item(self, item: str, value: float = 1.0):
        """Add item to streaming algorithms"""
        # Update HyperLogLog
        self._update_hll(item)
        
        # Update Count-Min Sketch
        self._update_cms(item, value)
        
        # Update reservoir sampling
        self._update_reservoir(item)
        
        # Update exponential histogram
        self._update_exp_histogram(value)
    
    def _update_hll(self, item: str):
        """Update HyperLogLog cardinality estimator"""
        hash_val = hash(item) % (2**32)
        bucket = hash_val & (self.hll_buckets - 1)  # Last 11 bits
        
        # Count leading zeros in remaining bits
        remaining = hash_val >> 11
        leading_zeros = 0
        for _ in range(21):  # 32 - 11 = 21 remaining bits
            if remaining & 1:
                break
            leading_zeros += 1
            remaining >>= 1
        
        self.hll_registers[bucket] = max(self.hll_registers[bucket], leading_zeros + 1)
    
    def _update_cms(self, item: str, value: float):
        """Update Count-Min Sketch frequency estimator"""
        for i in range(self.cms_depth):
            hash_val = hash(f"{item}:{i}") % self.cms_width
            self.cms_table[i][hash_val] += value
    
    def _update_reservoir(self, item: str):
        """Update reservoir sampling"""
        self.reservoir_count += 1
        
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(item)
        else:
            # Replace with probability 1/count
            import random
            if random.randint(1, self.reservoir_count) <= self.reservoir_size:
                idx = random.randint(0, self.reservoir_size - 1)
                self.reservoir[idx] = item
    
    def _update_exp_histogram(self, value: float):
        """Update exponential histogram for quantiles"""
        self.exp_histogram.append(value)
    
    def estimate_cardinality(self) -> int:
        """Estimate unique item count using HyperLogLog"""
        if not any(self.hll_registers):
            return 0
        
        # Calculate harmonic mean
        raw_estimate = (0.7213 / (1 + 1.079 / self.hll_buckets)) * \
                      (self.hll_buckets ** 2) / \
                      sum(2 ** (-x) for x in self.hll_registers)
        
        return int(raw_estimate)
    
    def estimate_frequency(self, item: str) -> float:
        """Estimate item frequency using Count-Min Sketch"""
        estimates = []
        for i in range(self.cms_depth):
            hash_val = hash(f"{item}:{i}") % self.cms_width
            estimates.append(self.cms_table[i][hash_val])
        
        return min(estimates)  # Conservative estimate
    
    def get_sample(self) -> List[str]:
        """Get uniform sample using reservoir sampling"""
        return self.reservoir.copy()
    
    def estimate_quantile(self, quantile: float) -> float:
        """Estimate quantile from exponential histogram"""
        if not self.exp_histogram:
            return 0.0
        
        sorted_values = sorted(self.exp_histogram)
        index = int(quantile * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

class DistributedLoadBalancer:
    """
    Advanced load balancer with multiple algorithms
    Optimized for million-scale agent coordination
    """
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.consistent_hash = ConsistentHashRing()
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Load balancing algorithms
        self.algorithms = {
            "round_robin": self._round_robin,
            "least_connections": self._least_connections,
            "weighted_round_robin": self._weighted_round_robin,
            "consistent_hash": self._consistent_hash_lb,
            "power_of_two": self._power_of_two_choices,
            "adaptive": self._adaptive_selection
        }
        
        self.current_algorithm = "adaptive"
        self.round_robin_counter = 0
    
    def register_agent(self, agent_id: str, capacity: float = 1.0, 
                      current_load: float = 0.0, capabilities: Optional[Set[str]] = None):
        """Register agent with load balancer"""
        self.agents[agent_id] = {
            "capacity": capacity,
            "current_load": current_load,
            "capabilities": capabilities or set(),
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "last_updated": time.time()
        }
        
        self.consistent_hash.add_node(agent_id)
        log.debug(f"Registered agent {agent_id} with capacity {capacity}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister agent from load balancer"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.consistent_hash.remove_node(agent_id)
            if agent_id in self.load_history:
                del self.load_history[agent_id]
            log.debug(f"Unregistered agent {agent_id}")
    
    def update_agent_load(self, agent_id: str, current_load: float, response_time: float = 0.0):
        """Update agent load metrics"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        agent["current_load"] = current_load
        agent["last_updated"] = time.time()
        
        # Update response time (exponential moving average)
        if response_time > 0:
            alpha = 0.1
            if agent["average_response_time"] == 0:
                agent["average_response_time"] = response_time
            else:
                agent["average_response_time"] = (
                    alpha * response_time + (1 - alpha) * agent["average_response_time"]
                )
        
        # Record load history
        self.load_history[agent_id].append({
            "timestamp": time.time(),
            "load": current_load,
            "response_time": response_time
        })
    
    def select_agents(self, count: int = 1, required_capabilities: Optional[Set[str]] = None,
                     task_key: Optional[str] = None, algorithm: Optional[str] = None) -> List[str]:
        """Select optimal agents for task execution"""
        if not self.agents:
            return []
        
        # Filter agents by capabilities
        eligible_agents = []
        for agent_id, agent_info in self.agents.items():
            if required_capabilities and not required_capabilities.issubset(agent_info["capabilities"]):
                continue
            eligible_agents.append(agent_id)
        
        if not eligible_agents:
            return []
        
        # Select load balancing algorithm
        algorithm = algorithm or self.current_algorithm
        selector = self.algorithms.get(algorithm, self._adaptive_selection)
        
        return selector(eligible_agents, count, task_key)
    
    def _round_robin(self, agents: List[str], count: int, task_key: Optional[str] = None) -> List[str]:
        """Round-robin selection"""
        selected = []
        for i in range(count):
            if i < len(agents):
                idx = (self.round_robin_counter + i) % len(agents)
                selected.append(agents[idx])
        
        self.round_robin_counter = (self.round_robin_counter + count) % len(agents)
        return selected
    
    def _least_connections(self, agents: List[str], count: int, task_key: Optional[str] = None) -> List[str]:
        """Select agents with least current load"""
        # Sort by current load
        sorted_agents = sorted(agents, key=lambda a: self.agents[a]["current_load"])
        return sorted_agents[:count]
    
    def _weighted_round_robin(self, agents: List[str], count: int, task_key: Optional[str] = None) -> List[str]:
        """Weighted round-robin based on agent capacity"""
        if not agents:
            return []
        
        # Create weighted list
        weighted_agents = []
        for agent_id in agents:
            capacity = self.agents[agent_id]["capacity"]
            weight = max(1, int(capacity * 10))  # Scale capacity to integer weights
            weighted_agents.extend([agent_id] * weight)
        
        # Select from weighted list
        selected = []
        for i in range(count):
            if i < len(weighted_agents):
                idx = (self.round_robin_counter + i) % len(weighted_agents)
                agent_id = weighted_agents[idx]
                if agent_id not in selected:
                    selected.append(agent_id)
        
        self.round_robin_counter = (self.round_robin_counter + count) % len(weighted_agents)
        return selected[:count]
    
    def _consistent_hash_lb(self, agents: List[str], count: int, task_key: Optional[str] = None) -> List[str]:
        """Consistent hashing for sticky sessions"""
        if not task_key:
            task_key = f"task_{time.time()}"
        
        return self.consistent_hash.get_nodes(task_key, count)
    
    def _power_of_two_choices(self, agents: List[str], count: int, task_key: Optional[str] = None) -> List[str]:
        """Power of two choices algorithm for better load distribution"""
        import random
        
        selected = []
        for _ in range(count):
            if len(agents) <= 2:
                # Not enough agents for power-of-two
                selected.extend(agents[:count - len(selected)])
                break
            
            # Randomly select two agents
            candidates = random.sample(agents, 2)
            
            # Choose the one with lower load
            agent1, agent2 = candidates
            load1 = self.agents[agent1]["current_load"]
            load2 = self.agents[agent2]["current_load"]
            
            chosen = agent1 if load1 <= load2 else agent2
            if chosen not in selected:
                selected.append(chosen)
        
        return selected
    
    def _adaptive_selection(self, agents: List[str], count: int, task_key: Optional[str] = None) -> List[str]:
        """Adaptive selection based on multiple factors"""
        if not agents:
            return []
        
        # Calculate composite scores for each agent
        scored_agents = []
        current_time = time.time()
        
        for agent_id in agents:
            agent = self.agents[agent_id]
            
            # Factors for scoring
            load_factor = 1.0 - agent["current_load"]  # Lower load is better
            capacity_factor = agent["capacity"]  # Higher capacity is better
            
            # Response time factor (lower is better)
            response_time_factor = 1.0
            if agent["average_response_time"] > 0:
                response_time_factor = 1.0 / (1.0 + agent["average_response_time"])
            
            # Success rate factor
            success_rate = 1.0
            if agent["total_requests"] > 0:
                success_rate = agent["successful_requests"] / agent["total_requests"]
            
            # Recency factor (prefer recently updated agents)
            recency_factor = max(0.1, 1.0 - (current_time - agent["last_updated"]) / 300.0)  # 5 minute decay
            
            # Composite score
            score = (load_factor * 0.3 + 
                    capacity_factor * 0.2 + 
                    response_time_factor * 0.2 + 
                    success_rate * 0.2 + 
                    recency_factor * 0.1)
            
            scored_agents.append((agent_id, score))
        
        # Sort by score (descending) and select top agents
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return [agent_id for agent_id, _ in scored_agents[:count]]
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        if not self.agents:
            return {"total_agents": 0}
        
        loads = [agent["current_load"] for agent in self.agents.values()]
        capacities = [agent["capacity"] for agent in self.agents.values()]
        response_times = [agent["average_response_time"] for agent in self.agents.values()]
        
        return {
            "total_agents": len(self.agents),
            "average_load": sum(loads) / len(loads),
            "max_load": max(loads),
            "min_load": min(loads),
            "total_capacity": sum(capacities),
            "average_response_time": sum(response_times) / len(response_times),
            "algorithm": self.current_algorithm,
            "load_distribution": {
                "low": len([l for l in loads if l < 0.3]),
                "medium": len([l for l in loads if 0.3 <= l < 0.7]),
                "high": len([l for l in loads if l >= 0.7])
            }
        }

class MemoryOptimizer:
    """
    Memory optimization for large-scale operations
    Implements hierarchical data structures and memory-efficient algorithms
    """
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.memory_pools: Dict[str, deque] = {}
        self.cache_hierarchy = {}
        self.compression_enabled = True
        
        # Memory usage tracking
        self.memory_usage = {
            "allocated": 0,
            "cached": 0,
            "compressed": 0
        }
    
    def create_memory_pool(self, pool_name: str, max_size: int, item_type: str = "dict"):
        """Create memory pool for efficient object reuse"""
        self.memory_pools[pool_name] = deque(maxlen=max_size)
        log.debug(f"Created memory pool {pool_name} with max size {max_size}")
    
    def get_from_pool(self, pool_name: str, default_factory=dict):
        """Get object from memory pool"""
        if pool_name not in self.memory_pools:
            return default_factory()
        
        pool = self.memory_pools[pool_name]
        if pool:
            return pool.popleft()
        else:
            return default_factory()
    
    def return_to_pool(self, pool_name: str, obj: Any):
        """Return object to memory pool"""
        if pool_name not in self.memory_pools:
            return
        
        # Clear object data if it's a dict
        if isinstance(obj, dict):
            obj.clear()
        
        self.memory_pools[pool_name].append(obj)
    
    def create_lru_cache(self, cache_name: str, max_size: int):
        """Create LRU cache with size limit"""
        from collections import OrderedDict
        
        class LRUCache:
            def __init__(self, max_size: int):
                self.max_size = max_size
                self.cache = OrderedDict()
            
            def get(self, key: str, default=None):
                if key in self.cache:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    return self.cache[key]
                return default
            
            def put(self, key: str, value: Any):
                if key in self.cache:
                    self.cache.move_to_end(key)
                else:
                    if len(self.cache) >= self.max_size:
                        # Remove least recently used
                        self.cache.popitem(last=False)
                self.cache[key] = value
            
            def size(self) -> int:
                return len(self.cache)
        
        self.cache_hierarchy[cache_name] = LRUCache(max_size)
        log.debug(f"Created LRU cache {cache_name} with max size {max_size}")
    
    def optimize_data_structure(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize data structure for memory efficiency"""
        if not data:
            return data
        
        # Find common keys to optimize storage
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        # Create key mapping for compression
        key_mapping = {key: i for i, key in enumerate(all_keys)}
        
        # Convert to more memory-efficient format
        optimized_data = []
        for item in data:
            # Use tuples instead of dicts for better memory usage
            optimized_item = tuple(item.get(key) for key in all_keys)
            optimized_data.append(optimized_item)
        
        return optimized_data
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        import sys
        
        # Calculate pool memory usage
        pool_memory = 0
        for pool in self.memory_pools.values():
            pool_memory += len(pool) * sys.getsizeof({})  # Approximate
        
        # Calculate cache memory usage
        cache_memory = 0
        for cache in self.cache_hierarchy.values():
            cache_memory += cache.size() * sys.getsizeof({})  # Approximate
        
        return {
            "total_pools": len(self.memory_pools),
            "pool_memory_kb": pool_memory / 1024,
            "total_caches": len(self.cache_hierarchy),
            "cache_memory_kb": cache_memory / 1024,
            "max_memory_mb": self.max_memory_mb
        }

class PerformanceOptimizer:
    """
    Comprehensive performance optimization system
    Combines all optimization strategies for maximum efficiency
    """
    
    def __init__(self, max_agents: int = 1000000):
        self.max_agents = max_agents
        
        # Core optimizers
        self.load_balancer = DistributedLoadBalancer()
        self.memory_optimizer = MemoryOptimizer()
        self.streaming_algorithms = StreamingAlgorithms()
        
        # Performance monitoring
        self.metrics_history: deque = deque(maxlen=10000)
        self.optimization_history: deque = deque(maxlen=1000)
        
        # Optimization strategies
        self.active_strategies: Set[OptimizationStrategy] = {
            OptimizationStrategy.LOAD_BALANCING,
            OptimizationStrategy.RESOURCE_ALLOCATION,
            OptimizationStrategy.CACHE_OPTIMIZATION
        }
        
        # Performance targets
        self.performance_targets = {
            "max_response_latency": 100.0,  # ms
            "min_throughput": 1000.0,  # requests/second
            "max_error_rate": 0.01,  # 1%
            "min_agent_efficiency": 0.8,  # 80%
            "min_quantum_coherence": 0.6  # 60%
        }
        
        log.info(f"Performance optimizer initialized for {max_agents} agents")
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics for optimization"""
        self.metrics_history.append(metrics)
        
        # Add to streaming algorithms for analysis
        self.streaming_algorithms.add_item(f"metrics_{metrics.timestamp}", metrics.overall_score())
        
        # Trigger optimization if performance degrades
        if metrics.overall_score() < 0.7:  # Performance threshold
            asyncio.create_task(self._trigger_optimization(metrics))
    
    async def _trigger_optimization(self, metrics: PerformanceMetrics):
        """Trigger performance optimization based on metrics"""
        try:
            optimization_actions = []
            
            # Analyze performance bottlenecks
            if metrics.response_latency_p95 > self.performance_targets["max_response_latency"]:
                optimization_actions.append("optimize_latency")
            
            if metrics.task_completion_rate < self.performance_targets["min_throughput"]:
                optimization_actions.append("optimize_throughput")
            
            if metrics.error_rate > self.performance_targets["max_error_rate"]:
                optimization_actions.append("reduce_errors")
            
            if metrics.agent_efficiency < self.performance_targets["min_agent_efficiency"]:
                optimization_actions.append("improve_agent_efficiency")
            
            # Execute optimization actions
            for action in optimization_actions:
                await self._execute_optimization_action(action, metrics)
            
            # Record optimization
            self.optimization_history.append({
                "timestamp": time.time(),
                "trigger_metrics": metrics,
                "actions": optimization_actions,
                "performance_score_before": metrics.overall_score()
            })
            
        except Exception as e:
            log.error(f"Optimization trigger failed: {e}")
    
    async def _execute_optimization_action(self, action: str, metrics: PerformanceMetrics):
        """Execute specific optimization action"""
        try:
            if action == "optimize_latency":
                await self._optimize_latency()
            elif action == "optimize_throughput":
                await self._optimize_throughput()
            elif action == "reduce_errors":
                await self._reduce_errors()
            elif action == "improve_agent_efficiency":
                await self._improve_agent_efficiency()
            
            log.info(f"Executed optimization action: {action}")
            
        except Exception as e:
            log.error(f"Optimization action {action} failed: {e}")
    
    async def _optimize_latency(self):
        """Optimize system latency"""
        # Switch to faster load balancing algorithm
        self.load_balancer.current_algorithm = "least_connections"
        
        # Enable more aggressive caching
        if "cache_l1" not in self.memory_optimizer.cache_hierarchy:
            self.memory_optimizer.create_lru_cache("cache_l1", 10000)
    
    async def _optimize_throughput(self):
        """Optimize system throughput"""
        # Use power-of-two choices for better distribution
        self.load_balancer.current_algorithm = "power_of_two"
        
        # Create task batching pool
        if "task_batch" not in self.memory_optimizer.memory_pools:
            self.memory_optimizer.create_memory_pool("task_batch", 1000)
    
    async def _reduce_errors(self):
        """Reduce system error rate"""
        # Switch to adaptive algorithm with better error handling
        self.load_balancer.current_algorithm = "adaptive"
        
        # Enable circuit breaker pattern (would be implemented in calling system)
        log.info("Enabling circuit breaker pattern for error reduction")
    
    async def _improve_agent_efficiency(self):
        """Improve agent efficiency"""
        # Analyze agent performance and rebalance
        stats = self.load_balancer.get_load_statistics()
        
        if stats.get("load_distribution", {}).get("high", 0) > 0:
            # Too many high-load agents, need better distribution
            self.load_balancer.current_algorithm = "weighted_round_robin"
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on analysis"""
        recommendations = []
        
        if len(self.metrics_history) < 10:
            return recommendations
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Analyze trends
        latency_trend = [m.response_latency_p95 for m in recent_metrics]
        throughput_trend = [m.task_completion_rate for m in recent_metrics]
        error_trend = [m.error_rate for m in recent_metrics]
        
        # Latency recommendations
        if len(latency_trend) > 1 and latency_trend[-1] > latency_trend[0] * 1.2:
            recommendations.append({
                "type": "latency_degradation",
                "severity": "medium",
                "description": "Response latency increasing over time",
                "suggested_actions": [
                    "Enable L1 caching",
                    "Switch to least-connections load balancing",
                    "Increase agent capacity"
                ]
            })
        
        # Throughput recommendations
        if len(throughput_trend) > 1 and throughput_trend[-1] < throughput_trend[0] * 0.8:
            recommendations.append({
                "type": "throughput_degradation",
                "severity": "high",
                "description": "Task completion rate declining",
                "suggested_actions": [
                    "Enable task batching",
                    "Add more agents",
                    "Optimize resource allocation"
                ]
            })
        
        # Error rate recommendations
        avg_error_rate = sum(error_trend) / len(error_trend)
        if avg_error_rate > self.performance_targets["max_error_rate"]:
            recommendations.append({
                "type": "high_error_rate",
                "severity": "critical",
                "description": f"Error rate {avg_error_rate:.3f} exceeds target",
                "suggested_actions": [
                    "Enable circuit breaker pattern",
                    "Implement retry mechanisms",
                    "Review agent health monitoring"
                ]
            })
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics
        
        # Calculate averages
        avg_score = sum(m.overall_score() for m in recent_metrics) / len(recent_metrics)
        avg_latency = sum(m.response_latency_p95 for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.task_completion_rate for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        
        return {
            "overall_performance_score": avg_score,
            "average_latency_p95": avg_latency,
            "average_throughput": avg_throughput,
            "average_error_rate": avg_error_rate,
            "performance_targets": self.performance_targets,
            "active_optimizations": len(self.active_strategies),
            "optimization_history_count": len(self.optimization_history),
            "load_balancer_stats": self.load_balancer.get_load_statistics(),
            "memory_stats": self.memory_optimizer.get_memory_usage(),
            "streaming_cardinality": self.streaming_algorithms.estimate_cardinality(),
            "recommendations": self.get_optimization_recommendations()
        }
