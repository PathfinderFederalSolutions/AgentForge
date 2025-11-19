"""
Performance Optimizer for Neural Mesh
Optimizes memory access, query performance, and resource utilization
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import psutil
import threading

log = logging.getLogger("performance-optimizer")

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    cpu_usage: float
    memory_usage: float
    query_latency: float
    throughput: float
    cache_hit_rate: float
    timestamp: float

class PerformanceOptimizer:
    """
    Performance optimizer for neural mesh operations
    """
    
    def __init__(self):
        self.metrics_history = []
        self.optimization_rules = {}
        self.monitoring_active = False
        self.monitor_thread = None
        
    async def optimize_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a query for better performance"""
        try:
            # Analyze query complexity
            complexity = self._analyze_query_complexity(query)
            
            # Apply optimization strategies
            optimized_query = self._apply_optimizations(query, complexity)
            
            return {
                "optimized_query": optimized_query,
                "optimization_applied": True,
                "estimated_improvement": f"{complexity * 10}% faster",
                "strategy": "query_optimization"
            }
            
        except Exception as e:
            log.error(f"Query optimization failed: {e}")
            return {
                "optimized_query": query,
                "optimization_applied": False,
                "error": str(e)
            }
    
    def _analyze_query_complexity(self, query: Dict[str, Any]) -> float:
        """Analyze query complexity (0.0 to 1.0)"""
        complexity = 0.1  # Base complexity
        
        # Add complexity based on query features
        if query.get("filters"):
            complexity += len(query["filters"]) * 0.1
        
        if query.get("joins"):
            complexity += len(query["joins"]) * 0.2
        
        if query.get("aggregations"):
            complexity += len(query["aggregations"]) * 0.15
        
        return min(complexity, 1.0)
    
    def _apply_optimizations(self, query: Dict[str, Any], complexity: float) -> Dict[str, Any]:
        """Apply optimization strategies based on complexity"""
        optimized = query.copy()
        
        # Add query hints for high complexity queries
        if complexity > 0.5:
            optimized["hints"] = ["use_index", "parallel_execution"]
        
        # Add caching for medium complexity
        if complexity > 0.3:
            optimized["cache_enabled"] = True
            optimized["cache_ttl"] = 300  # 5 minutes
        
        return optimized
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_performance)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            log.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        log.info("Performance monitoring stopped")
    
    def _monitor_performance(self):
        """Monitor system performance metrics"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                metrics = PerformanceMetrics(
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    query_latency=self._estimate_query_latency(),
                    throughput=self._estimate_throughput(),
                    cache_hit_rate=self._estimate_cache_hit_rate(),
                    timestamp=time.time()
                )
                
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                log.error(f"Performance monitoring error: {e}")
                time.sleep(10)
    
    def _estimate_query_latency(self) -> float:
        """Estimate current query latency"""
        # Simulate latency estimation based on system load
        cpu_usage = psutil.cpu_percent()
        base_latency = 0.1  # 100ms base
        load_factor = cpu_usage / 100.0
        return base_latency * (1 + load_factor)
    
    def _estimate_throughput(self) -> float:
        """Estimate current throughput (queries/second)"""
        # Simulate throughput estimation
        cpu_usage = psutil.cpu_percent()
        max_throughput = 1000  # queries/second
        efficiency = max(0.1, 1.0 - (cpu_usage / 100.0))
        return max_throughput * efficiency
    
    def _estimate_cache_hit_rate(self) -> float:
        """Estimate cache hit rate"""
        # Simulate cache hit rate (typically 80-95%)
        return 0.85 + (0.1 * (1.0 - psutil.cpu_percent() / 100.0))
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get current performance report"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No performance data available"}
        
        latest = self.metrics_history[-1]
        
        # Calculate averages over last 10 minutes
        recent_metrics = [m for m in self.metrics_history 
                         if time.time() - m.timestamp < 600]
        
        if recent_metrics:
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_latency = sum(m.query_latency for m in recent_metrics) / len(recent_metrics)
        else:
            avg_cpu = latest.cpu_usage
            avg_memory = latest.memory_usage
            avg_latency = latest.query_latency
        
        return {
            "current": {
                "cpu_usage": latest.cpu_usage,
                "memory_usage": latest.memory_usage,
                "query_latency": latest.query_latency,
                "throughput": latest.throughput,
                "cache_hit_rate": latest.cache_hit_rate
            },
            "averages_10min": {
                "cpu_usage": avg_cpu,
                "memory_usage": avg_memory,
                "query_latency": avg_latency
            },
            "status": "healthy" if avg_cpu < 80 and avg_memory < 85 else "warning",
            "recommendations": self._generate_recommendations(avg_cpu, avg_memory, avg_latency)
        }
    
    def _generate_recommendations(self, cpu: float, memory: float, latency: float) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if cpu > 80:
            recommendations.append("High CPU usage detected - consider scaling horizontally")
        
        if memory > 85:
            recommendations.append("High memory usage - consider increasing cache size or memory limits")
        
        if latency > 0.5:
            recommendations.append("High query latency - consider query optimization or indexing")
        
        if not recommendations:
            recommendations.append("Performance is optimal")
        
        return recommendations
