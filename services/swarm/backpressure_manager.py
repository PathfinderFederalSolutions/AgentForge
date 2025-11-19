"""
Advanced Backpressure Management for Million-Scale Agent Coordination
Implements sophisticated backpressure algorithms to prevent system overload
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import math

# Metrics imports (graceful degradation)
try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = lambda *args, **kwargs: None

log = logging.getLogger("backpressure-manager")

class BackpressureLevel(Enum):
    """Backpressure severity levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class BackpressureStrategy(Enum):
    """Backpressure mitigation strategies"""
    RATE_LIMIT = "rate_limit"
    BATCH_REDUCE = "batch_reduce"
    PRIORITY_SHED = "priority_shed"
    CIRCUIT_BREAK = "circuit_break"
    SCALE_OUT = "scale_out"

@dataclass
class BackpressureThresholds:
    """Configurable backpressure thresholds"""
    # Queue depth thresholds
    queue_low: int = 1000
    queue_medium: int = 5000
    queue_high: int = 10000
    queue_critical: int = 50000
    
    # Latency thresholds (seconds)
    latency_low: float = 0.1
    latency_medium: float = 0.5
    latency_high: float = 1.0
    latency_critical: float = 5.0
    
    # CPU/Memory thresholds (percentage)
    cpu_high: float = 70.0
    cpu_critical: float = 90.0
    memory_high: float = 80.0
    memory_critical: float = 95.0
    
    # Error rate thresholds (per second)
    error_rate_high: float = 10.0
    error_rate_critical: float = 50.0

@dataclass
class BackpressureMetrics:
    """Current system metrics for backpressure calculation"""
    queue_depth: int = 0
    avg_latency: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    active_agents: int = 0
    message_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class BackpressureAction:
    """Backpressure mitigation action"""
    strategy: BackpressureStrategy
    severity: BackpressureLevel
    parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0  # How long to apply action

class AdaptiveBatchController:
    """Advanced adaptive batching with backpressure awareness"""
    
    def __init__(self, initial_batch: int = 10, min_batch: int = 1, max_batch: int = 1000):
        self.current_batch = initial_batch
        self.min_batch = min_batch
        self.max_batch = max_batch
        
        # Performance tracking
        self.latency_history: List[float] = []
        self.throughput_history: List[float] = []
        self.success_rate_history: List[float] = []
        
        # Adaptive parameters
        self.alpha = 0.2  # Smoothing factor for EMA
        self.ema_latency = 0.0
        self.ema_throughput = 0.0
        self.ema_success_rate = 1.0
        
        # Metrics
        if METRICS_AVAILABLE:
            import uuid
            instance_id = str(uuid.uuid4())[:8]
            try:
                self.batch_size_gauge = Gauge(f'adaptive_batch_size_{instance_id}', 'Current adaptive batch size')
                self.batch_efficiency_gauge = Gauge(f'adaptive_batch_efficiency_{instance_id}', 'Batch processing efficiency')
            except Exception:
                # If metrics already exist, ignore (for testing)
                self.batch_size_gauge = None
                self.batch_efficiency_gauge = None
    
    def record_batch_result(self, batch_size: int, latency: float, success_count: int):
        """Record results from processing a batch"""
        # Calculate metrics
        throughput = batch_size / latency if latency > 0 else 0
        success_rate = success_count / batch_size if batch_size > 0 else 0
        
        # Update EMAs
        if self.ema_latency == 0.0:
            self.ema_latency = latency
            self.ema_throughput = throughput
            self.ema_success_rate = success_rate
        else:
            self.ema_latency = self.alpha * latency + (1 - self.alpha) * self.ema_latency
            self.ema_throughput = self.alpha * throughput + (1 - self.alpha) * self.ema_throughput
            self.ema_success_rate = self.alpha * success_rate + (1 - self.alpha) * self.ema_success_rate
        
        # Update histories (keep last 100 samples)
        self.latency_history.append(latency)
        self.throughput_history.append(throughput)
        self.success_rate_history.append(success_rate)
        
        for history in [self.latency_history, self.throughput_history, self.success_rate_history]:
            if len(history) > 100:
                history.pop(0)
        
        # Update metrics
        if METRICS_AVAILABLE and self.batch_size_gauge and self.batch_efficiency_gauge:
            self.batch_size_gauge.set(self.current_batch)
            efficiency = success_rate * throughput / (latency + 0.001)  # Avoid division by zero
            self.batch_efficiency_gauge.set(efficiency)
    
    def calculate_optimal_batch(self, backpressure_level: BackpressureLevel, queue_depth: int) -> int:
        """Calculate optimal batch size based on current conditions"""
        # Base adjustment based on backpressure level
        if backpressure_level == BackpressureLevel.CRITICAL:
            target_batch = max(1, self.current_batch // 4)
        elif backpressure_level == BackpressureLevel.HIGH:
            target_batch = max(1, self.current_batch // 2)
        elif backpressure_level == BackpressureLevel.MEDIUM:
            target_batch = max(1, int(self.current_batch * 0.8))
        elif backpressure_level == BackpressureLevel.LOW:
            target_batch = min(self.max_batch, int(self.current_batch * 1.1))
        else:  # NONE
            target_batch = min(self.max_batch, int(self.current_batch * 1.2))
        
        # Adjust based on queue depth
        if queue_depth > 1000:
            queue_factor = min(2.0, queue_depth / 1000)
            target_batch = max(1, int(target_batch / queue_factor))
        
        # Adjust based on performance history
        if len(self.latency_history) >= 10:
            recent_latency_trend = sum(self.latency_history[-5:]) / sum(self.latency_history[-10:-5])
            if recent_latency_trend > 1.2:  # Latency increasing
                target_batch = max(1, int(target_batch * 0.9))
            elif recent_latency_trend < 0.8:  # Latency decreasing
                target_batch = min(self.max_batch, int(target_batch * 1.1))
        
        # Smooth transitions
        if target_batch > self.current_batch:
            self.current_batch = min(target_batch, self.current_batch + max(1, self.current_batch // 10))
        else:
            self.current_batch = max(target_batch, self.current_batch - max(1, self.current_batch // 10))
        
        return max(self.min_batch, min(self.max_batch, self.current_batch))

class RateLimiter:
    """Token bucket rate limiter with burst capacity"""
    
    def __init__(self, rate: float, burst: int):
        self.rate = rate  # tokens per second
        self.burst = burst  # maximum burst size
        self.tokens = burst
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket"""
        async with self.lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: float = 10.0) -> bool:
        """Wait for tokens to become available"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self.acquire(tokens):
                return True
            await asyncio.sleep(0.01)  # 10ms polling
        return False

class BackpressureManager:
    """Comprehensive backpressure management system"""
    
    def __init__(self, thresholds: Optional[BackpressureThresholds] = None):
        self.thresholds = thresholds or BackpressureThresholds()
        self.current_level = BackpressureLevel.NONE
        self.active_actions: List[BackpressureAction] = []
        
        # Components
        self.batch_controller = AdaptiveBatchController()
        self.rate_limiters: Dict[str, RateLimiter] = {}
        
        # State tracking
        self.metrics_history: List[BackpressureMetrics] = []
        self.action_history: List[BackpressureAction] = []
        
        # Callbacks for actions
        self.action_callbacks: Dict[BackpressureStrategy, Callable] = {}
        
        # Metrics
        if METRICS_AVAILABLE:
            self.backpressure_level_gauge = Gauge('backpressure_level', 'Current backpressure level')
            self.backpressure_actions_counter = Counter('backpressure_actions_total', 'Backpressure actions taken', ['strategy'])
            self.queue_depth_gauge = Gauge('backpressure_queue_depth', 'Current queue depth')
            self.system_latency_gauge = Gauge('backpressure_system_latency_seconds', 'Current system latency')
    
    def register_action_callback(self, strategy: BackpressureStrategy, callback: Callable):
        """Register callback for backpressure action"""
        self.action_callbacks[strategy] = callback
    
    def update_metrics(self, metrics: BackpressureMetrics):
        """Update current system metrics"""
        self.metrics_history.append(metrics)
        
        # Keep last 1000 metrics (about 16 minutes at 1 update/second)
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
        
        # Update Prometheus metrics
        if METRICS_AVAILABLE:
            self.queue_depth_gauge.set(metrics.queue_depth)
            self.system_latency_gauge.set(metrics.avg_latency)
        
        # Evaluate backpressure level
        new_level = self._calculate_backpressure_level(metrics)
        if new_level != self.current_level:
            log.info(f"Backpressure level changed: {self.current_level.name} -> {new_level.name}")
            self.current_level = new_level
            
            if METRICS_AVAILABLE:
                self.backpressure_level_gauge.set(new_level.value)
        
        # Take actions if necessary
        self._evaluate_actions(metrics)
    
    def _calculate_backpressure_level(self, metrics: BackpressureMetrics) -> BackpressureLevel:
        """Calculate current backpressure level based on metrics"""
        level_scores = {
            BackpressureLevel.NONE: 0,
            BackpressureLevel.LOW: 0,
            BackpressureLevel.MEDIUM: 0,
            BackpressureLevel.HIGH: 0,
            BackpressureLevel.CRITICAL: 0
        }
        
        # Queue depth scoring
        if metrics.queue_depth >= self.thresholds.queue_critical:
            level_scores[BackpressureLevel.CRITICAL] += 3
        elif metrics.queue_depth >= self.thresholds.queue_high:
            level_scores[BackpressureLevel.HIGH] += 2
        elif metrics.queue_depth >= self.thresholds.queue_medium:
            level_scores[BackpressureLevel.MEDIUM] += 1
        elif metrics.queue_depth >= self.thresholds.queue_low:
            level_scores[BackpressureLevel.LOW] += 1
        
        # Latency scoring
        if metrics.avg_latency >= self.thresholds.latency_critical:
            level_scores[BackpressureLevel.CRITICAL] += 3
        elif metrics.avg_latency >= self.thresholds.latency_high:
            level_scores[BackpressureLevel.HIGH] += 2
        elif metrics.avg_latency >= self.thresholds.latency_medium:
            level_scores[BackpressureLevel.MEDIUM] += 1
        elif metrics.avg_latency >= self.thresholds.latency_low:
            level_scores[BackpressureLevel.LOW] += 1
        
        # Resource usage scoring
        if metrics.cpu_usage >= self.thresholds.cpu_critical or metrics.memory_usage >= self.thresholds.memory_critical:
            level_scores[BackpressureLevel.CRITICAL] += 2
        elif metrics.cpu_usage >= self.thresholds.cpu_high or metrics.memory_usage >= self.thresholds.memory_high:
            level_scores[BackpressureLevel.HIGH] += 1
        
        # Error rate scoring
        if metrics.error_rate >= self.thresholds.error_rate_critical:
            level_scores[BackpressureLevel.CRITICAL] += 2
        elif metrics.error_rate >= self.thresholds.error_rate_high:
            level_scores[BackpressureLevel.HIGH] += 1
        
        # Return highest scoring level
        for level in reversed(list(BackpressureLevel)):
            if level_scores[level] > 0:
                return level
        
        return BackpressureLevel.NONE
    
    def _evaluate_actions(self, metrics: BackpressureMetrics):
        """Evaluate and take backpressure actions"""
        actions_to_take = []
        
        # Remove expired actions
        current_time = time.time()
        self.active_actions = [
            action for action in self.active_actions 
            if current_time - action.timestamp < action.duration
        ]
        
        # Determine new actions based on level
        if self.current_level == BackpressureLevel.CRITICAL:
            if not any(a.strategy == BackpressureStrategy.CIRCUIT_BREAK for a in self.active_actions):
                actions_to_take.append(BackpressureAction(
                    strategy=BackpressureStrategy.CIRCUIT_BREAK,
                    severity=self.current_level,
                    parameters={"duration": 60.0},
                    duration=60.0
                ))
            
            if not any(a.strategy == BackpressureStrategy.PRIORITY_SHED for a in self.active_actions):
                actions_to_take.append(BackpressureAction(
                    strategy=BackpressureStrategy.PRIORITY_SHED,
                    severity=self.current_level,
                    parameters={"shed_percentage": 50},
                    duration=120.0
                ))
        
        elif self.current_level == BackpressureLevel.HIGH:
            if not any(a.strategy == BackpressureStrategy.RATE_LIMIT for a in self.active_actions):
                # Reduce rate to 50% of current
                target_rate = max(10, metrics.message_rate * 0.5)
                actions_to_take.append(BackpressureAction(
                    strategy=BackpressureStrategy.RATE_LIMIT,
                    severity=self.current_level,
                    parameters={"rate": target_rate, "burst": int(target_rate * 2)},
                    duration=300.0
                ))
            
            if not any(a.strategy == BackpressureStrategy.BATCH_REDUCE for a in self.active_actions):
                actions_to_take.append(BackpressureAction(
                    strategy=BackpressureStrategy.BATCH_REDUCE,
                    severity=self.current_level,
                    parameters={"reduction_factor": 0.5},
                    duration=180.0
                ))
        
        elif self.current_level == BackpressureLevel.MEDIUM:
            if not any(a.strategy == BackpressureStrategy.BATCH_REDUCE for a in self.active_actions):
                actions_to_take.append(BackpressureAction(
                    strategy=BackpressureStrategy.BATCH_REDUCE,
                    severity=self.current_level,
                    parameters={"reduction_factor": 0.8},
                    duration=120.0
                ))
        
        # Execute actions
        for action in actions_to_take:
            self._execute_action(action)
            self.active_actions.append(action)
            self.action_history.append(action)
            
            if METRICS_AVAILABLE:
                self.backpressure_actions_counter.labels(strategy=action.strategy.value).inc()
            
            # Keep action history limited
            if len(self.action_history) > 1000:
                self.action_history.pop(0)
    
    def _execute_action(self, action: BackpressureAction):
        """Execute a backpressure action"""
        log.info(f"Executing backpressure action: {action.strategy.value} with params {action.parameters}")
        
        if action.strategy == BackpressureStrategy.RATE_LIMIT:
            # Create or update rate limiter
            rate = action.parameters.get("rate", 100)
            burst = action.parameters.get("burst", rate * 2)
            self.rate_limiters["global"] = RateLimiter(rate, burst)
        
        elif action.strategy == BackpressureStrategy.BATCH_REDUCE:
            # Update batch controller
            reduction = action.parameters.get("reduction_factor", 0.8)
            current_batch = self.batch_controller.current_batch
            self.batch_controller.current_batch = max(1, int(current_batch * reduction))
        
        # Call registered callback if available
        if action.strategy in self.action_callbacks:
            try:
                self.action_callbacks[action.strategy](action)
            except Exception as e:
                log.error(f"Error executing action callback for {action.strategy}: {e}")
    
    async def should_process_message(self, priority: int = 1) -> bool:
        """Check if message should be processed based on current backpressure"""
        # Check rate limiting
        if "global" in self.rate_limiters:
            if not await self.rate_limiters["global"].acquire():
                return False
        
        # Check priority shedding
        for action in self.active_actions:
            if action.strategy == BackpressureStrategy.PRIORITY_SHED:
                shed_percentage = action.parameters.get("shed_percentage", 0)
                if priority <= 5 and (time.time() % 100) < shed_percentage:
                    return False
        
        # Check circuit breaking
        for action in self.active_actions:
            if action.strategy == BackpressureStrategy.CIRCUIT_BREAK:
                return False
        
        return True
    
    def get_optimal_batch_size(self, queue_depth: int) -> int:
        """Get optimal batch size for current conditions"""
        return self.batch_controller.calculate_optimal_batch(self.current_level, queue_depth)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive backpressure status"""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            "level": self.current_level.name,
            "level_value": self.current_level.value,
            "active_actions": [
                {
                    "strategy": action.strategy.value,
                    "severity": action.severity.name,
                    "parameters": action.parameters,
                    "age": time.time() - action.timestamp,
                    "remaining": max(0, action.duration - (time.time() - action.timestamp))
                }
                for action in self.active_actions
            ],
            "current_batch_size": self.batch_controller.current_batch,
            "rate_limiters": {
                name: {"rate": rl.rate, "burst": rl.burst, "tokens": rl.tokens}
                for name, rl in self.rate_limiters.items()
            },
            "latest_metrics": {
                "queue_depth": latest_metrics.queue_depth if latest_metrics else 0,
                "avg_latency": latest_metrics.avg_latency if latest_metrics else 0,
                "cpu_usage": latest_metrics.cpu_usage if latest_metrics else 0,
                "memory_usage": latest_metrics.memory_usage if latest_metrics else 0,
                "error_rate": latest_metrics.error_rate if latest_metrics else 0,
                "message_rate": latest_metrics.message_rate if latest_metrics else 0,
            }
        }

# Global backpressure manager instance
global_backpressure_manager: Optional[BackpressureManager] = None

def get_backpressure_manager() -> BackpressureManager:
    """Get or create global backpressure manager"""
    global global_backpressure_manager
    if global_backpressure_manager is None:
        global_backpressure_manager = BackpressureManager()
    return global_backpressure_manager
