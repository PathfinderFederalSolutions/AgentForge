"""
Reliability and Fault Tolerance Framework for Intelligence Fusion Systems
Comprehensive error recovery, graceful degradation, and system resilience
"""

import asyncio
import numpy as np
import time
import json
import traceback
import logging
import threading
import weakref
from typing import List, Dict, Any, Tuple, Optional, Set, Union, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
import concurrent.futures
import pickle
import hashlib

log = logging.getLogger("reliability-framework")

class SystemHealth(Enum):
    """System health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"

class FailureMode(Enum):
    """Types of system failures"""
    COMPONENT_FAILURE = "component_failure"
    NETWORK_FAILURE = "network_failure"
    DATA_CORRUPTION = "data_corruption"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT_FAILURE = "timeout_failure"
    SECURITY_BREACH = "security_breach"
    CONFIGURATION_ERROR = "configuration_error"
    EXTERNAL_DEPENDENCY = "external_dependency"

class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes"""
    RESTART_COMPONENT = "restart_component"
    FALLBACK_ALGORITHM = "fallback_algorithm"
    REDUNDANT_PROCESSING = "redundant_processing"
    CACHED_RESPONSE = "cached_response"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    MANUAL_INTERVENTION = "manual_intervention"
    SYSTEM_ROLLBACK = "system_rollback"

class DegradationLevel(Enum):
    """Levels of system degradation"""
    FULL_CAPABILITY = "full_capability"
    REDUCED_ACCURACY = "reduced_accuracy"
    REDUCED_THROUGHPUT = "reduced_throughput"
    BASIC_FUNCTIONALITY = "basic_functionality"
    EMERGENCY_MODE = "emergency_mode"
    SAFE_MODE = "safe_mode"

@dataclass
class FailureEvent:
    """Represents a system failure event"""
    failure_id: str
    timestamp: float
    failure_mode: FailureMode
    component: str
    severity: int  # 1-10 scale
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_time: Optional[float] = None
    
    def __post_init__(self):
        if not self.failure_id:
            self.failure_id = f"failure_{int(time.time() * 1000)}_{hash(self.component) % 10000}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "failure_id": self.failure_id,
            "timestamp": self.timestamp,
            "failure_mode": self.failure_mode.value,
            "component": self.component,
            "severity": self.severity,
            "description": self.description,
            "context": self.context,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "recovery_strategy": self.recovery_strategy.value if self.recovery_strategy else None,
            "recovery_time": self.recovery_time
        }

@dataclass
class SystemState:
    """Current system state snapshot"""
    timestamp: float
    health: SystemHealth
    degradation_level: DegradationLevel
    active_components: Set[str]
    failed_components: Set[str]
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    active_failures: List[str]  # failure_ids
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "health": self.health.value,
            "degradation_level": self.degradation_level.value,
            "active_components": list(self.active_components),
            "failed_components": list(self.failed_components),
            "performance_metrics": self.performance_metrics,
            "resource_usage": self.resource_usage,
            "active_failures": self.active_failures
        }

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        # State tracking
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
        
        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
        self._lock = threading.Lock()
    
    def __call__(self, func):
        """Decorator for circuit breaker"""
        
        async def async_wrapper(*args, **kwargs):
            return await self._execute_async(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return self._execute_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    async def _execute_async(self, func, *args, **kwargs):
        """Execute async function with circuit breaker"""
        
        with self._lock:
            self.total_calls += 1
            
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                    log.info(f"Circuit breaker {func.__name__} entering half-open state")
                else:
                    raise Exception(f"Circuit breaker {func.__name__} is open")
        
        try:
            result = await func(*args, **kwargs)
            
            with self._lock:
                self.successful_calls += 1
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                    log.info(f"Circuit breaker {func.__name__} recovered, closing")
            
            return result
            
        except self.expected_exception as e:
            with self._lock:
                self.failed_calls += 1
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    log.warning(f"Circuit breaker {func.__name__} opened after {self.failure_count} failures")
            
            raise e
    
    def _execute_sync(self, func, *args, **kwargs):
        """Execute sync function with circuit breaker"""
        
        with self._lock:
            self.total_calls += 1
            
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                    log.info(f"Circuit breaker {func.__name__} entering half-open state")
                else:
                    raise Exception(f"Circuit breaker {func.__name__} is open")
        
        try:
            result = func(*args, **kwargs)
            
            with self._lock:
                self.successful_calls += 1
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                    log.info(f"Circuit breaker {func.__name__} recovered, closing")
            
            return result
            
        except self.expected_exception as e:
            with self._lock:
                self.failed_calls += 1
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    log.warning(f"Circuit breaker {func.__name__} opened after {self.failure_count} failures")
            
            raise e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            success_rate = self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0
            
            return {
                "state": self.state,
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "success_rate": success_rate,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time
            }
    
    def reset(self):
        """Reset circuit breaker state"""
        with self._lock:
            self.state = "closed"
            self.failure_count = 0
            self.last_failure_time = 0.0
            log.info(f"Circuit breaker reset")

class RetryManager:
    """Advanced retry mechanism with exponential backoff"""
    
    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        # Statistics
        self.retry_attempts: Dict[str, int] = defaultdict(int)
        self.retry_successes: Dict[str, int] = defaultdict(int)
        self.retry_failures: Dict[str, int] = defaultdict(int)
    
    async def retry_async(self, 
                         func: Callable,
                         *args,
                         retryable_exceptions: Tuple[type, ...] = (Exception,),
                         operation_name: str = "operation",
                         **kwargs) -> Any:
        """Retry async operation with exponential backoff"""
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    log.debug(f"Retrying {operation_name} (attempt {attempt + 1}/{self.max_retries + 1}) after {delay:.2f}s")
                    await asyncio.sleep(delay)
                
                self.retry_attempts[operation_name] += 1
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    self.retry_successes[operation_name] += 1
                    log.info(f"Operation {operation_name} succeeded on attempt {attempt + 1}")
                
                return result
                
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    self.retry_failures[operation_name] += 1
                    log.error(f"Operation {operation_name} failed after {self.max_retries + 1} attempts: {e}")
                    break
                
                log.warning(f"Operation {operation_name} failed on attempt {attempt + 1}: {e}")
        
        if last_exception:
            raise last_exception
    
    def retry_sync(self,
                   func: Callable,
                   *args,
                   retryable_exceptions: Tuple[type, ...] = (Exception,),
                   operation_name: str = "operation",
                   **kwargs) -> Any:
        """Retry sync operation with exponential backoff"""
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    log.debug(f"Retrying {operation_name} (attempt {attempt + 1}/{self.max_retries + 1}) after {delay:.2f}s")
                    time.sleep(delay)
                
                self.retry_attempts[operation_name] += 1
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.retry_successes[operation_name] += 1
                    log.info(f"Operation {operation_name} succeeded on attempt {attempt + 1}")
                
                return result
                
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    self.retry_failures[operation_name] += 1
                    log.error(f"Operation {operation_name} failed after {self.max_retries + 1} attempts: {e}")
                    break
                
                log.warning(f"Operation {operation_name} failed on attempt {attempt + 1}: {e}")
        
        if last_exception:
            raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        # Add jitter to avoid thundering herd
        if self.jitter:
            import random
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.1, delay)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics"""
        
        total_attempts = sum(self.retry_attempts.values())
        total_successes = sum(self.retry_successes.values())
        total_failures = sum(self.retry_failures.values())
        
        return {
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate": total_successes / total_attempts if total_attempts > 0 else 0.0,
            "operations": dict(self.retry_attempts)
        }

class HealthMonitor:
    """System health monitoring and alerting"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.component_health: Dict[str, SystemHealth] = {}
        self.health_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # Thresholds
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 80.0  # %
        self.error_rate_threshold = 0.1  # 10%
        self.response_time_threshold = 5000.0  # ms
        
        # Monitoring state
        self._monitoring = False
        self._monitor_task = None
        
        log.info("Health monitor initialized")
    
    def register_health_check(self, component: str, check_func: Callable):
        """Register health check for component"""
        self.health_checks[component] = check_func
        self.component_health[component] = SystemHealth.HEALTHY
        log.info(f"Registered health check for component: {component}")
    
    def register_alert_callback(self, callback: Callable):
        """Register alert callback function"""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start health monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        log.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        log.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self._monitoring:
            try:
                # Perform health checks
                health_results = await self._perform_health_checks()
                
                # Update component health
                self._update_component_health(health_results)
                
                # Calculate overall system health
                system_health = self._calculate_system_health()
                
                # Store health snapshot
                health_snapshot = {
                    "timestamp": time.time(),
                    "system_health": system_health.value,
                    "component_health": {k: v.value for k, v in self.component_health.items()},
                    "health_results": health_results
                }
                
                self.health_history.append(health_snapshot)
                
                # Check for alerts
                await self._check_alerts(health_results, system_health)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                log.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Perform all registered health checks"""
        
        health_results = {}
        
        # Execute health checks in parallel
        check_tasks = []
        
        for component, check_func in self.health_checks.items():
            if asyncio.iscoroutinefunction(check_func):
                task = asyncio.create_task(self._execute_health_check_async(component, check_func))
            else:
                task = asyncio.create_task(self._execute_health_check_sync(component, check_func))
            
            check_tasks.append(task)
        
        # Wait for all checks to complete
        if check_tasks:
            results = await asyncio.gather(*check_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                component = list(self.health_checks.keys())[i]
                
                if isinstance(result, Exception):
                    health_results[component] = {
                        "healthy": False,
                        "error": str(result),
                        "metrics": {}
                    }
                else:
                    health_results[component] = result
        
        return health_results
    
    async def _execute_health_check_async(self, component: str, check_func: Callable) -> Dict[str, Any]:
        """Execute async health check"""
        try:
            return await check_func()
        except Exception as e:
            log.error(f"Health check failed for {component}: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "metrics": {}
            }
    
    async def _execute_health_check_sync(self, component: str, check_func: Callable) -> Dict[str, Any]:
        """Execute sync health check"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, check_func)
        except Exception as e:
            log.error(f"Health check failed for {component}: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "metrics": {}
            }
    
    def _update_component_health(self, health_results: Dict[str, Dict[str, Any]]):
        """Update component health status"""
        
        for component, result in health_results.items():
            if result.get("healthy", False):
                self.component_health[component] = SystemHealth.HEALTHY
            else:
                # Determine severity based on metrics
                metrics = result.get("metrics", {})
                error_rate = metrics.get("error_rate", 0.0)
                response_time = metrics.get("response_time_ms", 0.0)
                
                if error_rate > 0.5 or response_time > 10000:  # 50% errors or >10s response
                    self.component_health[component] = SystemHealth.FAILED
                elif error_rate > 0.2 or response_time > 5000:  # 20% errors or >5s response
                    self.component_health[component] = SystemHealth.CRITICAL
                else:
                    self.component_health[component] = SystemHealth.DEGRADED
    
    def _calculate_system_health(self) -> SystemHealth:
        """Calculate overall system health"""
        
        if not self.component_health:
            return SystemHealth.HEALTHY
        
        health_counts = defaultdict(int)
        for health in self.component_health.values():
            health_counts[health] += 1
        
        total_components = len(self.component_health)
        
        # System health logic
        if health_counts[SystemHealth.FAILED] > total_components * 0.5:
            return SystemHealth.FAILED
        elif health_counts[SystemHealth.CRITICAL] > total_components * 0.3:
            return SystemHealth.CRITICAL
        elif health_counts[SystemHealth.DEGRADED] > total_components * 0.5:
            return SystemHealth.DEGRADED
        else:
            return SystemHealth.HEALTHY
    
    async def _check_alerts(self, health_results: Dict[str, Dict[str, Any]], system_health: SystemHealth):
        """Check for alert conditions"""
        
        alerts = []
        
        # System-level alerts
        if system_health in [SystemHealth.CRITICAL, SystemHealth.FAILED]:
            alerts.append({
                "type": "system_health",
                "severity": "critical" if system_health == SystemHealth.CRITICAL else "emergency",
                "message": f"System health is {system_health.value}",
                "timestamp": time.time()
            })
        
        # Component-level alerts
        for component, result in health_results.items():
            if not result.get("healthy", True):
                alerts.append({
                    "type": "component_health",
                    "severity": "warning",
                    "component": component,
                    "message": f"Component {component} is unhealthy: {result.get('error', 'Unknown error')}",
                    "timestamp": time.time()
                })
        
        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert to registered callbacks"""
        
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                log.error(f"Alert callback failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        
        system_health = self._calculate_system_health()
        
        return {
            "timestamp": time.time(),
            "system_health": system_health.value,
            "component_health": {k: v.value for k, v in self.component_health.items()},
            "total_components": len(self.component_health),
            "healthy_components": sum(1 for h in self.component_health.values() if h == SystemHealth.HEALTHY),
            "degraded_components": sum(1 for h in self.component_health.values() if h == SystemHealth.DEGRADED),
            "critical_components": sum(1 for h in self.component_health.values() if h == SystemHealth.CRITICAL),
            "failed_components": sum(1 for h in self.component_health.values() if h == SystemHealth.FAILED)
        }

class FaultTolerantFusionProcessor:
    """Fault-tolerant fusion processor with graceful degradation"""
    
    def __init__(self):
        # Core components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager()
        self.health_monitor = HealthMonitor()
        
        # Failure tracking
        self.failure_history: deque = deque(maxlen=1000)
        self.recovery_strategies: Dict[FailureMode, RecoveryStrategy] = {}
        
        # Degradation management
        self.current_degradation = DegradationLevel.FULL_CAPABILITY
        self.degradation_thresholds: Dict[DegradationLevel, Dict[str, float]] = {}
        
        # Fallback mechanisms
        self.fallback_algorithms: Dict[str, Callable] = {}
        self.cached_results: Dict[str, Tuple[Any, float]] = {}  # result, timestamp
        self.cache_ttl = 300.0  # 5 minutes
        
        # State management
        self.system_state_history: deque = deque(maxlen=100)
        self._state_lock = threading.Lock()
        
        # Load default configurations
        self._load_default_configurations()
        
        log.info("Fault-tolerant fusion processor initialized")
    
    async def process_with_fault_tolerance(self,
                                         fusion_function: Callable,
                                         data: Dict[str, Any],
                                         operation_name: str = "fusion") -> Dict[str, Any]:
        """Process fusion with comprehensive fault tolerance"""
        
        start_time = time.time()
        
        try:
            # Check system health
            if not await self._check_system_readiness():
                return await self._handle_system_not_ready(data, operation_name)
            
            # Apply circuit breaker if registered
            if operation_name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[operation_name]
                result = await circuit_breaker._execute_async(fusion_function, data)
            else:
                # Execute with retry
                result = await self.retry_manager.retry_async(
                    fusion_function,
                    data,
                    operation_name=operation_name,
                    retryable_exceptions=(Exception,)
                )
            
            # Cache successful result
            self._cache_result(operation_name, result)
            
            # Update system state
            await self._update_system_state(success=True, operation=operation_name)
            
            # Add fault tolerance metadata
            result["fault_tolerance_metadata"] = {
                "processing_time_ms": (time.time() - start_time) * 1000,
                "degradation_level": self.current_degradation.value,
                "fallback_used": False,
                "cache_used": False
            }
            
            return result
            
        except Exception as e:
            log.error(f"Fault-tolerant processing failed for {operation_name}: {e}")
            
            # Record failure
            failure_event = FailureEvent(
                failure_id="",
                timestamp=time.time(),
                failure_mode=self._classify_failure(e),
                component=operation_name,
                severity=self._calculate_failure_severity(e),
                description=str(e),
                context={"data_keys": list(data.keys()) if isinstance(data, dict) else str(type(data))}
            )
            
            self.failure_history.append(failure_event)
            
            # Attempt recovery
            recovery_result = await self._attempt_recovery(failure_event, fusion_function, data, operation_name)
            
            if recovery_result:
                return recovery_result
            else:
                # Final fallback
                return await self._emergency_fallback(data, operation_name, str(e))
    
    async def _check_system_readiness(self) -> bool:
        """Check if system is ready for processing"""
        
        health_status = self.health_monitor.get_health_status()
        system_health = SystemHealth(health_status["system_health"])
        
        # System is ready unless completely failed
        return system_health != SystemHealth.FAILED
    
    async def _handle_system_not_ready(self, data: Dict[str, Any], operation_name: str) -> Dict[str, Any]:
        """Handle case when system is not ready"""
        
        log.warning(f"System not ready for {operation_name}, attempting cached response")
        
        # Try cached result
        cached_result = self._get_cached_result(operation_name)
        if cached_result:
            cached_result["fault_tolerance_metadata"] = {
                "processing_time_ms": 0,
                "degradation_level": DegradationLevel.EMERGENCY_MODE.value,
                "fallback_used": False,
                "cache_used": True,
                "system_not_ready": True
            }
            return cached_result
        
        # Emergency fallback
        return await self._emergency_fallback(data, operation_name, "System not ready")
    
    def _classify_failure(self, exception: Exception) -> FailureMode:
        """Classify failure based on exception type"""
        
        if isinstance(exception, TimeoutError):
            return FailureMode.TIMEOUT_FAILURE
        elif isinstance(exception, ConnectionError):
            return FailureMode.NETWORK_FAILURE
        elif isinstance(exception, MemoryError):
            return FailureMode.RESOURCE_EXHAUSTION
        elif isinstance(exception, (ValueError, TypeError)):
            return FailureMode.DATA_CORRUPTION
        elif isinstance(exception, PermissionError):
            return FailureMode.SECURITY_BREACH
        else:
            return FailureMode.COMPONENT_FAILURE
    
    def _calculate_failure_severity(self, exception: Exception) -> int:
        """Calculate failure severity (1-10 scale)"""
        
        if isinstance(exception, (MemoryError, SystemError)):
            return 9  # Critical system issues
        elif isinstance(exception, (ConnectionError, TimeoutError)):
            return 6  # Network/timeout issues
        elif isinstance(exception, (ValueError, TypeError)):
            return 4  # Data issues
        else:
            return 5  # Default moderate severity
    
    async def _attempt_recovery(self,
                               failure_event: FailureEvent,
                               fusion_function: Callable,
                               data: Dict[str, Any],
                               operation_name: str) -> Optional[Dict[str, Any]]:
        """Attempt recovery from failure"""
        
        failure_mode = failure_event.failure_mode
        recovery_strategy = self.recovery_strategies.get(failure_mode, RecoveryStrategy.FALLBACK_ALGORITHM)
        
        failure_event.recovery_attempted = True
        failure_event.recovery_strategy = recovery_strategy
        
        recovery_start = time.time()
        
        try:
            if recovery_strategy == RecoveryStrategy.CACHED_RESPONSE:
                result = self._get_cached_result(operation_name)
                if result:
                    result["fault_tolerance_metadata"] = {
                        "processing_time_ms": 0,
                        "degradation_level": self.current_degradation.value,
                        "fallback_used": False,
                        "cache_used": True,
                        "recovery_used": True
                    }
                    failure_event.recovery_successful = True
                    return result
            
            elif recovery_strategy == RecoveryStrategy.FALLBACK_ALGORITHM:
                if operation_name in self.fallback_algorithms:
                    fallback_func = self.fallback_algorithms[operation_name]
                    result = await fallback_func(data)
                    
                    result["fault_tolerance_metadata"] = {
                        "processing_time_ms": (time.time() - recovery_start) * 1000,
                        "degradation_level": self.current_degradation.value,
                        "fallback_used": True,
                        "cache_used": False,
                        "recovery_used": True
                    }
                    
                    failure_event.recovery_successful = True
                    return result
            
            elif recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                # Reduce system capability and retry
                await self._apply_degradation()
                
                # Retry with degraded system
                try:
                    result = await fusion_function(data)
                    failure_event.recovery_successful = True
                    
                    result["fault_tolerance_metadata"] = {
                        "processing_time_ms": (time.time() - recovery_start) * 1000,
                        "degradation_level": self.current_degradation.value,
                        "fallback_used": False,
                        "cache_used": False,
                        "recovery_used": True,
                        "degradation_applied": True
                    }
                    
                    return result
                    
                except Exception as e:
                    log.warning(f"Recovery with degradation failed: {e}")
            
        except Exception as e:
            log.error(f"Recovery attempt failed: {e}")
        
        finally:
            failure_event.recovery_time = time.time() - recovery_start
        
        failure_event.recovery_successful = False
        return None
    
    async def _apply_degradation(self):
        """Apply system degradation"""
        
        if self.current_degradation == DegradationLevel.FULL_CAPABILITY:
            self.current_degradation = DegradationLevel.REDUCED_ACCURACY
        elif self.current_degradation == DegradationLevel.REDUCED_ACCURACY:
            self.current_degradation = DegradationLevel.REDUCED_THROUGHPUT
        elif self.current_degradation == DegradationLevel.REDUCED_THROUGHPUT:
            self.current_degradation = DegradationLevel.BASIC_FUNCTIONALITY
        elif self.current_degradation == DegradationLevel.BASIC_FUNCTIONALITY:
            self.current_degradation = DegradationLevel.EMERGENCY_MODE
        
        log.warning(f"System degradation applied: {self.current_degradation.value}")
    
    async def _emergency_fallback(self, data: Dict[str, Any], operation_name: str, error: str) -> Dict[str, Any]:
        """Emergency fallback when all else fails"""
        
        log.critical(f"Emergency fallback activated for {operation_name}: {error}")
        
        # Provide minimal safe response
        return {
            "emergency_response": True,
            "error": error,
            "operation": operation_name,
            "timestamp": time.time(),
            "data_received": True,
            "fault_tolerance_metadata": {
                "processing_time_ms": 0,
                "degradation_level": DegradationLevel.SAFE_MODE.value,
                "fallback_used": True,
                "cache_used": False,
                "emergency_mode": True
            }
        }
    
    def _cache_result(self, operation_name: str, result: Dict[str, Any]):
        """Cache successful result"""
        
        # Don't cache results that are already cached or fallback results
        metadata = result.get("fault_tolerance_metadata", {})
        if metadata.get("cache_used") or metadata.get("fallback_used"):
            return
        
        self.cached_results[operation_name] = (result.copy(), time.time())
        
        # Clean old cache entries
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cached_results.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.cached_results[key]
    
    def _get_cached_result(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired"""
        
        if operation_name not in self.cached_results:
            return None
        
        result, timestamp = self.cached_results[operation_name]
        
        if time.time() - timestamp > self.cache_ttl:
            del self.cached_results[operation_name]
            return None
        
        return result.copy()
    
    async def _update_system_state(self, success: bool, operation: str):
        """Update system state tracking"""
        
        with self._state_lock:
            current_time = time.time()
            
            # Get current component health
            health_status = self.health_monitor.get_health_status()
            
            # Create state snapshot
            state_snapshot = SystemState(
                timestamp=current_time,
                health=SystemHealth(health_status["system_health"]),
                degradation_level=self.current_degradation,
                active_components=set(self.health_monitor.component_health.keys()),
                failed_components=set(
                    k for k, v in self.health_monitor.component_health.items()
                    if v == SystemHealth.FAILED
                ),
                performance_metrics={
                    "success_rate": 1.0 if success else 0.0,
                    "operation_count": 1
                },
                resource_usage={},  # Would be populated with actual resource metrics
                active_failures=[f.failure_id for f in list(self.failure_history)[-10:]]
            )
            
            self.system_state_history.append(state_snapshot)
    
    def register_circuit_breaker(self, operation_name: str, **kwargs):
        """Register circuit breaker for operation"""
        
        self.circuit_breakers[operation_name] = CircuitBreaker(**kwargs)
        log.info(f"Circuit breaker registered for {operation_name}")
    
    def register_fallback_algorithm(self, operation_name: str, fallback_func: Callable):
        """Register fallback algorithm for operation"""
        
        self.fallback_algorithms[operation_name] = fallback_func
        log.info(f"Fallback algorithm registered for {operation_name}")
    
    def register_health_check(self, component: str, check_func: Callable):
        """Register health check for component"""
        
        self.health_monitor.register_health_check(component, check_func)
    
    def _load_default_configurations(self):
        """Load default fault tolerance configurations"""
        
        # Default recovery strategies
        self.recovery_strategies = {
            FailureMode.COMPONENT_FAILURE: RecoveryStrategy.FALLBACK_ALGORITHM,
            FailureMode.NETWORK_FAILURE: RecoveryStrategy.CACHED_RESPONSE,
            FailureMode.DATA_CORRUPTION: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FailureMode.RESOURCE_EXHAUSTION: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FailureMode.TIMEOUT_FAILURE: RecoveryStrategy.CACHED_RESPONSE,
            FailureMode.SECURITY_BREACH: RecoveryStrategy.MANUAL_INTERVENTION,
            FailureMode.CONFIGURATION_ERROR: RecoveryStrategy.SYSTEM_ROLLBACK,
            FailureMode.EXTERNAL_DEPENDENCY: RecoveryStrategy.CACHED_RESPONSE
        }
        
        # Degradation thresholds
        self.degradation_thresholds = {
            DegradationLevel.FULL_CAPABILITY: {"error_rate": 0.01, "response_time_ms": 1000},
            DegradationLevel.REDUCED_ACCURACY: {"error_rate": 0.05, "response_time_ms": 2000},
            DegradationLevel.REDUCED_THROUGHPUT: {"error_rate": 0.1, "response_time_ms": 5000},
            DegradationLevel.BASIC_FUNCTIONALITY: {"error_rate": 0.2, "response_time_ms": 10000},
            DegradationLevel.EMERGENCY_MODE: {"error_rate": 0.5, "response_time_ms": 30000},
            DegradationLevel.SAFE_MODE: {"error_rate": 1.0, "response_time_ms": 60000}
        }
    
    def get_fault_tolerance_status(self) -> Dict[str, Any]:
        """Get comprehensive fault tolerance status"""
        
        try:
            # Circuit breaker stats
            circuit_breaker_stats = {
                name: cb.get_stats() for name, cb in self.circuit_breakers.items()
            }
            
            # Retry manager stats
            retry_stats = self.retry_manager.get_stats()
            
            # Health monitor stats
            health_stats = self.health_monitor.get_health_status()
            
            # Recent failures
            recent_failures = [
                f.to_dict() for f in list(self.failure_history)[-10:]
            ]
            
            # Cache stats
            cache_stats = {
                "cached_operations": len(self.cached_results),
                "cache_ttl_seconds": self.cache_ttl
            }
            
            # System state
            current_state = None
            if self.system_state_history:
                current_state = self.system_state_history[-1].to_dict()
            
            return {
                "timestamp": time.time(),
                "current_degradation": self.current_degradation.value,
                "circuit_breakers": circuit_breaker_stats,
                "retry_manager": retry_stats,
                "health_monitor": health_stats,
                "recent_failures": recent_failures,
                "cache_stats": cache_stats,
                "current_system_state": current_state,
                "registered_fallbacks": list(self.fallback_algorithms.keys())
            }
            
        except Exception as e:
            log.error(f"Fault tolerance status generation failed: {e}")
            return {"error": str(e), "timestamp": time.time()}

# Utility functions for creating fault-tolerant systems
async def create_fault_tolerant_fusion_system() -> FaultTolerantFusionProcessor:
    """Create comprehensive fault-tolerant fusion system"""
    
    processor = FaultTolerantFusionProcessor()
    
    # Register default health checks
    processor.register_health_check("fusion_engine", lambda: {"healthy": True, "metrics": {}})
    processor.register_health_check("data_pipeline", lambda: {"healthy": True, "metrics": {}})
    processor.register_health_check("storage_system", lambda: {"healthy": True, "metrics": {}})
    
    # Register default circuit breakers
    processor.register_circuit_breaker("bayesian_fusion", failure_threshold=3, recovery_timeout=30.0)
    processor.register_circuit_breaker("neural_mesh_integration", failure_threshold=5, recovery_timeout=60.0)
    
    # Register simple fallback algorithms
    async def simple_fusion_fallback(data):
        return {
            "fused_value": 0.5,
            "confidence": 0.3,
            "algorithm": "simple_fallback",
            "fallback_reason": "Primary fusion failed"
        }
    
    processor.register_fallback_algorithm("fusion", simple_fusion_fallback)
    
    # Start health monitoring
    await processor.health_monitor.start_monitoring()
    
    log.info("Fault-tolerant fusion system created and initialized")
    
    return processor
