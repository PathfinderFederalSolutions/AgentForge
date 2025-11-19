"""
Circuit Breaker Pattern Implementation
Production-grade circuit breaker with exponential backoff and graceful degradation
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import random

log = logging.getLogger("circuit-breaker")

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 3  # Successes to close circuit from half-open
    timeout_seconds: float = 60.0  # Time before trying half-open
    max_timeout_seconds: float = 300.0  # Maximum timeout
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    
class CircuitBreakerMetrics:
    """Circuit breaker metrics tracking"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.circuit_opens = 0
        self.circuit_closes = 0
        self.fast_failures = 0  # Requests that failed due to open circuit
        
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
        self.total_downtime = 0.0
        self.circuit_open_time = 0.0
    
    def record_success(self):
        """Record successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.last_success_time = time.time()
    
    def record_failure(self):
        """Record failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_failure_time = time.time()
    
    def record_circuit_open(self):
        """Record circuit opening"""
        self.circuit_opens += 1
        self.circuit_open_time = time.time()
    
    def record_circuit_close(self):
        """Record circuit closing"""
        self.circuit_closes += 1
        if self.circuit_open_time > 0:
            self.total_downtime += time.time() - self.circuit_open_time
            self.circuit_open_time = 0.0
    
    def record_fast_failure(self):
        """Record fast failure due to open circuit"""
        self.fast_failures += 1
    
    def get_success_rate(self) -> float:
        """Get success rate"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    def get_failure_rate(self) -> float:
        """Get failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

class CircuitBreaker:
    """
    Production-grade circuit breaker implementation
    Provides fault tolerance and graceful degradation
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.next_attempt_time = 0.0
        self.current_timeout = self.config.timeout_seconds
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        
        # Fallback function
        self.fallback_function: Optional[Callable] = None
        
        log.info(f"Circuit breaker '{name}' initialized")
    
    def set_fallback(self, fallback_function: Callable):
        """Set fallback function for when circuit is open"""
        self.fallback_function = fallback_function
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        current_time = time.time()
        
        # Check if circuit should transition states
        self._update_state(current_time)
        
        if self.state == CircuitState.OPEN:
            # Circuit is open, fail fast
            self.metrics.record_fast_failure()
            
            if self.fallback_function:
                log.debug(f"Circuit '{self.name}' is open, using fallback")
                return await self._execute_fallback(*args, **kwargs)
            else:
                raise CircuitBreakerOpenError(f"Circuit '{self.name}' is open")
        
        # Circuit is closed or half-open, attempt the call
        try:
            result = await self._execute_with_timeout(func, *args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            
            # If we have a fallback, use it
            if self.fallback_function:
                log.warning(f"Function failed in circuit '{self.name}', using fallback: {e}")
                return await self._execute_fallback(*args, **kwargs)
            else:
                raise
    
    async def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout"""
        try:
            # Add timeout to prevent hanging
            return await asyncio.wait_for(func(*args, **kwargs), timeout=30.0)
        except asyncio.TimeoutError:
            raise CircuitBreakerTimeoutError(f"Function timeout in circuit '{self.name}'")
    
    async def _execute_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback function"""
        try:
            if asyncio.iscoroutinefunction(self.fallback_function):
                return await self.fallback_function(*args, **kwargs)
            else:
                return self.fallback_function(*args, **kwargs)
        except Exception as e:
            log.error(f"Fallback function failed in circuit '{self.name}': {e}")
            raise CircuitBreakerFallbackError(f"Fallback failed: {e}")
    
    def _update_state(self, current_time: float):
        """Update circuit breaker state based on current conditions"""
        if self.state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if current_time >= self.next_attempt_time:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                log.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
        
        elif self.state == CircuitState.HALF_OPEN:
            # In half-open state, we're testing the service
            # State transitions happen in _record_success() and _record_failure()
            pass
    
    def _record_success(self):
        """Record successful execution"""
        self.metrics.record_success()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.config.success_threshold:
                # Service appears to be healthy, close the circuit
                self._close_circuit()
        
        # Reset failure count on success
        self.failure_count = 0
    
    def _record_failure(self):
        """Record failed execution"""
        self.metrics.record_failure()
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._open_circuit()
        
        elif self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state means service is still unhealthy
            self._open_circuit()
    
    def _open_circuit(self):
        """Open the circuit"""
        self.state = CircuitState.OPEN
        self.next_attempt_time = time.time() + self.current_timeout
        self.metrics.record_circuit_open()
        
        # Exponential backoff
        self.current_timeout = min(
            self.current_timeout * self.config.backoff_multiplier,
            self.config.max_timeout_seconds
        )
        
        log.warning(f"Circuit '{self.name}' OPENED (failures: {self.failure_count}, next attempt in {self.current_timeout}s)")
    
    def _close_circuit(self):
        """Close the circuit"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.current_timeout = self.config.timeout_seconds
        self.metrics.record_circuit_close()
        
        log.info(f"Circuit '{self.name}' CLOSED (service recovered)")
    
    def force_open(self):
        """Manually force circuit open"""
        self._open_circuit()
        log.warning(f"Circuit '{self.name}' manually forced OPEN")
    
    def force_close(self):
        """Manually force circuit closed"""
        self._close_circuit()
        log.info(f"Circuit '{self.name}' manually forced CLOSED")
    
    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "current_timeout": self.current_timeout,
            "next_attempt_time": self.next_attempt_time,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": self.metrics.get_success_rate(),
                "failure_rate": self.metrics.get_failure_rate(),
                "circuit_opens": self.metrics.circuit_opens,
                "circuit_closes": self.metrics.circuit_closes,
                "fast_failures": self.metrics.fast_failures,
                "total_downtime": self.metrics.total_downtime
            }
        }

class CircuitBreakerManager:
    """
    Manages multiple circuit breakers
    Provides centralized configuration and monitoring
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.global_config = CircuitBreakerConfig()
        
        # Health check configuration
        self.health_check_interval = 30.0  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        
        log.info("Circuit breaker manager initialized")
    
    def create_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create and register a circuit breaker"""
        if name in self.circuit_breakers:
            return self.circuit_breakers[name]
        
        circuit_config = config or self.global_config
        circuit_breaker = CircuitBreaker(name, circuit_config)
        self.circuit_breakers[name] = circuit_breaker
        
        log.info(f"Created circuit breaker: {name}")
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def remove_circuit_breaker(self, name: str):
        """Remove circuit breaker"""
        if name in self.circuit_breakers:
            del self.circuit_breakers[name]
            log.info(f"Removed circuit breaker: {name}")
    
    async def call_with_circuit_breaker(self, circuit_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        circuit_breaker = self.get_circuit_breaker(circuit_name)
        if not circuit_breaker:
            circuit_breaker = self.create_circuit_breaker(circuit_name)
        
        return await circuit_breaker.call(func, *args, **kwargs)
    
    def set_fallback(self, circuit_name: str, fallback_function: Callable):
        """Set fallback function for circuit breaker"""
        circuit_breaker = self.get_circuit_breaker(circuit_name)
        if circuit_breaker:
            circuit_breaker.set_fallback(fallback_function)
    
    async def start_health_monitoring(self):
        """Start health monitoring for all circuit breakers"""
        if self.health_check_task:
            return
        
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        log.info("Started circuit breaker health monitoring")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
            log.info("Stopped circuit breaker health monitoring")
    
    async def _health_check_loop(self):
        """Health check loop for monitoring circuit breakers"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all circuit breakers"""
        current_time = time.time()
        
        for name, circuit_breaker in self.circuit_breakers.items():
            try:
                # Check for circuits that have been open too long
                if (circuit_breaker.state == CircuitState.OPEN and 
                    current_time - circuit_breaker.metrics.circuit_open_time > 600):  # 10 minutes
                    
                    log.warning(f"Circuit '{name}' has been open for over 10 minutes")
                
                # Check for high failure rates
                failure_rate = circuit_breaker.metrics.get_failure_rate()
                if failure_rate > 0.5:  # 50% failure rate
                    log.warning(f"Circuit '{name}' has high failure rate: {failure_rate:.2%}")
                
            except Exception as e:
                log.error(f"Health check failed for circuit '{name}': {e}")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all circuit breakers"""
        metrics = {}
        
        for name, circuit_breaker in self.circuit_breakers.items():
            metrics[name] = circuit_breaker.get_metrics()
        
        return {
            "circuit_breakers": metrics,
            "total_circuits": len(self.circuit_breakers),
            "open_circuits": len([cb for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN]),
            "half_open_circuits": len([cb for cb in self.circuit_breakers.values() if cb.state == CircuitState.HALF_OPEN]),
            "closed_circuits": len([cb for cb in self.circuit_breakers.values() if cb.state == CircuitState.CLOSED])
        }
    
    def force_open_all(self):
        """Force open all circuit breakers (emergency use)"""
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.force_open()
        log.warning("All circuit breakers forced OPEN")
    
    def force_close_all(self):
        """Force close all circuit breakers"""
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.force_close()
        log.info("All circuit breakers forced CLOSED")

# Custom exceptions
class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors"""
    pass

class CircuitBreakerOpenError(CircuitBreakerError):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Exception raised when function times out"""
    pass

class CircuitBreakerFallbackError(CircuitBreakerError):
    """Exception raised when fallback function fails"""
    pass

# Decorator for easy circuit breaker usage
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None, 
                   fallback: Optional[Callable] = None):
    """Decorator to apply circuit breaker to a function"""
    def decorator(func: Callable):
        # Create global circuit breaker manager if not exists
        if not hasattr(circuit_breaker, '_manager'):
            circuit_breaker._manager = CircuitBreakerManager()
        
        # Create circuit breaker
        cb = circuit_breaker._manager.create_circuit_breaker(name, config)
        
        # Set fallback if provided
        if fallback:
            cb.set_fallback(fallback)
        
        async def wrapper(*args, **kwargs):
            return await cb.call(func, *args, **kwargs)
        
        return wrapper
    return decorator
