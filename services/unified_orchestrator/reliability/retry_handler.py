"""
Advanced Retry Handler with Exponential Backoff
Production-grade retry mechanisms with jitter and adaptive strategies
"""

from __future__ import annotations
import asyncio
import logging
import time
import random
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import inspect

log = logging.getLogger("retry-handler")

class RetryStrategy(Enum):
    """Retry strategies"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    ADAPTIVE = "adaptive"

class JitterType(Enum):
    """Types of jitter to add to retry delays"""
    NONE = "none"
    FULL = "full"          # Random jitter between 0 and calculated delay
    EQUAL = "equal"        # Half calculated delay + half random
    DECORRELATED = "decorrelated"  # Exponentially weighted random

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 300.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Base for exponential backoff
    jitter_type: JitterType = JitterType.FULL
    
    # Exception handling
    retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    
    # Adaptive parameters
    success_threshold: float = 0.8  # Success rate threshold for adaptive strategy
    failure_window: int = 10  # Number of recent attempts to consider
    
    # Circuit breaker integration
    enable_circuit_breaker: bool = False
    circuit_breaker_threshold: int = 5  # Failures before opening circuit

@dataclass
class RetryAttempt:
    """Information about a retry attempt"""
    attempt_number: int
    delay: float
    exception: Optional[Exception] = None
    timestamp: float = field(default_factory=time.time)
    success: bool = False

class RetryMetrics:
    """Metrics tracking for retry operations"""
    
    def __init__(self):
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_attempts = 0
        self.total_retry_time = 0.0
        
        # Per-exception metrics
        self.exception_counts: Dict[str, int] = {}
        
        # Recent history for adaptive behavior
        self.recent_attempts: List[RetryAttempt] = []
        self.max_history = 1000
    
    def record_operation_start(self):
        """Record start of retry operation"""
        self.total_operations += 1
    
    def record_attempt(self, attempt: RetryAttempt):
        """Record a retry attempt"""
        self.total_attempts += 1
        
        if attempt.exception:
            exception_name = type(attempt.exception).__name__
            self.exception_counts[exception_name] = self.exception_counts.get(exception_name, 0) + 1
        
        # Store in recent history
        self.recent_attempts.append(attempt)
        if len(self.recent_attempts) > self.max_history:
            self.recent_attempts.pop(0)
    
    def record_operation_success(self, total_time: float):
        """Record successful operation"""
        self.successful_operations += 1
        self.total_retry_time += total_time
    
    def record_operation_failure(self, total_time: float):
        """Record failed operation"""
        self.failed_operations += 1
        self.total_retry_time += total_time
    
    def get_success_rate(self) -> float:
        """Get overall success rate"""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations
    
    def get_recent_success_rate(self, window: int = 10) -> float:
        """Get recent success rate within window"""
        if not self.recent_attempts:
            return 1.0
        
        recent = self.recent_attempts[-window:]
        if not recent:
            return 1.0
        
        successful = sum(1 for attempt in recent if attempt.success)
        return successful / len(recent)
    
    def get_average_attempts(self) -> float:
        """Get average attempts per operation"""
        if self.total_operations == 0:
            return 0.0
        return self.total_attempts / self.total_operations

class RetryHandler:
    """
    Advanced retry handler with multiple strategies and adaptive behavior
    """
    
    def __init__(self, name: str, config: Optional[RetryConfig] = None):
        self.name = name
        self.config = config or RetryConfig()
        self.metrics = RetryMetrics()
        
        # Strategy implementations
        self.strategies = {
            RetryStrategy.FIXED_DELAY: self._fixed_delay,
            RetryStrategy.EXPONENTIAL_BACKOFF: self._exponential_backoff,
            RetryStrategy.LINEAR_BACKOFF: self._linear_backoff,
            RetryStrategy.FIBONACCI_BACKOFF: self._fibonacci_backoff,
            RetryStrategy.ADAPTIVE: self._adaptive_delay
        }
        
        # Current strategy
        self.current_strategy = RetryStrategy.EXPONENTIAL_BACKOFF
        
        # Fibonacci sequence cache
        self._fib_cache = [1, 1]
        
        log.info(f"Retry handler '{name}' initialized")
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        operation_start = time.time()
        self.metrics.record_operation_start()
        
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            attempt_start = time.time()
            
            try:
                # Execute the function
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success!
                attempt_info = RetryAttempt(
                    attempt_number=attempt,
                    delay=0.0,
                    success=True,
                    timestamp=attempt_start
                )
                self.metrics.record_attempt(attempt_info)
                
                operation_time = time.time() - operation_start
                self.metrics.record_operation_success(operation_time)
                
                log.debug(f"Retry handler '{self.name}' succeeded on attempt {attempt}")
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    log.info(f"Non-retryable exception in '{self.name}': {type(e).__name__}")
                    break
                
                # Calculate delay for next attempt (if not last attempt)
                delay = 0.0
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt, e)
                
                attempt_info = RetryAttempt(
                    attempt_number=attempt,
                    delay=delay,
                    exception=e,
                    timestamp=attempt_start
                )
                self.metrics.record_attempt(attempt_info)
                
                if attempt < self.config.max_attempts:
                    log.warning(f"Retry handler '{self.name}' attempt {attempt} failed: {type(e).__name__}, retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    log.error(f"Retry handler '{self.name}' exhausted all {self.config.max_attempts} attempts")
        
        # All attempts failed
        operation_time = time.time() - operation_start
        self.metrics.record_operation_failure(operation_time)
        
        if last_exception:
            raise RetryExhaustedError(f"All retry attempts failed for '{self.name}'") from last_exception
        else:
            raise RetryExhaustedError(f"All retry attempts failed for '{self.name}'")
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        # Check non-retryable exceptions first
        for exc_type in self.config.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # Check retryable exceptions
        for exc_type in self.config.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        return False
    
    def _calculate_delay(self, attempt: int, exception: Exception) -> float:
        """Calculate delay for next retry attempt"""
        strategy_func = self.strategies.get(self.current_strategy, self._exponential_backoff)
        base_delay = strategy_func(attempt, exception)
        
        # Apply jitter
        delay_with_jitter = self._apply_jitter(base_delay, attempt)
        
        # Ensure delay is within bounds
        return min(max(delay_with_jitter, 0.0), self.config.max_delay)
    
    def _fixed_delay(self, attempt: int, exception: Exception) -> float:
        """Fixed delay strategy"""
        return self.config.base_delay
    
    def _exponential_backoff(self, attempt: int, exception: Exception) -> float:
        """Exponential backoff strategy"""
        return self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
    
    def _linear_backoff(self, attempt: int, exception: Exception) -> float:
        """Linear backoff strategy"""
        return self.config.base_delay * attempt
    
    def _fibonacci_backoff(self, attempt: int, exception: Exception) -> float:
        """Fibonacci backoff strategy"""
        # Extend fibonacci cache if needed
        while len(self._fib_cache) < attempt:
            next_fib = self._fib_cache[-1] + self._fib_cache[-2]
            self._fib_cache.append(next_fib)
        
        fib_value = self._fib_cache[attempt - 1] if attempt <= len(self._fib_cache) else self._fib_cache[-1]
        return self.config.base_delay * fib_value
    
    def _adaptive_delay(self, attempt: int, exception: Exception) -> float:
        """Adaptive delay based on recent success rate"""
        recent_success_rate = self.metrics.get_recent_success_rate(self.config.failure_window)
        
        if recent_success_rate >= self.config.success_threshold:
            # High success rate, use shorter delays
            return self._fixed_delay(attempt, exception)
        else:
            # Low success rate, use longer delays
            return self._exponential_backoff(attempt, exception) * 2.0
    
    def _apply_jitter(self, delay: float, attempt: int) -> float:
        """Apply jitter to delay"""
        if self.config.jitter_type == JitterType.NONE:
            return delay
        
        elif self.config.jitter_type == JitterType.FULL:
            # Random jitter between 0 and calculated delay
            return random.uniform(0, delay)
        
        elif self.config.jitter_type == JitterType.EQUAL:
            # Half calculated delay + half random
            return delay * 0.5 + random.uniform(0, delay * 0.5)
        
        elif self.config.jitter_type == JitterType.DECORRELATED:
            # Exponentially weighted random jitter
            base = delay * 0.5
            jitter = random.uniform(0, delay)
            return base + jitter * (0.8 ** attempt)
        
        return delay
    
    def set_strategy(self, strategy: RetryStrategy):
        """Set retry strategy"""
        self.current_strategy = strategy
        log.info(f"Retry handler '{self.name}' strategy changed to {strategy.value}")
    
    def add_retryable_exception(self, exception_type: Type[Exception]):
        """Add exception type to retryable list"""
        if exception_type not in self.config.retryable_exceptions:
            self.config.retryable_exceptions.append(exception_type)
    
    def add_non_retryable_exception(self, exception_type: Type[Exception]):
        """Add exception type to non-retryable list"""
        if exception_type not in self.config.non_retryable_exceptions:
            self.config.non_retryable_exceptions.append(exception_type)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry handler metrics"""
        return {
            "name": self.name,
            "strategy": self.current_strategy.value,
            "config": {
                "max_attempts": self.config.max_attempts,
                "base_delay": self.config.base_delay,
                "max_delay": self.config.max_delay,
                "jitter_type": self.config.jitter_type.value
            },
            "metrics": {
                "total_operations": self.metrics.total_operations,
                "successful_operations": self.metrics.successful_operations,
                "failed_operations": self.metrics.failed_operations,
                "success_rate": self.metrics.get_success_rate(),
                "recent_success_rate": self.metrics.get_recent_success_rate(),
                "average_attempts": self.metrics.get_average_attempts(),
                "total_attempts": self.metrics.total_attempts,
                "total_retry_time": self.metrics.total_retry_time,
                "exception_counts": self.metrics.exception_counts
            }
        }

class RetryManager:
    """
    Manages multiple retry handlers
    Provides centralized configuration and monitoring
    """
    
    def __init__(self):
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.global_config = RetryConfig()
        
        log.info("Retry manager initialized")
    
    def create_retry_handler(self, name: str, config: Optional[RetryConfig] = None) -> RetryHandler:
        """Create and register a retry handler"""
        if name in self.retry_handlers:
            return self.retry_handlers[name]
        
        retry_config = config or self.global_config
        retry_handler = RetryHandler(name, retry_config)
        self.retry_handlers[name] = retry_handler
        
        log.info(f"Created retry handler: {name}")
        return retry_handler
    
    def get_retry_handler(self, name: str) -> Optional[RetryHandler]:
        """Get retry handler by name"""
        return self.retry_handlers.get(name)
    
    def remove_retry_handler(self, name: str):
        """Remove retry handler"""
        if name in self.retry_handlers:
            del self.retry_handlers[name]
            log.info(f"Removed retry handler: {name}")
    
    async def execute_with_retry(self, handler_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry handler"""
        retry_handler = self.get_retry_handler(handler_name)
        if not retry_handler:
            retry_handler = self.create_retry_handler(handler_name)
        
        return await retry_handler.execute(func, *args, **kwargs)
    
    def set_global_config(self, config: RetryConfig):
        """Set global retry configuration"""
        self.global_config = config
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all retry handlers"""
        metrics = {}
        
        for name, retry_handler in self.retry_handlers.items():
            metrics[name] = retry_handler.get_metrics()
        
        # Calculate aggregate metrics
        total_operations = sum(h.metrics.total_operations for h in self.retry_handlers.values())
        total_successful = sum(h.metrics.successful_operations for h in self.retry_handlers.values())
        
        aggregate_success_rate = (total_successful / total_operations) if total_operations > 0 else 1.0
        
        return {
            "retry_handlers": metrics,
            "total_handlers": len(self.retry_handlers),
            "aggregate_metrics": {
                "total_operations": total_operations,
                "total_successful": total_successful,
                "overall_success_rate": aggregate_success_rate
            }
        }

# Custom exceptions
class RetryError(Exception):
    """Base exception for retry errors"""
    pass

class RetryExhaustedError(RetryError):
    """Exception raised when all retry attempts are exhausted"""
    pass

# Decorator for easy retry usage
def retry(name: Optional[str] = None, config: Optional[RetryConfig] = None, 
         max_attempts: int = 3, base_delay: float = 1.0, strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF):
    """Decorator to apply retry logic to a function"""
    def decorator(func: Callable):
        # Create retry config from parameters
        retry_config = config or RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay
        )
        
        # Generate name if not provided
        handler_name = name or f"{func.__module__}.{func.__name__}"
        
        # Create global retry manager if not exists
        if not hasattr(retry, '_manager'):
            retry._manager = RetryManager()
        
        # Create retry handler
        retry_handler = retry._manager.create_retry_handler(handler_name, retry_config)
        retry_handler.set_strategy(strategy)
        
        async def wrapper(*args, **kwargs):
            return await retry_handler.execute(func, *args, **kwargs)
        
        return wrapper
    return decorator

# Common retry configurations
class CommonRetryConfigs:
    """Predefined retry configurations for common scenarios"""
    
    @staticmethod
    def network_operation() -> RetryConfig:
        """Configuration for network operations"""
        return RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter_type=JitterType.FULL,
            retryable_exceptions=[ConnectionError, TimeoutError, OSError],
            non_retryable_exceptions=[ValueError, TypeError]
        )
    
    @staticmethod
    def database_operation() -> RetryConfig:
        """Configuration for database operations"""
        return RetryConfig(
            max_attempts=3,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=2.0,
            jitter_type=JitterType.EQUAL,
            retryable_exceptions=[ConnectionError, TimeoutError],
            non_retryable_exceptions=[ValueError, TypeError, KeyError]
        )
    
    @staticmethod
    def api_call() -> RetryConfig:
        """Configuration for API calls"""
        return RetryConfig(
            max_attempts=4,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=1.5,
            jitter_type=JitterType.DECORRELATED,
            retryable_exceptions=[ConnectionError, TimeoutError, OSError]
        )
    
    @staticmethod
    def quick_operation() -> RetryConfig:
        """Configuration for quick operations that should fail fast"""
        return RetryConfig(
            max_attempts=2,
            base_delay=0.1,
            max_delay=1.0,
            exponential_base=2.0,
            jitter_type=JitterType.NONE
        )
