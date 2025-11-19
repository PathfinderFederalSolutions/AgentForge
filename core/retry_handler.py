#!/usr/bin/env python3
"""
Enhanced Retry Handler for AgentForge
Inspired by the TypeScript service's workflow retry configuration
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass

from core.enhanced_logging import log_info, log_error

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_backoff: bool = True
    jitter: bool = True

class RetryHandler:
    """Enhanced retry handler with exponential backoff and jitter"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    async def retry_async(
        self, 
        func: Callable, 
        *args, 
        config: Optional[RetryConfig] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Retry an async function with exponential backoff"""
        retry_config = config or self.config
        context = context or {}
        
        last_exception = None
        
        for attempt in range(retry_config.max_attempts):
            try:
                log_info(f"Attempting {func.__name__} (attempt {attempt + 1}/{retry_config.max_attempts})", {
                    "function": func.__name__,
                    "attempt": attempt + 1,
                    "context": context
                })
                
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    log_info(f"Function {func.__name__} succeeded after {attempt + 1} attempts")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == retry_config.max_attempts - 1:
                    # Last attempt failed
                    log_error(f"Function {func.__name__} failed after {retry_config.max_attempts} attempts: {str(e)}")
                    break
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt, retry_config)
                
                log_info(f"Function {func.__name__} failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {str(e)}")
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        raise last_exception or Exception(f"Function {func.__name__} failed after {retry_config.max_attempts} attempts")
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay before next retry attempt"""
        if config.exponential_backoff:
            delay = config.base_delay * (2 ** attempt)
        else:
            delay = config.base_delay
        
        # Cap at max delay
        delay = min(delay, config.max_delay)
        
        # Add jitter if enabled
        if config.jitter:
            import random
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay

# Global retry handler
retry_handler = RetryHandler()

async def retry_with_backoff(
    func: Callable,
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """Global function for retrying operations with backoff"""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay
    )
    return await retry_handler.retry_async(func, *args, config=config, context=context, **kwargs)
