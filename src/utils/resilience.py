"""Resilience patterns: retry, circuit breaker, etc."""

from functools import wraps
from typing import Callable, Any, TypeVar, Coroutine
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.logger import logger


def async_retry(
    max_attempts: int = 3,
    base_wait: float = 1,
    max_wait: float = 10,
    exceptions: tuple = (Exception,),
):
    """Decorator for async functions with exponential backoff."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_wait, min=1, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
    )


class CircuitBreaker:
    """Simple circuit breaker for API calls."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.is_open = False
    
    def record_success(self):
        """Record successful call."""
        self.failures = 0
        self.is_open = False
    
    def record_failure(self):
        """Record failed call."""
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker opened after {self.failures} failures")
    
    def is_available(self) -> bool:
        """Check if circuit is available."""
        if not self.is_open:
            return True
        
        # Check timeout
        import time
        if time.time() - self.last_failure_time > self.timeout:
            self.is_open = False
            self.failures = 0
            logger.info("Circuit breaker reset")
            return True
        
        return False
