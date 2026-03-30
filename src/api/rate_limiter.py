"""Redis-based rate limiting for API endpoints with in-memory fallback."""

import time
from collections import defaultdict

from fastapi import HTTPException, Request, status

from src.config import settings
from src.logger import logger
from src.storage.redis_cache import query_cache


class RateLimiter:
    """Fixed-window rate limiter: Redis primary, in-memory fallback."""

    def __init__(self):
        self.enabled = settings.enable_rate_limit
        self.limit = settings.rate_limit_requests_per_minute
        self.window = 60  # 1 minute
        # In-memory fallback: {ip: (count, window_start)}
        self._memory: dict[str, tuple[int, float]] = defaultdict(lambda: (0, time.monotonic()))

    def _check_memory(self, client_ip: str) -> None:
        """Enforce rate limit using in-memory counters (fallback path)."""
        count, window_start = self._memory[client_ip]
        now = time.monotonic()

        if now - window_start >= self.window:
            # Start a new window
            self._memory[client_ip] = (1, now)
            return

        count += 1
        self._memory[client_ip] = (count, window_start)

        if count > self.limit:
            logger.warning("Rate limit exceeded (memory) for %s: %d/%d", client_ip, count, self.limit)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests. Limit is {self.limit} per minute.",
                    "retry_after": self.window,
                },
            )

    async def check_rate_limit(self, request: Request):
        """Check rate limit. Falls back to in-memory counter when Redis is unavailable."""
        if not self.enabled:
            return

        client_ip = request.client.host if request.client else "unknown"

        if not query_cache.redis:
            self._check_memory(client_ip)
            return

        key = f"rate_limit:{client_ip}"
        try:
            async with query_cache.redis.pipeline(transaction=True) as pipe:
                await pipe.incr(key)
                await pipe.expire(key, self.window, nx=True)
                results = await pipe.execute()

            count = results[0]
            if count > self.limit:
                logger.warning("Rate limit exceeded for %s: %d/%d", client_ip, count, self.limit)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "rate_limit_exceeded",
                        "message": f"Too many requests. Limit is {self.limit} per minute.",
                        "retry_after": self.window,
                    },
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Rate limiter Redis error, falling back to in-memory: %s", e)
            self._check_memory(client_ip)


rate_limiter = RateLimiter()
