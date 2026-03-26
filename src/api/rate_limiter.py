"""Redis-based rate limiting for API endpoints."""

import time
from fastapi import Request, HTTPException, status
from src.config import settings
from src.storage.redis_cache import query_cache
from src.logger import logger


class RateLimiter:
    """Simple fixed-window rate limiter using Redis."""

    def __init__(self):
        self.enabled = settings.enable_rate_limit
        self.limit = settings.rate_limit_requests_per_minute
        self.window = 60  # 1 minute

    async def check_rate_limit(self, request: Request):
        """
        Check if the request exceeds the rate limit.
        Uses the client's IP address as the key.
        """
        if not self.enabled or not query_cache.redis:
            return

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:{client_ip}"

        try:
            # Use Redis INCR and EXPIRE for an atomic fixed-window counter
            # Optimization: Use a pipeline to reduce round-trips
            async with query_cache.redis.pipeline(transaction=True) as pipe:
                await pipe.incr(key)
                await pipe.expire(key, self.window, nx=True)  # Only set expiry if key is new
                results = await pipe.execute()

            count = results[0]

            if count > self.limit:
                logger.warning(f"Rate limit exceeded for {client_ip}: {count}/{self.limit}")
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
            logger.error(f"Rate limiter error: {e}")
            # Fail open if Redis is down? For now, we allow the request.
            return


rate_limiter = RateLimiter()
