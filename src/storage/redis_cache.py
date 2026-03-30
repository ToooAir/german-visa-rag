"""Redis-based semantic cache to save LLM API costs for repeated queries."""

import hashlib
import json
from typing import Any, Optional

import redis.asyncio as redis

from src.config import settings
from src.logger import logger


class QueryCache:
    def __init__(self):
        self.enabled = settings.enable_query_cache
        self.redis = None
        if self.enabled:
            self.redis = redis.from_url(settings.redis_url, decode_responses=True)
            self.ttl = settings.cache_ttl_seconds

    def _hash_query(self, query: str) -> str:
        """Create a deterministic hash for the query."""
        normalized = " ".join(query.lower().split())
        return f"rag_cache:{hashlib.md5(normalized.encode()).hexdigest()}"

    async def get(self, query: str) -> Optional[dict[str, Any]]:
        """Retrieve cached response for a query."""
        if not self.enabled or not self.redis:
            return None

        try:
            cached = await self.redis.get(self._hash_query(query))
            if cached:
                logger.info("Redis Cache HIT for query: %.30s...", query)
                return json.loads(cached)

            logger.debug("Redis Cache MISS for query: %.30s...", query)
            return None
        except Exception as e:
            logger.warning("Redis get failed: %s", e)
            return None

    async def set(self, query: str, response: dict[str, Any]):
        """Save response to cache."""
        if not self.enabled or not self.redis:
            return

        try:
            await self.redis.setex(self._hash_query(query), self.ttl, json.dumps(response, ensure_ascii=False))
            logger.debug("Query cached successfully to Redis")
        except Exception as e:
            logger.warning("Redis set failed: %s", e)

    async def close(self):
        """Close Redis connection pool."""
        if self.redis:
            await self.redis.aclose()
            logger.info("Redis connection closed")


# Singleton instance
query_cache = QueryCache()
