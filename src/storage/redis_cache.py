"""Redis-based semantic cache to save LLM API costs for repeated queries."""

import json
import hashlib
from typing import Optional, Dict, Any
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

    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response for a query."""
        if not self.enabled or not self.redis:
            return None
            
        try:
            cached = await self.redis.get(self._hash_query(query))
            if cached:
                logger.info(f"Redis Cache HIT for query: {query[:30]}...")
                return json.loads(cached)
            
            logger.debug(f"Redis Cache MISS for query: {query[:30]}...")
            return None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    async def set(self, query: str, response: Dict[str, Any]):
        """Save response to cache."""
        if not self.enabled or not self.redis:
            return
            
        try:
            await self.redis.setex(
                self._hash_query(query),
                self.ttl,
                json.dumps(response, ensure_ascii=False)
            )
            logger.debug("Query cached successfully to Redis")
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    async def close(self):
        """Close Redis connection pool."""
        if self.redis:
            await self.redis.aclose()
            logger.info("Redis connection closed")


# Singleton instance
query_cache = QueryCache()
