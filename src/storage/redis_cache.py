"""Redis-based semantic cache to save LLM API costs for repeated queries."""

import hashlib
import json
from typing import Any, Optional

import redis.asyncio as redis

from src.config import settings
from src.logger import logger
from src.utils.hash_utils import compute_query_fingerprint


class QueryCache:
    def __init__(self):
        self.enabled = settings.enable_query_cache
        self.redis = None
        if self.enabled:
            self.redis = redis.from_url(settings.redis_url, decode_responses=True)
            self.ttl = settings.cache_ttl_seconds

    def _hash_query(self, query: str, scope: dict[str, Any]) -> str:
        """Create a deterministic cache key for a query within a scope.

        The scope carries everything besides the question text that changes the
        answer — visa type, language, and the caller's conversation state. It is
        part of the key, so a cached answer can never be served to a caller in a
        different state. Scope values are compared as serialized, so a list whose
        order varies produces a miss rather than a wrong hit.
        """
        normalized = " ".join(query.lower().split())
        key_material = json.dumps([normalized, scope], sort_keys=True, ensure_ascii=False, default=str)
        return f"rag_cache:v2:{hashlib.sha256(key_material.encode()).hexdigest()}"

    async def get(self, query: str, *, scope: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Retrieve cached response for a query in the given scope."""
        if not self.enabled or not self.redis:
            return None

        try:
            cached = await self.redis.get(self._hash_query(query, scope))
            if cached:
                logger.info("Redis Cache HIT (query=%s)", compute_query_fingerprint(query))
                return json.loads(cached)

            logger.debug("Redis Cache MISS (query=%s)", compute_query_fingerprint(query))
            return None
        except Exception as e:
            logger.warning("Redis get failed: %s", e)
            return None

    async def set(self, query: str, response: dict[str, Any], *, scope: dict[str, Any]):
        """Save response to cache under the given scope."""
        if not self.enabled or not self.redis:
            return

        try:
            await self.redis.setex(self._hash_query(query, scope), self.ttl, json.dumps(response, ensure_ascii=False))
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
