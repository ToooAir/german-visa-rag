"""Unit tests for src/storage/redis_cache.py"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.storage.redis_cache import QueryCache


@pytest.fixture
def cache_enabled(monkeypatch):
    """Return a QueryCache with Redis mocked out."""
    mock_redis = AsyncMock()
    with patch("src.storage.redis_cache.settings") as mock_settings:
        mock_settings.enable_query_cache = True
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.cache_ttl_seconds = 3600
        cache = QueryCache.__new__(QueryCache)
        cache.enabled = True
        cache.redis = mock_redis
        cache.ttl = 3600
    return cache, mock_redis


@pytest.fixture
def cache_disabled():
    """Return a QueryCache with caching disabled."""
    cache = QueryCache.__new__(QueryCache)
    cache.enabled = False
    cache.redis = None
    return cache


class TestQueryHash:
    def test_deterministic(self):
        cache = QueryCache.__new__(QueryCache)
        cache.enabled = False
        h1 = cache._hash_query("what is chancenkarte?")
        h2 = cache._hash_query("what is chancenkarte?")
        assert h1 == h2

    def test_normalizes_whitespace_and_case(self):
        cache = QueryCache.__new__(QueryCache)
        cache.enabled = False
        h1 = cache._hash_query("  Hello   WORLD  ")
        h2 = cache._hash_query("hello world")
        assert h1 == h2

    def test_different_queries_differ(self):
        cache = QueryCache.__new__(QueryCache)
        cache.enabled = False
        assert cache._hash_query("foo") != cache._hash_query("bar")

    def test_prefix(self):
        cache = QueryCache.__new__(QueryCache)
        cache.enabled = False
        assert cache._hash_query("test").startswith("rag_cache:")


class TestCacheGet:
    @pytest.mark.asyncio
    async def test_hit_returns_parsed_json(self, cache_enabled):
        cache, mock_redis = cache_enabled
        payload = {"answer": "yes", "sources": []}
        mock_redis.get.return_value = json.dumps(payload)

        result = await cache.get("my query")
        assert result == payload

    @pytest.mark.asyncio
    async def test_miss_returns_none(self, cache_enabled):
        cache, mock_redis = cache_enabled
        mock_redis.get.return_value = None
        result = await cache.get("my query")
        assert result is None

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, cache_disabled):
        result = await cache_disabled.get("any query")
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_error_returns_none(self, cache_enabled):
        cache, mock_redis = cache_enabled
        mock_redis.get.side_effect = Exception("connection refused")
        result = await cache.get("query")
        assert result is None


class TestCacheSet:
    @pytest.mark.asyncio
    async def test_stores_with_ttl(self, cache_enabled):
        cache, mock_redis = cache_enabled
        payload = {"answer": "42"}
        await cache.set("my query", payload)

        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[1] == 3600  # TTL
        assert json.loads(args[2]) == payload

    @pytest.mark.asyncio
    async def test_disabled_does_nothing(self, cache_disabled):
        # Should complete without error and not attempt any Redis call
        await cache_disabled.set("query", {"answer": "x"})

    @pytest.mark.asyncio
    async def test_redis_error_is_silenced(self, cache_enabled):
        cache, mock_redis = cache_enabled
        mock_redis.setex.side_effect = Exception("timeout")
        # Should not raise
        await cache.set("query", {"answer": "x"})


class TestCacheClose:
    @pytest.mark.asyncio
    async def test_closes_redis(self, cache_enabled):
        cache, mock_redis = cache_enabled
        await cache.close()
        mock_redis.aclose.assert_called_once()
