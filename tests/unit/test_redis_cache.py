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


EMPTY_SCOPE: dict = {}


def _scope(visa_type=None, language="auto", requirements=None) -> dict:
    """Mirror AnswerGenerator._cache_scope for key tests."""
    return {"language": language, "visa_type": visa_type, "requirements": requirements or []}


class TestQueryHash:
    def test_deterministic(self):
        cache = QueryCache.__new__(QueryCache)
        cache.enabled = False
        h1 = cache._hash_query("what is chancenkarte?", EMPTY_SCOPE)
        h2 = cache._hash_query("what is chancenkarte?", EMPTY_SCOPE)
        assert h1 == h2

    def test_normalizes_whitespace_and_case(self):
        cache = QueryCache.__new__(QueryCache)
        cache.enabled = False
        h1 = cache._hash_query("  Hello   WORLD  ", EMPTY_SCOPE)
        h2 = cache._hash_query("hello world", EMPTY_SCOPE)
        assert h1 == h2

    def test_different_queries_differ(self):
        cache = QueryCache.__new__(QueryCache)
        cache.enabled = False
        assert cache._hash_query("foo", EMPTY_SCOPE) != cache._hash_query("bar", EMPTY_SCOPE)

    def test_prefix(self):
        cache = QueryCache.__new__(QueryCache)
        cache.enabled = False
        assert cache._hash_query("test", EMPTY_SCOPE).startswith("rag_cache:")

    def test_scope_is_order_independent(self):
        """Same scope, different dict insertion order -> same key."""
        cache = QueryCache.__new__(QueryCache)
        cache.enabled = False
        a = cache._hash_query("q", {"language": "de", "visa_type": "chancenkarte"})
        b = cache._hash_query("q", {"visa_type": "chancenkarte", "language": "de"})
        assert a == b


class TestScopeIsolation:
    """The cached payload carries per-user eligibility state, so the state must be
    part of the key: the same question asked in a different state must not hit."""

    def setup_method(self):
        self.cache = QueryCache.__new__(QueryCache)
        self.cache.enabled = False

    def test_conversation_state_changes_key(self):
        q = "do I qualify?"
        user_a = _scope(requirements=[{"id": "2-1", "value": "C1", "status": "required"}])
        user_b = _scope(requirements=[{"id": "2-1", "value": "A2", "status": "required"}])
        assert self.cache._hash_query(q, user_a) != self.cache._hash_query(q, user_b)

    def test_empty_state_differs_from_populated_state(self):
        q = "do I qualify?"
        fresh = _scope()
        with_state = _scope(requirements=[{"id": "1-1", "value": "MET", "status": "required"}])
        assert self.cache._hash_query(q, fresh) != self.cache._hash_query(q, with_state)

    def test_visa_type_changes_key(self):
        q = "what is the salary threshold?"
        assert self.cache._hash_query(q, _scope(visa_type="blue_card")) != self.cache._hash_query(
            q, _scope(visa_type="skilled_worker")
        )

    def test_language_changes_key(self):
        q = "what is the salary threshold?"
        assert self.cache._hash_query(q, _scope(language="de")) != self.cache._hash_query(q, _scope(language="zh"))

    def test_identical_scope_shares_key(self):
        """Stateless questions still share a cache entry -- the cache stays useful."""
        q = "what is the chancenkarte?"
        assert self.cache._hash_query(q, _scope(visa_type="chancenkarte")) == self.cache._hash_query(
            q, _scope(visa_type="chancenkarte")
        )

    @pytest.mark.asyncio
    async def test_write_in_one_state_is_not_read_in_another(self, cache_enabled):
        """End-to-end: user A's cached answer is unreachable from user B's state."""
        cache, mock_redis = cache_enabled
        user_a = _scope(requirements=[{"id": "2-1", "value": "C1", "status": "required"}])
        user_b = _scope(requirements=[{"id": "2-1", "value": "A2", "status": "required"}])

        await cache.set("do I qualify?", {"answer": "A's answer"}, scope=user_a)
        written_key = mock_redis.setex.call_args[0][0]

        await cache.get("do I qualify?", scope=user_b)
        read_key = mock_redis.get.call_args[0][0]

        assert written_key != read_key


class TestCacheGet:
    @pytest.mark.asyncio
    async def test_hit_returns_parsed_json(self, cache_enabled):
        cache, mock_redis = cache_enabled
        payload = {"answer": "yes", "sources": []}
        mock_redis.get.return_value = json.dumps(payload)

        result = await cache.get("my query", scope=EMPTY_SCOPE)
        assert result == payload

    @pytest.mark.asyncio
    async def test_miss_returns_none(self, cache_enabled):
        cache, mock_redis = cache_enabled
        mock_redis.get.return_value = None
        result = await cache.get("my query", scope=EMPTY_SCOPE)
        assert result is None

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, cache_disabled):
        result = await cache_disabled.get("any query", scope=EMPTY_SCOPE)
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_error_returns_none(self, cache_enabled):
        cache, mock_redis = cache_enabled
        mock_redis.get.side_effect = Exception("connection refused")
        result = await cache.get("query", scope=EMPTY_SCOPE)
        assert result is None


class TestCacheSet:
    @pytest.mark.asyncio
    async def test_stores_with_ttl(self, cache_enabled):
        cache, mock_redis = cache_enabled
        payload = {"answer": "42"}
        await cache.set("my query", payload, scope=EMPTY_SCOPE)

        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[1] == 3600  # TTL
        assert json.loads(args[2]) == payload

    @pytest.mark.asyncio
    async def test_disabled_does_nothing(self, cache_disabled):
        # Should complete without error and not attempt any Redis call
        await cache_disabled.set("query", {"answer": "x"}, scope=EMPTY_SCOPE)

    @pytest.mark.asyncio
    async def test_redis_error_is_silenced(self, cache_enabled):
        cache, mock_redis = cache_enabled
        mock_redis.setex.side_effect = Exception("timeout")
        # Should not raise
        await cache.set("query", {"answer": "x"}, scope=EMPTY_SCOPE)


class TestCacheClose:
    @pytest.mark.asyncio
    async def test_closes_redis(self, cache_enabled):
        cache, mock_redis = cache_enabled
        await cache.close()
        mock_redis.aclose.assert_called_once()
