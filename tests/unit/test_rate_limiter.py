"""Unit tests for src/api/rate_limiter.py"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from src.api.rate_limiter import RateLimiter


def make_request(ip: str = "1.2.3.4") -> MagicMock:
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = ip
    return req


def make_limiter(enabled: bool = True, limit: int = 60) -> RateLimiter:
    limiter = RateLimiter.__new__(RateLimiter)
    limiter.enabled = enabled
    limiter.limit = limit
    limiter.window = 60
    return limiter


@pytest.mark.asyncio
async def test_allows_request_under_limit(monkeypatch):
    limiter = make_limiter(limit=60)

    mock_redis = AsyncMock()
    mock_pipe = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=False)
    mock_pipe.execute = AsyncMock(return_value=[5, True])  # count=5, under limit
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)

    monkeypatch.setattr("src.api.rate_limiter.query_cache.redis", mock_redis)

    # Should not raise
    await limiter.check_rate_limit(make_request())


@pytest.mark.asyncio
async def test_raises_429_when_limit_exceeded(monkeypatch):
    limiter = make_limiter(limit=60)

    mock_redis = AsyncMock()
    mock_pipe = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=False)
    mock_pipe.execute = AsyncMock(return_value=[61, True])  # count=61, over limit
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)

    monkeypatch.setattr("src.api.rate_limiter.query_cache.redis", mock_redis)

    with pytest.raises(HTTPException) as exc:
        await limiter.check_rate_limit(make_request())
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_disabled_skips_redis(monkeypatch):
    limiter = make_limiter(enabled=False)
    # No Redis mock needed — disabled limiter must return early without touching Redis
    await limiter.check_rate_limit(make_request())


@pytest.mark.asyncio
async def test_no_redis_skips_check(monkeypatch):
    limiter = make_limiter(enabled=True)
    monkeypatch.setattr("src.api.rate_limiter.query_cache.redis", None)
    # Should return silently when redis is None
    await limiter.check_rate_limit(make_request())


@pytest.mark.asyncio
async def test_redis_error_fails_open(monkeypatch):
    limiter = make_limiter(limit=60)

    mock_redis = AsyncMock()
    mock_pipe = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=False)
    mock_pipe.execute = AsyncMock(side_effect=Exception("connection refused"))
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)

    monkeypatch.setattr("src.api.rate_limiter.query_cache.redis", mock_redis)

    # Should NOT raise — fail open behavior
    await limiter.check_rate_limit(make_request())


@pytest.mark.asyncio
async def test_uses_client_ip_as_key(monkeypatch):
    limiter = make_limiter(limit=60)
    captured_keys = []

    mock_redis = AsyncMock()
    mock_pipe = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=False)

    async def capture_incr(key):
        captured_keys.append(key)

    mock_pipe.incr = AsyncMock(side_effect=capture_incr)
    mock_pipe.expire = AsyncMock()
    mock_pipe.execute = AsyncMock(return_value=[1, True])
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)

    monkeypatch.setattr("src.api.rate_limiter.query_cache.redis", mock_redis)

    await limiter.check_rate_limit(make_request(ip="9.8.7.6"))
    assert any("9.8.7.6" in k for k in captured_keys)
