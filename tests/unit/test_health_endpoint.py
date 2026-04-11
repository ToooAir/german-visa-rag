"""Unit tests for src/api/endpoints/health.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.endpoints.health import health_check, health_check_detailed


def _patch_deps(qdrant_ok=True, db_ok=True, redis_enabled=False, redis_ok=True):
    """Return a context manager tuple that patches all health dependencies."""
    mock_qdrant = AsyncMock()
    mock_qdrant.health_check = AsyncMock(return_value=qdrant_ok)

    mock_store = MagicMock()
    mock_store.db_path = MagicMock()
    mock_store.db_path.exists.return_value = db_ok

    mock_redis = AsyncMock()
    if not redis_ok:
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("Redis down"))
    else:
        mock_redis.ping = AsyncMock(return_value=True)

    mock_cache = MagicMock()
    mock_cache.enabled = redis_enabled
    mock_cache.redis = mock_redis

    return mock_qdrant, mock_store, mock_cache


# ─── health_check (public) ────────────────────────────────────────────────────


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_all_healthy_returns_healthy(self):
        mock_qdrant, mock_store, mock_cache = _patch_deps()
        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        assert response.status == "healthy"

    @pytest.mark.asyncio
    async def test_no_dependencies_field_in_public_response(self):
        """Public endpoint must NOT expose dependency details."""
        mock_qdrant, mock_store, mock_cache = _patch_deps()
        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        assert not hasattr(response, "dependencies")

    @pytest.mark.asyncio
    async def test_qdrant_down_returns_degraded(self):
        mock_qdrant, mock_store, mock_cache = _patch_deps(qdrant_ok=False)
        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        assert response.status == "degraded"

    @pytest.mark.asyncio
    async def test_sqlite_missing_returns_degraded(self):
        mock_qdrant, mock_store, mock_cache = _patch_deps(db_ok=False)
        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        assert response.status == "degraded"

    @pytest.mark.asyncio
    async def test_redis_failure_returns_degraded(self):
        mock_qdrant, mock_store, mock_cache = _patch_deps(redis_enabled=True, redis_ok=False)
        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        assert response.status == "degraded"

    @pytest.mark.asyncio
    async def test_exception_returns_unhealthy(self):
        with patch("src.api.endpoints.health.get_qdrant_client", side_effect=RuntimeError("crash")):
            response = await health_check()

        assert response.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_redis_disabled_not_checked(self):
        mock_qdrant, mock_store, mock_cache = _patch_deps(redis_enabled=False)
        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        mock_cache.redis.ping.assert_not_called()
        assert response.status == "healthy"

    @pytest.mark.asyncio
    async def test_response_has_timestamp_and_version(self):
        mock_qdrant, mock_store, mock_cache = _patch_deps()
        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        assert response.timestamp
        assert response.version


# ─── health_check_detailed (authenticated) ───────────────────────────────────


class TestHealthCheckDetailed:
    @pytest.mark.asyncio
    async def test_all_healthy_includes_dependency_details(self):
        mock_qdrant, mock_store, mock_cache = _patch_deps()
        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check_detailed(x_api_key="test-key")

        assert response.status == "healthy"
        assert "qdrant" in response.dependencies
        assert "sqlite" in response.dependencies
        assert "redis" in response.dependencies
        assert "✓" in response.dependencies["qdrant"]
        assert "✓" in response.dependencies["sqlite"]

    @pytest.mark.asyncio
    async def test_qdrant_down_shows_failed_in_dependencies(self):
        mock_qdrant, mock_store, mock_cache = _patch_deps(qdrant_ok=False)
        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check_detailed(x_api_key="test-key")

        assert response.status == "degraded"
        assert "✗" in response.dependencies["qdrant"]

    @pytest.mark.asyncio
    async def test_sqlite_missing_shows_failed_in_dependencies(self):
        mock_qdrant, mock_store, mock_cache = _patch_deps(db_ok=False)
        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check_detailed(x_api_key="test-key")

        assert "✗" in response.dependencies["sqlite"]

    @pytest.mark.asyncio
    async def test_redis_ping_failure_shows_failed(self):
        mock_qdrant, mock_store, mock_cache = _patch_deps(redis_enabled=True, redis_ok=False)
        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check_detailed(x_api_key="test-key")

        assert "✗" in response.dependencies["redis"]

    @pytest.mark.asyncio
    async def test_redis_ping_success_shows_ok(self):
        mock_qdrant, mock_store, mock_cache = _patch_deps(redis_enabled=True, redis_ok=True)
        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check_detailed(x_api_key="test-key")

        assert "✓" in response.dependencies["redis"]

    @pytest.mark.asyncio
    async def test_exception_returns_unhealthy_with_error_key(self):
        with patch("src.api.endpoints.health.get_qdrant_client", side_effect=RuntimeError("crash")):
            response = await health_check_detailed(x_api_key="test-key")

        assert response.status == "unhealthy"
        assert "error" in response.dependencies
