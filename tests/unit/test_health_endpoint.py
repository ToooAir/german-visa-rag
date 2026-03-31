"""Unit tests for src/api/endpoints/health.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.endpoints.health import get_stats, health_check

# ─── health_check ─────────────────────────────────────────────────────────────


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_all_healthy(self):
        mock_qdrant = AsyncMock()
        mock_qdrant.health_check = AsyncMock(return_value=True)

        mock_store = MagicMock()
        mock_store.db_path = MagicMock()
        mock_store.db_path.exists.return_value = True

        mock_cache = MagicMock()
        mock_cache.enabled = False

        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        assert response.status == "healthy"
        assert "✓" in response.dependencies["qdrant"]
        assert "✓" in response.dependencies["sqlite"]

    @pytest.mark.asyncio
    async def test_qdrant_down_returns_degraded(self):
        mock_qdrant = AsyncMock()
        mock_qdrant.health_check = AsyncMock(return_value=False)

        mock_store = MagicMock()
        mock_store.db_path = MagicMock()
        mock_store.db_path.exists.return_value = True

        mock_cache = MagicMock()
        mock_cache.enabled = False

        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        assert response.status == "degraded"
        assert "✗" in response.dependencies["qdrant"]

    @pytest.mark.asyncio
    async def test_sqlite_missing_returns_degraded(self):
        mock_qdrant = AsyncMock()
        mock_qdrant.health_check = AsyncMock(return_value=True)

        mock_store = MagicMock()
        mock_store.db_path = MagicMock()
        mock_store.db_path.exists.return_value = False

        mock_cache = MagicMock()
        mock_cache.enabled = False

        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        assert response.status == "degraded"
        assert "✗" in response.dependencies["sqlite"]

    @pytest.mark.asyncio
    async def test_redis_ping_failure_returns_degraded(self):
        mock_qdrant = AsyncMock()
        mock_qdrant.health_check = AsyncMock(return_value=True)

        mock_store = MagicMock()
        mock_store.db_path = MagicMock()
        mock_store.db_path.exists.return_value = True

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("Redis down"))

        mock_cache = MagicMock()
        mock_cache.enabled = True
        mock_cache.redis = mock_redis

        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        assert response.status == "degraded"
        assert "✗" in response.dependencies["redis"]

    @pytest.mark.asyncio
    async def test_redis_ping_success_healthy(self):
        mock_qdrant = AsyncMock()
        mock_qdrant.health_check = AsyncMock(return_value=True)

        mock_store = MagicMock()
        mock_store.db_path = MagicMock()
        mock_store.db_path.exists.return_value = True

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        mock_cache = MagicMock()
        mock_cache.enabled = True
        mock_cache.redis = mock_redis

        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        assert response.status == "healthy"
        assert "✓" in response.dependencies["redis"]

    @pytest.mark.asyncio
    async def test_exception_returns_unhealthy(self):
        with patch("src.api.endpoints.health.get_qdrant_client", side_effect=RuntimeError("crash")):
            response = await health_check()

        assert response.status == "unhealthy"
        assert "error" in response.dependencies

    @pytest.mark.asyncio
    async def test_redis_disabled_not_checked(self):
        mock_qdrant = AsyncMock()
        mock_qdrant.health_check = AsyncMock(return_value=True)

        mock_store = MagicMock()
        mock_store.db_path = MagicMock()
        mock_store.db_path.exists.return_value = True

        mock_cache = MagicMock()
        mock_cache.enabled = False
        mock_cache.redis = AsyncMock()  # Should NOT be called

        with (
            patch("src.api.endpoints.health.get_qdrant_client", return_value=mock_qdrant),
            patch("src.api.endpoints.health.get_state_store", return_value=mock_store),
            patch("src.api.endpoints.health.query_cache", mock_cache),
        ):
            response = await health_check()

        mock_cache.redis.ping.assert_not_called()
        assert response.status == "healthy"


# ─── get_stats ────────────────────────────────────────────────────────────────


class TestGetStats:
    @pytest.mark.asyncio
    async def test_returns_stats(self):
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {
            "ingested_documents": 5,
            "active_chunks": 100,
            "total_ingestion_runs": 2,
        }

        with patch("src.api.endpoints.health.get_state_store", return_value=mock_store):
            result = await get_stats(x_api_key="test-key")

        assert "statistics" in result
        assert result["statistics"]["ingested_documents"] == 5
        assert "timestamp" in result
