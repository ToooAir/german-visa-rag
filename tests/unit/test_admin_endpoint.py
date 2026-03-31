"""Unit tests for src/api/endpoints/admin.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.endpoints.admin import get_ingestion_stats, trigger_ingestion


def _make_request(host: str = "127.0.0.1"):
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = host
    return req


# ─── trigger_ingestion ────────────────────────────────────────────────────────


class TestTriggerIngestion:
    @pytest.mark.asyncio
    async def test_returns_ingestion_started(self):
        mock_scheduler = AsyncMock()
        mock_scheduler.trigger_manual_ingestion = AsyncMock()

        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            result = await trigger_ingestion(request=_make_request(), x_api_key="test-key")

        assert result == {"status": "ingestion_started"}

    @pytest.mark.asyncio
    async def test_calls_trigger_manual_ingestion(self):
        mock_scheduler = AsyncMock()
        mock_scheduler.trigger_manual_ingestion = AsyncMock()

        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            await trigger_ingestion(request=_make_request(), x_api_key="sk-test")

        mock_scheduler.trigger_manual_ingestion.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_no_client(self):
        """request.client can be None (e.g. in certain test setups)."""
        mock_scheduler = AsyncMock()
        mock_scheduler.trigger_manual_ingestion = AsyncMock()

        req = MagicMock()
        req.client = None

        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            result = await trigger_ingestion(request=req, x_api_key="key")

        assert result["status"] == "ingestion_started"


# ─── get_ingestion_stats ──────────────────────────────────────────────────────


class TestGetIngestionStats:
    @pytest.mark.asyncio
    async def test_returns_statistics(self):
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {
            "ingested_documents": 10,
            "active_chunks": 200,
        }

        with patch("src.api.endpoints.admin.get_state_store", return_value=mock_store):
            result = await get_ingestion_stats(request=_make_request(), x_api_key="test-key")

        assert result["statistics"]["ingested_documents"] == 10

    @pytest.mark.asyncio
    async def test_calls_get_stats(self):
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {}

        with patch("src.api.endpoints.admin.get_state_store", return_value=mock_store):
            await get_ingestion_stats(request=_make_request(), x_api_key="key")

        mock_store.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_no_client(self):
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {}

        req = MagicMock()
        req.client = None

        with patch("src.api.endpoints.admin.get_state_store", return_value=mock_store):
            result = await get_ingestion_stats(request=req, x_api_key="key")

        assert "statistics" in result
