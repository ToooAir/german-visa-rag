"""Unit tests for src/api/endpoints/admin.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.endpoints.admin import (
    discover_urls,
    get_ingestion_stats,
    ingest_single_url,
    trigger_ingestion,
)


def _make_request(host: str = "127.0.0.1"):
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = host
    return req


def _make_scheduler(mode: str = "seed_urls"):
    mock = AsyncMock()
    mock.trigger_manual_ingestion = AsyncMock(return_value={"triggered": True, "mode": mode})
    mock.ingest_single_url = AsyncMock(return_value={"chunks_ingested": 3, "success": True})
    mock.discover_urls = AsyncMock(
        return_value=[{"domain": "example.com", "urls": ["https://example.com"], "total": 1}]
    )
    return mock


# ─── trigger_ingestion ────────────────────────────────────────────────────────


class TestTriggerIngestion:
    @pytest.mark.asyncio
    async def test_returns_ingestion_started(self):
        with patch("src.api.endpoints.admin.get_scheduler", return_value=_make_scheduler()):
            result = await trigger_ingestion(request=_make_request(), x_api_key="test-key")

        assert result["status"] == "ingestion_started"

    @pytest.mark.asyncio
    async def test_response_includes_force_flags_and_mode(self):
        with patch("src.api.endpoints.admin.get_scheduler", return_value=_make_scheduler("discovery")):
            result = await trigger_ingestion(
                request=_make_request(),
                force=True,
                force_discover=True,
                x_api_key="test-key",
            )

        assert result["force"] is True
        assert result["force_discover"] is True
        assert result["mode"] == "discovery"

    @pytest.mark.asyncio
    async def test_force_params_passed_to_scheduler(self):
        mock_scheduler = _make_scheduler()
        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            await trigger_ingestion(
                request=_make_request(),
                force=True,
                force_discover=True,
                auto_discover=False,
                x_api_key="key",
            )

        mock_scheduler.trigger_manual_ingestion.assert_called_once_with(
            force=True,
            force_discover=True,
            auto_discover=False,
        )

    @pytest.mark.asyncio
    async def test_defaults_are_false_and_none(self):
        mock_scheduler = _make_scheduler()
        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            # Pass explicit values to bypass FastAPI's Query() wrapper in direct calls
            await trigger_ingestion(
                request=_make_request(),
                force=False,
                force_discover=False,
                auto_discover=None,
                x_api_key="key",
            )

        mock_scheduler.trigger_manual_ingestion.assert_called_once_with(
            force=False,
            force_discover=False,
            auto_discover=None,
        )

    @pytest.mark.asyncio
    async def test_auto_discover_override_passed(self):
        mock_scheduler = _make_scheduler()
        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            await trigger_ingestion(
                request=_make_request(),
                auto_discover=True,
                x_api_key="key",
            )

        call_kwargs = mock_scheduler.trigger_manual_ingestion.call_args.kwargs
        assert call_kwargs["auto_discover"] is True

    @pytest.mark.asyncio
    async def test_handles_no_client(self):
        req = MagicMock()
        req.client = None
        with patch("src.api.endpoints.admin.get_scheduler", return_value=_make_scheduler()):
            result = await trigger_ingestion(request=req, x_api_key="key")

        assert result["status"] == "ingestion_started"


# ─── ingest_single_url ────────────────────────────────────────────────────────


class TestIngestSingleUrl:
    @pytest.mark.asyncio
    async def test_returns_completed_status(self):
        mock_scheduler = _make_scheduler()
        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            result = await ingest_single_url(
                request=_make_request(),
                url="https://example.com/page",
                x_api_key="key",
            )

        assert result["status"] == "ingestion_completed"
        assert result["url"] == "https://example.com/page"

    @pytest.mark.asyncio
    async def test_url_and_force_passed_to_scheduler(self):
        mock_scheduler = _make_scheduler()
        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            await ingest_single_url(
                request=_make_request(),
                url="https://example.com/page",
                force=True,
                x_api_key="key",
            )

        mock_scheduler.ingest_single_url.assert_called_once_with(
            url="https://example.com/page",
            force=True,
        )

    @pytest.mark.asyncio
    async def test_force_defaults_false(self):
        mock_scheduler = _make_scheduler()
        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            await ingest_single_url(
                request=_make_request(),
                url="https://example.com",
                force=False,
                x_api_key="key",
            )

        call_kwargs = mock_scheduler.ingest_single_url.call_args.kwargs
        assert call_kwargs["force"] is False

    @pytest.mark.asyncio
    async def test_result_included_in_response(self):
        mock_scheduler = _make_scheduler()
        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            result = await ingest_single_url(
                request=_make_request(),
                url="https://example.com",
                x_api_key="key",
            )

        assert result["result"]["chunks_ingested"] == 3


# ─── discover_urls ────────────────────────────────────────────────────────────


class TestDiscoverUrls:
    @pytest.mark.asyncio
    async def test_returns_discovery_completed(self):
        mock_scheduler = _make_scheduler()
        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            result = await discover_urls(request=_make_request(), x_api_key="key")

        assert result["status"] == "discovery_completed"
        assert result["total_urls"] == 1
        assert len(result["domains"]) == 1

    @pytest.mark.asyncio
    async def test_domain_passed_to_scheduler(self):
        mock_scheduler = _make_scheduler()
        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            await discover_urls(
                request=_make_request(),
                domain="make-it-in-germany.com",
                x_api_key="key",
            )

        mock_scheduler.discover_urls.assert_called_once_with(domain="make-it-in-germany.com")

    @pytest.mark.asyncio
    async def test_no_domain_passes_none(self):
        mock_scheduler = _make_scheduler()
        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            await discover_urls(request=_make_request(), domain=None, x_api_key="key")

        mock_scheduler.discover_urls.assert_called_once_with(domain=None)

    @pytest.mark.asyncio
    async def test_total_urls_aggregated_across_domains(self):
        mock_scheduler = AsyncMock()
        mock_scheduler.discover_urls = AsyncMock(
            return_value=[
                {"domain": "a.com", "urls": ["https://a.com/1", "https://a.com/2"], "total": 2},
                {"domain": "b.com", "urls": ["https://b.com/1"], "total": 1},
            ]
        )
        with patch("src.api.endpoints.admin.get_scheduler", return_value=mock_scheduler):
            result = await discover_urls(request=_make_request(), x_api_key="key")

        assert result["total_urls"] == 3

    @pytest.mark.asyncio
    async def test_handles_no_client(self):
        req = MagicMock()
        req.client = None
        with patch("src.api.endpoints.admin.get_scheduler", return_value=_make_scheduler()):
            result = await discover_urls(request=req, x_api_key="key")

        assert result["status"] == "discovery_completed"


# ─── get_ingestion_stats ──────────────────────────────────────────────────────


class TestGetIngestionStats:
    @pytest.mark.asyncio
    async def test_returns_statistics(self):
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"ingested_documents": 10, "active_chunks": 200}

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
