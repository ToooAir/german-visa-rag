"""Unit tests for src/ingestion/scheduler.py"""

from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

import src.ingestion.scheduler as scheduler_module
from src.ingestion.scheduler import IngestionScheduler, get_scheduler

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_scheduler() -> IngestionScheduler:
    """Build an IngestionScheduler with all heavy deps mocked out."""
    with (
        patch("src.ingestion.scheduler.AsyncIOScheduler"),
        patch("src.ingestion.scheduler.get_ingestion_pipeline"),
        patch("src.ingestion.scheduler.settings") as s,
        patch("builtins.open", mock_open(read_data="extra_urls: []")),
    ):
        s.seed_urls_path = "seed_urls.yml"
        s.crawler_discovery_enabled = False
        s.ingestion_schedule_interval_hours = 6
        sched = IngestionScheduler()

    sched.pipeline = MagicMock()
    sched.pipeline.run_full_ingestion = AsyncMock(return_value={"chunks_ingested": 5})
    return sched


# ─── _load_seed_urls ──────────────────────────────────────────────────────────


class TestLoadSeedUrls:
    def test_loads_extra_urls_key(self):
        yaml_content = "extra_urls:\n  - url: https://example.com\n    title: Example\n"
        with (
            patch("src.ingestion.scheduler.AsyncIOScheduler"),
            patch("src.ingestion.scheduler.get_ingestion_pipeline"),
            patch("src.ingestion.scheduler.settings") as s,
            patch("builtins.open", mock_open(read_data=yaml_content)),
        ):
            s.seed_urls_path = "seed_urls.yml"
            sched = IngestionScheduler()
        assert len(sched.seed_urls) == 1

    def test_falls_back_to_documents_key(self):
        yaml_content = "documents:\n  - url: https://example.com\n    title: Example\n"
        with (
            patch("src.ingestion.scheduler.AsyncIOScheduler"),
            patch("src.ingestion.scheduler.get_ingestion_pipeline"),
            patch("src.ingestion.scheduler.settings") as s,
            patch("builtins.open", mock_open(read_data=yaml_content)),
        ):
            s.seed_urls_path = "seed_urls.yml"
            sched = IngestionScheduler()
        assert len(sched.seed_urls) == 1

    def test_file_not_found_returns_empty(self):
        with (
            patch("src.ingestion.scheduler.AsyncIOScheduler"),
            patch("src.ingestion.scheduler.get_ingestion_pipeline"),
            patch("src.ingestion.scheduler.settings") as s,
            patch("builtins.open", side_effect=FileNotFoundError("not found")),
        ):
            s.seed_urls_path = "nonexistent.yml"
            sched = IngestionScheduler()
        assert sched.seed_urls == []


# ─── _ingestion_job ───────────────────────────────────────────────────────────


class TestIngestionJob:
    @pytest.mark.asyncio
    async def test_legacy_mode_calls_pipeline(self):
        sched = _make_scheduler()
        sched.seed_urls = [{"url": "https://example.com", "title": "Ex"}]

        with patch("src.ingestion.scheduler.settings") as s:
            s.crawler_discovery_enabled = False
            await sched._ingestion_job()

        sched.pipeline.run_full_ingestion.assert_called_once_with(
            sched.seed_urls,
            triggered_by="scheduler",
            force=False,
        )

    @pytest.mark.asyncio
    async def test_legacy_mode_passes_force_flag(self):
        sched = _make_scheduler()
        sched.seed_urls = [{"url": "https://example.com", "title": "Ex"}]

        with patch("src.ingestion.scheduler.settings") as s:
            s.crawler_discovery_enabled = False
            await sched._ingestion_job(force=True)

        call_kwargs = sched.pipeline.run_full_ingestion.call_args.kwargs
        assert call_kwargs["force"] is True

    @pytest.mark.asyncio
    async def test_discovery_mode_calls_discovery_job(self):
        sched = _make_scheduler()
        sched._discovery_ingestion_job = AsyncMock()

        with patch("src.ingestion.scheduler.settings") as s:
            s.crawler_discovery_enabled = True
            await sched._ingestion_job()

        sched._discovery_ingestion_job.assert_called_once_with(force_ingest=False, force_refresh=False)

    @pytest.mark.asyncio
    async def test_discovery_mode_passes_force_flags(self):
        sched = _make_scheduler()
        sched._discovery_ingestion_job = AsyncMock()

        with patch("src.ingestion.scheduler.settings") as s:
            s.crawler_discovery_enabled = True
            await sched._ingestion_job(force=True, force_discover=True)

        sched._discovery_ingestion_job.assert_called_once_with(force_ingest=True, force_refresh=True)

    @pytest.mark.asyncio
    async def test_exception_is_swallowed(self):
        sched = _make_scheduler()
        sched.pipeline.run_full_ingestion = AsyncMock(side_effect=RuntimeError("DB down"))

        with patch("src.ingestion.scheduler.settings") as s:
            s.crawler_discovery_enabled = False
            # Should not raise
            await sched._ingestion_job()


# ─── _discovery_ingestion_job ─────────────────────────────────────────────────


class TestDiscoveryIngestionJob:
    @pytest.mark.asyncio
    async def test_no_documents_returns_early(self):
        sched = _make_scheduler()
        mock_crawler = MagicMock()
        mock_crawler.crawl_with_discovery = AsyncMock(return_value=[])
        mock_crawler.reset_visited = MagicMock()

        with patch("src.ingestion.crawler.get_crawler", return_value=mock_crawler):
            await sched._discovery_ingestion_job()

        sched.pipeline.run_full_ingestion.assert_not_called()
        mock_crawler.reset_visited.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_refresh_passed_to_crawler(self):
        sched = _make_scheduler()
        mock_crawler = MagicMock()
        mock_crawler.crawl_with_discovery = AsyncMock(return_value=[])
        mock_crawler.reset_visited = MagicMock()

        with patch("src.ingestion.crawler.get_crawler", return_value=mock_crawler):
            await sched._discovery_ingestion_job(force_refresh=True)

        mock_crawler.crawl_with_discovery.assert_called_once_with(force_refresh=True)

    @pytest.mark.asyncio
    async def test_force_ingest_passed_to_pipeline(self):
        sched = _make_scheduler()
        sched.seed_urls = []
        crawled = [
            {
                "url": "https://example.com",
                "metadata": {"title": "T"},
                "authority_level": "official",
                "visa_types": ["general"],
            }
        ]
        mock_crawler = MagicMock()
        mock_crawler.crawl_with_discovery = AsyncMock(return_value=crawled)
        mock_crawler.reset_visited = MagicMock()

        with patch("src.ingestion.crawler.get_crawler", return_value=mock_crawler):
            await sched._discovery_ingestion_job(force_ingest=True)

        call_kwargs = sched.pipeline.run_full_ingestion.call_args.kwargs
        assert call_kwargs["force"] is True

    @pytest.mark.asyncio
    async def test_documents_passed_to_pipeline(self):
        sched = _make_scheduler()
        sched.seed_urls = []

        crawled = [
            {
                "url": "https://example.com/page",
                "metadata": {"title": "Page"},
                "authority_level": "official",
                "visa_types": ["chancenkarte"],
            }
        ]
        mock_crawler = MagicMock()
        mock_crawler.crawl_with_discovery = AsyncMock(return_value=crawled)
        mock_crawler.reset_visited = MagicMock()

        with patch("src.ingestion.crawler.get_crawler", return_value=mock_crawler):
            await sched._discovery_ingestion_job()

        sched.pipeline.run_full_ingestion.assert_called_once()
        call_args = sched.pipeline.run_full_ingestion.call_args
        source_docs = call_args[0][0]
        assert len(source_docs) == 1
        assert source_docs[0]["url"] == "https://example.com/page"
        mock_crawler.reset_visited.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_in_crawl_swallowed_reset_called(self):
        sched = _make_scheduler()
        mock_crawler = MagicMock()
        mock_crawler.crawl_with_discovery = AsyncMock(side_effect=RuntimeError("crawl error"))
        mock_crawler.reset_visited = MagicMock()

        with patch("src.ingestion.crawler.get_crawler", return_value=mock_crawler):
            await sched._discovery_ingestion_job()

        # reset_visited should be called even on error (finally block)
        mock_crawler.reset_visited.assert_called_once()


# ─── start / shutdown ─────────────────────────────────────────────────────────


class TestStartShutdown:
    def test_start_adds_job_and_starts_scheduler(self):
        sched = _make_scheduler()
        sched.scheduler = MagicMock()

        with patch("src.ingestion.scheduler.settings") as s:
            s.ingestion_schedule_interval_hours = 6
            s.crawler_discovery_enabled = False
            sched.start()

        sched.scheduler.add_job.assert_called_once()
        sched.scheduler.start.assert_called_once()

    def test_start_swallows_exception(self):
        sched = _make_scheduler()
        sched.scheduler = MagicMock()
        sched.scheduler.add_job.side_effect = RuntimeError("scheduler error")

        with patch("src.ingestion.scheduler.settings") as s:
            s.ingestion_schedule_interval_hours = 6
            s.crawler_discovery_enabled = False
            # Should not raise
            sched.start()

    @pytest.mark.asyncio
    async def test_shutdown_stops_running_scheduler(self):
        sched = _make_scheduler()
        sched.scheduler = MagicMock()
        sched.scheduler.running = True

        await sched.shutdown()
        sched.scheduler.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_not_running(self):
        sched = _make_scheduler()
        sched.scheduler = MagicMock()
        sched.scheduler.running = False

        await sched.shutdown()
        sched.scheduler.shutdown.assert_not_called()


# ─── trigger_manual_ingestion ─────────────────────────────────────────────────


class TestTriggerManualIngestion:
    @pytest.mark.asyncio
    async def test_calls_ingestion_job_with_defaults(self):
        sched = _make_scheduler()
        with patch("src.ingestion.scheduler.settings") as s:
            s.crawler_discovery_enabled = False
            await sched.trigger_manual_ingestion()
        sched.pipeline.run_full_ingestion.assert_called_once()
        call_kwargs = sched.pipeline.run_full_ingestion.call_args.kwargs
        assert call_kwargs["force"] is False

    @pytest.mark.asyncio
    async def test_force_flags_propagated(self):
        sched = _make_scheduler()
        with patch("src.ingestion.scheduler.settings") as s:
            s.crawler_discovery_enabled = False
            await sched.trigger_manual_ingestion(force=True, force_discover=True)
        call_kwargs = sched.pipeline.run_full_ingestion.call_args.kwargs
        assert call_kwargs["force"] is True

    @pytest.mark.asyncio
    async def test_auto_discover_true_overrides_settings(self):
        """auto_discover=True forces discovery mode regardless of settings."""
        sched = _make_scheduler()
        sched._discovery_ingestion_job = AsyncMock()
        with patch("src.ingestion.scheduler.settings") as s:
            s.crawler_discovery_enabled = False  # would normally use seed_urls
            result = await sched.trigger_manual_ingestion(auto_discover=True)
        sched._discovery_ingestion_job.assert_called_once()
        assert result["mode"] == "discovery"

    @pytest.mark.asyncio
    async def test_auto_discover_false_overrides_settings(self):
        """auto_discover=False forces seed_url mode regardless of settings."""
        sched = _make_scheduler()
        sched.pipeline.run_full_ingestion = AsyncMock(return_value={})
        with patch("src.ingestion.scheduler.settings") as s:
            s.crawler_discovery_enabled = True  # would normally use discovery
            result = await sched.trigger_manual_ingestion(auto_discover=False)
        assert result["mode"] == "seed_urls"

    @pytest.mark.asyncio
    async def test_returns_mode_dict(self):
        sched = _make_scheduler()
        sched._ingestion_job = AsyncMock()
        with patch("src.ingestion.scheduler.settings") as s:
            s.crawler_discovery_enabled = False
            result = await sched.trigger_manual_ingestion()
        assert "mode" in result
        assert result["triggered"] is True


# ─── ingest_single_url ────────────────────────────────────────────────────────


class TestIngestSingleUrl:
    @pytest.mark.asyncio
    async def test_calls_pipeline_with_url(self):
        sched = _make_scheduler()
        await sched.ingest_single_url(url="https://example.com/page")
        call_args = sched.pipeline.run_full_ingestion.call_args
        source_docs = call_args[0][0]
        assert source_docs[0]["url"] == "https://example.com/page"

    @pytest.mark.asyncio
    async def test_force_passed_to_pipeline(self):
        sched = _make_scheduler()
        await sched.ingest_single_url(url="https://example.com", force=True)
        call_kwargs = sched.pipeline.run_full_ingestion.call_args.kwargs
        assert call_kwargs["force"] is True

    @pytest.mark.asyncio
    async def test_triggered_by_api_single(self):
        sched = _make_scheduler()
        await sched.ingest_single_url(url="https://example.com")
        call_kwargs = sched.pipeline.run_full_ingestion.call_args.kwargs
        assert call_kwargs["triggered_by"] == "api_single"

    @pytest.mark.asyncio
    async def test_returns_pipeline_result(self):
        sched = _make_scheduler()
        sched.pipeline.run_full_ingestion = AsyncMock(return_value={"chunks_ingested": 7})
        result = await sched.ingest_single_url(url="https://example.com")
        assert result["chunks_ingested"] == 7


# ─── discover_urls ────────────────────────────────────────────────────────────


class TestDiscoverUrls:
    @pytest.mark.asyncio
    async def test_discover_all_domains(self):
        sched = _make_scheduler()
        mock_result = MagicMock()
        mock_result.domain = "example.com"
        mock_result.discovered_urls = ["https://example.com/a", "https://example.com/b"]

        mock_discoverer = AsyncMock()
        mock_discoverer.discover_all = AsyncMock(return_value=[mock_result])
        mock_discoverer.close = AsyncMock()

        with patch("src.ingestion.url_discoverer.get_url_discoverer", return_value=mock_discoverer):
            result = await sched.discover_urls()

        assert len(result) == 1
        assert result[0]["domain"] == "example.com"
        assert result[0]["total"] == 2

    @pytest.mark.asyncio
    async def test_discover_single_domain(self):
        sched = _make_scheduler()
        mock_result = MagicMock()
        mock_result.domain = "make-it-in-germany.com"
        mock_result.discovered_urls = ["https://make-it-in-germany.com/en/visa"]

        mock_discoverer = AsyncMock()
        mock_discoverer.discover_single_domain = AsyncMock(return_value=mock_result)
        mock_discoverer.close = AsyncMock()

        with patch("src.ingestion.url_discoverer.get_url_discoverer", return_value=mock_discoverer):
            result = await sched.discover_urls(domain="make-it-in-germany.com")

        mock_discoverer.discover_single_domain.assert_called_once_with("make-it-in-germany.com")
        assert result[0]["total"] == 1

    @pytest.mark.asyncio
    async def test_close_called_on_exception(self):
        sched = _make_scheduler()
        mock_discoverer = AsyncMock()
        mock_discoverer.discover_all = AsyncMock(side_effect=RuntimeError("discovery failed"))
        mock_discoverer.close = AsyncMock()

        with patch("src.ingestion.url_discoverer.get_url_discoverer", return_value=mock_discoverer):
            with pytest.raises(RuntimeError):
                await sched.discover_urls()

        mock_discoverer.close.assert_called_once()


# ─── Singleton ────────────────────────────────────────────────────────────────


class TestGetScheduler:
    def test_returns_same_instance(self):
        scheduler_module._scheduler = None
        with (
            patch("src.ingestion.scheduler.AsyncIOScheduler"),
            patch("src.ingestion.scheduler.get_ingestion_pipeline"),
            patch("src.ingestion.scheduler.settings") as s,
            patch("builtins.open", mock_open(read_data="extra_urls: []")),
        ):
            s.seed_urls_path = "seed_urls.yml"
            a = get_scheduler()
            b = get_scheduler()
        assert a is b
        scheduler_module._scheduler = None
