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
        )

    @pytest.mark.asyncio
    async def test_discovery_mode_calls_discovery_job(self):
        sched = _make_scheduler()
        sched._discovery_ingestion_job = AsyncMock()

        with patch("src.ingestion.scheduler.settings") as s:
            s.crawler_discovery_enabled = True
            await sched._ingestion_job()

        sched._discovery_ingestion_job.assert_called_once()

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
    async def test_calls_ingestion_job(self):
        sched = _make_scheduler()
        sched._ingestion_job = AsyncMock()
        await sched.trigger_manual_ingestion()
        sched._ingestion_job.assert_called_once()


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
