"""
APScheduler configuration for periodic ingestion tasks.
Manages background jobs for crawling and updating documents.
Supports both manual URL-based and auto-discovery ingestion.
"""

from typing import Optional
from datetime import datetime
import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import yaml

from src.config import settings
from src.logger import logger
from src.ingestion.ingestion_pipeline import get_ingestion_pipeline


class IngestionScheduler:
    """APScheduler wrapper for background ingestion tasks."""

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.pipeline = get_ingestion_pipeline()
        self.seed_urls = self._load_seed_urls()

    def _load_seed_urls(self):
        """Load source URLs from seed_urls.yml (extra_urls section)."""
        try:
            with open(settings.seed_urls_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                # Support both old 'documents' key and new 'extra_urls' key
                documents = config.get("extra_urls", config.get("documents", []))
                logger.info(f"Loaded {len(documents)} seed URLs")
                return documents
        except Exception as e:
            logger.error(f"Failed to load seed URLs: {e}")
            return []

    async def _ingestion_job(self):
        """Periodic ingestion job — uses auto-discovery if enabled."""
        logger.info("Starting scheduled ingestion job")
        try:
            if settings.crawler_discovery_enabled:
                # Auto-discovery mode
                await self._discovery_ingestion_job()
            else:
                # Legacy mode: ingest from seed URLs
                result = await self.pipeline.run_full_ingestion(
                    self.seed_urls,
                    triggered_by="scheduler",
                )
                logger.info("Scheduled ingestion completed", extra=result)
        except Exception as e:
            logger.error(f"Scheduled ingestion failed: {e}", exc_info=True)

    async def _discovery_ingestion_job(self):
        """Run auto-discovery then ingest all discovered pages."""
        from src.ingestion.crawler import get_crawler

        logger.info("Running discovery-based scheduled ingestion")
        crawler = get_crawler()

        try:
            crawled_docs = await crawler.crawl_with_discovery()

            if not crawled_docs:
                logger.warning("Discovery produced no documents")
                return

            # Convert to ingestion format
            source_docs = []
            for doc in crawled_docs:
                source_docs.append({
                    "url": doc["url"],
                    "title": doc.get("metadata", {}).get("title", doc["url"]),
                    "authority_level": doc.get("authority_level", "third_party"),
                    "visa_types": doc.get("visa_types", ["general"]),
                })

            # Also add the extra_urls from config
            source_docs.extend(self.seed_urls)

            result = await self.pipeline.run_full_ingestion(
                source_docs,
                triggered_by="scheduler_discovery",
            )
            logger.info("Discovery ingestion completed", extra=result)

        except Exception as e:
            logger.error(f"Discovery ingestion failed: {e}", exc_info=True)
        finally:
            crawler.reset_visited()

    def start(self):
        """Start scheduler."""
        try:
            # Add periodic ingestion job
            interval_hours = settings.ingestion_schedule_interval_hours

            self.scheduler.add_job(
                self._ingestion_job,
                trigger=IntervalTrigger(hours=interval_hours),
                id="ingestion_job",
                name="Periodic document ingestion",
                replace_existing=True,
            )

            mode = "discovery" if settings.crawler_discovery_enabled else "legacy"
            logger.info(
                f"Ingestion scheduler started",
                extra={"interval_hours": interval_hours, "mode": mode}
            )

            self.scheduler.start()

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")

    async def shutdown(self):
        """Graceful shutdown."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler shut down")

    async def trigger_manual_ingestion(self):
        """Manually trigger ingestion (for admin endpoint)."""
        logger.info("Manual ingestion triggered")
        await self._ingestion_job()


# Singleton instance
_scheduler: Optional[IngestionScheduler] = None


def get_scheduler() -> IngestionScheduler:
    """Get or create scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = IngestionScheduler()
    return _scheduler
