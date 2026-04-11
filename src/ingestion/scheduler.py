"""
APScheduler configuration for periodic ingestion tasks.
Manages background jobs for crawling and updating documents.
Supports both manual URL-based and auto-discovery ingestion.
"""

from typing import Optional

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.config import settings
from src.ingestion.ingestion_pipeline import get_ingestion_pipeline
from src.logger import logger


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

    async def _ingestion_job(self, force: bool = False, force_discover: bool = False):
        """Periodic ingestion job — uses auto-discovery if enabled."""
        logger.info("Starting scheduled ingestion job")
        try:
            if settings.crawler_discovery_enabled:
                # Auto-discovery mode
                await self._discovery_ingestion_job(force_ingest=force, force_refresh=force_discover)
            else:
                # Legacy mode: ingest from seed URLs
                result = await self.pipeline.run_full_ingestion(
                    self.seed_urls,
                    triggered_by="scheduler",
                    force=force,
                )
                logger.info("Scheduled ingestion completed", extra=result)
        except Exception as e:
            logger.error(f"Scheduled ingestion failed: {e}", exc_info=True)

    async def _discovery_ingestion_job(self, force_ingest: bool = False, force_refresh: bool = False):
        """Run auto-discovery then ingest all discovered pages."""
        from src.ingestion.crawler import get_crawler

        logger.info(
            "Running discovery-based scheduled ingestion | force_ingest=%s force_refresh=%s",
            force_ingest,
            force_refresh,
        )
        crawler = get_crawler()

        try:
            crawled_docs = await crawler.crawl_with_discovery(force_refresh=force_refresh)

            if not crawled_docs:
                logger.warning("Discovery produced no documents")
                return

            # Convert to ingestion format
            source_docs = []
            for doc in crawled_docs:
                source_docs.append(
                    {
                        "url": doc["url"],
                        "title": doc.get("metadata", {}).get("title", doc["url"]),
                        "authority_level": doc.get("authority_level", "third_party"),
                        "visa_types": doc.get("visa_types", ["general"]),
                    }
                )

            # Also add the extra_urls from config
            source_docs.extend(self.seed_urls)

            result = await self.pipeline.run_full_ingestion(
                source_docs,
                triggered_by="scheduler_discovery",
                force=force_ingest,
            )
            logger.info("Discovery ingestion completed", extra=result)

        except Exception as e:
            logger.error(f"Discovery ingestion failed: {e}", exc_info=True)
        finally:
            crawler.reset_visited()

    async def ingest_single_url(self, url: str, force: bool = False) -> dict:
        """Ingest a single URL (for admin endpoint --source equivalent)."""
        logger.info("Single URL ingestion triggered | url=%s force=%s", url, force)
        source_docs = [
            {"url": url, "title": "API Manual Ingest", "authority_level": "third_party", "visa_types": ["general"]}
        ]
        return await self.pipeline.run_full_ingestion(source_docs, triggered_by="api_single", force=force)

    async def discover_urls(self, domain: Optional[str] = None) -> list:
        """Dry-run URL discovery — returns discovered URLs without ingesting."""
        from src.ingestion.url_discoverer import get_url_discoverer

        logger.info("URL discovery dry-run triggered | domain=%s", domain or "all")
        discoverer = get_url_discoverer()
        try:
            if domain:
                result = await discoverer.discover_single_domain(domain)
                return [{"domain": result.domain, "urls": result.discovered_urls, "total": len(result.discovered_urls)}]
            else:
                results = await discoverer.discover_all()
                return [
                    {"domain": r.domain, "urls": r.discovered_urls, "total": len(r.discovered_urls)} for r in results
                ]
        finally:
            await discoverer.close()

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
            logger.info("Ingestion scheduler started", extra={"interval_hours": interval_hours, "mode": mode})

            self.scheduler.start()

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")

    async def shutdown(self):
        """Graceful shutdown."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler shut down")

    async def trigger_manual_ingestion(
        self,
        force: bool = False,
        force_discover: bool = False,
        auto_discover: Optional[bool] = None,
    ) -> dict:
        """Manually trigger ingestion (for admin endpoint).

        Args:
            force: Re-process all documents even if content hasn't changed.
            force_discover: Force fresh URL discovery, bypassing the visited-URL cache.
            auto_discover: Override CRAWLER_DISCOVERY_ENABLED for this run only.
                           If None, falls back to the settings value.
        """
        logger.info(
            "Manual ingestion triggered | force=%s force_discover=%s auto_discover=%s",
            force,
            force_discover,
            auto_discover,
        )
        use_discovery = auto_discover if auto_discover is not None else settings.crawler_discovery_enabled
        if use_discovery:
            await self._discovery_ingestion_job(force_ingest=force, force_refresh=force_discover)
        else:
            result = await self.pipeline.run_full_ingestion(
                self.seed_urls,
                triggered_by="api_manual",
                force=force,
            )
            logger.info("Manual ingestion completed", extra=result)
        return {"triggered": True, "mode": "discovery" if use_discovery else "seed_urls"}


# Singleton instance
_scheduler: Optional[IngestionScheduler] = None


def get_scheduler() -> IngestionScheduler:
    """Get or create scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = IngestionScheduler()
    return _scheduler
