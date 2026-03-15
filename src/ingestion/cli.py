"""
CLI tool for manual and scheduled data ingestion.
Supports both manual URL ingestion and automatic discovery mode.
Perfect for cronjobs or GCP Cloud Run Jobs.
"""

import asyncio
import typer
from typing import Optional
import yaml
from pathlib import Path

from src.logger import logger
from src.config import settings
from src.ingestion.ingestion_pipeline import get_ingestion_pipeline

app = typer.Typer(help="German Visa RAG Ingestion CLI")


def load_seed_urls(custom_path: Optional[str] = None):
    """Load URLs from config file (extra_urls section for backwards compat)."""
    path = custom_path or settings.seed_urls_path
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            # Support both old 'documents' key and new 'extra_urls' key
            docs = config.get("extra_urls", config.get("documents", []))
            return docs
    except Exception as e:
        logger.error(f"Failed to load URLs from {path}: {e}")
        return []


def load_domain_configs(custom_path: Optional[str] = None):
    """Load domain configurations from seed_urls.yml."""
    path = custom_path or settings.seed_urls_path
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config.get("domains", [])
    except Exception as e:
        logger.error(f"Failed to load domain configs from {path}: {e}")
        return []


@app.command()
def ingest(
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Specific URL to ingest"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to custom seed_urls.yml"),
    auto_discover: bool = typer.Option(False, "--auto-discover", "-a", help="Enable automatic URL discovery mode"),
    force_discover: bool = typer.Option(False, "--force-discover", "-f", help="Force fresh URL discovery (bypass cache)"),
    force: bool = typer.Option(False, "--force", help="Force re-processing of all documents even if content hasn't changed"),
):
    """Run the ingestion pipeline."""
    logger.info("Starting CLI Ingestion...")

    pipeline = get_ingestion_pipeline()

    if source:
        # 單一網址測試
        docs_to_process = [{
            "url": source,
            "title": "CLI Manual Ingest",
            "authority_level": "third_party",
            "visa_types": ["general"]
        }]
        logger.info(f"Ingesting single source: {source}")
    elif auto_discover:
        # 自動發現模式：先發現 URL 再爬取
        logger.info("🔍 Auto-discovery mode enabled. Discovering URLs...")
        result = asyncio.run(_run_discovery_ingestion(pipeline, force_refresh=force_discover, force_ingest=force))

        if result.get("quota_exhausted"):
            processed = result['documents_processed']
            skipped = result.get('documents_skipped_quota', 0)
            wait = result.get('wait_seconds', 0)
            wait_hrs = round(wait / 3600, 1) if wait else "unknown"
            logger.error(f"⚠️  Embedding API quota exhausted! Wait ~{wait_hrs} hours.")
            _print_ingestion_summary(result)
            raise typer.Exit(code=2)
        
        _print_ingestion_summary(result)
        if not result["success"]:
            raise typer.Exit(code=1)
        return
    else:
        # 讀取配置檔（extra_urls）
        docs_to_process = load_seed_urls(config)
        logger.info(f"Loaded {len(docs_to_process)} documents from config")

    if not docs_to_process:
        logger.error("No documents to process. Exiting.")
        raise typer.Exit(code=1)

    # 執行非同步 Pipeline
    result = asyncio.run(pipeline.run_full_ingestion(docs_to_process, triggered_by="cli", force=force))
    _print_ingestion_summary(result)

    if not result["success"]:
        raise typer.Exit(code=1)


def _print_ingestion_summary(result: dict):
    """Print a structured summary of the ingestion run."""
    from collections import Counter
    
    processed = result.get("documents_processed", 0)
    ingested = result.get("chunks_ingested", 0)
    skipped = result.get("chunks_skipped", 0)
    skipped_quota = result.get("documents_skipped_quota", 0)
    errors = result.get("errors", [])

    typer.echo("\n" + "="*40)
    typer.echo("📊 INGESTION SUMMARY")
    typer.echo("-" * 40)
    typer.echo(f"📄 Total Documents:    {processed + len(errors) + skipped_quota}")
    typer.echo(f"✅ Successfully Processed: {processed}")
    typer.echo(f"❌ Failed:              {len(errors)}")
    if skipped_quota > 0:
        typer.echo(f"⏭️  Skipped (Quota):     {skipped_quota}")
    typer.echo(f"🔗 Chunks Ingested:     {ingested}")
    typer.echo(f"⏭️  Chunks Skipped:      {skipped}")
    typer.echo("-" * 40)

    if errors:
        typer.echo("❌ ERROR SUMMARY (Top 5):")
        error_counts = Counter(errors)
        for msg, count in error_counts.most_common(5):
            # Shorten very long error messages
            display_msg = (msg[:75] + '...') if len(msg) > 75 else msg
            typer.echo(f" - {count:3d}x: {display_msg}")
        typer.echo("-" * 40)
    
    typer.echo("="*40 + "\n")


async def _run_discovery_ingestion(pipeline, force_refresh: bool = False, force_ingest: bool = False):
    """Run URL discovery then ingest all discovered pages."""
    from src.ingestion.crawler import get_crawler

    crawler = get_crawler()
    crawled_docs = await crawler.crawl_with_discovery(force_refresh=force_refresh)

    if not crawled_docs:
        return {"success": False, "documents_processed": 0, "chunks_ingested": 0, "errors": ["No documents discovered"]}

    # Convert crawled docs to ingestion format
    source_docs = []
    for doc in crawled_docs:
        source_docs.append({
            "url": doc["url"],
            "title": doc.get("metadata", {}).get("title", doc["url"]),
            "authority_level": doc.get("authority_level", "third_party"),
            "visa_types": doc.get("visa_types", ["general"]),
        })

    return await pipeline.run_full_ingestion(source_docs, triggered_by="cli_discovery", force=force_ingest)


@app.command()
def discover(
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Specific domain to discover"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to custom seed_urls.yml"),
):
    """
    Discover URLs without crawling (dry-run).
    Shows all URLs that would be discovered and crawled.
    """
    logger.info("🔍 Starting URL discovery (dry-run)...")
    results = asyncio.run(_run_discovery(domain))

    total = 0
    for result in results:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Domain: {result.domain}")
        typer.echo(f"  From sitemap:  {result.from_sitemap}")
        typer.echo(f"  From crawling: {result.from_crawling}")
        typer.echo(f"  Filtered out:  {result.filtered_out}")
        typer.echo(f"  Total URLs:    {len(result.discovered_urls)}")
        typer.echo(f"{'='*60}")

        for i, url in enumerate(result.discovered_urls, 1):
            typer.echo(f"  {i:3d}. {url}")

        total += len(result.discovered_urls)

    typer.echo(f"\n📊 Total discovered URLs: {total}")


async def _run_discovery(domain: Optional[str] = None):
    """Run URL discovery."""
    from src.ingestion.url_discoverer import get_url_discoverer

    discoverer = get_url_discoverer()

    try:
        if domain:
            result = await discoverer.discover_single_domain(domain)
            return [result]
        else:
            return await discoverer.discover_all()
    finally:
        await discoverer.close()


@app.command()
def status():
    """Show ingestion statistics."""
    from src.storage.sqlite_state_store import get_state_store
    store = get_state_store()
    stats = store.get_stats()
    typer.echo("=== Ingestion Statistics ===")
    for k, v in stats.items():
        typer.echo(f"{k}: {v}")

if __name__ == "__main__":
    app()
