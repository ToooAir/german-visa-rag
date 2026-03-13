"""
CLI tool for manual and scheduled data ingestion.
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
    """Load URLs from config file."""
    path = custom_path or settings.seed_urls_path
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config.get("documents", [])
    except Exception as e:
        logger.error(f"Failed to load URLs from {path}: {e}")
        return []

@app.command()
def ingest(
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Specific URL to ingest"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to custom seed_urls.yml"),
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
    else:
        # 讀取完整配置檔
        docs_to_process = load_seed_urls(config)
        logger.info(f"Loaded {len(docs_to_process)} documents from config")
        
    if not docs_to_process:
        logger.error("No documents to process. Exiting.")
        raise typer.Exit(code=1)

    # 執行非同步 Pipeline
    result = asyncio.run(pipeline.run_full_ingestion(docs_to_process, triggered_by="cli"))
    
    if result["success"]:
        logger.info(f"✅ Ingestion successful! Processed {result['documents_processed']} docs, ingested {result['chunks_ingested']} chunks.")
    else:
        logger.error(f"❌ Ingestion completed with errors. Check logs.")
        raise typer.Exit(code=1)

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
