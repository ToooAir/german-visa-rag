"""Integration tests for the ingestion pipeline."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from src.ingestion.ingestion_pipeline import IngestionPipeline

@pytest.mark.asyncio
async def test_full_ingestion_flow():
    """Test the ingestion pipeline from crawling to processing."""
    
    # 1. Read sample HTML fixture
    html_path = Path("tests/fixtures/sample_html.html")
    with open(html_path, "r") as f:
        mock_html = f.read()
    
    pipeline = IngestionPipeline()
    
    # 2. Mock external dependencies
    # Mock Crawler to return our fixture
    pipeline.crawler.crawl_document = AsyncMock(return_value={
        "url": "http://mock-visa.com",
        "markdown": "## Requirements\nNeeds 6 points.",
        "fetched_at": "2024-01-01T00:00:00",
        "metadata": {"title": "Test"}
    })
    
    # Mock Embedder to return dummy vectors
    with patch("src.ingestion.ingestion_pipeline.embedder.embed_texts", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [[0.1]*1536, [0.1]*1536] # Assume 2 chunks generated
        
        # Mock Qdrant Upsert
        pipeline.qdrant.ensure_collection_exists = AsyncMock()
        pipeline.qdrant.upsert_points = AsyncMock()
        
        # 3. Run Pipeline
        source_docs = [{
            "url": "http://mock-visa.com",
            "title": "Test Doc",
            "authority_level": "official",
            "visa_types": ["chancenkarte"]
        }]
        
        result = await pipeline.run_full_ingestion(source_docs, triggered_by="pytest")
        
        # 4. Assertions
        assert result["success"] == True
        assert result["documents_processed"] == 1
        assert result["chunks_ingested"] > 0
        
        # Verify Qdrant was called
        pipeline.qdrant.upsert_points.assert_called_once()
        
        # Verify State Store tracked the document
        stats = pipeline.state_store.get_stats()
        assert stats["total_ingestion_runs"] >= 1
