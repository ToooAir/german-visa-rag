"""Integration tests for the ingestion pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.ingestion_pipeline import IngestionPipeline


@pytest.mark.asyncio
async def test_full_ingestion_flow():
    """Test the ingestion pipeline from crawling to processing."""

    # 1. Read sample HTML fixture
    html_path = Path("tests/fixtures/sample_html.html")
    with open(html_path, "r") as f:
        f.read()

    pipeline = IngestionPipeline()

    # 2. Mock external dependencies
    # Mock Crawler to return our fixture
    pipeline.crawler.crawl_document = AsyncMock(
        return_value={
            "url": "http://mock-visa.com",
            "markdown": "## Requirements\nNeeds 6 points.",
            "fetched_at": "2024-01-01T00:00:00",
            "metadata": {"title": "Test"},
        }
    )

    # Mock Embedder to return dummy vectors
    with (
        patch("src.ingestion.ingestion_pipeline.embedder.embed_texts", new_callable=AsyncMock) as mock_embed,
        patch("src.ingestion.ingestion_pipeline.embedder.preflight_check", new_callable=AsyncMock) as mock_preflight,
    ):

        async def mock_embed_func(texts, **kwargs):
            return [[0.1] * 1536 for _ in texts]

        mock_embed.side_effect = mock_embed_func
        mock_preflight.return_value = True
        # Mock Qdrant Upsert
        pipeline.qdrant.ensure_collection_exists = AsyncMock()
        pipeline.qdrant.upsert_points = AsyncMock()
        pipeline.qdrant.delete_by_filter = AsyncMock()

        # Mock State Store
        pipeline.state_store.register_source_document = MagicMock(return_value="doc_123")
        pipeline.state_store.mark_document_processing = MagicMock()
        pipeline.state_store.get_document_metadata = MagicMock(
            return_value={"status": "ingested", "content_hash": "dummy_hash"}
        )
        pipeline.state_store.mark_document_ingested = MagicMock()
        pipeline.state_store.get_stats = MagicMock(return_value={"total_ingestion_runs": 1})

        # 3. Run Pipeline
        source_docs = [
            {
                "url": "http://mock-visa.com",
                "title": "Test Doc",
                "authority_level": "official",
                "visa_types": ["chancenkarte"],
            }
        ]

        result = await pipeline.run_full_ingestion(source_docs, triggered_by="pytest", force=True)

        # 4. Assertions
        assert result["success"]
        assert result["documents_processed"] == 1
        assert result["chunks_ingested"] > 0

        # Verify Qdrant was called
        pipeline.qdrant.upsert_points.assert_called_once()

        # Verify State Store tracked the document
        stats = pipeline.state_store.get_stats()
        assert stats["total_ingestion_runs"] >= 1
