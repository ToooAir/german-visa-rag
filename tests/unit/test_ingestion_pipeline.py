"""Unit tests for src/ingestion/ingestion_pipeline.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.ingestion.ingestion_pipeline as pipeline_module
from src.ingestion.ingestion_pipeline import IngestionPipeline, get_ingestion_pipeline
from src.vector_db.embedder import QuotaExhaustedError

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_pipeline() -> IngestionPipeline:
    """Build IngestionPipeline with all heavy deps mocked."""
    with (
        patch("src.ingestion.ingestion_pipeline.get_crawler"),
        patch("src.ingestion.ingestion_pipeline.get_chunker"),
        patch("src.ingestion.ingestion_pipeline.get_state_store"),
        patch("src.ingestion.ingestion_pipeline.get_qdrant_client"),
        patch("src.ingestion.ingestion_pipeline.get_mlflow_tracker"),
    ):
        p = IngestionPipeline()

    # Replace all deps with mocks
    p.crawler = MagicMock()
    p.crawler.crawl_document = AsyncMock(return_value=None)

    p.chunker = MagicMock()
    p.chunker.chunk_document = MagicMock(return_value=[])

    p.state_store = MagicMock()
    p.state_store.create_ingestion_run = MagicMock(return_value="run-001")
    p.state_store.register_source_document = MagicMock(return_value="doc-001")
    p.state_store.mark_document_processing = MagicMock()
    p.state_store.mark_document_ingested = MagicMock()
    p.state_store.mark_document_failed = MagicMock()
    p.state_store.get_document_metadata = MagicMock(return_value=None)
    p.state_store.check_chunk_duplicate = MagicMock(return_value=False)
    p.state_store.register_chunk = MagicMock(return_value=1)
    p.state_store.update_chunk_qdrant_id = MagicMock()
    p.state_store.finalize_ingestion_run = MagicMock()
    p.state_store.delete_document_chunks = MagicMock()

    p.qdrant = MagicMock()
    p.qdrant.ensure_collection_exists = AsyncMock()
    p.qdrant.upsert_points = AsyncMock()
    p.qdrant.delete_by_filter = AsyncMock()

    p.mlflow = MagicMock()
    p.mlflow.log_ingestion_run = MagicMock()

    return p


def _make_chunk(chunk_id: str = "c1", text_hash: str = "hash1", is_parent: bool = False):
    chunk = MagicMock()
    chunk.text = "Sample chunk text content"
    chunk.metadata = MagicMock()
    chunk.metadata.chunk_id = chunk_id
    chunk.metadata.text_hash = text_hash
    chunk.metadata.is_parent = is_parent
    chunk.metadata.section_header = "Section"
    chunk.metadata.language = "de"
    return chunk


_SOURCE_DOC = {
    "url": "https://example.com/visa",
    "title": "Visa Page",
    "authority_level": "official",
    "visa_types": ["chancenkarte"],
}


# ─── _compute_document_hash ───────────────────────────────────────────────────


class TestComputeDocumentHash:
    def test_deterministic(self):
        p = _make_pipeline()
        h1 = p._compute_document_hash("hello world")
        h2 = p._compute_document_hash("hello world")
        assert h1 == h2

    def test_different_text_different_hash(self):
        p = _make_pipeline()
        assert p._compute_document_hash("a") != p._compute_document_hash("b")

    def test_returns_hex_string(self):
        p = _make_pipeline()
        h = p._compute_document_hash("test")
        assert all(c in "0123456789abcdef" for c in h)


# ─── run_full_ingestion ───────────────────────────────────────────────────────


class TestRunFullIngestion:
    @pytest.mark.asyncio
    async def test_preflight_failure_returns_early(self):
        p = _make_pipeline()
        with patch("src.ingestion.ingestion_pipeline.embedder") as mock_emb:
            mock_emb.preflight_check = AsyncMock(return_value=False)
            result = await p.run_full_ingestion([_SOURCE_DOC])

        assert result["success"] is False
        assert result["chunks_ingested"] == 0

    @pytest.mark.asyncio
    async def test_quota_exhausted_on_preflight(self):
        p = _make_pipeline()
        with patch("src.ingestion.ingestion_pipeline.embedder") as mock_emb:
            mock_emb.preflight_check = AsyncMock(side_effect=QuotaExhaustedError(wait_seconds=60))
            result = await p.run_full_ingestion([_SOURCE_DOC])

        assert result["success"] is False
        assert result["quota_exhausted"] is True
        assert result["wait_seconds"] == 60

    @pytest.mark.asyncio
    async def test_empty_source_docs_succeeds(self):
        p = _make_pipeline()
        with patch("src.ingestion.ingestion_pipeline.embedder") as mock_emb:
            mock_emb.preflight_check = AsyncMock(return_value=True)
            result = await p.run_full_ingestion([])

        assert result["success"] is True
        assert result["documents_processed"] == 0

    @pytest.mark.asyncio
    async def test_crawl_failure_counted_as_error(self):
        p = _make_pipeline()
        p.crawler.crawl_document = AsyncMock(return_value=None)

        with patch("src.ingestion.ingestion_pipeline.embedder") as mock_emb:
            mock_emb.preflight_check = AsyncMock(return_value=True)
            result = await p.run_full_ingestion([_SOURCE_DOC])

        assert result["documents_processed"] == 0
        assert len(result["errors"]) == 1

    @pytest.mark.asyncio
    async def test_successful_ingestion_with_chunks(self):
        p = _make_pipeline()
        chunk = _make_chunk()
        p.crawler.crawl_document = AsyncMock(
            return_value={
                "markdown": "# Visa Requirements\n\nSome content here.",
                "html": "<h1>Visa</h1>",
                "metadata": {"title": "Visa Page"},
                "fetched_at": "2025-01-01T00:00:00Z",
            }
        )
        p.chunker.chunk_document = MagicMock(return_value=[chunk])

        with (
            patch("src.ingestion.ingestion_pipeline.embedder") as mock_emb,
            patch("src.ingestion.ingestion_pipeline.get_sparse_encoder") as mock_sparse,
            patch("src.ingestion.ingestion_pipeline.settings") as s,
        ):
            mock_emb.preflight_check = AsyncMock(return_value=True)
            mock_emb.embed_texts = AsyncMock(return_value=[[0.1, 0.2]])
            s.sparse_vocab_size = 8000
            s.qdrant_vector_size = 1536
            s.crawler_max_concurrent_requests = 5
            mock_enc = MagicMock()
            mock_enc.encode_batch = MagicMock(return_value=[{"indices": [1], "values": [0.5]}])
            mock_sparse.return_value = mock_enc

            from src.models.chunk import QdrantPayload

            with patch.object(QdrantPayload, "from_chunk", return_value=MagicMock(to_dict=MagicMock(return_value={}))):
                result = await p.run_full_ingestion([_SOURCE_DOC])

        assert result["documents_processed"] == 1
        assert result["chunks_ingested"] >= 1

    @pytest.mark.asyncio
    async def test_mlflow_called_on_completion(self):
        p = _make_pipeline()
        p.crawler.crawl_document = AsyncMock(return_value=None)

        with patch("src.ingestion.ingestion_pipeline.embedder") as mock_emb:
            mock_emb.preflight_check = AsyncMock(return_value=True)
            await p.run_full_ingestion([_SOURCE_DOC])

        p.mlflow.log_ingestion_run.assert_called_once()


# ─── _process_single_document ────────────────────────────────────────────────


class TestProcessSingleDocument:
    @pytest.mark.asyncio
    async def test_crawl_failure_returns_error(self):
        p = _make_pipeline()
        p.crawler.crawl_document = AsyncMock(return_value=None)
        result = await p._process_single_document(_SOURCE_DOC)
        assert result["success"] is False
        assert "crawl" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_content_unchanged_skips_ingestion(self):
        p = _make_pipeline()
        p.crawler.crawl_document = AsyncMock(
            return_value={
                "markdown": "same content",
                "metadata": {},
            }
        )
        content_hash = p._compute_document_hash("same content")
        p.state_store.get_document_metadata = MagicMock(
            return_value={
                "status": "ingested",
                "content_hash": content_hash,
            }
        )

        result = await p._process_single_document(_SOURCE_DOC, force=False)
        assert result["success"] is True
        assert result["chunks_ingested"] == 0

    @pytest.mark.asyncio
    async def test_force_reprocesses_unchanged_content(self):
        p = _make_pipeline()
        p.crawler.crawl_document = AsyncMock(
            return_value={
                "markdown": "same content",
                "metadata": {},
            }
        )
        content_hash = p._compute_document_hash("same content")
        p.state_store.get_document_metadata = MagicMock(
            return_value={
                "status": "ingested",
                "content_hash": content_hash,
            }
        )
        p.chunker.chunk_document = MagicMock(return_value=[])

        result = await p._process_single_document(_SOURCE_DOC, force=True)
        # Should proceed (even if no chunks to ingest)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_no_chunks_returns_success_skipped(self):
        p = _make_pipeline()
        p.crawler.crawl_document = AsyncMock(
            return_value={
                "markdown": "Short content",
                "metadata": {},
            }
        )
        p.chunker.chunk_document = MagicMock(return_value=[])

        result = await p._process_single_document(_SOURCE_DOC)
        assert result["success"] is True
        assert result.get("skipped_low_content") is True

    @pytest.mark.asyncio
    async def test_duplicate_chunks_skipped(self):
        p = _make_pipeline()
        chunk = _make_chunk()
        p.crawler.crawl_document = AsyncMock(
            return_value={
                "markdown": "content",
                "metadata": {},
            }
        )
        p.chunker.chunk_document = MagicMock(return_value=[chunk])
        p.state_store.check_chunk_duplicate = MagicMock(return_value=True)

        result = await p._process_single_document(_SOURCE_DOC)
        # With all chunks duplicated, should succeed but 0 ingested
        assert result["success"] is True
        assert result["chunks_ingested"] == 0

    @pytest.mark.asyncio
    async def test_existing_doc_triggers_replacement(self):
        """When doc exists in DB, old Qdrant + SQLite chunks should be deleted."""
        p = _make_pipeline()
        p.crawler.crawl_document = AsyncMock(
            return_value={
                "markdown": "new content",
                "metadata": {},
            }
        )
        p.state_store.get_document_metadata = MagicMock(
            return_value={
                "status": "ingested",
                "content_hash": "old_hash",  # Different → triggers replacement
            }
        )
        p.chunker.chunk_document = MagicMock(return_value=[])

        await p._process_single_document(_SOURCE_DOC, force=False)

        p.qdrant.delete_by_filter.assert_called_once()
        p.state_store.delete_document_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_during_processing_returns_error(self):
        """Lines 391-395: exception in _process_single_document returns error dict."""
        p = _make_pipeline()
        p.crawler.crawl_document = AsyncMock(side_effect=RuntimeError("unexpected crash"))
        result = await p._process_single_document(_SOURCE_DOC)
        assert result["success"] is False
        assert "unexpected crash" in result["error"]
        # doc_id was set before the error → mark_document_failed called
        p.state_store.mark_document_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_with_falsy_doc_id(self):
        """Line 393->395: doc_id=None/falsy → mark_document_failed NOT called."""
        p = _make_pipeline()
        # register returns None (falsy doc_id), then crawl raises
        p.state_store.register_source_document = MagicMock(return_value=None)
        p.crawler.crawl_document = AsyncMock(side_effect=RuntimeError("crash after register"))
        result = await p._process_single_document(_SOURCE_DOC)
        assert result["success"] is False
        assert "crash after register" in result["error"]
        p.state_store.mark_document_failed.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_duplicate_skipped(self):
        """Lines 298-300: second chunk with same hash in same batch is skipped."""
        p = _make_pipeline()
        # Use parent chunks so no embedding is needed
        chunk1 = _make_chunk("c1", "same_hash", is_parent=True)
        chunk2 = _make_chunk("c2", "same_hash", is_parent=True)  # same hash → batch dup
        p.crawler.crawl_document = AsyncMock(return_value={"markdown": "content", "metadata": {}})
        p.chunker.chunk_document = MagicMock(return_value=[chunk1, chunk2])
        p.state_store.check_chunk_duplicate = MagicMock(return_value=False)

        from src.models.chunk import QdrantPayload

        with (
            patch.object(QdrantPayload, "from_chunk", return_value=MagicMock(to_dict=MagicMock(return_value={}))),
            patch("src.ingestion.ingestion_pipeline.get_sparse_encoder") as mock_sparse,
            patch("src.ingestion.ingestion_pipeline.settings") as s,
        ):
            s.sparse_vocab_size = 8000
            s.qdrant_vector_size = 1536
            mock_enc = MagicMock()
            mock_enc.encode_batch = MagicMock(return_value=[])
            mock_sparse.return_value = mock_enc
            result = await p._process_single_document(_SOURCE_DOC)

        # Only 1 chunk ingested (duplicate skipped), 1 skipped
        assert result["success"] is True
        assert result["chunks_ingested"] == 1
        assert result["chunks_skipped"] == 1


class TestRunFullIngestionAdditional:
    @pytest.mark.asyncio
    async def test_quota_exhausted_during_task_covers_lines_134_148(self):
        """Lines 134-136, 146-148: QuotaExhaustedError raised in sem_process sets quota_exhausted."""
        p = _make_pipeline()

        async def _raise_quota(source_doc, force=False):
            raise QuotaExhaustedError(wait_seconds=60)

        with (
            patch("src.ingestion.ingestion_pipeline.embedder") as mock_emb,
            patch.object(p, "_process_single_document", side_effect=_raise_quota),
        ):
            mock_emb.preflight_check = AsyncMock(return_value=True)
            result = await p.run_full_ingestion([_SOURCE_DOC])

        assert result["quota_exhausted"] is True
        assert len(result["errors"]) >= 1

    @pytest.mark.asyncio
    async def test_unexpected_exception_in_task_counted_as_error(self):
        """Lines 150-151: unexpected Exception from asyncio.gather counted as error."""
        p = _make_pipeline()

        async def _raise_runtime(source_doc, force=False):
            raise RuntimeError("unexpected pipeline crash")

        with (
            patch("src.ingestion.ingestion_pipeline.embedder") as mock_emb,
            patch.object(p, "_process_single_document", side_effect=_raise_runtime),
        ):
            mock_emb.preflight_check = AsyncMock(return_value=True)
            result = await p.run_full_ingestion([_SOURCE_DOC])

        assert len(result["errors"]) == 1
        assert "unexpected pipeline crash" in result["errors"][0]["error"]

    @pytest.mark.asyncio
    async def test_no_mlflow_does_not_crash(self):
        """Line 192->195: mlflow=None skips mlflow logging."""
        p = _make_pipeline()
        p.mlflow = None
        p.crawler.crawl_document = AsyncMock(return_value=None)

        with patch("src.ingestion.ingestion_pipeline.embedder") as mock_emb:
            mock_emb.preflight_check = AsyncMock(return_value=True)
            result = await p.run_full_ingestion([_SOURCE_DOC])

        assert result is not None


# ─── Singleton ────────────────────────────────────────────────────────────────


class TestGetIngestionPipeline:
    def test_returns_same_instance(self):
        pipeline_module._pipeline = None
        with (
            patch("src.ingestion.ingestion_pipeline.get_crawler"),
            patch("src.ingestion.ingestion_pipeline.get_chunker"),
            patch("src.ingestion.ingestion_pipeline.get_state_store"),
            patch("src.ingestion.ingestion_pipeline.get_qdrant_client"),
            patch("src.ingestion.ingestion_pipeline.get_mlflow_tracker"),
        ):
            a = get_ingestion_pipeline()
            b = get_ingestion_pipeline()
        assert a is b
        pipeline_module._pipeline = None
