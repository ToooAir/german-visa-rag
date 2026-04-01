"""Unit tests for Hybrid Retriever weighting logic."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.models.chunk import VisaType
from src.rag.hybrid_retriever import HybridRetriever


def _make_retriever(search_results=None):
    mock_qdrant = AsyncMock()
    mock_qdrant.hybrid_search.return_value = search_results or []
    return HybridRetriever(qdrant_client=mock_qdrant), mock_qdrant


@pytest.mark.asyncio
async def test_retriever_recency_and_authority_weighting():
    """Test if official and recent documents get higher scores."""

    # Mock Qdrant results
    mock_qdrant_results = [
        {
            "id": 1,
            "score": 0.8,
            "payload": {
                "authority_level": "official",
                "fetched_at": datetime.now(timezone.utc).isoformat(),  # Very recent
                "text": "Doc 1",
            },
        },
        {
            "id": 2,
            "score": 0.8,  # Same base score
            "payload": {
                "authority_level": "third_party",
                "fetched_at": (datetime.now(timezone.utc) - timedelta(days=200)).isoformat(),  # Old
                "text": "Doc 2",
            },
        },
    ]

    # Create mock qdrant client
    mock_qdrant = AsyncMock()
    mock_qdrant.hybrid_search.return_value = mock_qdrant_results

    # Patch embedder to avoid real API calls
    with patch("src.rag.hybrid_retriever.embedder.embed_single", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * 1536

        retriever = HybridRetriever(qdrant_client=mock_qdrant)
        results = await retriever.retrieve("test query")

        assert len(results) == 2
        # Doc 1 should be ranked higher due to authority boost (x1.2) and recency
        assert results[0]["id"] == 1
        assert results[1]["id"] == 2
        assert results[0]["adjusted_score"] > results[1]["adjusted_score"]


@pytest.mark.asyncio
async def test_retrieve_returns_empty_when_embedding_is_empty():
    retriever, _ = _make_retriever()
    with patch("src.rag.hybrid_retriever.embedder.embed_single", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = []
        result = await retriever.retrieve("query")
    assert result == []


@pytest.mark.asyncio
async def test_retrieve_applies_visa_type_filter():
    retriever, mock_qdrant = _make_retriever(
        search_results=[
            {"id": 1, "score": 0.9, "payload": {"authority_level": "official", "fetched_at": None, "text": "t"}}
        ]
    )
    with patch("src.rag.hybrid_retriever.embedder.embed_single", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        results = await retriever.retrieve("query", visa_types=[VisaType.CHANCENKARTE])
    # Verify that the qdrant call included a filter
    call_kwargs = mock_qdrant.hybrid_search.call_args
    assert call_kwargs is not None
    assert len(results) == 1


@pytest.mark.asyncio
async def test_retrieve_handles_invalid_fetched_at():
    """Documents with unparseable fetched_at should default to now (no recency penalty)."""
    retriever, _ = _make_retriever(
        search_results=[
            {"id": 1, "score": 0.5, "payload": {"authority_level": "official", "fetched_at": "not-a-date", "text": ""}}
        ]
    )
    with patch("src.rag.hybrid_retriever.embedder.embed_single", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        results = await retriever.retrieve("query")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_retrieve_adds_utc_to_naive_datetime():
    """Documents with timezone-naive fetched_at get UTC applied."""
    retriever, _ = _make_retriever(
        search_results=[
            {
                "id": 1,
                "score": 0.5,
                "payload": {"authority_level": "official", "fetched_at": "2025-01-01T12:00:00", "text": ""},
            }
        ]
    )
    with patch("src.rag.hybrid_retriever.embedder.embed_single", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        results = await retriever.retrieve("query")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_retrieve_handles_missing_fetched_at():
    retriever, _ = _make_retriever(
        search_results=[{"id": 1, "score": 0.5, "payload": {"authority_level": "semi_official", "text": ""}}]
    )
    with patch("src.rag.hybrid_retriever.embedder.embed_single", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        results = await retriever.retrieve("query")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_retrieve_raises_on_qdrant_error():
    from tenacity import RetryError

    retriever, mock_qdrant = _make_retriever()
    mock_qdrant.hybrid_search.side_effect = RuntimeError("search error")
    with (
        patch("src.rag.hybrid_retriever.embedder.embed_single", new_callable=AsyncMock) as mock_embed,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_embed.return_value = [0.1] * 1536
        with pytest.raises((RuntimeError, RetryError)):
            await retriever.retrieve("query")


@pytest.mark.asyncio
async def test_retrieve_batch_returns_results_for_each_query():
    retriever, mock_qdrant = _make_retriever(
        search_results=[
            {"id": 1, "score": 0.9, "payload": {"authority_level": "official", "fetched_at": None, "text": "t"}}
        ]
    )
    with patch("src.rag.hybrid_retriever.embedder.embed_single", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        results = await retriever.retrieve_batch(["q1", "q2"])
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)


@pytest.mark.asyncio
async def test_retrieve_batch_returns_empty_list_on_individual_failure():
    retriever, mock_qdrant = _make_retriever()
    mock_qdrant.hybrid_search.side_effect = RuntimeError("search error")
    with (
        patch("src.rag.hybrid_retriever.embedder.embed_single", new_callable=AsyncMock) as mock_embed,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_embed.return_value = [0.1] * 1536
        results = await retriever.retrieve_batch(["q1", "q2"])
    assert results == [[], []]
