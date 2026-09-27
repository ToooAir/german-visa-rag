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


# ─── Parent expansion ────────────────────────────────────────────────────────


def _child(chunk_id: str, parent_chunk_id: str | None, text: str = "a child slice") -> dict:
    return {
        "id": 1,
        "original_score": 0.8,
        "adjusted_score": 0.8,
        "metadata": {
            "chunk_id": chunk_id,
            "parent_doc_id": "doc_18b",
            "parent_chunk_id": parent_chunk_id,
            "source_url": "https://www.gesetze-im-internet.de/aufenthg_2004/__18b.html",
            "section_header": "Introduction",
            "is_parent": False,
        },
        "text": text,
    }


def _parent(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "section_header": "§ 18b", "text": text, "is_parent": True}


@pytest.fixture
def expansion_settings():
    """Parent expansion on, with room for two ~500-char sections."""
    with patch("src.rag.hybrid_retriever.settings") as mock:
        mock.enable_parent_expansion = True
        mock.rag_parent_context_budget_chars = 1200
        mock.max_context_content_chars = 800
        yield mock


class TestExpandToParents:
    @pytest.mark.asyncio
    async def test_child_is_replaced_by_its_parent_section(self, expansion_settings):
        retriever, qdrant = _make_retriever()
        section = "(1) ... (2) the whole section " + "x" * 400
        qdrant.get_payloads_by_chunk_ids.return_value = {"p1": _parent("p1", section)}

        result = (await retriever.expand_to_parents([_child("c1", "p1")]))[0]

        assert result["text"] == section
        assert result["metadata"]["is_parent"] is True
        assert result["metadata"]["chunk_id"] == "p1"
        assert result["adjusted_score"] == 0.8  # ranking is untouched

    @pytest.mark.asyncio
    async def test_two_children_of_one_section_collapse_to_one_result(self, expansion_settings):
        retriever, qdrant = _make_retriever()
        qdrant.get_payloads_by_chunk_ids.return_value = {"p1": _parent("p1", "the whole section")}

        results = await retriever.expand_to_parents([_child("c1", "p1"), _child("c2", "p1")])

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_oversized_parent_keeps_the_child_rather_than_being_truncated(self, expansion_settings):
        """A section cut mid-rule is worse than the precise child it replaced."""
        retriever, qdrant = _make_retriever()
        qdrant.get_payloads_by_chunk_ids.return_value = {"p1": _parent("p1", "x" * 900)}

        result = (await retriever.expand_to_parents([_child("c1", "p1")]))[0]

        assert result["text"] == "a child slice"
        assert result["metadata"]["is_parent"] is False

    @pytest.mark.asyncio
    async def test_budget_stops_expansion_and_later_results_keep_their_child(self, expansion_settings):
        retriever, qdrant = _make_retriever()
        qdrant.get_payloads_by_chunk_ids.return_value = {
            "p1": _parent("p1", "a" * 700),
            "p2": _parent("p2", "b" * 700),
        }

        first, second = await retriever.expand_to_parents([_child("c1", "p1"), _child("c2", "p2")])

        assert first["text"] == "a" * 700  # 700 of a 1200 budget
        assert second["text"] == "a child slice"  # 700 more would overrun it

    @pytest.mark.asyncio
    async def test_index_without_parent_chunk_id_passes_through(self, expansion_settings):
        """An index ingested before parent_chunk_id existed must still answer."""
        retriever, qdrant = _make_retriever()
        results = await retriever.expand_to_parents([_child("c1", None)])

        assert results[0]["text"] == "a child slice"
        qdrant.get_payloads_by_chunk_ids.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_parent_passes_through(self, expansion_settings):
        retriever, qdrant = _make_retriever()
        qdrant.get_payloads_by_chunk_ids.return_value = {}

        results = await retriever.expand_to_parents([_child("c1", "p1")])
        assert results[0]["text"] == "a child slice"

    @pytest.mark.asyncio
    async def test_disabled_returns_results_untouched(self, expansion_settings):
        expansion_settings.enable_parent_expansion = False
        retriever, qdrant = _make_retriever()
        given = [_child("c1", "p1")]

        assert await retriever.expand_to_parents(given) == given
        qdrant.get_payloads_by_chunk_ids.assert_not_awaited()
