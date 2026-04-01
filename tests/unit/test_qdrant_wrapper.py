"""Unit tests for src/vector_db/qdrant_client_wrapper.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http.models import FieldCondition, Filter, MatchAny
from tenacity import RetryError

import src.vector_db.qdrant_client_wrapper as wrapper_module
from src.models.chunk import AuthorityLevel
from src.vector_db.qdrant_client_wrapper import QdrantWrapper, get_qdrant_client

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_wrapper() -> QdrantWrapper:
    """Create QdrantWrapper with mocked Qdrant clients."""
    with patch("src.vector_db.qdrant_client_wrapper.settings") as s:
        s.qdrant_url = "http://localhost:6333"
        s.qdrant_api_key = None
        s.qdrant_collection_name = "test_collection"
        s.qdrant_vector_size = 1536
        s.qdrant_prefer_grpc = False
        with (
            patch("src.vector_db.qdrant_client_wrapper.AsyncQdrantClient"),
            patch("src.vector_db.qdrant_client_wrapper.QdrantClient"),
        ):
            w = QdrantWrapper()

    w.client = AsyncMock()
    w.sync_client = MagicMock()
    w.collection_name = "test_collection"
    w.vector_size = 1536
    return w


def _make_point(point_id: int = 1, score: float = 0.9, payload: dict = None):
    p = MagicMock()
    p.id = point_id
    p.score = score
    p.payload = payload or {"source_url": "https://example.com", "source_title": "Title"}
    return p


# ─── ensure_collection_exists ─────────────────────────────────────────────────


class TestEnsureCollectionExists:
    @pytest.mark.asyncio
    async def test_skips_when_collection_already_exists(self):
        w = _make_wrapper()
        col = MagicMock()
        col.name = "test_collection"
        collections_resp = MagicMock()
        collections_resp.collections = [col]
        w.client.get_collections = AsyncMock(return_value=collections_resp)

        await w.ensure_collection_exists()

        w.client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_collection_when_missing(self):
        w = _make_wrapper()
        collections_resp = MagicMock()
        collections_resp.collections = []
        w.client.get_collections = AsyncMock(return_value=collections_resp)
        w.client.create_collection = AsyncMock()

        await w.ensure_collection_exists()

        w.client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_on_client_error(self):
        w = _make_wrapper()
        w.client.get_collections = AsyncMock(side_effect=RuntimeError("Qdrant down"))

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises((RuntimeError, RetryError)),
        ):
            await w.ensure_collection_exists()


# ─── upsert_points ────────────────────────────────────────────────────────────


class TestUpsertPoints:
    @pytest.mark.asyncio
    async def test_empty_points_skips_upsert(self):
        w = _make_wrapper()
        await w.upsert_points([])
        w.client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_upserts_points(self):
        w = _make_wrapper()
        w.client.upsert = AsyncMock()
        points = [MagicMock(), MagicMock()]

        await w.upsert_points(points)

        w.client.upsert.assert_called_once_with(
            collection_name="test_collection",
            points=points,
            wait=True,
        )

    @pytest.mark.asyncio
    async def test_raises_on_upsert_error(self):
        w = _make_wrapper()
        w.client.upsert = AsyncMock(side_effect=RuntimeError("upsert failed"))

        with pytest.raises(RuntimeError):
            await w.upsert_points([MagicMock()])

    @pytest.mark.asyncio
    async def test_wait_false_passed_through(self):
        w = _make_wrapper()
        w.client.upsert = AsyncMock()

        await w.upsert_points([MagicMock()], wait=False)

        _, kwargs = w.client.upsert.call_args
        assert kwargs["wait"] is False


# ─── hybrid_search ────────────────────────────────────────────────────────────


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_returns_formatted_results(self):
        w = _make_wrapper()
        point = _make_point(1, 0.9)
        search_result = MagicMock()
        search_result.points = [point]
        w.client.query_points = AsyncMock(return_value=search_result)

        with patch("src.vector_db.qdrant_client_wrapper.settings") as s:
            s.enable_sparse_search = False

            result = await w.hybrid_search(
                dense_vector=[0.1, 0.2],
                query_text="chancenkarte",
                top_k=5,
            )

        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_sparse_search_enabled_builds_two_prefetches(self):
        from qdrant_client.http.models import SparseVector as QdrantSparseVector

        w = _make_wrapper()
        search_result = MagicMock()
        search_result.points = []
        w.client.query_points = AsyncMock(return_value=search_result)

        sparse_vec = QdrantSparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2])
        mock_encoder = MagicMock()
        mock_encoder.encode = MagicMock(return_value=sparse_vec)

        with (
            patch("src.vector_db.qdrant_client_wrapper.settings") as s,
            patch("src.vector_db.sparse_encoder.get_sparse_encoder", return_value=mock_encoder),
        ):
            s.enable_sparse_search = True
            s.sparse_vocab_size = 8000
            await w.hybrid_search(dense_vector=[0.1], query_text="visa")

        _, kwargs = w.client.query_points.call_args
        assert len(kwargs["prefetch"]) == 2

    @pytest.mark.asyncio
    async def test_empty_sparse_vector_falls_back_to_dense_only(self):
        from qdrant_client.http.models import SparseVector as QdrantSparseVector

        w = _make_wrapper()
        search_result = MagicMock()
        search_result.points = []
        w.client.query_points = AsyncMock(return_value=search_result)

        sparse_vec = QdrantSparseVector(indices=[], values=[])  # Empty → skip sparse leg
        mock_encoder = MagicMock()
        mock_encoder.encode = MagicMock(return_value=sparse_vec)

        with (
            patch("src.vector_db.qdrant_client_wrapper.settings") as s,
            patch("src.vector_db.sparse_encoder.get_sparse_encoder", return_value=mock_encoder),
        ):
            s.enable_sparse_search = True
            s.sparse_vocab_size = 8000
            await w.hybrid_search(dense_vector=[0.1], query_text="visa")

        _, kwargs = w.client.query_points.call_args
        assert len(kwargs["prefetch"]) == 1  # Dense only

    @pytest.mark.asyncio
    async def test_raises_on_search_error(self):
        w = _make_wrapper()
        w.client.query_points = AsyncMock(side_effect=RuntimeError("search failed"))

        with (
            patch("src.vector_db.qdrant_client_wrapper.settings") as s,
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises((RuntimeError, RetryError)),
        ):
            s.enable_sparse_search = False
            await w.hybrid_search(dense_vector=[0.1], query_text="q")

    @pytest.mark.asyncio
    async def test_filters_passed_to_query(self):
        w = _make_wrapper()
        search_result = MagicMock()
        search_result.points = []
        w.client.query_points = AsyncMock(return_value=search_result)

        # Must be a real Filter — pydantic validates the Prefetch model
        my_filter = Filter(must=[FieldCondition(key="authority_level", match=MatchAny(any=["official"]))])
        with patch("src.vector_db.qdrant_client_wrapper.settings") as s:
            s.enable_sparse_search = False
            await w.hybrid_search(dense_vector=[0.1], query_text="q", filters=my_filter)

        _, kwargs = w.client.query_points.call_args
        assert kwargs["prefetch"][0].filter == my_filter


# ─── get_unique_sources ───────────────────────────────────────────────────────


class TestGetUniqueSources:
    @pytest.mark.asyncio
    async def test_returns_deduplicated_sources(self):
        w = _make_wrapper()
        p1 = MagicMock()
        p1.payload = {"source_url": "https://a.com", "source_title": "A", "authority_level": "official"}
        p2 = MagicMock()
        p2.payload = {"source_url": "https://a.com", "source_title": "A", "authority_level": "official"}  # duplicate
        p3 = MagicMock()
        p3.payload = {"source_url": "https://b.com", "source_title": "B", "authority_level": "third_party"}

        # First scroll returns 2 points; second returns empty (done)
        w.client.scroll = AsyncMock(
            side_effect=[
                ([p1, p2, p3], None),
            ]
        )

        result = await w.get_unique_sources()

        assert len(result) == 2
        urls = [r["url"] for r in result]
        assert "https://a.com" in urls
        assert "https://b.com" in urls

    @pytest.mark.asyncio
    async def test_sorted_official_first(self):
        w = _make_wrapper()
        p_third = MagicMock()
        p_third.payload = {"source_url": "https://c.com", "source_title": "C", "authority_level": "third_party"}
        p_official = MagicMock()
        p_official.payload = {"source_url": "https://d.com", "source_title": "D", "authority_level": "official"}

        w.client.scroll = AsyncMock(return_value=([p_third, p_official], None))

        result = await w.get_unique_sources()

        assert result[0]["authority_level"] == "official"

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_error(self):
        w = _make_wrapper()
        w.client.scroll = AsyncMock(side_effect=RuntimeError("scroll failed"))

        result = await w.get_unique_sources()

        assert result == []

    @pytest.mark.asyncio
    async def test_skips_points_without_url(self):
        w = _make_wrapper()
        p = MagicMock()
        p.payload = {"source_title": "No URL"}  # No source_url

        w.client.scroll = AsyncMock(return_value=([p], None))

        result = await w.get_unique_sources()

        assert result == []

    @pytest.mark.asyncio
    async def test_paginates_multiple_batches(self):
        w = _make_wrapper()
        p1 = MagicMock()
        p1.payload = {"source_url": "https://e.com", "source_title": "E", "authority_level": "official"}
        p2 = MagicMock()
        p2.payload = {"source_url": "https://f.com", "source_title": "F", "authority_level": "official"}

        # First call returns p1 with a next_offset; second call returns p2 with None
        w.client.scroll = AsyncMock(
            side_effect=[
                ([p1], "cursor-1"),
                ([p2], None),
            ]
        )

        result = await w.get_unique_sources()

        assert len(result) == 2


# ─── get_point_by_id ──────────────────────────────────────────────────────────


class TestGetPointById:
    @pytest.mark.asyncio
    async def test_returns_point_dict(self):
        w = _make_wrapper()
        mock_point = MagicMock()
        mock_point.model_dump.return_value = {"id": 42, "payload": {}}
        w.client.retrieve = AsyncMock(return_value=[mock_point])

        result = await w.get_point_by_id(42)

        assert result == {"id": 42, "payload": {}}

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        w = _make_wrapper()
        w.client.retrieve = AsyncMock(return_value=[])

        result = await w.get_point_by_id(99)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        w = _make_wrapper()
        w.client.retrieve = AsyncMock(side_effect=RuntimeError("retrieve failed"))

        result = await w.get_point_by_id(1)

        assert result is None


# ─── delete_by_filter ─────────────────────────────────────────────────────────


class TestDeleteByFilter:
    @pytest.mark.asyncio
    async def test_returns_true_when_completed(self):
        w = _make_wrapper()
        result_mock = MagicMock()
        result_mock.status = "completed"
        w.client.delete = AsyncMock(return_value=result_mock)

        ok = await w.delete_by_filter(MagicMock())

        assert ok is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_completed(self):
        w = _make_wrapper()
        result_mock = MagicMock()
        result_mock.status = "processing"
        w.client.delete = AsyncMock(return_value=result_mock)

        ok = await w.delete_by_filter(MagicMock())

        assert ok is False

    @pytest.mark.asyncio
    async def test_raises_on_error(self):
        w = _make_wrapper()
        w.client.delete = AsyncMock(side_effect=RuntimeError("delete failed"))

        with pytest.raises(RuntimeError):
            await w.delete_by_filter(MagicMock())


# ─── count_points ─────────────────────────────────────────────────────────────


class TestCountPoints:
    @pytest.mark.asyncio
    async def test_returns_count(self):
        w = _make_wrapper()
        col_info = MagicMock()
        col_info.points_count = 42
        w.client.get_collection = AsyncMock(return_value=col_info)

        count = await w.count_points()

        assert count == 42

    @pytest.mark.asyncio
    async def test_returns_zero_on_error(self):
        w = _make_wrapper()
        w.client.get_collection = AsyncMock(side_effect=RuntimeError("down"))

        count = await w.count_points()

        assert count == 0


# ─── health_check ─────────────────────────────────────────────────────────────


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_true_when_ok(self):
        w = _make_wrapper()
        w.client.get_collections = AsyncMock(return_value=MagicMock())

        assert await w.health_check() is True

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self):
        w = _make_wrapper()
        w.client.get_collections = AsyncMock(side_effect=RuntimeError("down"))

        assert await w.health_check() is False


# ─── close ────────────────────────────────────────────────────────────────────


class TestClose:
    @pytest.mark.asyncio
    async def test_closes_client(self):
        w = _make_wrapper()
        w.client.close = AsyncMock()

        await w.close()

        w.client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_swallows_close_error(self):
        w = _make_wrapper()
        w.client.close = AsyncMock(side_effect=RuntimeError("close error"))

        # Should not raise
        await w.close()


# ─── build_filter_authority_and_visa ─────────────────────────────────────────
# (Already tested in test_qdrant_filters.py but we add wrapper-specific tests)


class TestBuildFilterWrapper:
    def test_official_only_returns_field_condition(self):
        w = _make_wrapper()
        result = w.build_filter_authority_and_visa(
            min_authority_level=AuthorityLevel.OFFICIAL,
            visa_types=None,
        )
        assert isinstance(result, FieldCondition)

    def test_with_visa_types_returns_filter(self):
        w = _make_wrapper()
        result = w.build_filter_authority_and_visa(
            min_authority_level=AuthorityLevel.OFFICIAL,
            visa_types=["chancenkarte"],
        )
        assert isinstance(result, Filter)
        assert result.must is not None


# ─── get_qdrant_client singleton ─────────────────────────────────────────────


class TestGetQdrantClient:
    def setup_method(self):
        wrapper_module.qdrant_client = None

    def teardown_method(self):
        wrapper_module.qdrant_client = None

    def test_returns_same_instance(self):
        with patch("src.vector_db.qdrant_client_wrapper.settings") as s:
            s.qdrant_url = "http://localhost:6333"
            s.qdrant_api_key = None
            s.qdrant_collection_name = "col"
            s.qdrant_vector_size = 1536
            s.qdrant_prefer_grpc = False
            with (
                patch("src.vector_db.qdrant_client_wrapper.AsyncQdrantClient"),
                patch("src.vector_db.qdrant_client_wrapper.QdrantClient"),
            ):
                a = get_qdrant_client()
                b = get_qdrant_client()
        assert a is b

    def test_returns_qdrant_wrapper(self):
        with patch("src.vector_db.qdrant_client_wrapper.settings") as s:
            s.qdrant_url = "http://localhost:6333"
            s.qdrant_api_key = None
            s.qdrant_collection_name = "col"
            s.qdrant_vector_size = 1536
            s.qdrant_prefer_grpc = False
            with (
                patch("src.vector_db.qdrant_client_wrapper.AsyncQdrantClient"),
                patch("src.vector_db.qdrant_client_wrapper.QdrantClient"),
            ):
                client = get_qdrant_client()
        assert isinstance(client, QdrantWrapper)
