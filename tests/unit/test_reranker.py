"""Unit tests for src/rag/reranker.py"""

from unittest.mock import AsyncMock, patch

import pytest

from src.rag.reranker import (
    CohereReranker,
    JinaReranker,
    MockReranker,
    RerankerFactory,
    RerankerType,
    get_reranker,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _docs(n: int = 3) -> list[dict]:
    return [{"text": f"doc {i}", "adjusted_score": float(n - i), "metadata": {"chunk_id": str(i)}} for i in range(n)]


# ─── MockReranker ─────────────────────────────────────────────────────────────


class TestMockReranker:
    @pytest.mark.asyncio
    async def test_returns_top_k(self):
        r = MockReranker()
        result = await r.rerank("q", _docs(5), top_k=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        docs = _docs(3)
        r = MockReranker()
        result = await r.rerank("q", docs, top_k=3)
        assert result == docs

    @pytest.mark.asyncio
    async def test_empty_documents(self):
        r = MockReranker()
        result = await r.rerank("q", [], top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_top_k_larger_than_docs(self):
        r = MockReranker()
        result = await r.rerank("q", _docs(2), top_k=10)
        assert len(result) == 2


# ─── CohereReranker ───────────────────────────────────────────────────────────


class TestCohereReranker:
    def _make(self) -> CohereReranker:
        return CohereReranker(api_key="test-key", model="rerank-english-v2.0")

    @pytest.mark.asyncio
    async def test_empty_documents_returns_empty(self):
        r = self._make()
        assert await r.rerank("q", []) == []

    @pytest.mark.asyncio
    async def test_rerank_success(self):
        r = self._make()
        docs = _docs(3)
        api_results = [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.7},
        ]
        r._call_api = AsyncMock(return_value=api_results)

        result = await r.rerank("q", docs, top_k=2)
        assert len(result) == 2
        assert result[0]["rerank_score"] == 0.9
        assert result[0]["adjusted_score"] == 0.9
        assert result[1]["rerank_score"] == 0.7

    @pytest.mark.asyncio
    async def test_api_failure_falls_back_to_score_sort(self):
        r = self._make()
        docs = [
            {"text": "a", "adjusted_score": 0.3},
            {"text": "b", "adjusted_score": 0.9},
            {"text": "c", "adjusted_score": 0.6},
        ]
        r._call_api = AsyncMock(side_effect=RuntimeError("API down"))

        result = await r.rerank("q", docs, top_k=2)
        assert len(result) == 2
        assert result[0]["adjusted_score"] == 0.9
        assert result[1]["adjusted_score"] == 0.6

    @pytest.mark.asyncio
    async def test_uses_content_key_fallback(self):
        """Docs with 'content' key (not 'text') should still work."""
        r = self._make()
        docs = [{"content": "doc text", "adjusted_score": 0.5}]
        r._call_api = AsyncMock(return_value=[{"index": 0, "relevance_score": 0.8}])
        result = await r.rerank("q", docs, top_k=1)
        assert result[0]["rerank_score"] == 0.8

    @pytest.mark.asyncio
    async def test_close(self):
        r = self._make()
        r.client = AsyncMock()
        await r.close()
        r.client.aclose.assert_called_once()


# ─── JinaReranker ─────────────────────────────────────────────────────────────


class TestJinaReranker:
    def _make(self) -> JinaReranker:
        return JinaReranker(api_key="jina-key", model="jina-reranker-v1-base-en")

    @pytest.mark.asyncio
    async def test_empty_documents_returns_empty(self):
        r = self._make()
        assert await r.rerank("q", []) == []

    @pytest.mark.asyncio
    async def test_rerank_success(self):
        r = self._make()
        docs = _docs(2)
        r._call_api = AsyncMock(return_value=[{"index": 1, "relevance_score": 0.95}])

        result = await r.rerank("q", docs, top_k=1)
        assert result[0]["rerank_score"] == 0.95

    @pytest.mark.asyncio
    async def test_api_failure_falls_back_to_score_sort(self):
        r = self._make()
        docs = [
            {"text": "x", "adjusted_score": 0.1},
            {"text": "y", "adjusted_score": 0.8},
        ]
        r._call_api = AsyncMock(side_effect=RuntimeError("Jina down"))
        result = await r.rerank("q", docs, top_k=2)
        assert result[0]["adjusted_score"] == 0.8

    @pytest.mark.asyncio
    async def test_close(self):
        r = self._make()
        r.client = AsyncMock()
        await r.close()
        r.client.aclose.assert_called_once()


# ─── RerankerFactory ──────────────────────────────────────────────────────────


class TestRerankerFactory:
    def setup_method(self):
        RerankerFactory.reset()

    def teardown_method(self):
        RerankerFactory.reset()

    def test_default_mock_reranker(self):
        with patch("src.rag.reranker.settings") as s:
            s.reranker_api_type = "mock"
        r = RerankerFactory.get_reranker()
        assert isinstance(r, MockReranker)

    def test_cohere_with_key(self):
        with patch("src.rag.reranker.settings") as s:
            s.reranker_api_type = RerankerType.COHERE.value
            s.reranker_api_key = "cohere-key"
            s.reranker_model_name = "rerank-english-v2.0"
            r = RerankerFactory.get_reranker()
        assert isinstance(r, CohereReranker)

    def test_cohere_without_key_falls_back_to_mock(self):
        with patch("src.rag.reranker.settings") as s:
            s.reranker_api_type = RerankerType.COHERE.value
            s.reranker_api_key = ""
            r = RerankerFactory.get_reranker()
        assert isinstance(r, MockReranker)

    def test_jina_with_key(self):
        with patch("src.rag.reranker.settings") as s:
            s.reranker_api_type = RerankerType.JINA.value
            s.reranker_api_key = "jina-key"
            s.reranker_model_name = "jina-reranker-v1-base-en"
            r = RerankerFactory.get_reranker()
        assert isinstance(r, JinaReranker)

    def test_jina_without_key_falls_back_to_mock(self):
        with patch("src.rag.reranker.settings") as s:
            s.reranker_api_type = RerankerType.JINA.value
            s.reranker_api_key = ""
            r = RerankerFactory.get_reranker()
        assert isinstance(r, MockReranker)

    def test_unknown_type_uses_mock(self):
        with patch("src.rag.reranker.settings") as s:
            s.reranker_api_type = "unknown_backend"
            r = RerankerFactory.get_reranker()
        assert isinstance(r, MockReranker)

    def test_returns_same_instance_on_second_call(self):
        with patch("src.rag.reranker.settings") as s:
            s.reranker_api_type = "mock"
            a = RerankerFactory.get_reranker()
            b = RerankerFactory.get_reranker()
        assert a is b

    def test_reset_clears_singleton(self):
        with patch("src.rag.reranker.settings") as s:
            s.reranker_api_type = "mock"
            RerankerFactory.get_reranker()
        RerankerFactory.reset()
        assert RerankerFactory._instance is None


# ─── get_reranker ─────────────────────────────────────────────────────────────


class TestGetReranker:
    def setup_method(self):
        RerankerFactory.reset()

    def teardown_method(self):
        RerankerFactory.reset()

    def test_returns_reranker(self):
        with patch("src.rag.reranker.settings") as s:
            s.reranker_api_type = "mock"
            r = get_reranker()
        assert isinstance(r, MockReranker)
