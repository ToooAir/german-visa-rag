"""Unit tests for src/rag/answer_generator.py"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.answer_generator import AnswerGenerator

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_generator() -> AnswerGenerator:
    """Build an AnswerGenerator with all heavy dependencies mocked out."""
    retriever = MagicMock()
    with (
        patch("src.rag.answer_generator.get_query_transformer"),
        patch("src.rag.answer_generator.get_reranker"),
        patch("src.rag.answer_generator.get_prompt_builder"),
        patch("src.rag.answer_generator.get_llm_client"),
        patch("src.rag.answer_generator.get_token_counter"),
        patch("src.rag.answer_generator.get_mlflow_tracker"),
    ):
        gen = AnswerGenerator(retriever)

    # Replace all async deps with AsyncMocks
    gen.query_transformer = MagicMock()
    gen.query_transformer.transform_query = AsyncMock(return_value={"corrected_query": "corrected"})
    gen.query_transformer.get_search_queries = AsyncMock(return_value=["q1", "q2"])

    gen.reranker = MagicMock()
    gen.reranker.rerank = AsyncMock(return_value=[])

    gen.prompt_builder = MagicMock()
    gen.prompt_builder.build_context_from_retrieval = MagicMock(return_value="context text")
    gen.prompt_builder.build_system_prompt = MagicMock(return_value="system prompt")
    gen.prompt_builder.build_user_message = MagicMock(return_value={"role": "user", "content": "q"})

    gen.llm = MagicMock()
    gen.llm.call_non_streaming = AsyncMock(return_value="The answer.")

    gen.token_counter = MagicMock()
    gen.token_counter.count_messages = MagicMock(return_value=50)
    gen.token_counter.count_text = MagicMock(return_value=20)
    gen.token_counter.estimate_cost = MagicMock(return_value=0.001)

    gen.mlflow = MagicMock()
    gen.mlflow.log_query_result = MagicMock()

    return gen


def _make_doc(chunk_id: str = "c1", score: float = 0.9) -> dict:
    return {
        "metadata": {
            "chunk_id": chunk_id,
            "source_url": "https://example.com",
            "source_title": "Example",
            "authority_level": "official",
        },
        "adjusted_score": score,
    }


# ─── Static helpers ───────────────────────────────────────────────────────────


class TestFlattenAndDeduplicate:
    def test_deduplicates_by_chunk_id(self):
        doc = _make_doc("c1", 0.9)
        doc2 = _make_doc("c1", 0.5)  # same chunk_id
        result = AnswerGenerator._flatten_and_deduplicate([[doc, doc2]])
        assert len(result) == 1

    def test_sorts_by_score_descending(self):
        d1 = _make_doc("c1", 0.3)
        d2 = _make_doc("c2", 0.9)
        result = AnswerGenerator._flatten_and_deduplicate([[d1, d2]])
        assert result[0]["adjusted_score"] == 0.9

    def test_empty_batches(self):
        result = AnswerGenerator._flatten_and_deduplicate([[], []])
        assert result == []

    def test_multiple_batches_merged(self):
        d1 = _make_doc("c1", 0.8)
        d2 = _make_doc("c2", 0.6)
        result = AnswerGenerator._flatten_and_deduplicate([[d1], [d2]])
        assert len(result) == 2


class TestBuildSources:
    def test_extracts_fields(self):
        reranked = [_make_doc("c1")]
        sources = AnswerGenerator._build_sources(reranked)
        assert sources[0]["url"] == "https://example.com"
        assert sources[0]["title"] == "Example"
        assert sources[0]["authority"] == "official"

    def test_empty_list(self):
        assert AnswerGenerator._build_sources([]) == []


class TestFormatHelpers:
    def test_format_sse_chunk(self):
        out = AnswerGenerator._format_sse_chunk("hello")
        assert out.startswith("data: ")
        parsed = json.loads(out[len("data: ") :].strip())
        assert parsed["choices"][0]["delta"]["content"] == "hello"

    def test_format_milestone_chunk(self):
        out = AnswerGenerator._format_milestone_chunk("1-1", "completed")
        parsed = json.loads(out[len("data: ") :].strip())
        assert parsed["metadata"]["achieved_milestone"]["id"] == "1-1"
        assert parsed["metadata"]["achieved_milestone"]["status"] == "completed"

    def test_format_req_chunk(self):
        out = AnswerGenerator._format_req_chunk("age", "30", "valid")
        parsed = json.loads(out[len("data: ") :].strip())
        req = parsed["metadata"]["updated_requirement"]
        assert req["id"] == "age"
        assert req["value"] == "30"
        assert req["status"] == "valid"

    def test_format_status_chunk(self):
        out = AnswerGenerator._format_status_chunk("retrieving")
        parsed = json.loads(out[len("data: ") :].strip())
        assert parsed["metadata"]["status"] == "retrieving"

    def test_format_search_queries_chunk(self):
        out = AnswerGenerator._format_search_queries_chunk(["q1", "q2"])
        parsed = json.loads(out[len("data: ") :].strip())
        assert parsed["metadata"]["search_queries"] == ["q1", "q2"]


# ─── generate_answer ──────────────────────────────────────────────────────────


class TestGenerateAnswer:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self):
        gen = _make_generator()
        cached = {"answer": "cached!", "sources": [], "metadata": {"cache_hit": False}}

        with patch("src.rag.answer_generator.query_cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value=cached)
            result = await gen.generate_answer("what is Chancenkarte?")

        assert result["answer"] == "cached!"
        assert result["metadata"]["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_no_retrieval_results_returns_fallback(self):
        gen = _make_generator()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[]])

        with patch("src.rag.answer_generator.query_cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            with patch("src.rag.answer_generator.settings") as s:
                s.retrieval_top_k_reranked = 10
                s.retrieval_top_k_hybrid = 20
                s.openai_model = "gpt-4o-mini"
                s.max_response_tokens = 2000
                s.max_response_chars = 50000
                result = await gen.generate_answer("test query")

        assert (
            "couldn't find" in result["answer"]
            or "查閱" in result["answer"]
            or "Ich konnte" in result["answer"]
            or result["sources"] == []
        )

    @pytest.mark.asyncio
    async def test_normal_path_returns_answer(self):
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        with patch("src.rag.answer_generator.query_cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            with patch("src.rag.answer_generator.settings") as s:
                s.retrieval_top_k_reranked = 10
                s.retrieval_top_k_hybrid = 20
                s.openai_model = "gpt-4o-mini"
                s.max_response_tokens = 2000
                s.max_response_chars = 50000
                result = await gen.generate_answer("test query")

        assert result["answer"] == "The answer."
        assert result["metadata"]["cache_hit"] is False
        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_response_truncated_when_exceeds_limit(self):
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])
        gen.llm.call_non_streaming = AsyncMock(return_value="x" * 100)

        with patch("src.rag.answer_generator.query_cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            with patch("src.rag.answer_generator.settings") as s:
                s.retrieval_top_k_reranked = 10
                s.retrieval_top_k_hybrid = 20
                s.openai_model = "gpt-4o-mini"
                s.max_response_tokens = 2000
                s.max_response_chars = 50  # limit at 50 chars
                result = await gen.generate_answer("test query")

        assert len(result["answer"]) == 50

    @pytest.mark.asyncio
    async def test_query_transform_failure_falls_back(self):
        gen = _make_generator()
        gen.query_transformer.transform_query = AsyncMock(side_effect=RuntimeError("transform error"))
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[]])

        with patch("src.rag.answer_generator.query_cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            with patch("src.rag.answer_generator.settings") as s:
                s.retrieval_top_k_reranked = 10
                s.retrieval_top_k_hybrid = 20
                s.openai_model = "gpt-4o-mini"
                s.max_response_tokens = 2000
                s.max_response_chars = 50000
                # Should not crash; returns no-info fallback
                result = await gen.generate_answer("test query")

        assert result["sources"] == []


# ─── generate_answer_streaming ────────────────────────────────────────────────


class TestGenerateAnswerStreaming:
    @pytest.mark.asyncio
    async def test_cache_hit_streams_cached_answer(self):
        gen = _make_generator()
        cached = {
            "answer": "cached answer text",
            "sources": [],
            "milestones": [],
            "requirements": [],
            "metadata": {"query": "q", "retrieval_count": 1, "cache_hit": False},
        }

        with patch("src.rag.answer_generator.query_cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value=cached)
            chunks = [c async for c in gen.generate_answer_streaming("q")]

        full = "".join(chunks)
        assert "DONE" in full
        assert "cached answer text" in full

    @pytest.mark.asyncio
    async def test_no_results_streams_fallback(self):
        gen = _make_generator()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[]])

        with patch("src.rag.answer_generator.query_cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            with patch("src.rag.answer_generator.settings") as s:
                s.retrieval_top_k_reranked = 10
                s.retrieval_top_k_hybrid = 20
                s.openai_model = "gpt-4o-mini"
                s.max_response_tokens = 2000
                s.max_response_chars = 50000
                chunks = [c async for c in gen.generate_answer_streaming("q")]

        full = "".join(chunks)
        assert "DONE" in full

    @pytest.mark.asyncio
    async def test_normal_streaming_emits_done(self):
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        async def _fake_stream(messages, temperature, max_tokens):
            for tok in ["Hello", " world", "!"]:
                yield tok

        gen.llm.call_streaming = _fake_stream

        with patch("src.rag.answer_generator.query_cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            with patch("src.rag.answer_generator.settings") as s:
                s.retrieval_top_k_reranked = 10
                s.retrieval_top_k_hybrid = 20
                s.openai_model = "gpt-4o-mini"
                s.max_response_tokens = 2000
                s.max_response_chars = 50000
                chunks = [c async for c in gen.generate_answer_streaming("test")]

        full = "".join(chunks)
        assert "DONE" in full
        assert "Hello" in full or "world" in full

    @pytest.mark.asyncio
    async def test_streaming_truncates_at_size_limit(self):
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        # Each token is 30 chars → will exceed limit of 50 after second chunk
        async def _fake_stream(messages, temperature, max_tokens):
            for tok in ["A" * 30, "B" * 30, "C" * 30]:
                yield tok

        gen.llm.call_streaming = _fake_stream

        with patch("src.rag.answer_generator.query_cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            with patch("src.rag.answer_generator.settings") as s:
                s.retrieval_top_k_reranked = 10
                s.retrieval_top_k_hybrid = 20
                s.openai_model = "gpt-4o-mini"
                s.max_response_tokens = 2000
                s.max_response_chars = 50  # limit at 50 chars
                chunks = [c async for c in gen.generate_answer_streaming("test")]

        full = "".join(chunks)
        assert "truncated" in full.lower()
        assert "DONE" in full

    @pytest.mark.asyncio
    async def test_milestone_tags_extracted_during_streaming(self):
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        async def _fake_stream(messages, temperature, max_tokens):
            yield "Done! [MILESTONE:1-1:completed] Great."

        gen.llm.call_streaming = _fake_stream

        with patch("src.rag.answer_generator.query_cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            with patch("src.rag.answer_generator.settings") as s:
                s.retrieval_top_k_reranked = 10
                s.retrieval_top_k_hybrid = 20
                s.openai_model = "gpt-4o-mini"
                s.max_response_tokens = 2000
                s.max_response_chars = 50000
                chunks = [c async for c in gen.generate_answer_streaming("test")]

        full = "".join(chunks)
        # The milestone chunk should have been emitted
        assert "achieved_milestone" in full
        assert "1-1" in full
