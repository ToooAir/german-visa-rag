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


def _mock_settings():
    """Return a settings mock with sensible defaults."""
    s = MagicMock()
    s.retrieval_top_k_reranked = 10
    s.retrieval_top_k_hybrid = 20
    s.openai_model = "gpt-4o-mini"
    s.max_response_tokens = 2000
    s.max_response_chars = 50000
    return s


class TestGenerateAnswerParams:
    """Verify temperature, max_tokens, and top_k are correctly forwarded."""

    @pytest.mark.asyncio
    async def test_custom_temperature_passed_to_llm(self):
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            await gen.generate_answer("test", temperature=0.9)

        call_kwargs = gen.llm.call_non_streaming.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_default_temperature_is_0_3(self):
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            await gen.generate_answer("test")

        call_kwargs = gen.llm.call_non_streaming.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_custom_max_tokens_passed_to_llm(self):
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            await gen.generate_answer("test", max_tokens=512)

        call_kwargs = gen.llm.call_non_streaming.call_args.kwargs
        assert call_kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_max_tokens_none_falls_back_to_settings(self):
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])
        s = _mock_settings()
        s.max_response_tokens = 1500

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", s),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            await gen.generate_answer("test", max_tokens=None)

        call_kwargs = gen.llm.call_non_streaming.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1500

    @pytest.mark.asyncio
    async def test_top_k_passed_to_retriever(self):
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            await gen.generate_answer("test", top_k=5)

        call_kwargs = gen.retriever.retrieve_batch.call_args.kwargs
        assert call_kwargs["top_k"] == 5

    @pytest.mark.asyncio
    async def test_top_k_none_falls_back_to_settings(self):
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])
        s = _mock_settings()
        s.retrieval_top_k_hybrid = 20

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", s),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            await gen.generate_answer("test", top_k=None)

        call_kwargs = gen.retriever.retrieve_batch.call_args.kwargs
        assert call_kwargs["top_k"] == 20


class TestGenerateAnswerAdditional:
    @pytest.mark.asyncio
    async def test_visa_type_fallback_when_no_results_with_filter(self):
        """Lines 158-160: retry retrieve without visa_type when filtered returns nothing."""
        gen = _make_generator()
        doc = _make_doc()
        # First call (with filter) → empty; second call (no filter) → doc
        gen.retriever.retrieve_batch = AsyncMock(side_effect=[[[]], [[doc]]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            result = await gen.generate_answer("test", visa_type="chancenkarte")

        assert result["answer"] == "The answer."
        assert gen.retriever.retrieve_batch.call_count == 2

    @pytest.mark.asyncio
    async def test_mlflow_logged_on_normal_path(self):
        """Lines 225-236: mlflow.log_query_result called on success."""
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            await gen.generate_answer("test")

        gen.mlflow.log_query_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_mlflow_does_not_crash(self):
        """Line 225->238: mlflow=None path in generate_answer."""
        gen = _make_generator()
        gen.mlflow = None
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            result = await gen.generate_answer("test")

        assert result["answer"] == "The answer."

    @pytest.mark.asyncio
    async def test_generate_answer_raises_llm_generation_error(self):
        """Lines 240-242: exception path re-raises as LLMGenerationError."""
        from src.rag.answer_generator import LLMGenerationError

        gen = _make_generator()
        gen.query_transformer.transform_query = AsyncMock(side_effect=Exception("boom"))
        gen.retriever.retrieve_batch = AsyncMock(side_effect=RuntimeError("retrieval failed"))

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            with pytest.raises(LLMGenerationError):
                await gen.generate_answer("test")


class TestGenerateAnswerStreamingAdditional:
    @pytest.mark.asyncio
    async def test_cache_hit_with_milestone_and_req_tags_in_answer(self):
        """Lines 272, 274, 278, 280: milestone/req parsed from cached answer text."""
        gen = _make_generator()
        cached = {
            "answer": "Answer [MILESTONE:1-1:completed] done [REQ:123:30:valid]",
            "sources": [],
            "milestones": [{"id": "2-1", "status": "in_progress"}],
            "requirements": [{"id": "lang", "value": "B1", "status": "valid"}],
            "metadata": {"query": "q", "retrieval_count": 1, "cache_hit": False},
        }

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_cache.get = AsyncMock(return_value=cached)
            chunks = [c async for c in gen.generate_answer_streaming("q")]

        full = "".join(chunks)
        assert "achieved_milestone" in full
        assert "1-1" in full
        assert "updated_requirement" in full
        assert "2-1" in full  # from milestones list

    @pytest.mark.asyncio
    async def test_streaming_visa_type_fallback(self):
        """Lines 311-313: streaming retries retrieve without visa_type filter."""
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(side_effect=[[[]], [[doc]]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        async def _fake_stream(messages, temperature, max_tokens):
            yield "answer"

        gen.llm.call_streaming = _fake_stream

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            chunks = [c async for c in gen.generate_answer_streaming("q", visa_type="chancenkarte")]

        full = "".join(chunks)
        assert "DONE" in full
        assert gen.retriever.retrieve_batch.call_count == 2

    @pytest.mark.asyncio
    async def test_req_tags_extracted_during_streaming(self):
        """Lines 377-379: [REQ:...] tags emitted as req chunks."""
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        async def _fake_stream(messages, temperature, max_tokens):
            yield "Result [REQ:123:30:valid] done"

        gen.llm.call_streaming = _fake_stream

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            chunks = [c async for c in gen.generate_answer_streaming("q")]

        full = "".join(chunks)
        assert "updated_requirement" in full
        assert "123" in full

    @pytest.mark.asyncio
    async def test_partial_tag_buffer_held_until_complete(self):
        """Lines 385-395: tag buffer split on '[' waits for complete tag."""
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        async def _fake_stream(messages, temperature, max_tokens):
            # Send text with partial bracket (not a complete tag)
            yield "text [incomplete"
            yield "text without bracket"

        gen.llm.call_streaming = _fake_stream

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            chunks = [c async for c in gen.generate_answer_streaming("q")]

        full = "".join(chunks)
        assert "DONE" in full

    @pytest.mark.asyncio
    async def test_streaming_llm_error_emits_interruption(self):
        """Lines 426-428: LLM streaming error yields interruption message."""
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        async def _erroring_stream(messages, temperature, max_tokens):
            yield "Hello"
            raise RuntimeError("LLM crashed")

        gen.llm.call_streaming = _erroring_stream

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            chunks = [c async for c in gen.generate_answer_streaming("q")]

        full = "".join(chunks)
        assert "interrupted" in full.lower() or "Generation interrupted" in full

    @pytest.mark.asyncio
    async def test_streaming_mlflow_logged(self):
        """Lines 445-454: mlflow logged after streaming completes."""
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        async def _fake_stream(messages, temperature, max_tokens):
            yield "answer text"

        gen.llm.call_streaming = _fake_stream

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            async for _ in gen.generate_answer_streaming("q"):
                pass

        gen.mlflow.log_query_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_buffer_with_remaining_milestone_tag(self):
        """Lines 415-416, 418-419: flush remaining buffer containing milestone/req tags."""
        gen = _make_generator()
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        # Yield text that starts with "[" so buffer holds it; end with complete tag
        async def _fake_stream(messages, temperature, max_tokens):
            yield "[MILESTONE:3-1:completed][REQ:456:yes:valid]final text"

        gen.llm.call_streaming = _fake_stream

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            chunks = [c async for c in gen.generate_answer_streaming("q")]

        full = "".join(chunks)
        assert "DONE" in full

    @pytest.mark.asyncio
    async def test_streaming_no_mlflow_does_not_crash(self):
        """Lines 445->456: mlflow=None skips mlflow logging in streaming."""
        gen = _make_generator()
        gen.mlflow = None
        doc = _make_doc()
        gen.retriever.retrieve_batch = AsyncMock(return_value=[[doc]])
        gen.reranker.rerank = AsyncMock(return_value=[doc])

        async def _fake_stream(messages, temperature, max_tokens):
            yield "answer text"

        gen.llm.call_streaming = _fake_stream

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            chunks = [c async for c in gen.generate_answer_streaming("q")]

        full = "".join(chunks)
        assert "DONE" in full

    @pytest.mark.asyncio
    async def test_streaming_outer_exception_yields_error_chunk(self):
        """Lines 474-477: outer exception in streaming yields error chunk and DONE."""
        gen = _make_generator()
        gen.retriever.retrieve_batch = AsyncMock(side_effect=RuntimeError("retrieval crashed"))

        with (
            patch("src.rag.answer_generator.query_cache") as mock_cache,
            patch("src.rag.answer_generator.settings", _mock_settings()),
        ):
            mock_cache.get = AsyncMock(return_value=None)
            chunks = [c async for c in gen.generate_answer_streaming("q")]

        full = "".join(chunks)
        assert "DONE" in full
        assert "Error" in full or "error" in full
