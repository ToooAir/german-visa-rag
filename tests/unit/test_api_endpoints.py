"""
Unit tests for API endpoints — covers critical HTTP request paths.

Uses a minimal FastAPI test app with mocked dependencies (no real Qdrant/Redis/LLM).
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from src.api.auth import auth
from src.api.endpoints.chat import router as chat_router
from src.api.endpoints.dependencies import get_generator, get_qdrant
from src.api.endpoints.rag import router as rag_router
from src.api.sse import SSEFormatter, create_sse_response

# ─── Helpers ──────────────────────────────────────────────────────────────────


async def _async_gen(*items):
    """Yield items as an async generator."""
    for item in items:
        yield item


def _make_mock_generator(answer: str = "Test answer about Chancenkarte."):
    gen = AsyncMock()
    gen.generate_answer = AsyncMock(return_value={"answer": answer, "sources": [], "metadata": {}})
    gen.generate_answer_streaming = MagicMock(
        return_value=_async_gen(
            'data: {"type":"text","content":"Hello"}\n\n',
            "data: [DONE]\n\n",
        )
    )
    return gen


def _make_test_app(generator=None, qdrant=None, bypass_auth: bool = True) -> FastAPI:
    """
    Minimal FastAPI app with mocked dependencies — no lifespan startup.
    Routers are mounted at their original prefixes.
    """
    app = FastAPI()
    # chat_router already has prefix="/v1/chat"
    app.include_router(chat_router)
    # rag_router already has prefix="/query"
    app.include_router(rag_router)

    _gen = generator or _make_mock_generator()
    if qdrant is None:
        _qdrant = AsyncMock()
        _qdrant.get_unique_sources = AsyncMock(return_value=[])
    else:
        _qdrant = qdrant

    app.dependency_overrides[get_generator] = lambda: _gen
    app.dependency_overrides[get_qdrant] = lambda: _qdrant

    if bypass_auth:
        # Skip auth check; return a fixed key
        app.dependency_overrides[auth.verify_api_key] = lambda: "test-key"

    return app


# ─── chat.py — non-streaming path (lines 57-112) ─────────────────────────────


class TestChatCompletionsNonStreaming:
    @pytest.mark.asyncio
    async def test_returns_200_with_answer(self):
        """Lines 57-112: successful non-streaming completion returns OpenAI format."""
        app = _make_test_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Chancenkarte 條件？"}],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Test answer about Chancenkarte."
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] >= 0

    @pytest.mark.asyncio
    async def test_no_user_message_returns_400(self):
        """Lines 83-87: missing user message raises HTTP 400."""
        app = _make_test_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "system", "content": "You are a helper."}],
                },
            )
        assert resp.status_code == 400
        assert "No user message" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_streaming_flag_returns_streaming_response(self):
        """Lines 66-70: stream=True returns text/event-stream content type."""
        app = _make_test_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "What is Chancenkarte?"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


# ─── chat.py — streaming generator (lines 115-130) ───────────────────────────


class TestGenerateChatStream:
    @pytest.mark.asyncio
    async def test_stream_yields_generator_chunks(self):
        """Lines 128-130: stream passes generator chunks through."""
        from src.api.endpoints.chat import ChatCompletionRequest, ChatMessage, generate_chat_stream

        gen = _make_mock_generator()
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="test query")],
            stream=True,
        )
        chunks = []
        async for chunk in generate_chat_stream(request, gen):
            chunks.append(chunk)

        assert any("Hello" in c or "DONE" in c for c in chunks)
        gen.generate_answer_streaming.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_no_user_message_yields_error(self):
        """Lines 124-126: no user message yields error event and returns."""
        from src.api.endpoints.chat import ChatCompletionRequest, ChatMessage, generate_chat_stream

        gen = _make_mock_generator()
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="system", content="you are a bot")],
            stream=True,
        )
        chunks = []
        async for chunk in generate_chat_stream(request, gen):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "No user message" in chunks[0]
        gen.generate_answer_streaming.assert_not_called()


# ─── rag.py — /query endpoints (lines 49, 59-63, 73-77) ─────────────────────


class TestRagEndpoints:
    @pytest.mark.asyncio
    async def test_get_sources_returns_200(self):
        """Line 49: GET /query/sources calls qdrant.get_unique_sources."""
        mock_qdrant = AsyncMock()
        mock_qdrant.get_unique_sources = AsyncMock(
            return_value=[
                {
                    "title": "BAMF",
                    "url": "https://bamf.de",
                    "authority_level": "official",
                    "last_fetched": None,
                    "visa_types": ["chancenkarte"],
                }
            ]
        )
        app = _make_test_app(qdrant=mock_qdrant)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/query/sources")
        assert resp.status_code == 200
        sources = resp.json()
        assert len(sources) == 1
        assert sources[0]["title"] == "BAMF"

    @pytest.mark.asyncio
    async def test_ask_question_returns_answer(self):
        """Lines 59-63: POST /query/ask calls generator.generate_answer."""
        app = _make_test_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/query/ask",
                json={"query": "What is Chancenkarte?"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Test answer about Chancenkarte."
        assert "sources" in data
        assert "metadata" in data

    @pytest.mark.asyncio
    async def test_ask_stream_returns_event_stream(self):
        """Lines 73-77: POST /query/ask/stream returns SSE response."""
        app = _make_test_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/query/ask/stream",
                json={"query": "What is Chancenkarte?"},
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


# ─── Authentication (auth.py full flow via HTTP) ──────────────────────────────


class TestAuthentication:
    @pytest.mark.asyncio
    async def test_missing_api_key_returns_401(self):
        """Missing X-API-Key header → 401."""
        app = _make_test_app(bypass_auth=False)
        with (patch("src.api.auth.settings") as s,):
            s.require_api_key = True
            s.api_key = "secret-key"
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
        assert resp.status_code == 401
        assert "Missing API key" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_wrong_api_key_returns_403(self):
        """Wrong X-API-Key value → 403."""
        app = _make_test_app(bypass_auth=False)
        with patch("src.api.auth.settings") as s:
            s.require_api_key = True
            s.api_key = "correct-key"
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers={"X-API-Key": "wrong-key"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
        assert resp.status_code == 403
        assert "Invalid API key" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_valid_api_key_passes(self):
        """Correct X-API-Key → request is processed normally."""
        app = _make_test_app(bypass_auth=False)
        with patch("src.api.auth.settings") as s:
            s.require_api_key = True
            s.api_key = "correct-key"
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers={"X-API-Key": "correct-key"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_disabled_allows_any_key(self):
        """require_api_key=False → any key (or no key) allowed."""
        app = _make_test_app(bypass_auth=False)
        with patch("src.api.auth.settings") as s:
            s.require_api_key = False
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
        assert resp.status_code == 200


# ─── dependencies.py — get_generator fallback (lines 8-17) ───────────────────


class TestGetGeneratorFallback:
    def test_fallback_init_when_no_app_state(self):
        """Lines 8-15: fallback init when app.state has no answer_generator."""
        from src.api.endpoints.dependencies import get_generator
        from src.rag.answer_generator import AnswerGenerator

        app = FastAPI()  # Fresh app — no app.state.answer_generator set
        mock_request = MagicMock()
        mock_request.app = app  # Real FastAPI app with real State object

        # Imports are inside the function body, so patch at their source modules
        with (
            patch("src.vector_db.qdrant_client_wrapper.get_qdrant_client") as mock_qdrant_fn,
            patch("src.rag.hybrid_retriever.HybridRetriever") as mock_retriever_cls,
            patch("src.rag.answer_generator.AnswerGenerator") as mock_gen_cls,
        ):
            mock_qdrant_fn.return_value = AsyncMock()
            mock_retriever_cls.return_value = MagicMock()
            mock_gen_cls.return_value = MagicMock(spec=AnswerGenerator)

            result = get_generator(mock_request)

        assert result is not None

    def test_app_state_returned_directly(self):
        """Lines 16-17: existing app.state.answer_generator is returned as-is."""
        from src.api.endpoints.dependencies import get_generator

        app = FastAPI()
        mock_gen = MagicMock()
        app.state.answer_generator = mock_gen  # Pre-set

        mock_request = MagicMock()
        mock_request.app = app

        result = get_generator(mock_request)
        assert result is mock_gen

    def test_get_qdrant_returns_client(self):
        """Lines 22-24: get_qdrant calls get_qdrant_client and returns it."""
        from src.api.endpoints.dependencies import get_qdrant

        mock_request = MagicMock()
        mock_client = MagicMock()
        # get_qdrant_client is imported inside the function body
        with patch("src.vector_db.qdrant_client_wrapper.get_qdrant_client", return_value=mock_client):
            result = get_qdrant(mock_request)
        assert result is mock_client


# ─── rate_limiter.py — in-memory fallback (lines 30-31, 37-38) ───────────────


class TestRateLimiterMemoryFallback:
    def _make_limiter(self, limit: int = 60):
        with patch("src.api.rate_limiter.settings") as s:
            s.enable_rate_limit = True
            s.rate_limit_requests_per_minute = limit
            from src.api.rate_limiter import RateLimiter

            return RateLimiter()

    def test_check_memory_new_window_resets_counter(self):
        """Lines 29-31: expired window → counter resets to 1, no exception."""
        rl = self._make_limiter(limit=60)
        # Inject an expired window (start time far in the past)
        rl._memory["1.2.3.4"] = (999, 0.0)

        rl._check_memory("1.2.3.4")  # Should not raise

        count, _ = rl._memory["1.2.3.4"]
        assert count == 1

    def test_check_memory_over_limit_raises_429(self):
        """Lines 36-45: over limit → logs warning and raises HTTP 429."""
        rl = self._make_limiter(limit=5)
        # Already at limit (count=5, current window)
        rl._memory["1.2.3.4"] = (5, time.monotonic())

        with pytest.raises(HTTPException) as exc_info:
            rl._check_memory("1.2.3.4")

        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["error"] == "rate_limit_exceeded"

    def test_check_memory_within_limit_increments(self):
        """Counter increments normally when within limit."""
        rl = self._make_limiter(limit=60)
        rl._memory["1.2.3.4"] = (5, time.monotonic())

        rl._check_memory("1.2.3.4")  # Should not raise

        count, _ = rl._memory["1.2.3.4"]
        assert count == 6


# ─── sse.py (lines 24, 29, 40-41, 48) ────────────────────────────────────────


class TestSSEFormatter:
    def test_format_event_produces_data_line(self):
        """Line 24: format_event wraps data dict as SSE data line."""
        result = SSEFormatter.format_event("text", {"content": "hello"})
        assert result.startswith("data: ")
        assert '"content": "hello"' in result
        assert result.endswith("\n\n")

    def test_format_done_produces_done_marker(self):
        """Line 29: format_done returns [DONE] SSE line."""
        result = SSEFormatter.format_done()
        assert result == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_to_sse_passes_items_through(self):
        """Lines 40-41: stream_to_sse yields each item from the generator."""
        items = ["chunk1\n\n", "chunk2\n\n", "data: [DONE]\n\n"]
        collected = []
        async for item in SSEFormatter.stream_to_sse(_async_gen(*items)):
            collected.append(item)
        assert collected == items

    def test_create_sse_response_returns_streaming_response(self):
        """Line 48: create_sse_response returns StreamingResponse with correct headers."""
        from fastapi.responses import StreamingResponse

        response = create_sse_response(_async_gen("chunk\n\n"))

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
