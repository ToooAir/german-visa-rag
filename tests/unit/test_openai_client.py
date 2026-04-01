"""Unit tests for src/llm/openai_client.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import OpenAIError

from src.llm.openai_client import OpenAIClient


def _make_client(is_azure: bool = False) -> OpenAIClient:
    """Create OpenAIClient with patched settings and mocked underlying clients."""
    with (
        patch("src.llm.openai_client.settings") as s,
        patch("src.llm.openai_client.AsyncOpenAI") as mock_openai_cls,
        patch("src.llm.openai_client.AsyncAzureOpenAI") as mock_azure_cls,
    ):
        s.use_azure_openai = is_azure
        s.openai_api_base = None
        s.api_timeout_seconds = 30
        s.azure_llm_deployment = "gpt-4o-deploy"
        s.azure_openai_api_key = "az-key"
        s.azure_openai_endpoint = "https://example.openai.azure.com/"
        s.azure_openai_api_version = "2024-12-01-preview"
        mock_openai_cls.return_value = AsyncMock()
        mock_azure_cls.return_value = AsyncMock()
        client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini")
    return client


# ─── Init ─────────────────────────────────────────────────────────────────────


class TestOpenAIClientInit:
    def test_standard_init_creates_asyncopenai(self):
        with (
            patch("src.llm.openai_client.settings") as s,
            patch("src.llm.openai_client.AsyncOpenAI") as mock_cls,
            patch("src.llm.openai_client.AsyncAzureOpenAI"),
        ):
            s.use_azure_openai = False
            s.openai_api_base = None
            s.api_timeout_seconds = 30
            mock_cls.return_value = AsyncMock()
            c = OpenAIClient(api_key="sk-test", model="gpt-4o-mini")
        mock_cls.assert_called_once()
        assert c.model == "gpt-4o-mini"

    def test_azure_init_overrides_model_and_creates_azure_client(self):
        with (
            patch("src.llm.openai_client.settings") as s,
            patch("src.llm.openai_client.AsyncOpenAI"),
            patch("src.llm.openai_client.AsyncAzureOpenAI") as mock_azure_cls,
        ):
            s.use_azure_openai = True
            s.azure_llm_deployment = "my-deploy"
            s.azure_openai_api_key = "az-key"
            s.azure_openai_endpoint = "https://example.openai.azure.com/"
            s.azure_openai_api_version = "2024-12-01-preview"
            s.openai_api_base = None
            s.api_timeout_seconds = 30
            mock_azure_cls.return_value = AsyncMock()
            c = OpenAIClient(api_key="sk-test", model="gpt-4o-mini")
        mock_azure_cls.assert_called_once()
        assert c.model == "my-deploy"

    def test_init_encoding_fallback_on_key_error(self):
        """KeyError from tiktoken.encoding_for_model falls back to cl100k_base."""
        with (
            patch("src.llm.openai_client.settings") as s,
            patch("src.llm.openai_client.AsyncOpenAI"),
            patch("src.llm.openai_client.AsyncAzureOpenAI"),
            patch("src.llm.openai_client.tiktoken") as mock_tiktoken,
        ):
            s.use_azure_openai = False
            s.openai_api_base = None
            s.api_timeout_seconds = 30
            mock_tiktoken.encoding_for_model.side_effect = KeyError("unknown")
            mock_tiktoken.get_encoding.return_value = MagicMock()
            OpenAIClient(api_key="sk-test", model="custom-model")
        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")


# ─── count_tokens / estimate_cost ─────────────────────────────────────────────


class TestCountTokens:
    def test_returns_positive_count(self):
        c = _make_client()
        with patch.object(c.encoding, "encode", return_value=[1, 2, 3]):
            assert c.count_tokens("hello world") == 3

    def test_exception_falls_back_to_approximation(self):
        c = _make_client()
        with patch.object(c.encoding, "encode", side_effect=RuntimeError("err")):
            text = "hello world"
            assert c.count_tokens(text) == len(text) // 4


class TestEstimateCost:
    def test_known_model_returns_nonzero(self):
        c = _make_client()
        cost = c.estimate_cost(1000, 500)
        assert cost > 0

    def test_unknown_model_returns_zero(self):
        c = _make_client()
        c.model = "unknown-xyz"
        assert c.estimate_cost(1000, 500) == 0.0


# ─── call ─────────────────────────────────────────────────────────────────────


class TestOpenAIClientCall:
    @pytest.mark.asyncio
    async def test_call_returns_response_on_success(self):
        c = _make_client()
        mock_response = MagicMock()
        c.client.chat = MagicMock()
        c.client.chat.completions = MagicMock()
        c.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await c.call([{"role": "user", "content": "hi"}])
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_call_raises_openai_error(self):
        from tenacity import RetryError

        c = _make_client()
        err = OpenAIError("api error")
        c.client.chat = MagicMock()
        c.client.chat.completions = MagicMock()
        c.client.chat.completions.create = AsyncMock(side_effect=err)

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises((OpenAIError, RetryError)),
        ):
            await c.call([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_call_raises_unexpected_error(self):
        c = _make_client()
        c.client.chat = MagicMock()
        c.client.chat.completions = MagicMock()
        c.client.chat.completions.create = AsyncMock(side_effect=ValueError("unexpected"))

        with pytest.raises(ValueError, match="unexpected"):
            await c.call([{"role": "user", "content": "hi"}])


# ─── call_non_streaming ───────────────────────────────────────────────────────


class TestCallNonStreaming:
    @pytest.mark.asyncio
    async def test_returns_message_content(self):
        c = _make_client()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "answer text"
        c.call = AsyncMock(return_value=mock_response)

        result = await c.call_non_streaming([{"role": "user", "content": "q"}])
        assert result == "answer text"


# ─── call_streaming ───────────────────────────────────────────────────────────


class TestCallStreaming:
    @pytest.mark.asyncio
    async def test_yields_delta_content(self):
        c = _make_client()

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        async def _fake_stream():
            yield chunk1
            yield chunk2

        c.call = AsyncMock(return_value=_fake_stream())
        chunks = []
        async for text in c.call_streaming([{"role": "user", "content": "hi"}]):
            chunks.append(text)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_skips_chunk_with_no_delta_content(self):
        """Lines 165->164: delta.content is None/falsy → chunk skipped."""
        c = _make_client()

        chunk_no_content = MagicMock()
        chunk_no_content.choices = [MagicMock()]
        chunk_no_content.choices[0].delta.content = None

        chunk_with_content = MagicMock()
        chunk_with_content.choices = [MagicMock()]
        chunk_with_content.choices[0].delta.content = "data"

        async def _fake_stream():
            yield chunk_no_content
            yield chunk_with_content

        c.call = AsyncMock(return_value=_fake_stream())
        chunks = []
        async for text in c.call_streaming([{"role": "user", "content": "hi"}]):
            chunks.append(text)

        assert chunks == ["data"]

    @pytest.mark.asyncio
    async def test_skips_chunk_with_empty_choices(self):
        """Chunk with no choices is skipped."""
        c = _make_client()

        chunk_empty = MagicMock()
        chunk_empty.choices = []

        async def _fake_stream():
            yield chunk_empty

        c.call = AsyncMock(return_value=_fake_stream())
        chunks = []
        async for text in c.call_streaming([{"role": "user", "content": "hi"}]):
            chunks.append(text)

        assert chunks == []
