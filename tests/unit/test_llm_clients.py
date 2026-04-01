"""Unit tests for src/llm/openai_client.py and src/llm/ollama_client.py"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.ollama_client import OllamaClient
from src.llm.openai_client import OpenAIClient

# ─── OpenAIClient ─────────────────────────────────────────────────────────────


def _make_openai_client(model: str = "gpt-4o-mini") -> OpenAIClient:
    with (
        patch("src.llm.openai_client.settings") as s,
        patch("src.llm.openai_client.AsyncOpenAI"),
    ):
        s.use_azure_openai = False
        s.openai_api_base = None
        s.api_timeout_seconds = 30
        client = OpenAIClient(api_key="test-key", model=model)
    return client


class TestOpenAIClientInit:
    def test_standard_init(self):
        with (
            patch("src.llm.openai_client.settings") as s,
            patch("src.llm.openai_client.AsyncOpenAI") as mock_oai,
        ):
            s.use_azure_openai = False
            s.openai_api_base = None
            s.api_timeout_seconds = 30
            client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini")
        mock_oai.assert_called_once()
        assert client.model == "gpt-4o-mini"

    def test_azure_init(self):
        with (
            patch("src.llm.openai_client.settings") as s,
            patch("src.llm.openai_client.AsyncAzureOpenAI") as mock_azure,
        ):
            s.use_azure_openai = True
            s.azure_openai_api_key = "az-key"
            s.azure_openai_endpoint = "https://example.openai.azure.com/"
            s.azure_openai_api_version = "2024-12-01-preview"
            s.azure_llm_deployment = "gpt-4o-mini-az"
            s.openai_api_base = None
            s.api_timeout_seconds = 30
            client = OpenAIClient(api_key="sk-test")
        mock_azure.assert_called_once()
        assert client.model == "gpt-4o-mini-az"

    def test_unknown_model_uses_fallback_encoding(self):
        with (
            patch("src.llm.openai_client.settings") as s,
            patch("src.llm.openai_client.AsyncOpenAI"),
        ):
            s.use_azure_openai = False
            s.openai_api_base = None
            s.api_timeout_seconds = 30
            # "unknown-model-xyz" doesn't exist in tiktoken
            client = OpenAIClient(api_key="sk-test", model="unknown-model-xyz")
        # encoding should fall back to cl100k_base — count_tokens should work
        count = client.count_tokens("hello world")
        assert isinstance(count, int)
        assert count > 0


class TestOpenAIClientCountTokens:
    def test_counts_tokens(self):
        client = _make_openai_client()
        n = client.count_tokens("Hello, world!")
        assert isinstance(n, int)
        assert n > 0

    def test_empty_string(self):
        client = _make_openai_client()
        assert client.count_tokens("") == 0

    def test_encoding_error_falls_back(self):
        client = _make_openai_client()
        client.encoding = MagicMock()
        client.encoding.encode.side_effect = Exception("encode error")
        # Should not raise; returns len(text)//4
        result = client.count_tokens("a" * 40)
        assert result == 10


class TestOpenAIClientEstimateCost:
    def test_known_model(self):
        client = _make_openai_client("gpt-4o-mini")
        cost = client.estimate_cost(input_tokens=1000, output_tokens=1000)
        assert cost > 0.0

    def test_unknown_model_returns_zero(self):
        client = _make_openai_client("my-custom-model")
        # model not in costs dict → 0+0
        cost = client.estimate_cost(input_tokens=1000, output_tokens=1000)
        assert cost == 0.0

    def test_zero_tokens(self):
        client = _make_openai_client("gpt-4o-mini")
        assert client.estimate_cost(0, 0) == 0.0


class TestOpenAIClientCallNonStreaming:
    @pytest.mark.asyncio
    async def test_returns_content(self):
        client = _make_openai_client()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "The answer is 42."
        client.call = AsyncMock(return_value=mock_response)

        result = await client.call_non_streaming([{"role": "user", "content": "?"}])
        assert result == "The answer is 42."
        client.call.assert_called_once()


class TestOpenAIClientCallStreaming:
    @pytest.mark.asyncio
    async def test_yields_chunks(self):
        client = _make_openai_client()

        # Build fake async iterator of stream chunks
        def _make_chunk(content):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = content
            return chunk

        async def _fake_stream():
            for text in ["Hello", " world", "!"]:
                yield _make_chunk(text)

        client.call = AsyncMock(return_value=_fake_stream())
        chunks = [c async for c in client.call_streaming([{"role": "user", "content": "hi"}])]
        assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_skips_empty_delta(self):
        client = _make_openai_client()

        async def _fake_stream():
            chunk_with_content = MagicMock()
            chunk_with_content.choices = [MagicMock()]
            chunk_with_content.choices[0].delta.content = "hi"
            yield chunk_with_content

            chunk_empty = MagicMock()
            chunk_empty.choices = [MagicMock()]
            chunk_empty.choices[0].delta.content = None
            yield chunk_empty

        client.call = AsyncMock(return_value=_fake_stream())
        chunks = [c async for c in client.call_streaming([])]
        assert chunks == ["hi"]


# ─── OllamaClient ─────────────────────────────────────────────────────────────


class TestOllamaClientInit:
    def test_default_values(self):
        client = OllamaClient()
        assert "11434" in client.base_url
        assert client.model == "mistral"

    def test_strips_trailing_slash(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert not client.base_url.endswith("/")


class TestOllamaClientNonStreaming:
    @pytest.mark.asyncio
    async def test_returns_content(self):
        client = OllamaClient()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Ollama reply"}}

        with patch.object(client.client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.call_non_streaming([{"role": "user", "content": "hi"}])
        assert result == "Ollama reply"

    @pytest.mark.asyncio
    async def test_missing_message_key_returns_empty(self):
        client = OllamaClient()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {}

        with patch.object(client.client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.call_non_streaming([])
        assert result == ""

    @pytest.mark.asyncio
    async def test_http_error_raises(self):
        import httpx

        client = OllamaClient()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )

        with patch.object(client.client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError):
                await client.call_non_streaming([])


class TestOllamaClientStreaming:
    @pytest.mark.asyncio
    async def test_yields_chunks(self):
        client = OllamaClient()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        lines = [
            json.dumps({"message": {"content": "tok1"}}),
            json.dumps({"message": {"content": "tok2"}}),
            json.dumps({"message": {"content": ""}}),  # empty — should be skipped
        ]

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_response.aiter_lines = _aiter_lines

        with patch.object(client.client, "post", new_callable=AsyncMock, return_value=mock_response):
            chunks = [c async for c in client.call_streaming([])]

        assert chunks == ["tok1", "tok2"]

    @pytest.mark.asyncio
    async def test_invalid_json_skipped(self):
        client = OllamaClient()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def _aiter_lines():
            yield "not-json"
            yield json.dumps({"message": {"content": "good"}})

        mock_response.aiter_lines = _aiter_lines

        with patch.object(client.client, "post", new_callable=AsyncMock, return_value=mock_response):
            chunks = [c async for c in client.call_streaming([])]

        assert chunks == ["good"]

    @pytest.mark.asyncio
    async def test_empty_line_skipped(self):
        """Line 80->79: empty string from aiter_lines is falsy and skipped."""
        client = OllamaClient()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def _aiter_lines():
            yield ""  # empty → if line: is False, branch 80->79
            yield json.dumps({"message": {"content": "data"}})

        mock_response.aiter_lines = _aiter_lines
        with patch.object(client.client, "post", new_callable=AsyncMock, return_value=mock_response):
            chunks = [c async for c in client.call_streaming([])]

        assert chunks == ["data"]

    @pytest.mark.asyncio
    async def test_streaming_exception_reraises(self):
        """Lines 89-91: exception during streaming is logged and reraised."""
        client = OllamaClient()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = RuntimeError("stream error")
        with patch.object(client.client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="stream error"):
                _ = [c async for c in client.call_streaming([])]

    @pytest.mark.asyncio
    async def test_close(self):
        client = OllamaClient()
        client.client = AsyncMock()
        await client.close()
        client.client.aclose.assert_called_once()
