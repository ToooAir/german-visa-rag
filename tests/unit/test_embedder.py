"""Unit tests for src/vector_db/embedder.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.vector_db.embedder import (
    EmbedderFactory,
    OllamaEmbedder,
    OpenAIEmbedder,
    QuotaExhaustedError,
)

# ─── QuotaExhaustedError ──────────────────────────────────────────────────────


class TestQuotaExhaustedError:
    def test_basic_attributes(self):
        err = QuotaExhaustedError(wait_seconds=30, message="rate limit hit")
        assert err.wait_seconds == 30
        assert "30s" in str(err)

    def test_with_reset_fields(self):
        err = QuotaExhaustedError(
            wait_seconds=0,
            reset_requests="10s",
            reset_tokens="5s",
        )
        assert err.reset_requests == "10s"
        assert err.reset_tokens == "5s"

    def test_no_wait_message(self):
        err = QuotaExhaustedError()
        assert err.wait_seconds == 0
        assert "quota exhausted" in str(err).lower()


# ─── OpenAIEmbedder ───────────────────────────────────────────────────────────


def _make_openai_embedder() -> OpenAIEmbedder:
    with patch("src.vector_db.embedder.settings") as s:
        s.openai_api_base = None
        s.api_timeout_seconds = 30
        e = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-small")
    return e


class TestOpenAIEmbedderParseWaitSeconds:
    def test_parses_seconds(self):
        assert OpenAIEmbedder._parse_wait_seconds("Please wait 45 seconds") == 45

    def test_no_match_returns_zero(self):
        assert OpenAIEmbedder._parse_wait_seconds("error occurred") == 0

    def test_case_insensitive(self):
        assert OpenAIEmbedder._parse_wait_seconds("Wait 10 SECONDS") == 10


class TestOpenAIEmbedderEmbedTexts:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self):
        e = _make_openai_embedder()
        assert await e.embed_texts([]) == []

    @pytest.mark.asyncio
    async def test_returns_embeddings(self):
        e = _make_openai_embedder()
        mock_client = AsyncMock()
        emb_data = MagicMock()
        emb_data.index = 0
        emb_data.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [emb_data]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        e.client = mock_client

        result = await e.embed_texts(["hello world"])
        assert result == [[0.1, 0.2, 0.3]]

    @pytest.mark.asyncio
    async def test_rate_limit_error_raises_quota_exhausted(self):
        import openai as oai

        e = _make_openai_embedder()
        mock_client = AsyncMock()
        err = oai.RateLimitError(
            "rate limit",
            response=MagicMock(headers={}),
            body={},
        )
        mock_client.embeddings.create = AsyncMock(side_effect=err)
        e.client = mock_client

        with pytest.raises(QuotaExhaustedError):
            await e.embed_texts(["test"])

    @pytest.mark.asyncio
    async def test_embed_single_returns_first_embedding(self):
        e = _make_openai_embedder()
        e.embed_texts = AsyncMock(return_value=[[0.5, 0.6]])
        result = await e.embed_single("single text")
        assert result == [0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_single_empty_returns_empty(self):
        e = _make_openai_embedder()
        e.embed_texts = AsyncMock(return_value=[])
        result = await e.embed_single("x")
        assert result == []

    @pytest.mark.asyncio
    async def test_lazy_init_standard(self):
        e = _make_openai_embedder()
        assert e.client is None
        with patch("src.vector_db.embedder.openai.AsyncOpenAI") as mock_cls:
            mock_instance = AsyncMock()
            mock_cls.return_value = mock_instance
            with patch("src.vector_db.embedder.settings") as s:
                s.openai_api_base = None
                s.api_timeout_seconds = 30
                client = await e._get_client()
        assert client is not None

    @pytest.mark.asyncio
    async def test_lazy_init_azure(self):
        with patch("src.vector_db.embedder.settings") as s:
            s.openai_api_base = None
            s.api_timeout_seconds = 30
            e = OpenAIEmbedder(
                api_key="az-key",
                is_azure=True,
                azure_endpoint="https://example.openai.azure.com/",
                azure_api_version="2024-12-01-preview",
                azure_deployment="embed-model",
            )
        with patch("src.vector_db.embedder.openai.AsyncAzureOpenAI") as mock_cls:
            mock_cls.return_value = AsyncMock()
            client = await e._get_client()
        assert client is not None


# ─── OllamaEmbedder ───────────────────────────────────────────────────────────


class TestOllamaEmbedder:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self):
        e = OllamaEmbedder()
        assert await e.embed_texts([]) == []

    @pytest.mark.asyncio
    async def test_returns_embeddings(self):
        e = OllamaEmbedder()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2]}

        with patch.object(e.client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await e.embed_texts(["hello"])
        assert result == [[0.1, 0.2]]

    @pytest.mark.asyncio
    async def test_strips_trailing_slash(self):
        e = OllamaEmbedder(base_url="http://localhost:11434/")
        assert not e.base_url.endswith("/")

    @pytest.mark.asyncio
    async def test_embed_single(self):
        e = OllamaEmbedder()
        e.embed_texts = AsyncMock(return_value=[[0.9, 0.8]])
        result = await e.embed_single("text")
        assert result == [0.9, 0.8]

    @pytest.mark.asyncio
    async def test_close(self):
        e = OllamaEmbedder()
        e.client = AsyncMock()
        await e.close()
        e.client.aclose.assert_called_once()


# ─── Embedder (unified) ───────────────────────────────────────────────────────


def _make_embedder():
    """Create Embedder with mocked settings, bypassing real API calls."""
    from src.vector_db.embedder import Embedder

    with patch("src.vector_db.embedder.settings") as s:
        s.use_azure_openai = False
        s.openai_api_key = "sk-test"
        s.embedding_model = "text-embedding-3-small"
        s.openai_api_base = None
        s.use_ollama = False
        e = Embedder()
    return e


class TestEmbedderPreflight:
    @pytest.mark.asyncio
    async def test_preflight_success_returns_true(self):
        e = _make_embedder()
        e.primary = AsyncMock()
        e.primary.embed_texts = AsyncMock(return_value=[[0.1]])
        assert await e.preflight_check() is True

    @pytest.mark.asyncio
    async def test_preflight_failure_returns_false(self):
        e = _make_embedder()
        e.primary = AsyncMock()
        e.primary.embed_texts = AsyncMock(side_effect=RuntimeError("API down"))
        assert await e.preflight_check() is False

    @pytest.mark.asyncio
    async def test_preflight_quota_error_reraises(self):
        e = _make_embedder()
        e.primary = AsyncMock()
        e.primary.embed_texts = AsyncMock(side_effect=QuotaExhaustedError(wait_seconds=60))
        with pytest.raises(QuotaExhaustedError):
            await e.preflight_check()


class TestEmbedderEmbedTexts:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self):
        e = _make_embedder()
        assert await e.embed_texts([]) == []

    @pytest.mark.asyncio
    async def test_returns_primary_embeddings(self):
        e = _make_embedder()
        e.primary = AsyncMock()
        e.primary.embed_texts = AsyncMock(return_value=[[0.1, 0.2]])
        result = await e.embed_texts(["hello"])
        assert result == [[0.1, 0.2]]

    @pytest.mark.asyncio
    async def test_fallback_used_on_primary_failure(self):
        e = _make_embedder()
        e.primary = AsyncMock()
        e.primary.embed_texts = AsyncMock(side_effect=RuntimeError("primary down"))
        e.fallback = AsyncMock()
        e.fallback.embed_texts = AsyncMock(return_value=[[0.9]])
        result = await e.embed_texts(["hello"])
        assert result == [[0.9]]

    @pytest.mark.asyncio
    async def test_no_fallback_raises_on_primary_failure(self):
        e = _make_embedder()
        e.primary = AsyncMock()
        e.primary.embed_texts = AsyncMock(side_effect=RuntimeError("primary down"))
        e.fallback = None
        with pytest.raises(RuntimeError):
            await e.embed_texts(["hello"])

    @pytest.mark.asyncio
    async def test_quota_exhausted_raises_after_max_retries(self):
        e = _make_embedder()
        e.primary = AsyncMock()
        e.primary.embed_texts = AsyncMock(side_effect=QuotaExhaustedError(wait_seconds=1))
        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(QuotaExhaustedError),
        ):
            await e.embed_texts(["hello"], _quota_retry=3)  # already at max

    @pytest.mark.asyncio
    async def test_embed_single(self):
        e = _make_embedder()
        e.embed_texts = AsyncMock(return_value=[[0.3, 0.4]])
        result = await e.embed_single("text")
        assert result == [0.3, 0.4]


# ─── EmbedderFactory ──────────────────────────────────────────────────────────


class TestEmbedderFactory:
    def setup_method(self):
        EmbedderFactory.reset()

    def teardown_method(self):
        EmbedderFactory.reset()

    def test_returns_same_instance(self):
        with patch("src.vector_db.embedder.settings") as s:
            s.use_azure_openai = False
            s.openai_api_key = "sk-test"
            s.embedding_model = "text-embedding-3-small"
            s.openai_api_base = None
            s.use_ollama = False
            a = EmbedderFactory.get_embedder()
            b = EmbedderFactory.get_embedder()
        assert a is b

    def test_reset_clears_singleton(self):
        with patch("src.vector_db.embedder.settings") as s:
            s.use_azure_openai = False
            s.openai_api_key = "sk-test"
            s.embedding_model = "text-embedding-3-small"
            s.openai_api_base = None
            s.use_ollama = False
            EmbedderFactory.get_embedder()
        EmbedderFactory.reset()
        assert EmbedderFactory._instance is None
