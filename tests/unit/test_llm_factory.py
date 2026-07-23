"""Unit tests for src/llm/__init__.py (LLMFactory)."""

from unittest.mock import patch

from src.llm import LLMFactory, get_llm_client
from src.llm.ollama_client import OllamaClient
from src.llm.openai_client import OpenAIClient


def _reset():
    LLMFactory._instance = None


class TestLLMFactory:
    def setup_method(self):
        _reset()

    def teardown_method(self):
        _reset()

    def test_returns_cached_singleton(self):
        with patch("src.llm.settings") as s:
            s.openai_api_key = "sk-real-key"
            s.openai_model = "gpt-4o-mini"
            s.openai_api_base = None
            a = LLMFactory.get_client()
            b = LLMFactory.get_client()
        assert a is b

    def test_creates_openai_client_when_valid_key(self):
        with patch("src.llm.settings") as s:
            s.openai_api_key = "sk-real-key"
            s.openai_model = "gpt-4o-mini"
            s.openai_api_base = None
            client = LLMFactory.get_client()
        assert isinstance(client, OpenAIClient)

    def test_creates_ollama_client_when_use_ollama_true(self):
        with patch("src.llm.settings") as s:
            s.openai_api_key = ""
            s.use_ollama = True
            s.ollama_base_url = "http://localhost:11434"
            s.ollama_model = "llama3"
            client = LLMFactory.get_client()
        assert isinstance(client, OllamaClient)

    def test_placeholder_key_falls_through_to_ollama(self):
        """The default placeholder key 'sk-...your-key-here...' is treated as unset."""
        with patch("src.llm.settings") as s:
            s.openai_api_key = "sk-...your-key-here..."
            s.use_ollama = True
            s.ollama_base_url = "http://localhost:11434"
            s.ollama_model = "llama3"
            client = LLMFactory.get_client()
        assert isinstance(client, OllamaClient)

    def test_defaults_to_openai_when_no_config(self):
        """No valid key, no ollama → still returns an OpenAIClient wrapper.

        The underlying SDK client is mocked so this verifies factory routing
        independently of the installed openai SDK. Newer openai versions raise
        OpenAIError at construction time when the api_key is empty/None, whereas
        this test intentionally exercises the empty-key path — the wrapper must
        still be an OpenAIClient (real credential failure is deferred to call time).
        """
        with (
            patch("src.llm.settings") as s,
            patch("src.llm.openai_client.AsyncOpenAI"),
            patch("src.llm.openai_client.AsyncAzureOpenAI"),
        ):
            s.openai_api_key = ""
            s.use_ollama = False
            s.openai_model = "gpt-4o-mini"
            client = LLMFactory.get_client()
        assert isinstance(client, OpenAIClient)

    def test_placeholder_key_no_ollama_defaults_to_openai(self):
        with patch("src.llm.settings") as s:
            s.openai_api_key = "sk-...your-key-here..."
            s.use_ollama = False
            s.openai_model = "gpt-4o-mini"
            client = LLMFactory.get_client()
        assert isinstance(client, OpenAIClient)


class TestGetLLMClient:
    def setup_method(self):
        _reset()

    def teardown_method(self):
        _reset()

    def test_returns_client(self):
        with patch("src.llm.settings") as s:
            s.openai_api_key = "sk-real-key"
            s.openai_model = "gpt-4o-mini"
            s.openai_api_base = None
            result = get_llm_client()
        assert result is not None
