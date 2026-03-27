"""Pytest fixtures and configuration."""

from typing import AsyncGenerator

import pytest
from httpx import AsyncClient

from src.config import settings
from src.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Provide an async client for FastAPI testing."""
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def mock_api_key():
    return settings.api_key


@pytest.fixture
def mock_llm_client(monkeypatch):
    """Mock the LLM client to avoid real API calls in unit tests."""
    from unittest.mock import AsyncMock

    mock = AsyncMock()
    # Default response for QueryTransformer tests
    mock.call_non_streaming.return_value = """
    {
      "corrected_query": "chancenkarte applications",
      "english_query": "chancenkarte applications",
      "german_query": "chancenkarte anträge",
      "query_variants": ["chancenkarte requirements", "opportunity card process"],
      "detected_visa_types": ["chancenkarte"],
      "languages_detected": ["en"],
      "confidence": 0.95
    }
    """
    monkeypatch.setattr("src.rag.query_transformer.get_llm_client", lambda: mock)
    return mock
