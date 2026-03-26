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
