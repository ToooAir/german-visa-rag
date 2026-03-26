"""Integration tests for API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]


@pytest.mark.asyncio
async def test_chat_completions_endpoint():
    """Test OpenAI-compatible chat completions endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "dev-key-12345"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Chancenkarte 的條件？"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0


@pytest.mark.asyncio
async def test_auth_required():
    """Test API key authentication."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # No API key
        response = await client.get("/v1/health", headers={"X-API-Key": ""})
        # Wait, the health endpoint doesn't actually require an API key by default right? Let me check main.py. If health endpoint DOES require an API key, then this assert works. If not, maybe use /v1/chat/completions without key.
        response = await client.post("/v1/chat/completions", json={"messages": []})
        assert response.status_code == 401
