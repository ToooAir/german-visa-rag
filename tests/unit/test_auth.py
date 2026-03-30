"""Unit tests for src/api/auth.py"""

import pytest
from fastapi import HTTPException

from src.api.auth import APIKeyAuth
from src.config import settings


@pytest.fixture
def auth():
    return APIKeyAuth()


@pytest.mark.asyncio
async def test_valid_key_accepted(auth, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    monkeypatch.setattr(settings, "api_key", "secret-key")
    result = await auth.verify_api_key(x_api_key="secret-key")
    assert result == "secret-key"


@pytest.mark.asyncio
async def test_missing_key_raises_401(auth, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    with pytest.raises(HTTPException) as exc:
        await auth.verify_api_key(x_api_key=None)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_wrong_key_raises_403(auth, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    monkeypatch.setattr(settings, "api_key", "correct-key")
    with pytest.raises(HTTPException) as exc:
        await auth.verify_api_key(x_api_key="wrong-key")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_auth_disabled_allows_any_key(auth, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", False)
    result = await auth.verify_api_key(x_api_key="any-key")
    assert result == "any-key"


@pytest.mark.asyncio
async def test_auth_disabled_allows_no_key(auth, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", False)
    result = await auth.verify_api_key(x_api_key=None)
    assert result == "anonymous"
