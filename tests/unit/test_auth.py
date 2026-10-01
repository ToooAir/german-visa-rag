"""Unit tests for src/api/auth.py"""

from unittest.mock import patch

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


# ─── Admin key ────────────────────────────────────────────────────────────────


class TestVerifyAdminKey:
    """/admin/* can rebuild the corpus and ingest an arbitrary URL, so this
    dependency fails closed. Production had shipped with require_api_key unset,
    which left every admin endpoint answering with no key at all."""

    @pytest.mark.asyncio
    async def test_accepts_the_configured_key(self):
        with patch("src.api.auth.settings") as s:
            s.admin_api_key = "admin-secret"
            assert await APIKeyAuth.verify_admin_key(x_admin_key="admin-secret") == "admin-secret"

    @pytest.mark.asyncio
    async def test_missing_key_is_rejected(self):
        with patch("src.api.auth.settings") as s:
            s.admin_api_key = "admin-secret"
            with pytest.raises(HTTPException) as e:
                await APIKeyAuth.verify_admin_key(x_admin_key=None)
        assert e.value.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_key_is_rejected(self):
        with patch("src.api.auth.settings") as s:
            s.admin_api_key = "admin-secret"
            with pytest.raises(HTTPException) as e:
                await APIKeyAuth.verify_admin_key(x_admin_key="guess")
        assert e.value.status_code == 403

    @pytest.mark.asyncio
    async def test_unconfigured_fails_closed(self):
        """A missing setting must not mean "allow everyone" -- that was the bug."""
        for value in (None, ""):
            with patch("src.api.auth.settings") as s:
                s.admin_api_key = value
                with pytest.raises(HTTPException) as e:
                    await APIKeyAuth.verify_admin_key(x_admin_key="anything")
            assert e.value.status_code == 503

    @pytest.mark.asyncio
    async def test_require_api_key_cannot_switch_it_off(self):
        """require_api_key disables verification for every other endpoint at once.
        The admin gate deliberately ignores it."""
        with patch("src.api.auth.settings") as s:
            s.require_api_key = False
            s.admin_api_key = "admin-secret"
            with pytest.raises(HTTPException) as e:
                await APIKeyAuth.verify_admin_key(x_admin_key=None)
        assert e.value.status_code == 401

    @pytest.mark.asyncio
    async def test_the_public_api_key_does_not_open_the_admin_gate(self):
        """api_key reaches the browser bundle, so it must never satisfy this."""
        with patch("src.api.auth.settings") as s:
            s.api_key = "frontend-key"
            s.admin_api_key = "admin-secret"
            with pytest.raises(HTTPException) as e:
                await APIKeyAuth.verify_admin_key(x_admin_key="frontend-key")
        assert e.value.status_code == 403
