"""Unit tests for src/scripts/purge_orphans.py

This script deletes documents from both Qdrant and SQLite, so the cases that
matter are the ones where it must NOT delete.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.scripts.purge_orphans as purge_module
from src.scripts.purge_orphans import _pinned_urls, purge_orphans

PINNED = "https://anabin.kmk.org/anabin.html"
STALE = "https://chancenkarte.com/en/candidates"


def _rows(*urls):
    return [{"id": i, "source_url": u} for i, u in enumerate(urls, 1)]


@pytest.fixture
def wired():
    """Registry that disallows everything, so only the pin guard can save a URL."""
    store = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.fetchall.return_value = _rows(PINNED, STALE)
    store._get_connection.return_value.__enter__.return_value = conn

    qdrant = MagicMock()
    qdrant.delete_by_filter = AsyncMock()

    registry = MagicMock()
    registry.get_strategy.return_value.is_url_allowed.return_value = False

    with (
        patch.object(purge_module, "get_state_store", return_value=store),
        patch.object(purge_module, "get_qdrant_client", return_value=qdrant),
        patch.object(purge_module, "get_strategy_registry", return_value=registry),
        patch.object(purge_module, "load_seed_urls", return_value=[{"url": PINNED}]),
    ):
        yield store, qdrant


class TestPinnedUrls:
    def test_collects_urls_from_the_seed_list(self):
        with patch.object(purge_module, "load_seed_urls", return_value=[{"url": "a"}, {"url": "b"}]):
            assert _pinned_urls() == {"a", "b"}

    def test_tolerates_entries_without_a_url(self):
        with patch.object(purge_module, "load_seed_urls", return_value=[{"title": "no url"}, {"url": "a"}]):
            assert _pinned_urls() == {"a"}


class TestPurgeOrphans:
    @pytest.mark.asyncio
    async def test_pinned_url_is_never_purged(self, wired):
        """A pin exists because discovery cannot reach the page; deleting it would
        only make the next ingest add it back."""
        store, qdrant = wired
        purged = await purge_orphans()

        assert purged == 1  # the stale URL only
        assert qdrant.delete_by_filter.await_count == 1
        deleted_ids = {c.args[0].must[0].match.value for c in qdrant.delete_by_filter.await_args_list}
        assert deleted_ids == {"2"}  # STALE is row 2; PINNED is row 1

    @pytest.mark.asyncio
    async def test_dry_run_counts_without_deleting(self, wired):
        store, qdrant = wired
        purged = await purge_orphans(dry_run=True)

        assert purged == 1
        qdrant.delete_by_filter.assert_not_awaited()
        store.delete_document_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_allowed_url_is_kept(self, wired):
        store, qdrant = wired
        with patch.object(purge_module, "get_strategy_registry") as reg:
            reg.return_value.get_strategy.return_value.is_url_allowed.return_value = True
            assert await purge_orphans() == 0
        qdrant.delete_by_filter.assert_not_awaited()
