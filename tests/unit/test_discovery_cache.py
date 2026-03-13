"""Unit tests for URL discovery cache in SQLite state store."""

import pytest
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from src.storage.sqlite_state_store import SQLiteStateStore
from src.ingestion.url_discoverer import URLDiscoverer, DiscoveryResult
from src.ingestion.crawl_strategy import DomainCrawlStrategy


# ============================================
# SQLite Discovery Cache Tests
# ============================================

class TestDiscoveryCacheStore:
    """Test SQLite-backed discovery URL cache."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a fresh SQLite state store in a temp directory."""
        db_path = tmp_path / "test_state.db"
        return SQLiteStateStore(db_path=db_path)

    @pytest.fixture
    def sample_urls(self):
        return [
            {
                "url": "https://example.com/page1",
                "authority_level": "official",
                "visa_types": ["general"],
                "relevance_score": 0.8,
                "from_sitemap": True,
                "from_crawling": False,
            },
            {
                "url": "https://example.com/page2",
                "authority_level": "official",
                "visa_types": ["chancenkarte"],
                "relevance_score": 0.6,
                "from_sitemap": False,
                "from_crawling": True,
            },
        ]

    def test_save_and_retrieve_cached_urls(self, store, sample_urls):
        """Round-trip: save URLs then retrieve them."""
        store.save_discovered_urls("example.com", sample_urls)
        cached = store.get_cached_discovery("example.com", max_age_hours=1)

        assert cached is not None
        assert len(cached) == 2
        # Should be ordered by relevance_score DESC
        assert cached[0]["url"] == "https://example.com/page1"
        assert cached[0]["authority_level"] == "official"
        assert cached[0]["visa_types"] == ["general"]
        assert cached[0]["relevance_score"] == 0.8
        assert cached[0]["from_sitemap"] is True

    def test_cache_returns_none_when_empty(self, store):
        """No cached data returns None."""
        result = store.get_cached_discovery("nonexistent.com", max_age_hours=24)
        assert result is None

    def test_cache_ttl_expiry(self, store, sample_urls):
        """Cache should expire after max_age_hours."""
        store.save_discovered_urls("example.com", sample_urls)

        # Manually backdate the discovered_at timestamp
        with store._get_connection() as conn:
            cursor = conn.cursor()
            old_time = (datetime.utcnow() - timedelta(hours=25)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            cursor.execute(
                "UPDATE discovered_urls SET discovered_at = ?", (old_time,)
            )
            conn.commit()

        # Should be stale now
        result = store.get_cached_discovery("example.com", max_age_hours=24)
        assert result is None

    def test_cache_not_expired(self, store, sample_urls):
        """Cache within TTL should still be returned."""
        store.save_discovered_urls("example.com", sample_urls)
        result = store.get_cached_discovery("example.com", max_age_hours=24)
        assert result is not None
        assert len(result) == 2

    def test_save_replaces_old_entries(self, store, sample_urls):
        """Re-saving for same domain replaces old entries."""
        store.save_discovered_urls("example.com", sample_urls)

        new_urls = [
            {
                "url": "https://example.com/new-page",
                "authority_level": "official",
                "visa_types": ["general"],
                "relevance_score": 0.9,
                "from_sitemap": True,
                "from_crawling": False,
            }
        ]
        store.save_discovered_urls("example.com", new_urls)

        cached = store.get_cached_discovery("example.com", max_age_hours=1)
        assert len(cached) == 1
        assert cached[0]["url"] == "https://example.com/new-page"

    def test_clear_cache_single_domain(self, store, sample_urls):
        """Clear cache for a single domain."""
        store.save_discovered_urls("example.com", sample_urls)
        store.save_discovered_urls("other.com", sample_urls)

        store.clear_discovery_cache(domain="example.com")

        assert store.get_cached_discovery("example.com", max_age_hours=1) is None
        assert store.get_cached_discovery("other.com", max_age_hours=1) is not None

    def test_clear_cache_all(self, store, sample_urls):
        """Clear all cached discovery data."""
        store.save_discovered_urls("example.com", sample_urls)
        store.save_discovered_urls("other.com", sample_urls)

        store.clear_discovery_cache()

        assert store.get_cached_discovery("example.com", max_age_hours=1) is None
        assert store.get_cached_discovery("other.com", max_age_hours=1) is None

    def test_stats_include_cached_domains(self, store, sample_urls):
        """Stats should include count of cached discovery domains."""
        store.save_discovered_urls("example.com", sample_urls)
        store.save_discovered_urls("other.com", sample_urls)

        stats = store.get_stats()
        assert stats["cached_discovery_domains"] == 2


# ============================================
# URLDiscoverer Cache Integration Tests
# ============================================

class TestURLDiscovererCache:
    """Test that URLDiscoverer uses cache correctly."""

    @pytest.fixture
    def mock_strategy(self):
        return DomainCrawlStrategy(
            domain="test.com",
            language_prefixes=[],
            max_pages=50,
        )

    @pytest.mark.asyncio
    async def test_uses_cache_when_fresh(self, mock_strategy):
        """discover_domain should return cached result without BFS/sitemap."""
        cached_result = DiscoveryResult(
            domain="test.com",
            discovered_urls=["https://test.com/page1"],
            from_sitemap=1,
            from_crawling=0,
            from_cache=True,
        )

        discoverer = URLDiscoverer()
        discoverer._get_cached_result = MagicMock(return_value=cached_result)
        discoverer.sitemap_parser.discover_from_sitemap = AsyncMock()

        result = await discoverer.discover_domain(mock_strategy)

        assert result.from_cache is True
        assert len(result.discovered_urls) == 1
        assert result.discovered_urls[0] == "https://test.com/page1"
        # Sitemap should NOT have been called
        discoverer.sitemap_parser.discover_from_sitemap.assert_not_called()

    @pytest.mark.asyncio
    async def test_bypasses_cache_on_force_refresh(self, mock_strategy):
        """force_refresh=True should skip cache and run full discovery."""
        discoverer = URLDiscoverer()
        discoverer._get_cached_result = MagicMock(return_value=None)
        discoverer._save_to_cache = MagicMock()
        discoverer.sitemap_parser.discover_from_sitemap = AsyncMock(
            return_value={"https://test.com/fresh"}
        )
        discoverer._bfs_discover = AsyncMock(return_value=(set(), 0))

        result = await discoverer.discover_domain(
            mock_strategy, force_refresh=True
        )

        assert result.from_cache is False
        # _get_cached_result should NOT have been called
        discoverer._get_cached_result.assert_not_called()
        # Sitemap SHOULD have been called
        discoverer.sitemap_parser.discover_from_sitemap.assert_called_once()
        # Should save to cache after discovery
        discoverer._save_to_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_runs_discovery_on_cache_miss(self, mock_strategy):
        """When cache returns None, full discovery should run."""
        discoverer = URLDiscoverer()
        discoverer._get_cached_result = MagicMock(return_value=None)
        discoverer._save_to_cache = MagicMock()
        discoverer.sitemap_parser.discover_from_sitemap = AsyncMock(
            return_value={"https://test.com/new-page"}
        )
        discoverer._bfs_discover = AsyncMock(return_value=(set(), 0))

        result = await discoverer.discover_domain(mock_strategy)

        assert result.from_cache is False
        assert "https://test.com/new-page" in result.discovered_urls
        # Should save to cache after discovery
        discoverer._save_to_cache.assert_called_once()

