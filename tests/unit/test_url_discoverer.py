"""Unit tests for URL discoverer, crawl strategy, and link extractor."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from urllib.parse import urlparse

from src.ingestion.crawl_strategy import (
    DomainCrawlStrategy,
    StrategyRegistry,
    MAKE_IT_IN_GERMANY_STRATEGY,
    CHANCENKARTE_COM_STRATEGY,
    get_strategy_registry,
)
from src.ingestion.url_discoverer import (
    LinkExtractor,
    SitemapParser,
    DiscoveryResult,
)


# ============================================
# DomainCrawlStrategy Tests
# ============================================

class TestDomainCrawlStrategy:
    """Test domain crawl strategy filtering and scoring."""

    def test_allowed_url_passes_filter(self):
        strategy = MAKE_IT_IN_GERMANY_STRATEGY
        url = "https://www.make-it-in-germany.com/en/visa-residence/opportunity-card"
        assert strategy.is_url_allowed(url) is True

    def test_blocked_url_rejected(self):
        strategy = MAKE_IT_IN_GERMANY_STRATEGY
        url = "https://www.make-it-in-germany.com/en/newsletter"
        assert strategy.is_url_allowed(url) is False

    def test_wrong_language_rejected(self):
        strategy = MAKE_IT_IN_GERMANY_STRATEGY
        url = "https://www.make-it-in-germany.com/fr/visa-residence/something"
        assert strategy.is_url_allowed(url) is False

    def test_pdf_blocked(self):
        strategy = MAKE_IT_IN_GERMANY_STRATEGY
        url = "https://www.make-it-in-germany.com/en/visa-residence/doc.pdf"
        assert strategy.is_url_allowed(url) is False

    def test_chancenkarte_allowed_path(self):
        strategy = CHANCENKARTE_COM_STRATEGY
        url = "https://chancenkarte.com/en/guides/how-to-apply"
        assert strategy.is_url_allowed(url) is True

    def test_chancenkarte_wp_blocked(self):
        strategy = CHANCENKARTE_COM_STRATEGY
        url = "https://chancenkarte.com/wp-admin/something"
        assert strategy.is_url_allowed(url) is False

    def test_relevance_score_high_for_visa_url(self):
        strategy = MAKE_IT_IN_GERMANY_STRATEGY
        url = "https://www.make-it-in-germany.com/en/visa-residence/chancenkarte-requirements"
        score = strategy.get_relevance_score(url)
        assert score > 0.3

    def test_relevance_score_low_for_generic_url(self):
        strategy = MAKE_IT_IN_GERMANY_STRATEGY
        url = "https://www.make-it-in-germany.com/en/about-us"
        score = strategy.get_relevance_score(url)
        assert score <= 0.3

    def test_no_language_prefix_strategy(self):
        """Strategy with no language prefix should allow all paths."""
        strategy = DomainCrawlStrategy(
            domain="example.com",
            language_prefixes=[],
        )
        url = "https://example.com/any/path"
        assert strategy.is_url_allowed(url) is True

    def test_custom_blocked_pattern(self):
        strategy = DomainCrawlStrategy(
            domain="example.com",
            blocked_path_patterns=[r"/admin/", r"/login"],
            language_prefixes=[],
        )
        assert strategy.is_url_allowed("https://example.com/admin/dashboard") is False
        assert strategy.is_url_allowed("https://example.com/login") is False
        assert strategy.is_url_allowed("https://example.com/public/page") is True


# ============================================
# StrategyRegistry Tests
# ============================================

class TestStrategyRegistry:
    """Test strategy registry lookup."""

    def test_exact_domain_match(self):
        registry = StrategyRegistry()
        strategy = registry.get_strategy("make-it-in-germany.com")
        assert strategy.domain == "make-it-in-germany.com"

    def test_www_prefix_match(self):
        registry = StrategyRegistry()
        strategy = registry.get_strategy("www.make-it-in-germany.com")
        assert strategy.domain == "make-it-in-germany.com"

    def test_url_based_lookup(self):
        registry = StrategyRegistry()
        strategy = registry.get_strategy(
            "https://www.make-it-in-germany.com/en/visa-residence/"
        )
        assert strategy.domain == "make-it-in-germany.com"

    def test_unknown_domain_returns_default(self):
        registry = StrategyRegistry()
        strategy = registry.get_strategy("unknown-domain.com")
        assert strategy.domain == "unknown-domain.com"
        assert strategy.max_depth == 2
        assert strategy.max_pages == 50

    def test_register_custom_strategy(self):
        registry = StrategyRegistry()
        custom = DomainCrawlStrategy(
            domain="custom.de",
            max_depth=5,
        )
        registry.register(custom)
        result = registry.get_strategy("custom.de")
        assert result.max_depth == 5

    def test_get_all_domains(self):
        registry = StrategyRegistry()
        domains = registry.get_all_domains()
        assert "make-it-in-germany.com" in domains
        assert "chancenkarte.com" in domains


# ============================================
# LinkExtractor Tests
# ============================================

class TestLinkExtractor:
    """Test link extraction from HTML."""

    SAMPLE_HTML = """
    <html>
    <body>
        <a href="/en/visa-residence/types/blue-card">Blue Card</a>
        <a href="/en/visa-residence/chancenkarte">Chancenkarte</a>
        <a href="https://external.com/page">External</a>
        <a href="/en/newsletter">Newsletter</a>
        <a href="#section1">Anchor</a>
        <a href="mailto:test@test.com">Email</a>
        <a href="/en/visa-residence/doc.pdf">PDF</a>
        <a href="/fr/visa-residence/page">French</a>
    </body>
    </html>
    """

    def test_extracts_same_domain_links(self):
        links = LinkExtractor.extract_links(
            self.SAMPLE_HTML,
            "https://www.make-it-in-germany.com/en/visa-residence/",
            MAKE_IT_IN_GERMANY_STRATEGY,
        )
        # Should find blue-card and chancenkarte (same domain, allowed paths)
        assert any("blue-card" in url for url in links)
        assert any("chancenkarte" in url for url in links)

    def test_excludes_external_links(self):
        links = LinkExtractor.extract_links(
            self.SAMPLE_HTML,
            "https://www.make-it-in-germany.com/en/",
            MAKE_IT_IN_GERMANY_STRATEGY,
        )
        assert not any("external.com" in url for url in links)

    def test_excludes_blocked_paths(self):
        links = LinkExtractor.extract_links(
            self.SAMPLE_HTML,
            "https://www.make-it-in-germany.com/en/",
            MAKE_IT_IN_GERMANY_STRATEGY,
        )
        assert not any("newsletter" in url for url in links)

    def test_skips_anchors_and_mailto(self):
        links = LinkExtractor.extract_links(
            self.SAMPLE_HTML,
            "https://www.make-it-in-germany.com/en/",
            MAKE_IT_IN_GERMANY_STRATEGY,
        )
        assert not any("mailto" in url for url in links)
        assert not any(url == "#section1" for url in links)

    def test_excludes_pdf_links(self):
        links = LinkExtractor.extract_links(
            self.SAMPLE_HTML,
            "https://www.make-it-in-germany.com/en/",
            MAKE_IT_IN_GERMANY_STRATEGY,
        )
        assert not any(".pdf" in url for url in links)

    def test_excludes_wrong_language(self):
        links = LinkExtractor.extract_links(
            self.SAMPLE_HTML,
            "https://www.make-it-in-germany.com/en/",
            MAKE_IT_IN_GERMANY_STRATEGY,
        )
        assert not any("/fr/" in url for url in links)

    def test_empty_html_returns_empty(self):
        links = LinkExtractor.extract_links(
            "<html></html>",
            "https://example.com/",
            DomainCrawlStrategy(domain="example.com", language_prefixes=[]),
        )
        assert len(links) == 0


# ============================================
# SitemapParser Tests
# ============================================

class TestSitemapParser:
    """Test sitemap XML parsing."""

    SAMPLE_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://www.make-it-in-germany.com/en/visa-residence/opportunity-card</loc></url>
        <url><loc>https://www.make-it-in-germany.com/en/visa-residence/types/blue-card</loc></url>
        <url><loc>https://www.make-it-in-germany.com/en/newsletter</loc></url>
        <url><loc>https://www.make-it-in-germany.com/fr/visa-residence/page</loc></url>
    </urlset>
    """

    SAMPLE_SITEMAP_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <sitemap><loc>https://www.make-it-in-germany.com/sitemap-en.xml</loc></sitemap>
    </sitemapindex>
    """

    @pytest.mark.asyncio
    async def test_parse_regular_sitemap(self):
        """Test parsing a regular sitemap with URL filtering."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = self.SAMPLE_SITEMAP
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        parser = SitemapParser(mock_client)
        urls = await parser._fetch_and_parse_sitemap(
            "https://www.make-it-in-germany.com/sitemap.xml",
            MAKE_IT_IN_GERMANY_STRATEGY,
        )

        # Should include visa-related English URLs, exclude newsletter and French
        assert any("opportunity-card" in u for u in urls)
        assert any("blue-card" in u for u in urls)
        assert not any("newsletter" in u for u in urls)
        assert not any("/fr/" in u for u in urls)
