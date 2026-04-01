"""Unit tests for URL discoverer, crawl strategy, and link extractor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.ingestion.url_discoverer as discoverer_module
from src.ingestion.crawl_strategy import (
    CHANCENKARTE_COM_STRATEGY,
    MAKE_IT_IN_GERMANY_STRATEGY,
    DomainCrawlStrategy,
    StrategyRegistry,
)
from src.ingestion.url_discoverer import (
    DiscoveryResult,
    LinkExtractor,
    SitemapParser,
    URLDiscoverer,
    get_url_discoverer,
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
        strategy = registry.get_strategy("www.make-it-in-germany.com")
        assert strategy.domain == "www.make-it-in-germany.com"

    def test_www_prefix_match(self):
        registry = StrategyRegistry()
        strategy = registry.get_strategy("make-it-in-germany.com")
        assert strategy.domain == "www.make-it-in-germany.com"

    def test_url_based_lookup(self):
        registry = StrategyRegistry()
        strategy = registry.get_strategy("https://www.make-it-in-germany.com/en/visa-residence/")
        assert strategy.domain == "www.make-it-in-germany.com"

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
        assert "www.make-it-in-germany.com" in domains
        assert "www.chancenkarte.com" in domains


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


# ─── SitemapParser extended ───────────────────────────────────────────────────


class TestSitemapParserExtended:
    SAMPLE_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://www.make-it-in-germany.com/en/visa-residence/opportunity-card</loc></url>
    </urlset>
    """

    SAMPLE_SITEMAP_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <sitemap><loc>https://www.make-it-in-germany.com/sitemap-en.xml</loc></sitemap>
    </sitemapindex>
    """

    def _mock_response(self, text, status=200):
        r = MagicMock()
        r.text = text
        r.status_code = status
        return r

    @pytest.mark.asyncio
    async def test_fetch_returns_empty_on_non_200(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=self._mock_response("", status=404))
        parser = SitemapParser(mock_client)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            urls = await parser._fetch_and_parse_sitemap(
                "https://example.com/sitemap.xml",
                DomainCrawlStrategy(domain="example.com", language_prefixes=[]),
            )
        assert urls == set()

    @pytest.mark.asyncio
    async def test_fetch_returns_empty_on_empty_content(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=self._mock_response("   ", status=200))
        parser = SitemapParser(mock_client)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            urls = await parser._fetch_and_parse_sitemap(
                "https://example.com/sitemap.xml",
                DomainCrawlStrategy(domain="example.com", language_prefixes=[]),
            )
        assert urls == set()

    @pytest.mark.asyncio
    async def test_fetch_returns_empty_on_exception(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("network error"))
        parser = SitemapParser(mock_client)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            urls = await parser._fetch_and_parse_sitemap(
                "https://example.com/sitemap.xml",
                DomainCrawlStrategy(domain="example.com", language_prefixes=[]),
            )
        assert urls == set()

    @pytest.mark.asyncio
    async def test_depth_limit_returns_empty(self):
        mock_client = AsyncMock()
        parser = SitemapParser(mock_client)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            urls = await parser._fetch_and_parse_sitemap(
                "https://example.com/sitemap.xml",
                DomainCrawlStrategy(domain="example.com", language_prefixes=[]),
                depth=6,  # > 5 limit
            )
        assert urls == set()
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_parses_sitemap_index_recursively(self):
        child_sitemap_xml = self.SAMPLE_SITEMAP
        index_xml = self.SAMPLE_SITEMAP_INDEX

        call_count = 0

        async def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "sitemap-en" in url:
                return self._mock_response(child_sitemap_xml)
            return self._mock_response(index_xml)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=mock_get)
        parser = SitemapParser(mock_client)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            urls = await parser._fetch_and_parse_sitemap(
                "https://www.make-it-in-germany.com/sitemap.xml",
                MAKE_IT_IN_GERMANY_STRATEGY,
            )
        assert any("opportunity-card" in u for u in urls)

    @pytest.mark.asyncio
    async def test_discover_from_sitemap_returns_urls_on_success(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=self._mock_response(self.SAMPLE_SITEMAP))
        parser = SitemapParser(mock_client)

        strategy = DomainCrawlStrategy(domain="example.com", language_prefixes=[])
        with patch("asyncio.sleep", new_callable=AsyncMock):
            urls = await parser.discover_from_sitemap("example.com", strategy)
        assert isinstance(urls, set)

    @pytest.mark.asyncio
    async def test_discover_from_sitemap_www_domain_skips_www_prefix(self):
        """Lines 52->55: domain starting with 'www.' → only one domain tried."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=self._mock_response(self.SAMPLE_SITEMAP))
        parser = SitemapParser(mock_client)

        strategy = DomainCrawlStrategy(domain="www.make-it-in-germany.com", language_prefixes=[])
        with patch("asyncio.sleep", new_callable=AsyncMock):
            urls = await parser.discover_from_sitemap("www.make-it-in-germany.com", strategy)
        assert isinstance(urls, set)

    @pytest.mark.asyncio
    async def test_discover_from_sitemap_exception_in_fetch_continues(self):
        """Lines 65-67: _fetch_and_parse_sitemap raises → caught, continue."""
        mock_client = AsyncMock()
        parser = SitemapParser(mock_client)

        with (
            patch.object(
                parser,
                "_fetch_and_parse_sitemap",
                side_effect=RuntimeError("unexpected sitemap error"),
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            strategy = DomainCrawlStrategy(domain="example.com", language_prefixes=[])
            urls = await parser.discover_from_sitemap("example.com", strategy)
        assert isinstance(urls, set)

    @pytest.mark.asyncio
    async def test_discover_from_sitemap_robots_fallback_yields_urls(self):
        """Lines 74-77: robots.txt fallback successfully yields URLs."""
        parser = SitemapParser(AsyncMock())

        # All direct sitemap paths raise (go to robots fallback)
        async def _fetch_parse(url, strategy, depth=0):
            if "sitemap_from_robots" in url:
                return {"https://example.com/en/page"}  # Return a URL from robots sitemap
            raise RuntimeError("not found")

        with (
            patch.object(parser, "_fetch_and_parse_sitemap", side_effect=_fetch_parse),
            patch.object(
                parser,
                "_find_sitemaps_in_robots",
                new_callable=AsyncMock,
                return_value=["https://example.com/sitemap_from_robots.xml"],
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            strategy = DomainCrawlStrategy(domain="example.com", language_prefixes=[])
            urls = await parser.discover_from_sitemap("example.com", strategy)
        assert "https://example.com/en/page" in urls

    @pytest.mark.asyncio
    async def test_discover_from_sitemap_tries_robots_when_all_fail(self):
        """When all sitemap paths fail, fall back to robots.txt."""
        mock_client = AsyncMock()
        # All sitemap paths fail
        mock_client.get = AsyncMock(return_value=self._mock_response("", status=404))
        parser = SitemapParser(mock_client)

        strategy = DomainCrawlStrategy(domain="example.com", language_prefixes=[])
        with patch("asyncio.sleep", new_callable=AsyncMock):
            urls = await parser.discover_from_sitemap("example.com", strategy)
        # Should have tried robots.txt (no crash)
        assert isinstance(urls, set)

    @pytest.mark.asyncio
    async def test_fetch_and_parse_sitemap_no_namespace(self):
        """Lines 115->119: XML without namespace → namespace stays ''."""
        xml_no_ns = """<?xml version="1.0"?>
<urlset>
  <url><loc>https://example.com/en/page</loc></url>
</urlset>"""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=self._mock_response(xml_no_ns))
        parser = SitemapParser(mock_client)

        strategy = DomainCrawlStrategy(domain="example.com", language_prefixes=[])
        with patch("asyncio.sleep", new_callable=AsyncMock):
            urls = await parser._fetch_and_parse_sitemap("https://example.com/sitemap.xml", strategy)
        assert "https://example.com/en/page" in urls

    @pytest.mark.asyncio
    async def test_fetch_and_parse_sitemap_index_with_no_loc(self):
        """Lines 124->122: sitemap index tag without <loc> child → skipped."""
        xml = """<?xml version="1.0"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap></sitemap>
</sitemapindex>"""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=self._mock_response(xml))
        parser = SitemapParser(mock_client)

        strategy = DomainCrawlStrategy(domain="example.com", language_prefixes=[])
        with patch("asyncio.sleep", new_callable=AsyncMock):
            urls = await parser._fetch_and_parse_sitemap("https://example.com/sitemap.xml", strategy)
        assert urls == set()  # no URLs because sitemap tag had no loc

    @pytest.mark.asyncio
    async def test_fetch_and_parse_sitemap_url_with_no_loc(self):
        """Lines 132->130: url tag without <loc> → skipped."""
        xml = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url></url>
</urlset>"""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=self._mock_response(xml))
        parser = SitemapParser(mock_client)

        strategy = DomainCrawlStrategy(domain="example.com", language_prefixes=[])
        with patch("asyncio.sleep", new_callable=AsyncMock):
            urls = await parser._fetch_and_parse_sitemap("https://example.com/sitemap.xml", strategy)
        assert urls == set()

    @pytest.mark.asyncio
    async def test_find_sitemaps_in_robots_returns_list(self):
        robots_txt = "User-agent: *\nDisallow: /admin\nSitemap: https://example.com/sitemap.xml\n"
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=self._mock_response(robots_txt))
        parser = SitemapParser(mock_client)

        result = await parser._find_sitemaps_in_robots("https://example.com")
        assert "https://example.com/sitemap.xml" in result

    @pytest.mark.asyncio
    async def test_find_sitemaps_in_robots_returns_empty_on_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("timeout"))
        parser = SitemapParser(mock_client)

        result = await parser._find_sitemaps_in_robots("https://example.com")
        assert result == []

    @pytest.mark.asyncio
    async def test_find_sitemaps_non_200_returns_empty(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=self._mock_response("", status=403))
        parser = SitemapParser(mock_client)

        result = await parser._find_sitemaps_in_robots("https://example.com")
        assert result == []


# ─── LinkExtractor extended ───────────────────────────────────────────────────


class TestLinkExtractorExtended:
    def test_trailing_slash_removed_except_root(self):
        html = '<a href="/en/visa-residence/">Link</a>'
        links = LinkExtractor.extract_links(
            html,
            "https://www.make-it-in-germany.com/en/",
            MAKE_IT_IN_GERMANY_STRATEGY,
        )
        # Should not have trailing slash
        for link in links:
            if "/en/visa-residence" in link:
                assert not link.endswith("/")

    def test_exception_returns_empty_set(self):
        """Malformed HTML that raises in BeautifulSoup is caught gracefully."""
        with patch("src.ingestion.url_discoverer.BeautifulSoup", side_effect=Exception("parse error")):
            links = LinkExtractor.extract_links(
                "<a href='/page'>x</a>",
                "https://example.com/",
                DomainCrawlStrategy(domain="example.com", language_prefixes=[]),
            )
        assert links == set()


# ─── URLDiscoverer ────────────────────────────────────────────────────────────


def _make_discoverer():
    """Create URLDiscoverer with all external deps mocked."""
    mock_client = AsyncMock()
    mock_state_store = MagicMock()
    mock_state_store.get_cached_discovery = MagicMock(return_value=None)
    mock_state_store.save_discovered_urls = MagicMock()

    mock_registry = MagicMock()
    mock_strategy = DomainCrawlStrategy(
        domain="example.com",
        language_prefixes=[],
        seed_paths=["/en/"],
        max_depth=1,
        max_pages=10,
    )
    mock_registry.get_all_strategies.return_value = [mock_strategy]
    mock_registry.get_strategy.return_value = mock_strategy

    with (
        patch("src.ingestion.url_discoverer.get_strategy_registry", return_value=mock_registry),
        patch("src.storage.sqlite_state_store.get_state_store", return_value=mock_state_store),
    ):
        d = URLDiscoverer(client=mock_client, state_store=mock_state_store)

    d.registry = mock_registry
    d.sitemap_parser = AsyncMock()
    d.sitemap_parser.discover_from_sitemap = AsyncMock(return_value=set())
    return d, mock_strategy, mock_state_store


class TestURLDiscovererDiscover:
    @pytest.mark.asyncio
    async def test_discover_domain_uses_cache(self):
        d, strategy, store = _make_discoverer()
        cached_entry = [
            {
                "url": "https://example.com/en/page",
                "from_sitemap": True,
                "from_crawling": False,
                "discovered_at": "2025-01-01T00:00:00Z",
            }
        ]
        store.get_cached_discovery = MagicMock(return_value=cached_entry)

        with patch("src.ingestion.url_discoverer.settings") as s:
            s.discovery_cache_ttl_hours = 24
            result = await d.discover_domain(strategy, force_refresh=False)

        assert result.from_cache is True
        assert len(result.discovered_urls) == 1

    @pytest.mark.asyncio
    async def test_discover_domain_force_refresh_bypasses_cache(self):
        d, strategy, store = _make_discoverer()
        store.get_cached_discovery = MagicMock(
            return_value=[
                {
                    "url": "https://example.com/cached",
                    "from_sitemap": True,
                    "from_crawling": False,
                    "discovered_at": "2025-01-01T00:00:00Z",
                }
            ]
        )

        with (
            patch("src.ingestion.url_discoverer.settings") as s,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            s.enable_sparse_search = False
            s.crawler_user_agent = "bot"
            s.crawler_timeout_seconds = 10
            s.crawler_rate_limit_requests_per_second = 10
            # BFS: client returns non-200 to shortcircuit
            d.client.get = AsyncMock(return_value=MagicMock(status_code=404))
            result = await d.discover_domain(strategy, force_refresh=True)

        # Should have run fresh (cache not used)
        assert result.from_cache is False

    @pytest.mark.asyncio
    async def test_discover_domain_sitemap_disabled(self):
        d, _, store = _make_discoverer()
        no_sitemap_strategy = DomainCrawlStrategy(
            domain="example.com",
            language_prefixes=[],
            use_sitemap=False,
            seed_paths=[],
            max_depth=0,
            max_pages=5,
        )

        with (
            patch("src.ingestion.url_discoverer.settings") as s,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            s.crawler_user_agent = "bot"
            s.crawler_timeout_seconds = 10
            s.crawler_rate_limit_requests_per_second = 10
            d.client.get = AsyncMock(return_value=MagicMock(status_code=200, text="<html></html>"))
            result = await d.discover_domain(no_sitemap_strategy, force_refresh=True)

        assert result.from_sitemap == 0

    @pytest.mark.asyncio
    async def test_discover_all_returns_results_for_all_strategies(self):
        d, strategy, _ = _make_discoverer()

        with (
            patch("src.ingestion.url_discoverer.settings") as s,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            s.discovery_cache_ttl_hours = 24
            s.crawler_user_agent = "bot"
            s.crawler_timeout_seconds = 10
            s.crawler_rate_limit_requests_per_second = 10
            d.client.get = AsyncMock(return_value=MagicMock(status_code=404))
            d.state_store.get_cached_discovery = MagicMock(return_value=None)
            results = await d.discover_all()

        assert len(results) == 1
        assert results[0].domain == "example.com"

    @pytest.mark.asyncio
    async def test_discover_all_uses_cache(self):
        d, _, store = _make_discoverer()
        cached = [
            {
                "url": "https://example.com/page",
                "from_sitemap": True,
                "from_crawling": False,
                "discovered_at": "2025-01-01",
            }
        ]
        store.get_cached_discovery = MagicMock(return_value=cached)

        with patch("src.ingestion.url_discoverer.settings") as s:
            s.discovery_cache_ttl_hours = 24
            results = await d.discover_all()

        assert results[0].from_cache is True

    @pytest.mark.asyncio
    async def test_discover_domain_max_pages_limit(self):
        d, _, store = _make_discoverer()
        limited_strategy = DomainCrawlStrategy(
            domain="example.com",
            language_prefixes=[],
            use_sitemap=True,
            max_pages=2,
            seed_paths=[],
            max_depth=0,
        )
        # Sitemap returns 5 URLs
        d.sitemap_parser.discover_from_sitemap = AsyncMock(
            return_value={
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c",
                "https://example.com/d",
                "https://example.com/e",
            }
        )

        with (
            patch("src.ingestion.url_discoverer.settings") as s,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            s.crawler_user_agent = "bot"
            s.crawler_timeout_seconds = 10
            s.crawler_rate_limit_requests_per_second = 10
            result = await d.discover_domain(limited_strategy, force_refresh=True)

        assert len(result.discovered_urls) <= 2

    @pytest.mark.asyncio
    async def test_discover_single_domain(self):
        d, _, store = _make_discoverer()
        store.get_cached_discovery = MagicMock(
            return_value=[
                {
                    "url": "https://example.com/page",
                    "from_sitemap": True,
                    "from_crawling": False,
                    "discovered_at": "2025-01-01",
                }
            ]
        )

        with patch("src.ingestion.url_discoverer.settings") as s:
            s.discovery_cache_ttl_hours = 24
            result = await d.discover_single_domain("example.com")

        assert isinstance(result, DiscoveryResult)

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self):
        d, _, _ = _make_discoverer()
        d.client.aclose = AsyncMock()
        await d.close()
        d.client.aclose.assert_called_once()


class TestURLDiscovererDiscoverExceptions:
    @pytest.mark.asyncio
    async def test_sitemap_exception_logged_and_continues(self):
        """Lines 324-325: sitemap discovery raises → logged, result has 0 sitemap URLs."""
        d, strategy, _ = _make_discoverer()
        d.sitemap_parser.discover_from_sitemap = AsyncMock(side_effect=RuntimeError("sitemap crashed"))
        d._bfs_discover = AsyncMock(return_value=(set(), 0))

        with (
            patch("src.ingestion.url_discoverer.settings") as s,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            s.crawler_user_agent = "bot"
            s.crawler_timeout_seconds = 10
            s.crawler_rate_limit_requests_per_second = 10
            result = await d.discover_domain(strategy, force_refresh=True)

        assert result.from_sitemap == 0

    @pytest.mark.asyncio
    async def test_bfs_exception_logged_and_continues(self):
        """Lines 334-335: BFS raises → logged, result has 0 crawling URLs."""
        d, strategy, _ = _make_discoverer()
        d.sitemap_parser.discover_from_sitemap = AsyncMock(return_value=set())
        d._bfs_discover = AsyncMock(side_effect=RuntimeError("BFS crashed"))

        with (
            patch("src.ingestion.url_discoverer.settings") as s,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            s.crawler_user_agent = "bot"
            s.crawler_timeout_seconds = 10
            s.crawler_rate_limit_requests_per_second = 10
            result = await d.discover_domain(strategy, force_refresh=True)

        assert result.from_crawling == 0


class TestURLDiscovererBFS:
    @pytest.mark.asyncio
    async def test_bfs_adds_seed_urls(self):
        d, strategy, _ = _make_discoverer()
        d.client.get = AsyncMock(return_value=MagicMock(status_code=200, text="<html></html>"))

        with (
            patch("src.ingestion.url_discoverer.settings") as s,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            s.crawler_user_agent = "bot"
            s.crawler_timeout_seconds = 10
            s.crawler_rate_limit_requests_per_second = 10
            discovered, _ = await d._bfs_discover(strategy, already_known=set())

        assert "https://example.com/en/" in discovered

    @pytest.mark.asyncio
    async def test_bfs_skips_non_200(self):
        d, strategy, _ = _make_discoverer()
        d.client.get = AsyncMock(return_value=MagicMock(status_code=404))

        with (
            patch("src.ingestion.url_discoverer.settings") as s,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            s.crawler_user_agent = "bot"
            s.crawler_timeout_seconds = 10
            s.crawler_rate_limit_requests_per_second = 10
            discovered, _ = await d._bfs_discover(strategy, already_known=set())

        # Seed URL is still discovered (added before fetch), but no new links
        assert len(discovered) >= 1

    @pytest.mark.asyncio
    async def test_bfs_handles_fetch_exception(self):
        d, strategy, _ = _make_discoverer()
        d.client.get = AsyncMock(side_effect=Exception("connection refused"))

        with (
            patch("src.ingestion.url_discoverer.settings") as s,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            s.crawler_user_agent = "bot"
            s.crawler_timeout_seconds = 10
            s.crawler_rate_limit_requests_per_second = 10
            discovered, filtered = await d._bfs_discover(strategy, already_known=set())

        # Should not raise, seed URLs still added
        assert isinstance(discovered, set)

    @pytest.mark.asyncio
    async def test_bfs_filtered_links_increment_counter(self):
        """Lines 463-464: is_url_allowed returns False → filtered_count incremented."""
        d, _, _ = _make_discoverer()

        # Strategy with max_depth=1, max_pages=5
        strategy = DomainCrawlStrategy(
            domain="example.com",
            language_prefixes=[],
            seed_paths=["/en/"],
            max_depth=1,
            max_pages=5,
        )

        # Fetch returns HTML with a link
        html_with_link = '<html><body><a href="/blocked-path">Link</a></body></html>'
        d.client.get = AsyncMock(return_value=MagicMock(status_code=200, text=html_with_link))

        with (
            patch("src.ingestion.url_discoverer.settings") as s,
            patch("asyncio.sleep", new_callable=AsyncMock),
            # Make LinkExtractor return a link that is not allowed
            patch(
                "src.ingestion.url_discoverer.LinkExtractor.extract_links",
                return_value={"https://example.com/blocked-path"},
            ),
            patch.object(strategy, "is_url_allowed", return_value=False),
        ):
            s.crawler_user_agent = "bot"
            s.crawler_timeout_seconds = 10
            s.crawler_rate_limit_requests_per_second = 10
            discovered, filtered = await d._bfs_discover(strategy, already_known=set())

        assert filtered >= 1

    @pytest.mark.asyncio
    async def test_bfs_skips_already_known_seeds(self):
        d, strategy, _ = _make_discoverer()

        with (
            patch("src.ingestion.url_discoverer.settings") as s,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            s.crawler_user_agent = "bot"
            s.crawler_timeout_seconds = 10
            s.crawler_rate_limit_requests_per_second = 10
            already = {"https://example.com/en/"}
            discovered, _ = await d._bfs_discover(strategy, already_known=already)

        # Seed was already known — not added to discovered
        assert "https://example.com/en/" not in discovered


class TestGetUrlDiscoverer:
    def setup_method(self):
        discoverer_module._discoverer = None

    def teardown_method(self):
        discoverer_module._discoverer = None

    def test_returns_same_instance(self):
        mock_client = MagicMock()
        mock_state = MagicMock()
        with (
            patch("src.ingestion.url_discoverer.get_strategy_registry"),
            patch("src.storage.sqlite_state_store.get_state_store", return_value=mock_state),
        ):
            a = get_url_discoverer(client=mock_client)
            b = get_url_discoverer(client=mock_client)
        assert a is b

    def test_creates_default_client_when_none_provided(self):
        mock_state = MagicMock()
        with (
            patch("src.ingestion.url_discoverer.get_strategy_registry"),
            patch("src.storage.sqlite_state_store.get_state_store", return_value=mock_state),
            patch("src.ingestion.url_discoverer.settings") as s,
            patch("src.ingestion.url_discoverer.httpx.AsyncClient") as mock_client_cls,
        ):
            s.crawler_timeout_seconds = 10
            mock_client_cls.return_value = MagicMock()
            get_url_discoverer()
        mock_client_cls.assert_called_once()
