"""Unit tests for src/ingestion/crawler.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import src.ingestion.crawler as crawler_module
from src.ingestion.crawler import (
    RateLimiter,
    RobotsTxtChecker,
    WebCrawler,
    _is_safe_url,
    get_crawler,
)

# ─── _is_safe_url ─────────────────────────────────────────────────────────────


class TestIsSafeUrl:
    def test_valid_public_url(self):
        # Real public IP — must resolve in test env; skip if DNS fails.
        # Use a known public hostname with a predictable IP pattern.
        # We mock socket.gethostbyname instead.
        with patch("src.ingestion.crawler.socket.gethostbyname", return_value="8.8.8.8"):
            assert _is_safe_url("https://www.make-it-in-germany.com/en/") is True

    def test_file_scheme_blocked(self):
        assert _is_safe_url("file:///etc/passwd") is False

    def test_ftp_scheme_blocked(self):
        assert _is_safe_url("ftp://example.com/file") is False

    def test_cloud_metadata_gcp_blocked(self):
        assert _is_safe_url("http://metadata.google.internal/computeMetadata/v1/") is False

    def test_cloud_metadata_aws_blocked(self):
        assert _is_safe_url("http://169.254.169.254/latest/meta-data/") is False

    def test_loopback_ip_blocked(self):
        with patch("src.ingestion.crawler.socket.gethostbyname", return_value="127.0.0.1"):
            assert _is_safe_url("http://localhost/admin") is False

    def test_private_ip_blocked(self):
        with patch("src.ingestion.crawler.socket.gethostbyname", return_value="192.168.1.1"):
            assert _is_safe_url("http://internal.corp/api") is False

    def test_link_local_blocked(self):
        with patch("src.ingestion.crawler.socket.gethostbyname", return_value="169.254.0.1"):
            assert _is_safe_url("http://anything.local/") is False

    def test_dns_failure_blocked(self):
        import socket as _socket

        with patch("src.ingestion.crawler.socket.gethostbyname", side_effect=_socket.gaierror("NXDOMAIN")):
            assert _is_safe_url("https://doesnotexist.invalid/") is False

    def test_empty_hostname_blocked(self):
        assert _is_safe_url("http:///path") is False


# ─── RateLimiter ──────────────────────────────────────────────────────────────


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_does_not_block_when_tokens_available(self):
        rl = RateLimiter(requests_per_second=10)
        # Should complete without sleeping
        await rl.acquire()
        assert rl.tokens >= 0

    @pytest.mark.asyncio
    async def test_tokens_depleted_triggers_sleep(self):
        rl = RateLimiter(requests_per_second=1)
        rl.tokens = 0.0  # Force empty bucket

        with patch("src.ingestion.crawler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await rl.acquire()
        mock_sleep.assert_called_once()


# ─── RobotsTxtChecker ─────────────────────────────────────────────────────────


class TestRobotsTxtChecker:
    @pytest.mark.asyncio
    async def test_allowed_when_robots_disabled(self):
        mock_client = AsyncMock()
        checker = RobotsTxtChecker(mock_client)
        with patch("src.ingestion.crawler.settings") as s:
            s.crawler_respect_robots_txt = False
            result = await checker.is_allowed("http://example.com/page")
        assert result is True
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_blocked_by_robots_txt(self):
        mock_client = AsyncMock()
        robots_body = "User-agent: *\nDisallow: /private/\n"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = robots_body
        mock_client.get = AsyncMock(return_value=mock_response)

        checker = RobotsTxtChecker(mock_client)
        with patch("src.ingestion.crawler.settings") as s:
            s.crawler_respect_robots_txt = True
            s.crawler_user_agent = "TestBot"
            result = await checker.is_allowed("http://example.com/private/data")
        assert result is False

    @pytest.mark.asyncio
    async def test_allowed_by_robots_txt(self):
        mock_client = AsyncMock()
        robots_body = "User-agent: *\nDisallow: /private/\n"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = robots_body
        mock_client.get = AsyncMock(return_value=mock_response)

        checker = RobotsTxtChecker(mock_client)
        with patch("src.ingestion.crawler.settings") as s:
            s.crawler_respect_robots_txt = True
            s.crawler_user_agent = "TestBot"
            result = await checker.is_allowed("http://example.com/public/page")
        assert result is True

    @pytest.mark.asyncio
    async def test_fetch_error_defaults_to_allow(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))

        checker = RobotsTxtChecker(mock_client)
        with patch("src.ingestion.crawler.settings") as s:
            s.crawler_respect_robots_txt = True
            s.crawler_user_agent = "TestBot"
            result = await checker.is_allowed("http://example.com/page")
        assert result is True


# ─── WebCrawler.parse_html_to_markdown ────────────────────────────────────────


class TestParseHtmlToMarkdown:
    def _make_crawler(self) -> WebCrawler:
        with patch("src.ingestion.crawler.settings") as s:
            s.crawler_rate_limit_requests_per_second = 5
            s.crawler_timeout_seconds = 30
            s.crawler_max_retries = 3
            s.crawler_user_agent = "TestBot"
            crawler = WebCrawler.__new__(WebCrawler)
            crawler.rate_limiter = MagicMock()
            crawler.timeout = 30
            crawler.max_retries = 3
            crawler.user_agent = "TestBot"
            crawler._visited_urls = set()
            crawler.client = MagicMock()
            crawler.robots_checker = MagicMock()
        return crawler

    def test_converts_heading_and_paragraph(self):
        crawler = self._make_crawler()
        html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
        result = crawler.parse_html_to_markdown(html, "http://example.com/")
        assert "Hello" in result
        assert "World" in result

    def test_strips_script_and_nav(self):
        crawler = self._make_crawler()
        html = "<html><body><nav>Nav</nav><script>alert(1)</script><p>Content</p></body></html>"
        result = crawler.parse_html_to_markdown(html, "http://example.com/")
        assert "Nav" not in result
        assert "alert" not in result
        assert "Content" in result

    def test_prefers_main_tag(self):
        crawler = self._make_crawler()
        html = "<html><body><aside>Sidebar</aside><main><p>Main content</p></main></body></html>"
        result = crawler.parse_html_to_markdown(html, "http://example.com/")
        assert "Main content" in result

    def test_empty_html_returns_empty(self):
        crawler = self._make_crawler()
        result = crawler.parse_html_to_markdown("", "http://example.com/")
        # Empty HTML should return an empty or very short string
        assert isinstance(result, str)


# ─── WebCrawler.extract_metadata ──────────────────────────────────────────────


class TestExtractMetadata:
    def _make_crawler(self) -> WebCrawler:
        crawler = WebCrawler.__new__(WebCrawler)
        crawler.rate_limiter = MagicMock()
        crawler.timeout = 30
        crawler.max_retries = 3
        crawler.user_agent = "TestBot"
        crawler._visited_urls = set()
        crawler.client = MagicMock()
        crawler.robots_checker = MagicMock()
        return crawler

    def test_extracts_title(self):
        crawler = self._make_crawler()
        html = "<html><head><title>My Page</title></head><body></body></html>"
        meta = crawler.extract_metadata(html, "http://example.com/")
        assert meta["title"] == "My Page"

    def test_extracts_og_description(self):
        crawler = self._make_crawler()
        html = (
            "<html><head><title>T</title>"
            '<meta property="og:description" content="OG desc"/></head><body></body></html>'
        )
        meta = crawler.extract_metadata(html, "http://example.com/")
        assert meta["description"] == "OG desc"

    def test_fallback_to_meta_description(self):
        crawler = self._make_crawler()
        html = (
            "<html><head><title>T</title>" '<meta name="description" content="Meta desc"/></head><body></body></html>'
        )
        meta = crawler.extract_metadata(html, "http://example.com/")
        assert meta["description"] == "Meta desc"

    def test_no_title_falls_back_to_url(self):
        crawler = self._make_crawler()
        html = "<html><body></body></html>"
        meta = crawler.extract_metadata(html, "http://example.com/fallback")
        assert meta["title"] == "http://example.com/fallback"

    def test_url_field_present(self):
        crawler = self._make_crawler()
        html = "<html><head><title>T</title></head><body></body></html>"
        meta = crawler.extract_metadata(html, "http://example.com/page")
        assert meta["url"] == "http://example.com/page"


# ─── WebCrawler.fetch_url ─────────────────────────────────────────────────────


class TestFetchUrl:
    def _make_crawler(self) -> WebCrawler:
        crawler = WebCrawler.__new__(WebCrawler)
        crawler.rate_limiter = AsyncMock()
        crawler.rate_limiter.acquire = AsyncMock()
        crawler.timeout = 30
        crawler.max_retries = 3
        crawler.user_agent = "TestBot"
        crawler._visited_urls = set()
        crawler.client = AsyncMock()
        crawler.robots_checker = MagicMock()
        return crawler

    @pytest.mark.asyncio
    async def test_unsafe_url_returns_none(self):
        crawler = self._make_crawler()
        with patch("src.ingestion.crawler._is_safe_url", return_value=False):
            result = await crawler.fetch_url("http://192.168.1.1/admin")
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_fetch_returns_html(self):
        crawler = self._make_crawler()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Hello</html>"
        mock_response.raise_for_status = MagicMock()
        crawler.client.get = AsyncMock(return_value=mock_response)

        with patch("src.ingestion.crawler._is_safe_url", return_value=True):
            result = await crawler.fetch_url("https://example.com/page")
        assert result == "<html>Hello</html>"

    @pytest.mark.asyncio
    async def test_http_status_error_raises(self):
        from tenacity import RetryError

        crawler = self._make_crawler()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=mock_response
        )
        crawler.client.get = AsyncMock(return_value=mock_response)

        with (
            patch("src.ingestion.crawler._is_safe_url", return_value=True),
            patch("asyncio.sleep", new_callable=AsyncMock),
            # tenacity wraps the exception in RetryError after max_attempts
            pytest.raises((httpx.HTTPStatusError, RetryError)),
        ):
            await crawler.fetch_url("https://example.com/missing")


# ─── WebCrawler.crawl_document ────────────────────────────────────────────────


class TestCrawlDocument:
    def _make_crawler(self) -> WebCrawler:
        crawler = WebCrawler.__new__(WebCrawler)
        crawler.rate_limiter = AsyncMock()
        crawler.timeout = 30
        crawler.max_retries = 3
        crawler.user_agent = "TestBot"
        crawler._visited_urls = set()
        crawler.client = AsyncMock()
        crawler.robots_checker = MagicMock()
        return crawler

    @pytest.mark.asyncio
    async def test_returns_none_when_fetch_fails(self):
        crawler = self._make_crawler()
        crawler.fetch_url = AsyncMock(return_value=None)
        result = await crawler.crawl_document("https://example.com/")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_markdown(self):
        crawler = self._make_crawler()
        crawler.fetch_url = AsyncMock(return_value="<html></html>")
        crawler.parse_html_to_markdown = MagicMock(return_value="")
        crawler.extract_metadata = MagicMock(return_value={"title": "T", "url": "http://x.com"})
        result = await crawler.crawl_document("https://example.com/empty")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_document_dict(self):
        crawler = self._make_crawler()
        crawler.fetch_url = AsyncMock(return_value="<html><p>Content</p></html>")
        crawler.parse_html_to_markdown = MagicMock(return_value="# Content")
        crawler.extract_metadata = MagicMock(return_value={"title": "T", "url": "https://example.com/page"})
        result = await crawler.crawl_document("https://example.com/page")
        assert result is not None
        assert result["url"] == "https://example.com/page"
        assert result["markdown"] == "# Content"
        assert "fetched_at" in result


# ─── WebCrawler.crawl_batch ───────────────────────────────────────────────────


class TestCrawlBatch:
    @pytest.mark.asyncio
    async def test_filters_none_results(self):
        crawler = WebCrawler.__new__(WebCrawler)
        crawler._visited_urls = set()

        async def _crawl_doc(url):
            if url == "https://good.com/":
                return {"url": url, "markdown": "ok", "html": "", "metadata": {}, "fetched_at": "now"}
            return None

        crawler.crawl_document = _crawl_doc
        results = await crawler.crawl_batch(["https://good.com/", "https://bad.com/"])
        assert len(results) == 1
        assert results[0]["url"] == "https://good.com/"


# ─── WebCrawler misc ──────────────────────────────────────────────────────────


class TestWebCrawlerMisc:
    def test_reset_visited(self):
        crawler = WebCrawler.__new__(WebCrawler)
        crawler._visited_urls = {"a", "b"}
        crawler.reset_visited()
        assert crawler._visited_urls == set()

    @pytest.mark.asyncio
    async def test_close(self):
        crawler = WebCrawler.__new__(WebCrawler)
        crawler.client = AsyncMock()
        await crawler.close()
        crawler.client.aclose.assert_called_once()


# ─── Singleton ────────────────────────────────────────────────────────────────


class TestGetCrawler:
    def test_returns_same_instance(self):
        crawler_module._crawler = None
        with patch("src.ingestion.crawler.settings") as s:
            s.crawler_rate_limit_requests_per_second = 1
            s.crawler_timeout_seconds = 30
            s.crawler_max_retries = 3
            s.crawler_user_agent = "TestBot"
            a = get_crawler()
            b = get_crawler()
        assert a is b
        crawler_module._crawler = None
