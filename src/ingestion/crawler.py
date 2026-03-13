"""
Web crawler for fetching and parsing German visa regulation documents.
Includes rate limiting, retry logic, HTML-to-Markdown conversion,
robots.txt compliance, and recursive discovery-based crawling.
"""

from typing import Optional, Dict, Any, List, Set
from datetime import datetime
import asyncio
from urllib.parse import urlparse, urljoin
import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import settings
from src.logger import logger
from src.utils.text_utils import normalize_whitespace


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_second: float):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_update = datetime.utcnow()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make request."""
        async with self.lock:
            now = datetime.utcnow()
            elapsed = (now - self.last_update).total_seconds()
            
            # Refill tokens
            self.tokens = min(
                self.rate,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(sleep_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class RobotsTxtChecker:
    """Check robots.txt rules for crawl permission."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self._cache: Dict[str, List[str]] = {}  # domain -> disallowed paths

    async def is_allowed(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        if not settings.crawler_respect_robots_txt:
            return True

        parsed = urlparse(url)
        domain = parsed.netloc

        if domain not in self._cache:
            await self._fetch_robots(domain)

        disallowed = self._cache.get(domain, [])
        path = parsed.path
        for rule in disallowed:
            if path.startswith(rule):
                logger.debug(f"Blocked by robots.txt: {url}")
                return False
        return True

    async def _fetch_robots(self, domain: str):
        """Fetch and parse robots.txt for a domain."""
        disallowed = []
        try:
            response = await self.client.get(
                f"https://{domain}/robots.txt",
                headers={"User-Agent": settings.crawler_user_agent},
                timeout=10,
            )
            if response.status_code == 200:
                in_our_agent = False
                in_wildcard = False
                for line in response.text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("user-agent:"):
                        agent = line.split(":", 1)[1].strip().lower()
                        in_wildcard = agent == "*"
                        in_our_agent = "german-visa-rag" in agent
                    elif line.lower().startswith("disallow:"):
                        if in_our_agent or in_wildcard:
                            path = line.split(":", 1)[1].strip()
                            if path:
                                disallowed.append(path)
        except Exception as e:
            logger.debug(f"Could not fetch robots.txt for {domain}: {e}")

        self._cache[domain] = disallowed


class WebCrawler:
    """
    Web crawler for fetching visa regulation documents.
    
    Features:
    - Rate limiting (requests/sec)
    - User-Agent rotation
    - Exponential backoff on errors
    - HTML to Markdown conversion
    - Automatic link extraction
    - robots.txt compliance
    - Discovery-based recursive crawling
    """

    def __init__(self):
        self.rate_limiter = RateLimiter(
            settings.crawler_rate_limit_requests_per_second
        )
        self.timeout = settings.crawler_timeout_seconds
        self.max_retries = settings.crawler_max_retries
        self.user_agent = settings.crawler_user_agent
        self._visited_urls: Set[str] = set()
        
        # HTTP client with pooling
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
            ),
        )
        
        # robots.txt checker
        self.robots_checker = RobotsTxtChecker(self.client)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    async def fetch_url(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Fetch URL content with rate limiting and retries.
        
        Args:
            url: URL to fetch
            headers: Optional custom headers
            
        Returns:
            HTML content or None if fetch failed
        """
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        try:
            request_headers = headers or {}
            request_headers["User-Agent"] = self.user_agent
            
            logger.debug(f"Fetching URL: {url}")
            
            response = await self.client.get(url, headers=request_headers)
            response.raise_for_status()
            
            logger.info(f"Successfully fetched {url}", extra={"status_code": response.status_code})
            return response.text
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for {url}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error for {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            raise

    def parse_html_to_markdown(self, html_content: str, url: str) -> str:
        """
        Convert HTML to Markdown with structure preservation.
        
        Args:
            html_content: Raw HTML
            url: Source URL (for logging)
            
        Returns:
            Cleaned Markdown text
        """
        try:
            soup = BeautifulSoup(html_content, "lxml")
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer"]):
                element.decompose()
            
            # Extract main content area (heuristic)
            main_content = soup.find("main") or soup.find("article") or soup.find("body")
            
            if not main_content:
                logger.warning(f"Could not find main content in {url}")
                main_content = soup
            
            # Convert to Markdown
            markdown_text = md(str(main_content), heading_style="atx")
            
            # Clean up
            markdown_text = normalize_whitespace(markdown_text)
            
            logger.debug(f"Converted {url} to Markdown ({len(markdown_text)} chars)")
            return markdown_text
            
        except Exception as e:
            logger.error(f"HTML parsing failed for {url}: {e}")
            return ""

    def extract_metadata(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML (title, og:tags, etc.)."""
        try:
            soup = BeautifulSoup(html_content, "lxml")
            
            # Extract title
            title_tag = soup.find("title")
            title = title_tag.get_text() if title_tag else url
            
            # Extract open graph properties
            og_description = None
            og_tag = soup.find("meta", property="og:description")
            if og_tag:
                og_description = og_tag.get("content")
            
            # Fallback to meta description
            if not og_description:
                desc_tag = soup.find("meta", attrs={"name": "description"})
                og_description = desc_tag.get("content") if desc_tag else None
            
            # Extract published/modified dates
            published_date = None
            date_tag = soup.find("meta", attrs={"property": "article:published_time"})
            if date_tag:
                try:
                    published_date = datetime.fromisoformat(date_tag.get("content"))
                except:
                    pass
            
            return {
                "title": title,
                "description": og_description,
                "published_date": published_date,
                "url": url,
            }
        except Exception as e:
            logger.warning(f"Metadata extraction failed for {url}: {e}")
            return {"title": url, "url": url}

    async def crawl_document(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Crawl a single document.
        
        Returns:
            Document dict with html, markdown, metadata, or None if failed
        """
        try:
            # Fetch HTML
            html_content = await self.fetch_url(url)
            if not html_content:
                return None
            
            # Extract metadata
            metadata = self.extract_metadata(html_content, url)
            
            # Convert to Markdown
            markdown = self.parse_html_to_markdown(html_content, url)
            
            if not markdown:
                logger.warning(f"No markdown content extracted from {url}")
                return None
            
            return {
                "url": url,
                "html": html_content,
                "markdown": markdown,
                "metadata": metadata,
                "fetched_at": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Document crawl failed for {url}: {e}")
            return None

    async def crawl_batch(self, urls: list) -> list:
        """Crawl multiple URLs concurrently with rate limiting."""
        results = await asyncio.gather(
            *[self.crawl_document(url) for url in urls],
            return_exceptions=False,
        )
        return [r for r in results if r is not None]

    async def crawl_with_discovery(
        self,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run URL discovery and then crawl all discovered pages.

        Args:
            force_refresh: If True, bypass discovery cache

        Returns:
            List of crawled document dicts
        """
        from src.ingestion.url_discoverer import get_url_discoverer

        discoverer = get_url_discoverer()
        results = await discoverer.discover_all(force_refresh=force_refresh)

        all_urls = []
        url_metadata = {}  # url -> {authority_level, visa_types}

        for result in results:
            strategy = discoverer.registry.get_strategy(result.domain)
            for url in result.discovered_urls:
                if url not in self._visited_urls:
                    all_urls.append(url)
                    url_metadata[url] = {
                        "authority_level": strategy.authority_level,
                        "visa_types": strategy.default_visa_types,
                    }

        logger.info(
            f"Discovery complete. Crawling {len(all_urls)} URLs across "
            f"{len(results)} domains"
        )

        # Crawl all discovered URLs with metadata enrichment
        crawled_docs = []
        for url in all_urls:
            # Check robots.txt
            if not await self.robots_checker.is_allowed(url):
                continue

            doc = await self.crawl_document(url)
            if doc:
                # Enrich with strategy metadata
                meta = url_metadata.get(url, {})
                doc["authority_level"] = meta.get("authority_level", "third_party")
                doc["visa_types"] = meta.get("visa_types", ["general"])
                crawled_docs.append(doc)
                self._visited_urls.add(url)

        logger.info(f"Crawled {len(crawled_docs)} documents successfully")
        return crawled_docs

    def reset_visited(self):
        """Reset visited URL tracking."""
        self._visited_urls.clear()

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Singleton instance
_crawler = None


def get_crawler() -> WebCrawler:
    """Get or create crawler singleton."""
    global _crawler
    if _crawler is None:
        _crawler = WebCrawler()
    return _crawler
