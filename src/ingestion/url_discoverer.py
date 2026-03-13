"""
Automatic URL discovery engine.
Discovers relevant pages via sitemap parsing, page link extraction,
and keyword-based filtering.
Supports persistent caching of discovered URLs via SQLite.
"""

import asyncio
import xml.etree.ElementTree as ET
from typing import List, Set, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

from src.config import settings
from src.logger import logger
from src.ingestion.crawl_strategy import DomainCrawlStrategy, get_strategy_registry


# ============================================
# Sitemap Parser
# ============================================

class SitemapParser:
    """Parse XML sitemaps to discover URLs."""

    SITEMAP_PATHS = [
        "/sitemap.xml",
        "/sitemap_index.xml",
        "/sitemap/sitemap.xml",
        "/wp-sitemap.xml",
    ]

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def discover_from_sitemap(
        self,
        domain: str,
        strategy: DomainCrawlStrategy,
    ) -> Set[str]:
        """
        Try to find and parse sitemaps for a domain.
        """
        discovered = set()
        # Try both with and without www
        domains_to_try = [domain]
        if not domain.startswith("www."):
            domains_to_try.append(f"www.{domain}")

        for d in domains_to_try:
            base_url = f"https://{d}"
            for sitemap_path in self.SITEMAP_PATHS:
                sitemap_url = f"{base_url}{sitemap_path}"
                try:
                    urls = await self._fetch_and_parse_sitemap(sitemap_url, strategy)
                    if urls:
                        discovered.update(urls)
                        logger.info(f"Found {len(urls)} URLs from sitemap: {sitemap_url}")
                        return discovered # Success!
                except Exception as e:
                    logger.debug(f"Sitemap not found or error at {sitemap_url}: {e}")
                    continue

        # Also try robots.txt if sitemaps not found in common paths
        if not discovered:
            try:
                robots_sitemaps = await self._find_sitemaps_in_robots(f"https://{domain}")
                for sitemap_url in robots_sitemaps:
                    urls = await self._fetch_and_parse_sitemap(sitemap_url, strategy)
                    discovered.update(urls)
            except Exception as e:
                logger.debug(f"Error checking robots.txt for sitemaps: {e}")

        return discovered

    async def _fetch_and_parse_sitemap(
        self,
        sitemap_url: str,
        strategy: DomainCrawlStrategy,
        depth: int = 0
    ) -> Set[str]:
        """Fetch and parse a single sitemap file with recursion support."""
        if depth > 5: # Prevent infinite sitemap loops
            return set()

        urls = set()
        
        # Apply rate limiting even for sitemaps
        await asyncio.sleep(1.0 / settings.crawler_rate_limit_requests_per_second)

        try:
            response = await self.client.get(
                sitemap_url,
                headers={"User-Agent": settings.crawler_user_agent},
                timeout=settings.crawler_timeout_seconds,
            )
            
            if response.status_code != 200:
                logger.debug(f"Sitemap {sitemap_url} returned status {response.status_code}")
                return set()

            content = response.text
            if not content.strip():
                return set()
                
            root = ET.fromstring(content)
        except Exception as e:
            logger.debug(f"Failed to fetch or parse sitemap {sitemap_url}: {e}")
            return set()

        # Remove namespace for easier parsing
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

        # Check if this is a sitemap index
        sitemap_tags = root.findall(f".//{namespace}sitemap")
        if sitemap_tags:
            # It's an index — recursively parse child sitemaps
            for sitemap_tag in sitemap_tags:
                loc = sitemap_tag.find(f"{namespace}loc")
                if loc is not None and loc.text:
                    child_urls = await self._fetch_and_parse_sitemap(
                        loc.text.strip(), strategy, depth + 1
                    )
                    urls.update(child_urls)
        else:
            # It's a regular sitemap — extract URLs
            url_tags = root.findall(f".//{namespace}url")
            for url_tag in url_tags:
                loc = url_tag.find(f"{namespace}loc")
                if loc is not None and loc.text:
                    url = loc.text.strip()
                    if strategy.is_url_allowed(url):
                        urls.add(url)

        return urls

    async def _find_sitemaps_in_robots(self, base_url: str) -> List[str]:
        """Extract sitemap URLs from robots.txt."""
        sitemaps = []
        try:
            response = await self.client.get(
                f"{base_url}/robots.txt",
                headers={"User-Agent": settings.crawler_user_agent},
                timeout=10,
            )
            if response.status_code == 200:
                for line in response.text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("sitemap:"):
                        sitemap_url = line.split(":", 1)[1].strip()
                        sitemaps.append(sitemap_url)
        except Exception:
            pass
        return sitemaps


# ============================================
# Link Extractor
# ============================================

class LinkExtractor:
    """Extract and filter links from HTML pages."""

    @staticmethod
    def extract_links(
        html_content: str,
        base_url: str,
        strategy: DomainCrawlStrategy,
    ) -> Set[str]:
        """
        Extract same-domain links from HTML that pass the strategy filter.

        Args:
            html_content: Raw HTML
            base_url: URL of the page (for resolving relative links)
            strategy: Domain crawl strategy for filtering

        Returns:
            Set of absolute URLs that pass filtering
        """
        links = set()
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc.replace("www.", "")

        try:
            soup = BeautifulSoup(html_content, "lxml")

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()

                # Skip non-HTTP links
                if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                    continue

                # Resolve relative URLs
                absolute_url = urljoin(base_url, href)
                parsed = urlparse(absolute_url)

                # Remove fragment and query for deduplication
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                # Remove trailing slash for consistent dedup (but keep root)
                if clean_url.endswith("/") and len(parsed.path) > 1:
                    clean_url = clean_url.rstrip("/")

                # Same domain check
                link_domain = parsed.netloc.replace("www.", "")
                if link_domain != base_domain:
                    continue

                # Apply strategy filter
                if strategy.is_url_allowed(clean_url):
                    links.add(clean_url)

        except Exception as e:
            logger.warning(f"Link extraction failed for {base_url}: {e}")

        return links


# ============================================
# URL Discoverer (main orchestrator)
# ============================================

@dataclass
class DiscoveryResult:
    """Result of a URL discovery run."""
    domain: str
    discovered_urls: List[str] = field(default_factory=list)
    from_sitemap: int = 0
    from_crawling: int = 0
    filtered_out: int = 0
    from_cache: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class URLDiscoverer:
    """
    Automatic URL discovery engine.

    Combines sitemap parsing + recursive link extraction
    to discover all relevant pages from seed domains.
    Uses SQLite-backed cache to avoid expensive re-discovery.
    """

    def __init__(self):
        self.registry = get_strategy_registry()
        self.client = httpx.AsyncClient(
            timeout=settings.crawler_timeout_seconds,
            follow_redirects=True,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
            ),
        )
        self.sitemap_parser = SitemapParser(self.client)

    async def discover_all(
        self,
        force_refresh: bool = False,
    ) -> List[DiscoveryResult]:
        """
        Run discovery on all registered domains.
        Uses cached results if available and fresh.

        Args:
            force_refresh: If True, bypass cache and re-crawl

        Returns:
            List of DiscoveryResult per domain
        """
        results = []
        strategies = self.registry.get_all_strategies()

        for strategy in strategies:
            logger.info(f"Starting URL discovery for {strategy.domain}")
            result = await self.discover_domain(strategy, force_refresh=force_refresh)
            results.append(result)

            cache_label = " (from cache)" if result.from_cache else ""
            logger.info(
                f"Discovered {len(result.discovered_urls)} URLs for {strategy.domain} "
                f"(sitemap: {result.from_sitemap}, crawling: {result.from_crawling})"
                f"{cache_label}"
            )

        return results

    async def discover_domain(
        self,
        strategy: DomainCrawlStrategy,
        force_refresh: bool = False,
    ) -> DiscoveryResult:
        """
        Discover all relevant URLs for a single domain.

        Strategy:
        1. Check SQLite cache first (unless force_refresh)
        2. Try sitemap (bulk discovery)
        3. BFS crawl from seed paths (find pages sitemap missed)
        4. Merge + deduplicate + sort by relevance
        5. Save results to cache
        """
        # --- Phase 0: Check cache ---
        if not force_refresh:
            cached = self._get_cached_result(strategy)
            if cached is not None:
                return cached

        # --- Full discovery (cache miss or forced) ---
        logger.info(f"[{strategy.domain}] Cache miss or forced refresh, running fresh discovery...")

        result = DiscoveryResult(domain=strategy.domain)
        all_discovered: Set[str] = set()
        total_filtered = 0

        # Phase 1: Sitemap discovery
        if strategy.use_sitemap:
            try:
                sitemap_urls = await self.sitemap_parser.discover_from_sitemap(
                    strategy.domain, strategy
                )
                result.from_sitemap = len(sitemap_urls)
                all_discovered.update(sitemap_urls)
                logger.info(
                    f"[{strategy.domain}] Sitemap: found {len(sitemap_urls)} URLs"
                )
            except Exception as e:
                logger.warning(
                    f"[{strategy.domain}] Sitemap discovery failed: {e}"
                )

        # Phase 2: BFS crawl from seed paths
        try:
            crawled_urls, filtered = await self._bfs_discover(
                strategy, already_known=all_discovered
            )
            result.from_crawling = len(crawled_urls)
            total_filtered += filtered
            all_discovered.update(crawled_urls)
            logger.info(
                f"[{strategy.domain}] BFS crawl: found {len(crawled_urls)} new URLs"
            )
        except Exception as e:
            logger.warning(f"[{strategy.domain}] BFS discovery failed: {e}")

        # Sort by relevance score
        scored_urls = [
            (url, strategy.get_relevance_score(url))
            for url in all_discovered
        ]
        scored_urls.sort(key=lambda x: x[1], reverse=True)

        # Apply max_pages limit
        if len(scored_urls) > strategy.max_pages:
            total_filtered += len(scored_urls) - strategy.max_pages
            scored_urls = scored_urls[: strategy.max_pages]

        result.discovered_urls = [url for url, _ in scored_urls]
        result.filtered_out = total_filtered

        # --- Save to cache ---
        self._save_to_cache(strategy, result, scored_urls)

        return result

    def _get_cached_result(
        self,
        strategy: DomainCrawlStrategy,
    ) -> Optional[DiscoveryResult]:
        """Check SQLite cache for fresh discovery results."""
        from src.storage.sqlite_state_store import get_state_store

        state_store = get_state_store()
        ttl = settings.discovery_cache_ttl_hours
        cached = state_store.get_cached_discovery(strategy.domain, max_age_hours=ttl)

        if cached is None:
            return None

        # Rebuild DiscoveryResult from cache
        result = DiscoveryResult(
            domain=strategy.domain,
            discovered_urls=[entry["url"] for entry in cached],
            from_sitemap=sum(1 for e in cached if e["from_sitemap"]),
            from_crawling=sum(1 for e in cached if e["from_crawling"]),
            filtered_out=0,
            from_cache=True,
        )

        age_str = cached[0]["discovered_at"] if cached else "unknown"
        logger.info(
            f"[{strategy.domain}] Using cached discovery "
            f"({len(cached)} URLs, cached at: {age_str})"
        )
        return result

    def _save_to_cache(
        self,
        strategy: DomainCrawlStrategy,
        result: DiscoveryResult,
        scored_urls: List[tuple],
    ):
        """Persist discovery results to SQLite cache."""
        from src.storage.sqlite_state_store import get_state_store

        state_store = get_state_store()
        urls_with_metadata = []

        # Build a set for quick lookup of sitemap-discovered URLs
        # We don't have per-URL provenance, so approximate:
        # all URLs are from the combined discovery
        for url, score in scored_urls:
            urls_with_metadata.append({
                "url": url,
                "authority_level": strategy.authority_level,
                "visa_types": strategy.default_visa_types,
                "relevance_score": score,
                "from_sitemap": result.from_sitemap > 0,
                "from_crawling": result.from_crawling > 0,
            })

        state_store.save_discovered_urls(strategy.domain, urls_with_metadata)

    async def _bfs_discover(
        self,
        strategy: DomainCrawlStrategy,
        already_known: Set[str],
    ) -> tuple[Set[str], int]:
        """
        BFS crawl from seed paths to discover new URLs.

        Returns:
            (new URLs found, count of filtered-out URLs)
        """
        base_url = f"https://{strategy.domain}"
        visited: Set[str] = set(already_known)
        discovered: Set[str] = set()
        filtered_count = 0

        # Initialize queue with seed paths and depth
        queue: deque[tuple[str, int]] = deque()
        for seed_path in strategy.seed_paths:
            seed_url = f"{base_url}{seed_path}"
            if seed_url not in visited:
                queue.append((seed_url, 0))
                visited.add(seed_url)
                discovered.add(seed_url)

        pages_crawled = 0

        while queue and pages_crawled < strategy.max_pages:
            url, depth = queue.popleft()

            if depth > strategy.max_depth:
                continue

            try:
                # Fetch page
                response = await self.client.get(
                    url,
                    headers={"User-Agent": settings.crawler_user_agent},
                    timeout=settings.crawler_timeout_seconds,
                )

                if response.status_code != 200:
                    continue

                pages_crawled += 1

                # Extract links
                new_links = LinkExtractor.extract_links(
                    response.text, url, strategy
                )

                for link in new_links:
                    if link not in visited:
                        visited.add(link)
                        if strategy.is_url_allowed(link):
                            discovered.add(link)
                            if depth + 1 <= strategy.max_depth:
                                queue.append((link, depth + 1))
                        else:
                            filtered_count += 1

                # Rate limiting
                await asyncio.sleep(1.0 / settings.crawler_rate_limit_requests_per_second)

            except Exception as e:
                logger.debug(f"BFS fetch failed for {url}: {e}")
                continue

        return discovered, filtered_count

    async def discover_single_domain(
        self,
        domain: str,
        force_refresh: bool = False,
    ) -> DiscoveryResult:
        """Discover URLs for a specific domain by name."""
        strategy = self.registry.get_strategy(domain)
        return await self.discover_domain(strategy, force_refresh=force_refresh)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Singleton
_discoverer = None


def get_url_discoverer() -> URLDiscoverer:
    """Get or create URL discoverer singleton."""
    global _discoverer
    if _discoverer is None:
        _discoverer = URLDiscoverer()
    return _discoverer
