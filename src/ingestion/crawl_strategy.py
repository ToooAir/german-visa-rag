"""
Domain-specific crawl strategies.
Each domain can have custom rules for path filtering, depth, authority level, etc.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass, field
from urllib.parse import urlparse
import re

from src.logger import logger


@dataclass
class DomainCrawlStrategy:
    """Crawl strategy for a specific domain."""

    domain: str
    seed_paths: List[str] = field(default_factory=lambda: ["/"])
    authority_level: str = "third_party"
    default_visa_types: List[str] = field(default_factory=lambda: ["general"])
    max_depth: int = 3
    max_pages: int = 100

    # Path filtering
    allowed_path_patterns: List[str] = field(default_factory=list)
    blocked_path_patterns: List[str] = field(default_factory=list)

    # Content relevance keywords in URL path
    relevance_keywords: List[str] = field(default_factory=list)

    # Language filter — only keep URLs matching these lang path prefixes
    language_prefixes: List[str] = field(default_factory=lambda: ["/en/", "/de/"])

    # Whether to parse sitemap for this domain
    use_sitemap: bool = True

    def is_url_allowed(self, url: str) -> bool:
        """Check if a URL is allowed by this strategy."""
        parsed = urlparse(url)
        path = parsed.path.lower()

        # Check blocked patterns first
        for pattern in self.blocked_path_patterns:
            if re.search(pattern, path):
                return False

        # Check allowed patterns (if defined, URL must match at least one)
        if self.allowed_path_patterns:
            if not any(re.search(p, path) for p in self.allowed_path_patterns):
                return False

        # Check language prefix (if defined)
        if self.language_prefixes:
            if not any(path.startswith(prefix) for prefix in self.language_prefixes):
                # Allow root path
                if path not in ("/", ""):
                    return False

        return True

    def get_relevance_score(self, url: str) -> float:
        """Score URL relevance (0.0 - 1.0) based on path keywords."""
        if not self.relevance_keywords:
            return 0.5  # neutral if no keywords defined

        parsed = urlparse(url)
        path = parsed.path.lower()
        matches = sum(1 for kw in self.relevance_keywords if kw in path)
        return min(1.0, matches / max(len(self.relevance_keywords) * 0.3, 1))


# ============================================
# Pre-defined strategies for known domains
# ============================================

MAKE_IT_IN_GERMANY_STRATEGY = DomainCrawlStrategy(
    domain="make-it-in-germany.com",
    seed_paths=[
        "/en/visa-residence/",
        "/en/visa-residence/opportunity-card",
        "/en/visa-residence/types/",
        "/en/visa-residence/procedure/",
        "/en/visa-residence/skilled-immigration-act",
        "/en/working-in-germany/",
    ],
    authority_level="official",
    default_visa_types=["general"],
    max_depth=4,
    max_pages=200,
    allowed_path_patterns=[
        r"/en/visa-residence/",
        r"/en/working-in-germany/",
        r"/en/living-in-germany/",
    ],
    blocked_path_patterns=[
        r"/newsletter",
        r"/contact",
        r"/press",
        r"/imprint",
        r"/privacy",
        r"/print$",
        r"\.(pdf|jpg|png|gif|svg|css|js)$",
    ],
    relevance_keywords=[
        "visa", "residence", "chancenkarte", "opportunity-card",
        "blue-card", "work-permit", "skilled", "immigration",
        "application", "requirements", "procedure",
    ],
    language_prefixes=["/en/"],
    use_sitemap=True,
)

CHANCENKARTE_COM_STRATEGY = DomainCrawlStrategy(
    domain="chancenkarte.com",
    seed_paths=[
        "/en/",
        "/en/guides/",
        "/en/news/",
        "/en/calculator/",
    ],
    authority_level="semi_official",
    default_visa_types=["chancenkarte"],
    max_depth=3,
    max_pages=100,
    allowed_path_patterns=[
        r"/en/",
    ],
    blocked_path_patterns=[
        r"/tag/",
        r"/author/",
        r"/wp-",
        r"/feed",
        r"\.(pdf|jpg|png|gif|svg|css|js)$",
    ],
    relevance_keywords=[
        "chancenkarte", "opportunity-card", "guide", "requirement",
        "point", "calculator", "recognition", "application",
    ],
    language_prefixes=["/en/"],
    use_sitemap=True,
)

GERMANY_VISA_STRATEGY = DomainCrawlStrategy(
    domain="www.germany-visa.org",
    seed_paths=[
        "/",
        "/work-employment-visa/",
    ],
    authority_level="third_party",
    default_visa_types=["general"],
    max_depth=3,
    max_pages=50,
    allowed_path_patterns=[
        r"/work",
        r"/visa",
        r"/residence",
        r"/blue-card",
        r"/chancenkarte",
    ],
    blocked_path_patterns=[
        r"/blog/",
        r"/contact",
        r"\.(pdf|jpg|png|gif|svg|css|js)$",
    ],
    relevance_keywords=[
        "visa", "work", "residence", "blue-card", "chancenkarte",
    ],
    language_prefixes=[],  # no language prefix structure
    use_sitemap=True,
)


# ============================================
# Strategy Registry
# ============================================

class StrategyRegistry:
    """Registry for domain-specific crawl strategies."""

    def __init__(self):
        self._strategies: Dict[str, DomainCrawlStrategy] = {}
        # Register built-in strategies
        self._register_defaults()

    def _register_defaults(self):
        """Register pre-defined strategies."""
        self.register(MAKE_IT_IN_GERMANY_STRATEGY)
        self.register(CHANCENKARTE_COM_STRATEGY)
        self.register(GERMANY_VISA_STRATEGY)

    def register(self, strategy: DomainCrawlStrategy):
        """Register a crawl strategy for a domain."""
        self._strategies[strategy.domain] = strategy
        logger.debug(f"Registered crawl strategy for {strategy.domain}")

    def get_strategy(self, url_or_domain: str) -> DomainCrawlStrategy:
        """Get strategy for a URL or domain. Returns default if not found."""
        # Extract domain from URL if needed
        if url_or_domain.startswith("http"):
            domain = urlparse(url_or_domain).netloc
        else:
            domain = url_or_domain

        # Remove www. prefix for matching
        domain_clean = domain.replace("www.", "")

        # Try exact match
        if domain in self._strategies:
            return self._strategies[domain]
        if domain_clean in self._strategies:
            return self._strategies[domain_clean]

        # Try partial match (subdomain)
        for key, strategy in self._strategies.items():
            if domain.endswith(key) or domain_clean.endswith(key):
                return strategy

        # Return a generic default strategy
        logger.info(f"No specific strategy for {domain}, using default")
        return DomainCrawlStrategy(
            domain=domain,
            max_depth=2,
            max_pages=50,
        )

    def get_all_domains(self) -> List[str]:
        """Get all registered domain names."""
        return list(self._strategies.keys())

    def get_all_strategies(self) -> List[DomainCrawlStrategy]:
        """Get all registered strategies."""
        return list(self._strategies.values())


# Singleton registry
_registry = None


def get_strategy_registry() -> StrategyRegistry:
    """Get or create strategy registry singleton."""
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry
