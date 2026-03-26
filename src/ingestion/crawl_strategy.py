"""
Domain-specific crawl strategies.
Each domain can have custom rules for path filtering, depth, authority level, etc.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import yaml

from src.config import settings
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
    domain="www.make-it-in-germany.com",
    seed_paths=[
        "/en/visa-residence/",
        "/en/visa-residence/opportunity-card",
        "/en/visa-residence/types/",
        "/en/visa-residence/procedure/",
        "/en/visa-residence/skilled-immigration-act",
        "/en/working-in-germany/",
        # German-language equivalents (often contain additional legal detail)
        "/de/visum-aufenthalt/",
        "/de/visum-aufenthalt/chancenkarte/chancenkarte-zur-jobsuche",
        "/de/visum-aufenthalt/fachkraefteeinwanderungsgesetz/",
    ],
    authority_level="official",
    default_visa_types=["general", "chancenkarte", "blue_card", "work_visa", "student_visa"],
    max_depth=4,
    max_pages=200,
    allowed_path_patterns=[
        r"/en/visa-residence/",
        r"/en/working-in-germany/",
        r"/en/living-in-germany/",
        r"/de/visum-aufenthalt/",
        r"/de/arbeit-in-deutschland/",
    ],
    blocked_path_patterns=[
        r"/newsletter",
        r"/contact",
        r"/press",
        r"/imprint",
        r"/privacy",
        r"/print$",
        # ── Confirmed noise from 400-sample analysis ──
        r"/working-in-germany/job-listings/job/",  # individual job postings (not regulations)
        r"/service/newsletter",  # newsletter archives (en)
        r"/service/kurz-newsletter",  # short newsletter (en)
        r"/de/newsletter-",  # newsletter archives (de)
        r"/de/service/Newsletter",
        r"/service/glossar",  # thin glossary entries (de)
        r"/service/glossary",  # thin glossary entries (en)
        r"/service/advisory-contact-services/",  # worldwide contact list
        r"/footer-meta/",  # privacy policy, imprint, etc.
        r"/living-in-germany/discover-germany/",  # culture/politics (not visa law)
        r"/living-in-germany/housing-mobility/",  # driving licence, housing
        r"/living-in-germany/learn-german/",  # german classes
        r"/living-in-germany/family-life/",  # childcare, schools — not visa law
        r"/living-in-germany/family-reunification/parental-leave",
        r"/de/unternehmen/integrieren/",  # employer integration guides
        r"/index\.html$",  # generic entry pages
        r"\.(pdf|jpg|png|gif|svg|css|js)$",
    ],
    relevance_keywords=[
        "visa",
        "residence",
        "chancenkarte",
        "opportunity-card",
        "blue-card",
        "work-permit",
        "skilled",
        "immigration",
        "application",
        "requirements",
        "procedure",
        "visum",
        "aufenthalt",
        "fachkraft",
    ],
    language_prefixes=["/en/", "/de/"],
    use_sitemap=True,
)

CHANCENKARTE_COM_STRATEGY = DomainCrawlStrategy(
    domain="www.chancenkarte.com",
    seed_paths=[
        "/en/",
        "/en/guides/",
        "/en/news/",
        "/en/calculator/",
    ],
    authority_level="third_party",  # commercial site, not government
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
        r"/privacy-policy",
        r"/cookie-policy",
        r"/imprint",
        r"/candidates",  # B2B marketing
        r"/employers",  # B2B marketing
        r"/offers/jobs",  # job board noise
        r"/index\.html$",
        r"\.(pdf|jpg|png|gif|svg|css|js)$",
    ],
    relevance_keywords=[
        "chancenkarte",
        "opportunity-card",
        "guide",
        "requirement",
        "point",
        "calculator",
        "recognition",
        "application",
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
        r"/out/",  # affiliate/external redirects
        r"/privacy-policy",
        r"/terms-of-service",
        r"/index\.(php|html)$",
        r"\.(pdf|jpg|png|gif|svg|css|js)$",
    ],
    relevance_keywords=[
        "visa",
        "work",
        "residence",
        "blue-card",
        "chancenkarte",
    ],
    language_prefixes=[],  # no language prefix structure
    use_sitemap=True,
)

BAMF_STRATEGY = DomainCrawlStrategy(
    domain="www.bamf.de",
    # BAMF: Bundesamt für Migration und Flüchtlinge
    # Primary authority for visa regulations, Blue Card, Chancenkarte, skilled worker law
    seed_paths=[
        "/EN/Themen/MigrationAufenthalt/ZuwandererDrittstaaten/",
        "/EN/Themen/MigrationAufenthalt/ZuwandererDrittstaaten/Fachkraefte/fachkraefte-node.html",
        "/EN/Themen/MigrationAufenthalt/ZuwandererDrittstaaten/BlaueKarteEU/",
        "/EN/Themen/MigrationAufenthalt/ZuwandererDrittstaaten/Fachkraefte/Chancenkarte/",
        "/Shared/Publikationen/",
    ],
    authority_level="official",
    default_visa_types=["general", "blue_card", "chancenkarte", "work_visa"],
    max_depth=3,
    max_pages=150,
    allowed_path_patterns=[
        r"/EN/Themen/MigrationAufenthalt/",
        r"/Shared/Publikationen/",
    ],
    blocked_path_patterns=[
        r"/Presse/",
        r"/Veranstaltungen/",
        r"/SharedDocs/Downloads/",
        r"/SiteGlobals/",
        r"/data-protection",  # GDPR noise
        r"-node$",  # internal navigation/index nodes
        r"\.(pdf|jpg|png|gif|svg|css|js)$",
    ],
    relevance_keywords=[
        "migration",
        "aufenthalt",
        "drittstaaten",
        "fachkraefte",
        "blaue-karte",
        "bluecard",
        "chancenkarte",
        "visum",
        "skilled",
        "immigration",
        "residence",
    ],
    language_prefixes=["/EN/", "/DE/"],
    use_sitemap=True,
)

BA_STRATEGY = DomainCrawlStrategy(
    domain="www.arbeitsagentur.de",
    # BA: Bundesagentur für Arbeit
    # Labour market authority: Zustimmung checks and shortage occupation lists
    seed_paths=[
        "/en/",
        "/web/content/EN/findingajob/",
        "/web/content/EN/institutionunternehmen/",
    ],
    authority_level="official",
    default_visa_types=["work_visa", "blue_card"],
    max_depth=3,
    max_pages=100,
    allowed_path_patterns=[
        r"/en/",
        r"/web/content/EN/",
    ],
    blocked_path_patterns=[
        r"/en/press/",  # press releases
        r"/de/presse/",  # press releases (de)
        r"/Buergerinnen/",  # citizen benefit (irrelevant)
        r"/Vordrucke/",  # forms
        r"\.(pdf|jpg|png|gif|svg|css|js)$",
    ],
    relevance_keywords=[
        "skilled",
        "worker",
        "immigration",
        "qualified",
        "professional",
        "zustimmung",
        "employment",
        "labour",
        "market",
        "shortage",
    ],
    language_prefixes=["/en/", "/web/content/EN/"],
    use_sitemap=True,
)

KMK_STRATEGY = DomainCrawlStrategy(
    domain="www.kmk.org",
    # KMK: Kultusministerkonferenz — education policy & credential recognition
    # Also covers ZAB (Zentralstelle für ausländisches Bildungswesen) under /zab/
    seed_paths=[
        "/en/",
        "/en/themen/recognition-and-transparency/recognition/recognition-of-foreign-qualifications.html",
        "/zab/en/",
        "/zab/en/statement-of-comparability.html",
    ],
    authority_level="official",
    default_visa_types=["general", "work_visa", "blue_card", "student_visa"],
    max_depth=3,
    max_pages=100,
    allowed_path_patterns=[
        r"/en/",
        r"/zab/en/",
    ],
    blocked_path_patterns=[
        r"/presse/",
        r"/aktuell/",  # news — time-sensitive, low RAG value
        r"/dokumentation/",
        # Confirmed noise from chunk quality analysis
        r"/en/aktuelles/",  # PISA news, press archive
        r"/en/wissenschaftsministerkonferenz/",  # science ministry (unrelated to visa)
        r"/en/bildungsministerkonferenz/",  # K-12 education ministry (not immigration)
        r"/en/kultusministerkonferenz/",  # teacher exchange, school topics
        r"/en/kulturministerkonferenz/",  # culture topics (UNESCO, etc.)
        r"/en/service/servicebereich-schule",  # K-12 school services
        r"/en/downloads-dokumente/statistik/",  # statistics archives
        r"/en/downloads-dokumente/beschluesse-und-veroeffentlichungen/bildung-/-schule/",
        r"/en/downloads-dokumente/beschluesse-und-veroeffentlichungen/kunst-",
        r"/en/downloads-dokumente/beschluesse-und-veroeffentlichungen/internationales",
        r"downloadbereich-rahmenlehrplaene",  # Huge page (40k tokens) — Table of contents
        r"downloads-berufsfachschulen",  # Low-value giant list
        r"/en/inhalt\.html",  # Generic site map
        # NOTE: /zab/ and /recognition/ paths are explicitly allowed above — not affected
        r"\.(pdf|jpg|png|gif|svg|css|js)$",
    ],
    relevance_keywords=[
        "recognition",
        "qualification",
        "foreign",
        "degree",
        "comparability",
        "statement",
        "credential",
        "anabin",
        "university",
        "higher-education",
        "bilateral",
    ],
    language_prefixes=["/en/", "/zab/en/"],
    use_sitemap=True,
)


# ============================================
# Strategy Registry
# ============================================


class StrategyRegistry:
    """Registry for domain-specific crawl strategies."""

    def __init__(self, config_path: Optional[Path] = None):
        self._strategies: Dict[str, DomainCrawlStrategy] = {}
        # 1. Register built-in hardcoded strategies (base rules)
        self._register_defaults()
        # 2. Load and override/extend from YAML
        self.load_from_yaml(config_path or settings.seed_urls_path)

    def _register_defaults(self):
        """Register pre-defined strategies."""
        self.register(MAKE_IT_IN_GERMANY_STRATEGY)
        self.register(CHANCENKARTE_COM_STRATEGY)
        self.register(GERMANY_VISA_STRATEGY)
        self.register(BAMF_STRATEGY)
        self.register(BA_STRATEGY)
        self.register(KMK_STRATEGY)

    def load_from_yaml(self, path: Path):
        """Load domain configurations from a YAML file.

        YAML fields supported per domain:
          seed_paths, authority_level, default_visa_types,
          max_depth, max_pages, use_sitemap,
          allowed_path_patterns, blocked_path_patterns,
          relevance_keywords, language_prefixes
        """
        if not path.exists():
            logger.warning(f"Seed URLs config not found at {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                domain_configs = config.get("domains", [])

                for dc in domain_configs:
                    domain = dc.get("domain")
                    if not domain:
                        continue

                    if domain in self._strategies:
                        # Merge YAML values into existing hardcoded strategy
                        strategy = self._strategies[domain]
                        strategy.seed_paths = dc.get("seed_paths", strategy.seed_paths)
                        strategy.authority_level = dc.get("authority_level", strategy.authority_level)
                        strategy.default_visa_types = dc.get("default_visa_types", strategy.default_visa_types)
                        strategy.max_depth = dc.get("max_depth", strategy.max_depth)
                        strategy.max_pages = dc.get("max_pages", strategy.max_pages)
                        strategy.use_sitemap = dc.get("use_sitemap", strategy.use_sitemap)
                        # Now also honour filter fields from YAML
                        if "allowed_path_patterns" in dc:
                            strategy.allowed_path_patterns = dc["allowed_path_patterns"]
                        if "blocked_path_patterns" in dc:
                            strategy.blocked_path_patterns = dc["blocked_path_patterns"]
                        if "relevance_keywords" in dc:
                            strategy.relevance_keywords = dc["relevance_keywords"]
                        if "language_prefixes" in dc:
                            strategy.language_prefixes = dc["language_prefixes"]
                    else:
                        # Create a new strategy from YAML for unknown domains
                        strategy = DomainCrawlStrategy(
                            domain=domain,
                            seed_paths=dc.get("seed_paths", ["/"]),
                            authority_level=dc.get("authority_level", "third_party"),
                            default_visa_types=dc.get("default_visa_types", ["general"]),
                            max_depth=dc.get("max_depth", 3),
                            max_pages=dc.get("max_pages", 100),
                            use_sitemap=dc.get("use_sitemap", True),
                            allowed_path_patterns=dc.get("allowed_path_patterns", []),
                            blocked_path_patterns=dc.get("blocked_path_patterns", []),
                            relevance_keywords=dc.get("relevance_keywords", []),
                            language_prefixes=dc.get("language_prefixes", ["/en/", "/de/"]),
                        )
                        self.register(strategy)

            logger.info(f"Loaded {len(domain_configs)} domain strategies from {path}")
        except Exception as e:
            logger.error(f"Failed to load domain strategies from {path}: {e}")

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

        # Try exact match first
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
