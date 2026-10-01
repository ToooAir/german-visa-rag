"""Unit tests for src/scripts/audit_coverage.py"""

from src.ingestion.crawl_strategy import DomainCrawlStrategy
from src.scripts.audit_coverage import explain_rejection


def _strategy(**kw) -> DomainCrawlStrategy:
    return DomainCrawlStrategy(domain="example.com", **kw)


class TestExplainRejectionAgreesWithIsUrlAllowed:
    """explain_rejection restates is_url_allowed in order to name the failing rule.
    Two copies of one decision drift apart, so the contract is that they always
    give the same verdict."""

    CASES = [
        (_strategy(), "https://example.com/anything"),
        (_strategy(allowed_path_patterns=["/en/"]), "https://example.com/en/visa"),
        (_strategy(allowed_path_patterns=["/en/"]), "https://example.com/de/visum"),
        (_strategy(blocked_path_patterns=[r"\.pdf$"]), "https://example.com/en/form.pdf"),
        (_strategy(language_prefixes=["/en/"]), "https://example.com/vor-ort/zav/page"),
        (_strategy(language_prefixes=["/en/"]), "https://example.com/"),
        (_strategy(language_prefixes=[]), "https://example.com/anywhere"),
        (
            _strategy(allowed_path_patterns=["/en/"], blocked_path_patterns=["/press/"]),
            "https://example.com/en/press/release",
        ),
    ]

    def test_verdicts_match(self):
        for strategy, url in self.CASES:
            allowed = strategy.is_url_allowed(url)
            reason = explain_rejection(strategy, url)
            assert allowed is (reason is None), f"{url}: is_url_allowed={allowed} but reason={reason!r}"


class TestExplainRejectionNamesTheRule:
    def test_allowed_pattern_miss(self):
        s = _strategy(allowed_path_patterns=["/en/"], language_prefixes=[])
        assert "allowed_path_patterns" in explain_rejection(s, "https://example.com/vor-ort/zav/x")

    def test_blocked_pattern_takes_precedence(self):
        """Checked first, mirroring is_url_allowed, so the reported rule is the
        one that actually decided."""
        s = _strategy(allowed_path_patterns=["/en/"], blocked_path_patterns=["/press/"])
        reason = explain_rejection(s, "https://example.com/en/press/x")
        assert reason.startswith("blocked_path_patterns")

    def test_language_prefix(self):
        s = _strategy(language_prefixes=["/en/"])
        assert "language_prefixes" in explain_rejection(s, "https://example.com/de/visum")

    def test_root_path_survives_the_language_prefix(self):
        s = _strategy(language_prefixes=["/en/"])
        assert explain_rejection(s, "https://example.com/") is None

    def test_allowed_url_has_no_reason(self):
        s = _strategy(allowed_path_patterns=["/en/"], language_prefixes=["/en/"])
        assert explain_rejection(s, "https://example.com/en/visa-residence") is None


class TestTheRealBlindSpot:
    """The case that prompted the script: the Arbeitsagentur page carrying the
    annual EU Blue Card salary thresholds. It was excluded by two rules at once --
    allowed_path_patterns and language_prefixes -- because both assume the useful
    pages sit under /en/, and this one is a German-language ZAV newsletter."""

    URL = "https://www.arbeitsagentur.de/vor-ort/zav/working-and-living-in-germany/newsletter-iss/03-2026/blaue-karte"

    def test_the_strategy_now_admits_it(self):
        from src.ingestion.crawl_strategy import get_strategy_registry

        strategy = get_strategy_registry().get_strategy(self.URL)
        assert explain_rejection(strategy, self.URL) is None

    def test_it_scores_above_zero_so_it_survives_the_max_pages_cut(self):
        """Permitting a URL is not enough: discovery ranks by relevance and keeps
        the top max_pages. At 0.0 this page lost the cut even when allowed."""
        from src.ingestion.crawl_strategy import get_strategy_registry

        strategy = get_strategy_registry().get_strategy(self.URL)
        assert strategy.get_relevance_score(self.URL) > 0.0

    def test_either_rule_alone_would_still_exclude_it(self):
        """Both had to change, which is why fixing one was not enough."""
        only_pattern = _strategy(allowed_path_patterns=["/en/", "/vor-ort/zav/"], language_prefixes=["/en/"])
        only_prefix = _strategy(allowed_path_patterns=["/en/"], language_prefixes=["/en/", "/vor-ort/zav/"])
        assert "language_prefixes" in explain_rejection(only_pattern, self.URL)
        assert "allowed_path_patterns" in explain_rejection(only_prefix, self.URL)
