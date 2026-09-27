"""Validate seed_urls.yml against the models it is loaded into.

Every value here is turned into an enum member during ingestion, so a typo in the
config surfaces only as a failed document mid-run. These checks fail at test time
instead: "'skilled_worker' is not a valid VisaType" cost a 786-document run.
"""

import yaml

from src.ingestion.cli import load_domain_configs, load_seed_urls
from src.models.chunk import AuthorityLevel, VisaType

VALID_VISA_TYPES = {v.value for v in VisaType}
VALID_AUTHORITY = {a.value for a in AuthorityLevel}


def test_pinned_urls_declare_only_real_visa_types():
    offenders = [
        (doc["url"], vt) for doc in load_seed_urls() for vt in doc.get("visa_types", []) if vt not in VALID_VISA_TYPES
    ]
    assert offenders == []


def test_domain_defaults_declare_only_real_visa_types():
    offenders = [
        (d["domain"], vt)
        for d in load_domain_configs()
        for vt in d.get("default_visa_types", [])
        if vt not in VALID_VISA_TYPES
    ]
    assert offenders == []


def test_authority_levels_are_real():
    offenders = [
        (d.get("domain") or d.get("url"), d["authority_level"])
        for d in load_domain_configs() + load_seed_urls()
        if d.get("authority_level") and d["authority_level"] not in VALID_AUTHORITY
    ]
    assert offenders == []


def test_python_strategy_defaults_declare_only_real_visa_types():
    """The YAML overrides these, which is how an invalid value hid in the gesetze
    strategy until a pinned URL used it directly."""
    from src.ingestion import crawl_strategy

    offenders = []
    for name in dir(crawl_strategy):
        obj = getattr(crawl_strategy, name)
        if isinstance(obj, crawl_strategy.DomainCrawlStrategy):
            offenders += [(obj.domain, vt) for vt in obj.default_visa_types if vt not in VALID_VISA_TYPES]
    assert offenders == []


def test_pinned_urls_are_unique():
    urls = [doc["url"] for doc in load_seed_urls()]
    assert len(urls) == len(set(urls))


def test_yaml_stays_parseable_and_keeps_both_sections():
    from src.config import settings

    config = yaml.safe_load(open(settings.seed_urls_path, encoding="utf-8"))
    assert config["extra_urls"] and config["domains"]
