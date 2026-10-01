"""Audit whether authoritative sources can reach the corpus at all.

A crawl strategy decides what to fetch from a URL's path, which stands in for
relevance. The assumption breaks quietly: the Arbeitsagentur page carrying the
annual EU Blue Card salary thresholds lives under /vor-ort/zav/, while the
strategy only allows /en/ and /web/content/EN/, so it was never fetched. Nothing
recorded that -- a page the crawler never considers produces no error, no skip
count and no log line, and the corpus looks healthy right up until someone asks
a question it cannot answer.

This reads a list of URLs that *should* be answerable and reports, for each,
whether the strategy would let it in and whether it is in the index today. It is
a one-shot check to run after changing strategies or when an answer looks stale,
not something to wire into CI.

    python -m src.scripts.audit_coverage                      # eval/authoritative_sources.json
    python -m src.scripts.audit_coverage URL [URL ...]
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from src.ingestion.crawl_strategy import DomainCrawlStrategy, get_strategy_registry

DEFAULT_SOURCES = Path("eval/authoritative_sources.json")


def explain_rejection(strategy: DomainCrawlStrategy, url: str) -> Optional[str]:
    """Name the rule that excludes this URL, or None if the strategy allows it.

    Mirrors DomainCrawlStrategy.is_url_allowed, which returns a bare bool; the
    point of the audit is knowing which rule to change.
    """
    path = urlparse(url).path.lower()

    for pattern in strategy.blocked_path_patterns:
        if re.search(pattern, path):
            return f"blocked_path_patterns: {pattern}"

    if strategy.allowed_path_patterns and not any(re.search(p, path) for p in strategy.allowed_path_patterns):
        return f"no allowed_path_patterns match (have {strategy.allowed_path_patterns})"

    if strategy.language_prefixes and path not in ("/", ""):
        if not any(path.startswith(p) for p in strategy.language_prefixes):
            return f"language_prefixes: {strategy.language_prefixes}"

    return None


def indexed_urls() -> Optional[set[str]]:
    """Every source_url in the local collection, or None if Qdrant is unreachable."""
    import urllib.error
    import urllib.request

    from src.config import settings

    urls: set[str] = set()
    offset = None
    try:
        while True:
            body: dict = {"limit": 4000, "with_payload": ["source_url"], "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            req = urllib.request.Request(
                f"{settings.qdrant_url}/collections/{settings.qdrant_collection_name}/points/scroll",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            result = json.load(urllib.request.urlopen(req, timeout=60))["result"]
            urls |= {p["payload"].get("source_url") for p in result["points"]}
            offset = result.get("next_page_offset")
            if not offset:
                return urls
    except (urllib.error.URLError, OSError, KeyError):
        return None


def load_sources(argv: list[str]) -> list[dict]:
    if argv:
        return [{"url": u, "note": ""} for u in argv]
    if not DEFAULT_SOURCES.exists():
        sys.exit(f"No URLs given and {DEFAULT_SOURCES} does not exist.")
    return json.loads(DEFAULT_SOURCES.read_text(encoding="utf-8"))["sources"]


def main() -> int:
    sources = load_sources(sys.argv[1:])
    registry = get_strategy_registry()
    present = indexed_urls()
    if present is None:
        print("! Qdrant unreachable — reporting strategy coverage only\n")

    blind, missing, ok = [], [], []
    for entry in sources:
        url = entry["url"]
        strategy = registry.get_strategy(url)
        reason = explain_rejection(strategy, url)
        score = strategy.get_relevance_score(url)

        if reason:
            blind.append((url, strategy.domain, reason, score, entry.get("note", "")))
        elif present is not None and url not in present:
            missing.append((url, strategy.domain, score, entry.get("note", "")))
        else:
            ok.append(url)

    print(f"{len(sources)} authoritative URLs checked\n")

    if blind:
        print(f"✗ BLIND SPOT — the strategy excludes these ({len(blind)})")
        for url, domain, reason, score, note in blind:
            print(f"    {url}")
            print(f"      {domain}  ·  {reason}  ·  relevance {score:.2f}")
            if note:
                print(f"      {note}")
        print()

    if missing:
        print(f"? ALLOWED BUT ABSENT — discovery never reached these ({len(missing)})")
        print("    Usually max_pages, max_depth, or a relevance score too low to survive the cut.")
        for url, domain, score, note in missing:
            print(f"    {url}")
            print(f"      {domain}  ·  relevance {score:.2f}")
            if note:
                print(f"      {note}")
        print()

    if ok:
        print(f"✓ IN CORPUS ({len(ok)})")
        for url in ok:
            print(f"    {url}")

    return 1 if blind else 0


if __name__ == "__main__":
    sys.exit(main())
