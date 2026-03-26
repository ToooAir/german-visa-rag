"""
Query Transformer module for query expansion and multilingual support.
Uses LLM for spell-checking, intent expansion, and query enrichment.
"""

from typing import Any, Optional
from enum import Enum
import json
import re
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import logger
from src.llm import get_llm_client


class QueryTransformType(str, Enum):
    """Types of query transformations."""

    SPELL_CHECK = "spell_check"
    EXPANSION = "expansion"
    REFORMULATION = "reformulation"
    SUMMARIZATION = "summarization"


QUERY_TRANSFORMER_PROMPT_TEMPLATE = """
You are an expert on German visa and work permit regulations. Given a user query, perform the following tasks:

1. **Spell Correction**: Fix obvious spelling or grammar errors while preserving the original language.
2. **Intent Expansion**: If the query is short or ambiguous, generate 2-3 alternative phrasings that cover different angles.
3. **Disambiguation**: Identify terms that may refer to multiple visa categories.

**Original Query**:
{query}

**Task Instructions**:
1. Correct spelling and grammar while keeping the original language.
2. In `english_query` and `german_query`, prioritize **legal and domain-specific terminology**
   (e.g., Zulassung, Zusatzblatt, Chancenkarte, Verpflichtungserklärung, Fachkräfteeinwanderungsgesetz).
   Precise terminology is critical for vector retrieval accuracy.
3. Generate 2-3 variants with distinct focus areas.

**Output Format** (strict JSON, no markdown wrapping):
{{
  "corrected_query": "Spell-corrected version of the original query (same language as input)",
  "english_query": "Technical English translation optimized for retrieval",
  "german_query": "Präzise deutsche juristische Übersetzung für die Vektorssuche",
  "query_variants": [
    "Variant 1: Focus on procedure and application process",
    "Variant 2: Focus on eligibility criteria and thresholds",
    "Variant 3: Focus on financial proof requirements"
  ],
  "detected_visa_types": ["chancenkarte", "work_visa"],
  "languages_detected": ["zh"],
  "confidence": 0.95
}}
"""


class QueryTransformer:
    """
    Query transformation pipeline for improving RAG retrieval.

    Features:
    - Multi-language support (DE, EN, ZH)
    - Query expansion for short or ambiguous queries
    - Spell checking and normalization
    - Visa type detection for downstream filtering
    """

    def __init__(self):
        self.llm = get_llm_client()

    async def transform_query(
        self,
        query: str,
        apply_expansion: bool = True,
    ) -> dict[str, Any]:
        """Transform and enrich a query."""
        logger.debug("Transforming query: %.100s", query)

        try:
            # Short queries always expand; longer queries respect apply_expansion
            if len(query.split()) < 3 or apply_expansion:
                return await self._expand_query_with_llm(query)
            else:
                return {
                    "corrected_query": query,
                    "query_variants": [query],
                    "detected_visa_types": self._detect_visa_types(query),
                    "languages_detected": self._detect_languages(query),
                    "confidence": 0.9,
                }
        except Exception as e:
            logger.warning("Query transformation failed, using original: %s", e)
            return {
                "corrected_query": query,
                "query_variants": [query],
                "detected_visa_types": [],
                "languages_detected": ["de"],
                "confidence": 0.5,
            }

    # @retry moved here — where the actual LLM call happens
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def _expand_query_with_llm(self, query: str) -> dict[str, Any]:
        """Use LLM to expand and enrich query."""
        prompt = QUERY_TRANSFORMER_PROMPT_TEMPLATE.format(query=query)

        response_text = await self.llm.call_non_streaming(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        response_text = response_text.strip()

        # Strip markdown code fences if present
        if "```json" in response_text:
            response_text = response_text.split("```json").split("```").strip()[1]
        elif "```" in response_text:
            response_text = response_text.split("```")[11].split("```")[0].strip()

        result = json.loads(response_text)
        logger.debug("Query expansion result: %s", result)
        return result

    def _detect_visa_types(self, query: str) -> list[str]:
        """Detect visa types mentioned in query."""
        query_lower = query.lower()
        visa_patterns = {
            "chancenkarte": [r"chancenkarte", r"opportunity card", r"機會卡"],
            "work_visa": [r"work permit", r"arbeitserlaubnis", r"work visa", r"技術人才", r"工作簽證"],
            "student_visa": [r"student visa", r"studentenvisum", r"學生簽證", r"就學簽證", r"留學"],
            "blue_card": [r"blue card", r"blaue karte", r"藍卡"],
            "freelance_visa": [r"freelance", r"freiberufler", r"自由業"],
            "entrepreneur_visa": [r"entrepreneur", r"unternehmer", r"創業"],
        }
        detected = []
        for visa_type, patterns in visa_patterns.items():
            if any(re.search(p, query_lower) for p in patterns):
                detected.append(visa_type)
        return detected

    def _detect_languages(self, query: str) -> list[str]:
        """Simple heuristic language detection."""
        languages = []
        if any("\u4e00" <= char <= "\u9fff" for char in query):
            languages.append("zh")
        if any(char in query.lower() for char in ["ä", "ö", "ü", "ß"]) or re.search(
            r"\b(der|die|das|und|zu|mit|ein|eine|wie|ich)\b", query.lower()
        ):
            languages.append("de")
        if not languages:
            languages.append("en")
        return languages

    async def get_search_queries(self, query: str) -> list[str]:
        """
        Return search queries (main + variants) for hybrid retrieval.
        Caps output at 3 queries: [corrected, english, german].
        """
        if not settings.enable_query_expansion:
            return [query]

        if len(query) > 100:
            logger.debug("Fast Mode: skipping expansion for long query (len=%d)", len(query))
            return [query]

        try:
            transformed = await self.transform_query(query)
            search_queries = [transformed.get("corrected_query", query)]

            for key in ("english_query", "german_query"):
                val = transformed.get(key)
                if val and val not in search_queries:
                    search_queries.append(val)

            if len(search_queries) < 3:
                for variant in transformed.get("query_variants", []):
                    if variant and variant not in search_queries:
                        search_queries.append(variant)
                        break

            return list(dict.fromkeys(filter(None, search_queries)))[:3]

        except Exception as e:
            logger.warning("Expansion failed, falling back to original: %s", e)
            return [query]


_transformer: Optional[QueryTransformer] = None


def get_query_transformer() -> QueryTransformer:
    """Get or create query transformer singleton."""
    global _transformer
    if _transformer is None:
        _transformer = QueryTransformer()
    return _transformer
