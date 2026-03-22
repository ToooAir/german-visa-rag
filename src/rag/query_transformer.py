"""
Query Transformer module for query expansion and multilingual support.
Uses LLM for spell-checking, intent expansion, and query enrichment.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
import re
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import logger
from src.llm import get_llm_client  # <--- 修改點 1：指向 LLM Factory


class QueryTransformType(str, Enum):
    """Types of query transformations."""
    SPELL_CHECK = "spell_check"
    EXPANSION = "expansion"
    REFORMULATION = "reformulation"
    SUMMARIZATION = "summarization"


QUERY_TRANSFORMER_PROMPT_TEMPLATE = """
你是一位德國簽證與工作許可專家。用戶提出了一個問題，請執行以下任務：

1. **拼字修正**：糾正明顯的拼字/語法錯誤（保持原語言）
2. **意圖擴充**：如果查詢簡短或模糊，生成 2-3 個相關的替代提問
3. **去歧義**：識別可能指代多個簽證類別的術語

**原始查詢**：
{query}

**輸出格式**（JSON）：
{{
  "corrected_query": "修正後的主查詢",
  "english_query": "English translation of the query for better retrieval",
  "german_query": "Deutsche Übersetzung der Suchanfrage",
  "query_variants": [
    "變體1：關注於簽證申請程序",
    "變體2：關注於 Chancenkarte 資格",
    "變體3：關注於財務要求"
  ],
  "detected_visa_types": ["chancenkarte", "work_visa"],
  "languages_detected": ["zh"],
  "confidence": 0.95
}}
"""


class QueryTransformer:
    """
    Query transformation pipeline for improving RAG retrieval.
    
    Optimizations:
    - Multi-language support (DE, EN, ZH)
    - Query expansion for poor/ambiguous queries
    - Spell checking and normalization
    - Visa type detection for filtering
    """

    def __init__(self):
        self.llm = get_llm_client()
        self.enable_expansion = True

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def transform_query(
        self,
        query: str,
        apply_expansion: bool = True,
    ) -> Dict[str, Any]:
        """
        Transform and enrich query.
        """
        logger.debug(f"Transforming query: {query[:100]}")
        
        try:
            if not apply_expansion or len(query.split()) < 3:
                # For very short queries, always expand
                return await self._expand_query_with_llm(query)
            else:
                # For longer queries, do lightweight normalization only
                return {
                    "corrected_query": query,
                    "query_variants": [query],
                    "detected_visa_types": self._detect_visa_types(query),
                    "languages_detected": self._detect_languages(query),
                    "confidence": 0.9,
                }
        except Exception as e:
            logger.warning(f"Query transformation failed, using original: {e}")
            return {
                "corrected_query": query,
                "query_variants": [query],
                "detected_visa_types": [],
                "languages_detected": ["de"],
                "confidence": 0.5,
            }

    async def _expand_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to expand query."""
        try:
            prompt = QUERY_TRANSFORMER_PROMPT_TEMPLATE.format(query=query)
            
            # <--- 修改點 2：使用標準化介面 call_non_streaming，確保相容 OpenAI 與 Ollama
            response_text = await self.llm.call_non_streaming(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.3,  # Low temp for deterministic expansion
                max_tokens=500,
            )
            
            # Parse JSON response safely
            import json
            response_text = response_text.strip()
            
            # Extract JSON if wrapped in markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            logger.debug(f"Query expansion result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Query expansion LLM call failed: {e}")
            raise

    def _detect_visa_types(self, query: str) -> List[str]:
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
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected.append(visa_type)
                    break
        
        return list(set(detected))

    def _detect_languages(self, query: str) -> List[str]:
        """Simple language detection."""
        languages = []
        
        # Chinese
        if any("\u4e00" <= char <= "\u9fff" for char in query):
            languages.append("zh")
        
        # German
        if any(char in query.lower() for char in ["ä", "ö", "ü", "ß"]):
            languages.append("de")
        elif re.search(r"\b(der|die|das|und|in|zu|mit|ein|eine|wie|ich)\b", query.lower()):
            languages.append("de")
        
        # English (default if no other detected)
        if not languages:
            languages.append("en")
        
        return languages

    async def get_search_queries(self, query: str) -> List[str]:
        """
        Get list of search queries (main + variants) for hybrid search.
        
        Quota Optimizations:
        1. Respect global ENABLE_QUERY_EXPANSION setting.
        2. Fast Mode: Skip full LLM expansion for very long or specialized queries.
        3. Strict capping: Max 2 queries total (original + 1 variant) to save embeddings quota.
        """
        if not settings.enable_query_expansion:
            return [query]

        # Use 100 chars as threshold for "specific enough"
        if len(query) > 100:
            logger.debug("Fast Mode: Skipping expansion for long query")
            return [query]

        try:
            # Short-circuit logic: if it's very short, we need expansion
            # but we cap the overhead
            transformed = await self.transform_query(query)
            
            search_queries = [transformed.get("corrected_query", query)]
            
            # Capping: Add only ONE variant (preferring English if available)
            if "english_query" in transformed and transformed["english_query"] != search_queries[0]:
                search_queries.append(transformed["english_query"])
            elif transformed.get("query_variants"):
                search_queries.append(transformed["query_variants"][0])
                
            # Final unique list, max size 2
            return list(set(filter(None, search_queries)))[:2]
            
        except Exception as e:
            logger.warning(f"Expansion failed, falling back to original: {e}")
            return [query]


# Singleton instance
_transformer = None

def get_query_transformer() -> QueryTransformer:
    """Get or create query transformer singleton."""
    global _transformer
    if _transformer is None:
        _transformer = QueryTransformer()
    return _transformer
