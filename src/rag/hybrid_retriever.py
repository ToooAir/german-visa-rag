"""
Hybrid retrieval pipeline combining dense vector search with sparse BM25,
authority-based filtering, and recency weighting.
"""

from typing import Any, Optional
from datetime import datetime, timezone
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from qdrant_client import models
from src.config import settings
from src.logger import logger
from src.vector_db.qdrant_client_wrapper import get_qdrant_client, QdrantWrapper
from src.vector_db.embedder import embedder
from src.models.chunk import AuthorityLevel, VisaType


# Maps each minimum authority level to the set of accepted levels in Qdrant filter.
# Follows a hierarchical inclusion pattern: lower minimum → more levels accepted.
_AUTHORITY_HIERARCHY: dict[AuthorityLevel, list[str]] = {
    AuthorityLevel.OFFICIAL:      [AuthorityLevel.OFFICIAL.value],
    AuthorityLevel.SEMI_OFFICIAL: [AuthorityLevel.OFFICIAL.value, AuthorityLevel.SEMI_OFFICIAL.value],
    AuthorityLevel.THIRD_PARTY:   [AuthorityLevel.OFFICIAL.value, AuthorityLevel.SEMI_OFFICIAL.value, AuthorityLevel.THIRD_PARTY.value],
}


class HybridRetriever:
    """
    Hybrid retrieval using Qdrant with dense vector + sparse BM25 search.

    Pipeline:
    1. Embed query with text-embedding-3-small
    2. Perform hybrid search (dense + sparse BM25)
    3. Apply authority filtering and recency weighting
    4. Return top-k with enriched metadata
    """

    def __init__(self, qdrant_client: Optional[QdrantWrapper] = None):
        self.qdrant = qdrant_client or get_qdrant_client()
        self.top_k_hybrid  = settings.retrieval_top_k_hybrid
        self.top_k_reranked = settings.retrieval_top_k_reranked
        self.dense_weight  = settings.retrieval_dense_weight
        self.sparse_weight = settings.retrieval_sparse_weight

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def retrieve(
        self,
        query: str,
        visa_types: Optional[list[VisaType]] = None,
        min_authority: AuthorityLevel = AuthorityLevel.SEMI_OFFICIAL,
        top_k: Optional[int] = None,
        language: str = "de",  # Reserved for future language-specific optimizations
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search.

        Args:
            query:         User query string (can be multilingual).
            visa_types:    Filter by specific visa categories.
            min_authority: Minimum authority level; all higher levels are included.
            top_k:         Override default top-k.
            language:      Query language hint (reserved for future use).

        Returns:
            Ranked retrieval results with adjusted scores and metadata.
        """
        top_k = top_k or self.top_k_hybrid
        logger.debug("Starting hybrid retrieval for query: %.100s", query)

        try:
            query_embedding = await embedder.embed_single(query)
            if not query_embedding:
                logger.warning("Query embedding is empty, aborting retrieval")
                return []

            # ── Build Qdrant filter ──────────────────────────────────────────
            must_filters = []

            if visa_types:
                must_filters.append(
                    models.FieldCondition(
                        key="visa_types",
                        match=models.MatchAny(
                            any=[v.value if hasattr(v, "value") else v for v in visa_types]
                        ),
                    )
                )

            allowed_levels = _AUTHORITY_HIERARCHY.get(
                min_authority, [AuthorityLevel.OFFICIAL.value]
            )
            must_filters.append(
                models.FieldCondition(
                    key="authority_level",
                    match=models.MatchAny(any=allowed_levels),
                )
            )

            filters = models.Filter(must=must_filters)

            # ── Hybrid search ────────────────────────────────────────────────
            logger.debug("Performing hybrid search (top_k=%d)", top_k)
            results = await self.qdrant.hybrid_search(
                dense_vector=query_embedding,
                query_text=query,
                top_k=top_k,
                filters=filters,
                dense_weight=self.dense_weight,
                sparse_weight=self.sparse_weight,
            )

            # ── Enrich results with recency + authority scoring ──────────────
            now = datetime.now(timezone.utc)
            enriched_results = []

            for result in results:
                payload = result["payload"]

                # Recency penalty: normalised decay over configured window
                fetched_at_raw = payload.get("fetched_at")
                if fetched_at_raw:
                    try:
                        fetched_at = datetime.fromisoformat(fetched_at_raw)
                        if fetched_at.tzinfo is None:
                            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        fetched_at = now
                else:
                    fetched_at = now

                days_old = (now - fetched_at).days
                max_days = settings.rag_recency_penalty_days
                penalty_max = settings.rag_recency_penalty_max
                recency_penalty = 1.0 - (min(days_old, max_days) / float(max_days)) * penalty_max

                # Authority boost
                authority_level = payload.get("authority_level", "third_party")
                authority_boost = {
                    "official":      settings.rag_authority_boost_official,
                    "semi_official": settings.rag_authority_boost_semi,
                    "third_party":   settings.rag_authority_boost_third_party,
                }.get(authority_level, settings.rag_authority_boost_third_party)

                adjusted_score = result["score"] * recency_penalty * authority_boost

                enriched_results.append({
                    "id":             result["id"],
                    "original_score": result["score"],
                    "adjusted_score": adjusted_score,
                    "metadata": {
                        "chunk_id":      payload.get("chunk_id"),
                        "parent_doc_id": payload.get("parent_doc_id"),
                        "source_url":    payload.get("source_url"),
                        "source_title":  payload.get("source_title"),
                        "authority_level": authority_level,
                        "visa_types":    payload.get("visa_types", []),
                        "published_at":  payload.get("published_at"),
                        "fetched_at":    payload.get("fetched_at"),
                        "section_header": payload.get("section_header"),
                        "is_parent":     payload.get("is_parent", False),
                        "language":      payload.get("language", "de"),
                    },
                    "text": payload.get("text", ""),
                })

            enriched_results.sort(key=lambda x: x["adjusted_score"], reverse=True)

            logger.info(
                "Hybrid retrieval completed: %d results for query length %d",
                len(enriched_results),
                len(query),
                extra={"top_scores": [r["adjusted_score"] for r in enriched_results[:3]]},
            )
            return enriched_results

        except Exception as e:
            logger.error("Hybrid retrieval failed: %s", e, extra={"query": query[:100]})
            raise

    async def retrieve_batch(
        self,
        queries: list[str],
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """
        Batch retrieve for multiple queries concurrently.
        Failed individual queries are replaced with empty lists.
        """
        results = await asyncio.gather(
            *[self.retrieve(q, **kwargs) for q in queries],
            return_exceptions=True,
        )
        return [r if not isinstance(r, Exception) else [] for r in results]