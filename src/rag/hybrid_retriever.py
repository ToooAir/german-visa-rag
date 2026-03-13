"""
Hybrid retrieval pipeline combining dense vector search with sparse BM25,
authority-based filtering, and recency weighting.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import logger
from src.vector_db.qdrant_client_wrapper import get_qdrant_client, QdrantWrapper
from src.vector_db.embedder import embedder
from src.models.chunk import AuthorityLevel, VisaType


class HybridRetriever:
    """
    Hybrid retrieval using Qdrant with vector + sparse search.
    
    Pipeline:
    1. Embed query with text-embedding-3-small
    2. Perform hybrid search (dense + sparse BM25)
    3. Apply authority filtering and recency weighting
    4. Return top-k with metadata
    """

    def __init__(
        self,
        qdrant_client: Optional[QdrantWrapper] = None,
    ):
        self.qdrant = qdrant_client or get_qdrant_client()
        
        # Retrieval parameters
        self.top_k_hybrid = settings.retrieval_top_k_hybrid
        self.top_k_reranked = settings.retrieval_top_k_reranked
        self.dense_weight = settings.retrieval_dense_weight
        self.sparse_weight = settings.retrieval_sparse_weight

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def retrieve(
        self,
        query: str,
        visa_types: Optional[List[VisaType]] = None,
        min_authority: AuthorityLevel = AuthorityLevel.SEMI_OFFICIAL,
        top_k: Optional[int] = None,
        language: str = "de",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: User query string (can be multilingual)
            visa_types: Filter by specific visa categories
            min_authority: Minimum authority level filter
            top_k: Override default top-k
            language: Query language (for potential future optimization)
            
        Returns:
            List of ranked retrieval results with scores and metadata
        """
        top_k = top_k or self.top_k_hybrid
        
        logger.debug(f"Starting hybrid retrieval for query: {query[:100]}")
        
        try:
            # Step 1: Embed query
            logger.debug("Embedding query")
            query_embedding = await embedder.embed_single(query)
            
            if not query_embedding:
                logger.warning("Query embedding is empty")
                return []
            
            # Step 2: Build filter
            # Note: Simplified filter for now
            filters = None  # TODO: Implement authority + visa type filtering in Phase 3
            
            # Step 3: Perform hybrid search
            logger.debug(f"Performing hybrid search with top_k={top_k}")
            results = await self.qdrant.hybrid_search(
                dense_vector=query_embedding,
                query_text=query,
                top_k=top_k,
                filters=filters,
                dense_weight=self.dense_weight,
                sparse_weight=self.sparse_weight,
            )
            
            # Step 4: Enrich with computed metadata and recency weighting
            enriched_results = []
            now = datetime.utcnow()
            
            for result in results:
                payload = result["payload"]
                
                # Recency weighting: prefer more recent documents
                fetched_at = datetime.fromisoformat(payload.get("fetched_at", now.isoformat()))
                days_old = (now - fetched_at).days
                recency_penalty = 1.0 - (min(days_old, 365) / 365.0) * 0.1  # Max 10% penalty
                
                # Authority boost: prefer official sources
                authority_level = payload.get("authority_level", "third_party")
                authority_boost = {
                    "official": 1.2,
                    "semi_official": 1.0,
                    "third_party": 0.8,
                }.get(authority_level, 0.8)
                
                adjusted_score = result["score"] * recency_penalty * authority_boost
                
                enriched_results.append({
                    "id": result["id"],
                    "original_score": result["score"],
                    "adjusted_score": adjusted_score,
                    "metadata": {
                        "chunk_id": payload.get("chunk_id"),
                        "parent_doc_id": payload.get("parent_doc_id"),
                        "source_url": payload.get("source_url"),
                        "source_title": payload.get("source_title"),
                        "authority_level": authority_level,
                        "visa_types": payload.get("visa_types", []),
                        "published_at": payload.get("published_at"),
                        "fetched_at": payload.get("fetched_at"),
                        "section_header": payload.get("section_header"),
                        "is_parent": payload.get("is_parent", False),
                        "language": payload.get("language", "de"),
                    },
                    "text": payload.get("text", ""),
                })
            
            # Re-sort by adjusted score
            enriched_results = sorted(
                enriched_results,
                key=lambda x: x["adjusted_score"],
                reverse=True,
            )
            
            logger.info(
                f"Hybrid retrieval completed",
                extra={
                    "query_length": len(query),
                    "results_count": len(enriched_results),
                    "top_scores": [r["adjusted_score"] for r in enriched_results[:3]],
                },
            )
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}", extra={"query": query[:100]})
            raise

    async def retrieve_batch(
        self,
        queries: List[str],
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """Batch retrieve for multiple queries."""
        results = await asyncio.gather(
            *[self.retrieve(q, **kwargs) for q in queries],
            return_exceptions=True,
        )
        
        # Handle exceptions
        return [
            r if not isinstance(r, Exception) else []
            for r in results
        ]


# Singleton instance
retriever = None


def get_retriever(qdrant_client: Optional[QdrantWrapper] = None) -> HybridRetriever:
    """Get or create retriever singleton."""
    global retriever
    if retriever is None:
        retriever = HybridRetriever(qdrant_client)
    return retriever


# Import asyncio at module level for batch operations
import asyncio
