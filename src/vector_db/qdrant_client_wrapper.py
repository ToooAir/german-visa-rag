
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    PointStruct,
    VectorParams,
    Distance,
    HnswConfigDiff,
    CollectionStatus,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    HasIdCondition,
    SparseVector,
)

from src.config import settings
from src.logger import logger
from src.models.chunk import QdrantPayload, AuthorityLevel


class QdrantWrapper:
    """Async wrapper for Qdrant operations with connection pooling."""

    def __init__(self):
        """Initialize Qdrant client."""
        self.url = settings.qdrant_url
        self.api_key = settings.qdrant_api_key
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = settings.qdrant_vector_size
        
        # Async client (preferred)
        self.client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=settings.qdrant_prefer_grpc,
        )
        
        # Sync client fallback
        self.sync_client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=settings.qdrant_prefer_grpc,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
            
            # Create collection with hybrid search setup
            logger.info(f"Creating collection '{self.collection_name}'")
            
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
                # Enable sparse vectors for BM25
                sparse_vectors_config={
                    "text_sparse": {
                        "index": {
                            "on_disk": True,
                        }
                    }
                },
            )
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
            
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    async def upsert_points(
        self,
        points: List[PointStruct],
        wait: bool = True,
    ) -> None:
        """
        Upsert points into Qdrant collection.
        
        Args:
            points: List of PointStruct objects
            wait: Wait for operation to complete
        """
        try:
            if not points:
                logger.info("No points to upsert, skipping.")
                return

            logger.debug(f"Upserting {len(points)} points to Qdrant")
            
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=wait,
            )
            
            logger.info(f"Successfully upserted {len(points)} points")
            
        except Exception as e:
            logger.error(f"Upsert failed: {e}", extra={"points_count": len(points)})
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def hybrid_search(
        self,
        dense_vector: List[float],
        query_text: str,
        top_k: int = 5,
        filters: Optional[Filter] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense vector + sparse BM25 via Qdrant RRF fusion.

        Uses Qdrant's native `prefetch` + `FusionQuery(Fusion.RRF)` for server-side
        Reciprocal Rank Fusion (RRF). If sparse search is disabled or the sparse
        vector is empty, falls back to pure dense search.

        Args:
            dense_vector: Query embedding vector
            query_text: Query text for BM25 sparse encoding
            top_k: Number of results to return
            filters: Optional Qdrant filters (authority_level, visa_types, date range)
            dense_weight: Weight hint for dense search (used in logging; RRF handles fusion)
            sparse_weight: Weight hint for sparse search (used in logging)

        Returns:
            List of scored search results with metadata
        """
        try:
            logger.debug(f"Performing hybrid search with top_k={top_k}")

            # --- Build sparse vector from query text ---
            sparse_vec: Optional[SparseVector] = None
            if settings.enable_sparse_search:
                from src.vector_db.sparse_encoder import get_sparse_encoder
                encoder = get_sparse_encoder(vocab_size=settings.sparse_vocab_size)
                sparse_vec = encoder.encode(query_text)
                if not sparse_vec.indices:
                    sparse_vec = None  # Empty sparse vector — skip sparse leg

            # --- Build prefetch legs ---
            candidate_limit = top_k * 2
            prefetches = [
                models.Prefetch(
                    query=dense_vector,
                    using="",  # unnamed default dense vector
                    filter=filters,
                    limit=candidate_limit,
                ),
            ]
            if sparse_vec is not None:
                prefetches.append(
                    models.Prefetch(
                        query=sparse_vec,
                        using="text_sparse",
                        filter=filters,
                        limit=candidate_limit,
                    )
                )

            # --- Execute RRF hybrid query ---
            if len(prefetches) > 1:
                # True hybrid: RRF fusion of dense + sparse
                search_result = await self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetches,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    with_payload=True,
                    limit=top_k,
                )
                logger.debug(
                    f"Hybrid RRF search: dense + sparse legs, top_k={top_k}"
                )
            else:
                # Dense-only fallback (sparse disabled or empty vector)
                search_result = await self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetches,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    with_payload=True,
                    limit=top_k,
                )
                logger.debug(
                    f"Dense-only fallback search (sparse disabled or empty), top_k={top_k}"
                )

            # --- Format results ---
            ranked = [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload,
                }
                for point in search_result.points
            ]

            logger.debug(f"Hybrid search returned {len(ranked)} results")
            return ranked

        except Exception as e:
            logger.error(f"Hybrid search failed: {type(e).__name__}: {e}", exc_info=True)
            raise

    async def get_unique_sources(self) -> List[Dict[str, Any]]:
        """
        Retrieve unique sources (URLs and titles) from the collection.
        Since Qdrant doesn't have a direct 'distinct' query on payloads,
        we scroll through points and deduplicate in memory (fine for small/medium datasets).
        """
        try:
            unique_sources = {}  # key: source_url, value: metadata
            offset = None
            limit = 100  # Batch size for scrolling
            max_points = 2000  # Safety limit for UI exhibition
            points_scanned = 0

            while points_scanned < max_points:
                # Use scroll to iterate through all points
                points, next_offset = await self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                if not points:
                    break

                for point in points:
                    payload = point.payload
                    url = payload.get("source_url")
                    if url and url not in unique_sources:
                        unique_sources[url] = {
                            "title": payload.get("source_title", url),
                            "url": url,
                            "authority_level": payload.get("authority_level", "third_party"),
                            "last_fetched": payload.get("fetched_at"),
                            "visa_types": payload.get("visa_types", []),
                        }
                    
                points_scanned += len(points)
                offset = next_offset
                if not offset:
                    break

            # Convert to list and sort by authority then title
            result = list(unique_sources.values())
            # Simple sorting: official first
            authority_rank = {"official": 0, "semi_official": 1, "third_party": 2}
            result.sort(key=lambda x: (authority_rank.get(x["authority_level"], 3), x["title"]))
            
            logger.info(f"Retrieved {len(result)} unique sources from Qdrant")
            return result

        except Exception as e:
            logger.error(f"Failed to get unique sources: {e}")
            return []

    def build_filter_authority_and_visa(
        self,
        min_authority_level: AuthorityLevel = AuthorityLevel.SEMI_OFFICIAL,
        visa_types: Optional[List[str]] = None,
        max_days_old: int = 365,
    ) -> Filter:
        """
        Build composite filter for authority level, visa types, and recency.
        
        Args:
            min_authority_level: Minimum authority level (prioritize official)
            visa_types: List of relevant visa types to filter
            max_days_old: Only return documents fetched within this many days
            
        Returns:
            Qdrant Filter object
        """
        conditions = []
        
        # Authority level filter (prioritize official sources)
        authority_mapping = {
            AuthorityLevel.OFFICIAL: ["official"],
            AuthorityLevel.SEMI_OFFICIAL: ["official", "semi_official"],
            AuthorityLevel.THIRD_PARTY: ["official", "semi_official", "third_party"],
        }
        
        authority_values = authority_mapping.get(
            min_authority_level,
            ["official", "semi_official", "third_party"]
        )
        
        conditions.append(
            FieldCondition(
                key="authority_level",
                match=MatchValue(value=authority_values),
            )
        )
        
        # Visa type filter
        if visa_types:
            conditions.append(
                FieldCondition(
                    key="visa_types",
                    match=MatchValue(value=visa_types),
                )
            )
        
        # Recency filter (documents fetched within max_days_old)
        if max_days_old:
            cutoff_date = (datetime.now(timezone.utc).timestamp() - max_days_old * 86400)
            # Note: Qdrant doesn't have native datetime filtering in v0.x
            # This would need custom filtering logic
        
        # Combine conditions with OR logic
        if len(conditions) == 1:
            return conditions[0]
        else:
            return Filter(
                must=conditions if len(conditions) > 1 else None,
                # Use OR if multiple types
            )

    async def get_point_by_id(self, point_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single point by ID."""
        try:
            result = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
            )
            return result[0].model_dump() if result else None
        except Exception as e:
            logger.error(f"Failed to retrieve point {point_id}: {e}")
            return None

    async def delete_by_filter(self, filters: Filter) -> bool:
        """Delete points matching filter (for pruning old documents)."""
        try:
            result = await self.client.delete(
                collection_name=self.collection_name,
                points_selector=filters,
            )
            logger.info(f"Deleted points matching filter (Status: {result.status})")
            return result.status == "completed"
        except Exception as e:
            logger.error(f"Delete by filter failed: {e}")
            raise

    async def count_points(self) -> int:
        """Get total number of points in collection."""
        try:
            collection = await self.client.get_collection(
                collection_name=self.collection_name
            )
            return collection.points_count
        except Exception as e:
            logger.error(f"Failed to count points: {e}")
            return 0

    async def health_check(self) -> bool:
        """Check Qdrant service health."""
        try:
            _ = await self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def close(self):
        """Close client connections."""
        try:
            await self.client.close()
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")


# Singleton instance
qdrant_client = None


def get_qdrant_client() -> QdrantWrapper:
    """Get or create Qdrant client singleton."""
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantWrapper()
    return qdrant_client
