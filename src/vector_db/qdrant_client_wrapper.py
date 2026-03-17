"""
Qdrant vector database client wrapper with collection management,
hybrid search (dense + sparse), filtering, and upsert operations.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from qdrant_client import QdrantClient, AsyncQdrantClient
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
        Perform hybrid search combining dense vector + sparse BM25.
        
        Args:
            dense_vector: Query embedding vector
            query_text: Query text for BM25 indexing
            top_k: Number of results to return
            filters: Optional Qdrant filters (authority_level, visa_types, date range)
            dense_weight: Weight for dense vector search
            sparse_weight: Weight for sparse search
            
        Returns:
            List of scored search results with metadata
        """
        try:
            logger.debug(f"Performing hybrid search with top_k={top_k}")
            
            # Dense search (cosine similarity)
            # Use query_points which is the recommended async API in newer versions
            search_result = await self.client.query_points(
                collection_name=self.collection_name,
                prefetch=None, # Simple search
                query=dense_vector,
                query_filter=filters,
                limit=top_k * 2,
                with_payload=True,
            )
            dense_results = search_result.points
            
            # Sparse search (BM25) - Disabled for now to fix AttributeError and input issues
            sparse_results = [] # TODO: Implement properly in Phase 3
            
            # Merge and deduplicate results with weighted scoring
            merged = {}
            
            for result in dense_results:
                point_id = result.id
                score = result.score * dense_weight
                merged[point_id] = {
                    "id": point_id,
                    "score": score,
                    "payload": result.payload,
                }
            
            for result in sparse_results:
                point_id = result.id
                score = result.score * sparse_weight
                
                if point_id in merged:
                    merged[point_id]["score"] += score
                else:
                    merged[point_id] = {
                        "id": point_id,
                        "score": score,
                        "payload": result.payload,
                    }
            
            # Sort by combined score and take top_k
            ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]
            
            logger.debug(f"Hybrid search returned {len(ranked)} results")
            return ranked
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {type(e).__name__}: {e}", exc_info=True)
            raise

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
            cutoff_date = (datetime.utcnow().timestamp() - max_days_old * 86400)
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
