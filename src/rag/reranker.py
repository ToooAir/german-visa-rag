"""
Reranking module using cross-encoder models to improve retrieval precision.
Supports mock, Cohere, and Jina reranker APIs.
"""

from typing import List, Dict, Any, Optional, Literal
from enum import Enum
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

from src.config import settings
from src.logger import logger


class RerankerType(str, Enum):
    """Supported reranker backends."""
    MOCK = "mock"
    COHERE = "cohere"
    JINA = "jina"


class Reranker:
    """Abstract base for rerankers."""

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: User query
            documents: List of document chunks with 'text' key
            top_k: Number of top results to return
            
        Returns:
            Top-k reranked documents with updated scores
        """
        raise NotImplementedError


class MockReranker(Reranker):
    """
    Mock reranker for development/testing.
    Returns documents in order without modification.
    """

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Mock reranking - returns top-k by original score."""
        logger.debug(f"MockReranker: reranking {len(documents)} docs, top_k={top_k}")
        
        # Simply return top-k by existing score
        return documents[:top_k]


class CohereReranker(Reranker):
    """
    Cohere Reranker API for semantic reranking.
    Requires COHERE_API_KEY and uses their /rerank endpoint.
    """

    def __init__(self, api_key: str, model: str = "rerank-english-v2.0"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.cohere.com/v1"
        self.client = httpx.AsyncClient(timeout=30.0)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Call Cohere API for reranking."""
        if not documents:
            return []

        try:
            # Extract texts for reranking
            texts = [doc.get("text", "") for doc in documents]
            
            logger.debug(
                f"Cohere reranker: reranking {len(documents)} docs with query: {query[:50]}"
            )
            
            response = await self.client.post(
                f"{self.base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": texts,
                    "top_n": top_k,
                    "return_documents": False,  # Only return scores
                },
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Map Cohere results back to documents
            reranked = []
            for result in data.get("results", []):
                idx = result["index"]
                score = result["relevance_score"]
                
                doc = documents[idx].copy()
                doc["rerank_score"] = score
                doc["adjusted_score"] = score  # Replace with rerank score
                reranked.append(doc)
            
            logger.info(f"Cohere reranker returned {len(reranked)} results")
            return reranked
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            # Fallback: return original documents sorted by original score
            return sorted(
                documents,
                key=lambda x: x.get("adjusted_score", 0),
                reverse=True,
            )[:top_k]

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class JinaReranker(Reranker):
    """
    Jina Reranker API for multilingual semantic reranking.
    Supports Chinese, German, English out of the box.
    """

    def __init__(self, api_key: str, model: str = "jina-reranker-v1-base-en"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.jina.ai/v1"
        self.client = httpx.AsyncClient(timeout=30.0)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Call Jina API for reranking."""
        if not documents:
            return []

        try:
            texts = [doc.get("text", "") for doc in documents]
            
            logger.debug(
                f"Jina reranker: reranking {len(documents)} docs with query: {query[:50]}"
            )
            
            response = await self.client.post(
                f"{self.base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": texts,
                    "top_n": top_k,
                },
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Map Jina results
            reranked = []
            for result in data.get("results", []):
                idx = result["index"]
                score = result["relevance_score"]
                
                doc = documents[idx].copy()
                doc["rerank_score"] = score
                doc["adjusted_score"] = score
                reranked.append(doc)
            
            logger.info(f"Jina reranker returned {len(reranked)} results")
            return reranked
            
        except Exception as e:
            logger.error(f"Jina reranking failed: {e}")
            return sorted(
                documents,
                key=lambda x: x.get("adjusted_score", 0),
                reverse=True,
            )[:top_k]

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class RerankerFactory:
    """Factory for creating reranker instances."""

    _instance: Optional[Reranker] = None

    @classmethod
    def get_reranker(cls) -> Reranker:
        """Get or create reranker singleton based on config."""
        if cls._instance is None:
            reranker_type = settings.reranker_api_type
            
            if reranker_type == RerankerType.COHERE.value:
                if not settings.reranker_api_key:
                    logger.warning("Cohere API key not found, using mock reranker")
                    cls._instance = MockReranker()
                else:
                    cls._instance = CohereReranker(
                        api_key=settings.reranker_api_key,
                        model=settings.reranker_model_name,
                    )
                    logger.info("Cohere reranker initialized")
                    
            elif reranker_type == RerankerType.JINA.value:
                if not settings.reranker_api_key:
                    logger.warning("Jina API key not found, using mock reranker")
                    cls._instance = MockReranker()
                else:
                    cls._instance = JinaReranker(
                        api_key=settings.reranker_api_key,
                        model=settings.reranker_model_name,
                    )
                    logger.info("Jina reranker initialized")
            else:
                logger.info("Using mock reranker")
                cls._instance = MockReranker()
        
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None


# Singleton getter
def get_reranker() -> Reranker:
    """Get reranker instance."""
    return RerankerFactory.get_reranker()


# Singleton instance for import
reranker = RerankerFactory.get_reranker()
