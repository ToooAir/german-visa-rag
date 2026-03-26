"""
Reranking module using cross-encoder models to improve retrieval precision.
Supports mock, Cohere, and Jina reranker APIs.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import logger


class RerankerType(str, Enum):
    """Supported reranker backends."""

    MOCK = "mock"
    COHERE = "cohere"
    JINA = "jina"


class Reranker(ABC):
    """Abstract base for reranker backends."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: User query string.
            documents: Document chunks; must contain a 'text' or 'content' key.
            top_k: Maximum number of results to return.

        Returns:
            Top-k reranked documents with 'rerank_score' and 'adjusted_score' fields.
        """


class MockReranker(Reranker):
    """
    Mock reranker for development/testing.
    Returns documents in original order without modification.
    """

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        logger.debug("MockReranker: returning top-%d of %d docs", top_k, len(documents))
        return documents[:top_k]


class CohereReranker(Reranker):
    """
    Cohere Reranker API for semantic reranking.
    Requires COHERE_API_KEY. Uses /v1/rerank endpoint.
    """

    def __init__(self, api_key: str, model: str = "rerank-english-v2.0"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.cohere.com/v1"
        self.client = httpx.AsyncClient(timeout=30.0)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def _call_api(self, query: str, texts: list[str], top_k: int) -> list[dict[str, Any]]:
        """Internal HTTP call — retry lives here, not in rerank()."""
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
                "return_documents": False,
            },
        )
        response.raise_for_status()
        return response.json().get("results", [])

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        try:
            # Support both 'text' (old-style) and 'content' (new-style) keys
            texts = [(doc.get("text") or doc.get("content", "")) for doc in documents]
            logger.debug("Cohere reranker: %d docs, query=%.50s", len(documents), query)

            results = await self._call_api(query, texts, top_k)

            reranked = []
            for result in results:
                doc = documents[result["index"]].copy()
                doc["rerank_score"] = result["relevance_score"]
                doc["adjusted_score"] = result["relevance_score"]
                reranked.append(doc)

            logger.info("Cohere reranker returned %d results", len(reranked))
            return reranked

        except Exception as e:
            logger.error("Cohere reranking failed, falling back to score sort: %s", e)
            return sorted(
                documents,
                key=lambda x: x.get("adjusted_score", 0),
                reverse=True,
            )[:top_k]

    async def close(self) -> None:
        """Close the underlying HTTP client. Call on application shutdown."""
        await self.client.aclose()


class JinaReranker(Reranker):
    """
    Jina Reranker API for multilingual semantic reranking.
    Supports Chinese, German, and English out of the box.
    """

    def __init__(self, api_key: str, model: str = "jina-reranker-v1-base-en"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.jina.ai/v1"
        self.client = httpx.AsyncClient(timeout=30.0)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def _call_api(self, query: str, texts: list[str], top_k: int) -> list[dict[str, Any]]:
        """Internal HTTP call — retry lives here, not in rerank()."""
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
        return response.json().get("results", [])

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        try:
            texts = [(doc.get("text") or doc.get("content", "")) for doc in documents]
            logger.debug("Jina reranker: %d docs, query=%.50s", len(documents), query)

            results = await self._call_api(query, texts, top_k)

            reranked = []
            for result in results:
                doc = documents[result["index"]].copy()
                doc["rerank_score"] = result["relevance_score"]
                doc["adjusted_score"] = result["relevance_score"]
                reranked.append(doc)

            logger.info("Jina reranker returned %d results", len(reranked))
            return reranked

        except Exception as e:
            logger.error("Jina reranking failed, falling back to score sort: %s", e)
            return sorted(
                documents,
                key=lambda x: x.get("adjusted_score", 0),
                reverse=True,
            )[:top_k]

    async def close(self) -> None:
        """Close the underlying HTTP client. Call on application shutdown."""
        await self.client.aclose()


class RerankerFactory:
    """Factory for creating and caching the reranker singleton."""

    _instance: Optional[Reranker] = None

    @classmethod
    def get_reranker(cls) -> Reranker:
        """Lazily create and return the reranker singleton based on config."""
        if cls._instance is not None:
            return cls._instance

        reranker_type = settings.reranker_api_type

        if reranker_type == RerankerType.COHERE.value:
            if not settings.reranker_api_key:
                logger.warning("Cohere API key not configured, falling back to MockReranker")
                cls._instance = MockReranker()
            else:
                cls._instance = CohereReranker(
                    api_key=settings.reranker_api_key,
                    model=settings.reranker_model_name,
                )
                logger.info("CohereReranker initialized (model=%s)", settings.reranker_model_name)

        elif reranker_type == RerankerType.JINA.value:
            if not settings.reranker_api_key:
                logger.warning("Jina API key not configured, falling back to MockReranker")
                cls._instance = MockReranker()
            else:
                cls._instance = JinaReranker(
                    api_key=settings.reranker_api_key,
                    model=settings.reranker_model_name,
                )
                logger.info("JinaReranker initialized (model=%s)", settings.reranker_model_name)

        else:
            logger.info("No reranker configured, using MockReranker")
            cls._instance = MockReranker()

        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton. Intended for testing only."""
        cls._instance = None


def get_reranker() -> Reranker:
    """Return the application-wide reranker singleton."""
    return RerankerFactory.get_reranker()
