"""
Embedding service with OpenAI + Ollama fallback support.
Handles dense vector generation for RAG retrieval.
Includes quota guard (preflight check) and 429 circuit breaker.
"""

import asyncio
import re
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import logger


class QuotaExhaustedError(Exception):
    """Raised when embedding API quota is exhausted (HTTP 429)."""

    def __init__(
        self,
        wait_seconds: int = 0,
        message: str = "",
        reset_requests: Optional[str] = None,
        reset_tokens: Optional[str] = None,
    ):
        self.wait_seconds = wait_seconds
        self.reset_requests = reset_requests
        self.reset_tokens = reset_tokens

        details = []
        if wait_seconds:
            details.append(f"retry in {wait_seconds}s")
        if reset_requests:
            details.append(f"reset requests: {reset_requests}")
        if reset_tokens:
            details.append(f"reset tokens: {reset_tokens}")

        detail_str = f" [{', '.join(details)}]" if details else ""
        super().__init__(f"Embedding API quota exhausted.{detail_str} {message}".strip())


class EmbedderBase(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts."""
        pass

    @abstractmethod
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        pass


class OpenAIEmbedder(EmbedderBase):
    """OpenAI Embedding Service using text-embedding-3-small."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        is_azure: bool = False,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or settings.openai_api_base
        self.is_azure = is_azure
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.azure_deployment = azure_deployment
        self.client = None

    async def _get_client(self):
        """Lazy-initialize async OpenAI or AzureOpenAI client."""
        if self.client is None:
            if self.is_azure:
                from openai import AsyncAzureOpenAI

                self.client = AsyncAzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_api_version,
                    azure_deployment=self.azure_deployment,
                )
            else:
                from openai import AsyncOpenAI

                self.client = AsyncOpenAI(
                    api_key=self.api_key, base_url=self.base_url, timeout=settings.api_timeout_seconds
                )
        return self.client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts with OpenAI API with retry logic and automatic batching.
        OpenAI has a total token limit per request (8192), so we split large lists.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors (1536-dimensional)
        """
        if not texts:
            return []

        # Batch size of 10 is extremely conservative.
        # Even with long context prefixes and dense text, this ensures we stay under 8192.
        BATCH_SIZE = 10
        all_embeddings = []

        client = await self._get_client()

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_chars = sum(len(t) for t in batch)
            logger.debug(f"OpenAI embedding batch {i//BATCH_SIZE}: {len(batch)} items, ~{batch_chars} chars")

            try:
                response = await client.embeddings.create(
                    input=batch,
                    model=self.model,
                )

                # Sort by index to maintain order within batch
                batch_embeddings = sorted(response.data, key=lambda x: x.index)
                all_embeddings.extend([emb.embedding for emb in batch_embeddings])

            except Exception as e:
                # Detect rate limit errors
                import openai

                if isinstance(e, openai.RateLimitError):
                    headers = getattr(e, "response", None).headers if hasattr(e, "response") else {}
                    raise QuotaExhaustedError(
                        wait_seconds=self._parse_wait_seconds(str(e)),
                        reset_requests=headers.get("x-ratelimit-reset-requests"),
                        reset_tokens=headers.get("x-ratelimit-reset-tokens"),
                        message=f"Provider: {'Azure' if self.is_azure else 'OpenAI'}",
                    ) from e

                logger.error(f"OpenAI batch embedding failed (batch {i//BATCH_SIZE}): {e}")
                raise

        return all_embeddings

    @staticmethod
    def _parse_wait_seconds(error_message: str) -> int:
        """Extract wait time from 429 error messages."""
        match = re.search(r"wait\s+(\d+)\s*seconds?", error_message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0

    async def embed_single(self, text: str) -> List[float]:
        """Embed single text."""
        result = await self.embed_texts([text])
        return result[0] if result else []


class OllamaEmbedder(EmbedderBase):
    """Local Ollama Embedding Service (fallback)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using local Ollama API."""
        if not texts:
            return []

        embeddings = []
        for text in texts:
            try:
                response = await self.client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data.get("embedding", []))

            except Exception as e:
                logger.error(f"Ollama embedding failed for text: {e}")
                raise

        return embeddings

    async def embed_single(self, text: str) -> List[float]:
        """Embed single text."""
        result = await self.embed_texts([text])
        return result[0] if result else []

    async def close(self):
        """Close async client."""
        await self.client.aclose()


class EmbedderFactory:
    """Factory for creating embedder instances."""

    _instance: Optional["Embedder"] = None

    @classmethod
    def get_embedder(cls) -> "Embedder":
        """Get or create singleton embedder instance."""
        if cls._instance is None:
            cls._instance = Embedder()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton instance (for testing)."""
        cls._instance = None


class Embedder:
    """Unified embedder interface with automatic fallback."""

    def __init__(self):
        if settings.use_azure_openai:
            self.primary = OpenAIEmbedder(
                api_key=settings.azure_openai_api_key,
                model=settings.embedding_model,
                is_azure=True,
                azure_endpoint=settings.azure_openai_endpoint,
                azure_api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_embedding_deployment,
            )
            logger.info(f"Azure OpenAI embedder initialized (deployment: {settings.azure_embedding_deployment})")
        else:
            self.primary = OpenAIEmbedder(
                api_key=settings.openai_api_key,
                model=settings.embedding_model,
                base_url=settings.openai_api_base,
            )
            logger.info("Standard OpenAI embedder initialized")

        self.fallback = None
        if settings.use_ollama:
            self.fallback = OllamaEmbedder(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )
            logger.info("Ollama embedder initialized as fallback")

    async def preflight_check(self) -> bool:
        """
        Pre-flight check: verify embedding API is reachable and has quota.
        Sends a single-word embedding request before committing to full ingestion.

        Returns:
            True if API is available, False otherwise

        Raises:
            QuotaExhaustedError: If the API returns 429
        """
        try:
            logger.info("Running embedding API preflight check...")
            await self.primary.embed_texts(["preflight"])
            logger.info("✅ Embedding API preflight check passed")
            return True
        except QuotaExhaustedError:
            # Re-raise quota errors — caller should handle these
            raise
        except Exception as e:
            logger.error(f"❌ Embedding API preflight check failed: {e}")
            return False

    async def embed_texts(
        self,
        texts: List[str],
        _quota_retry: int = 0,
    ) -> List[List[float]]:
        """
        Embed texts with fallback and quota-aware retry support.

        On QuotaExhaustedError (HTTP 429):
        - Waits for the reset window returned by the API (or defaults to 60s)
        - Retries up to MAX_QUOTA_RETRIES times before raising
        """
        MAX_QUOTA_RETRIES = 3
        DEFAULT_WAIT_SECONDS = 60

        if not texts:
            return []

        try:
            logger.debug(f"Embedding {len(texts)} texts with primary embedder")
            return await self.primary.embed_texts(texts)

        except QuotaExhaustedError as qe:
            if _quota_retry >= MAX_QUOTA_RETRIES:
                logger.error(f"⛔ Quota exhausted after {MAX_QUOTA_RETRIES} retries. Giving up.")
                raise

            wait_secs = qe.wait_seconds or DEFAULT_WAIT_SECONDS
            logger.warning(
                f"⏳ Quota exhausted (retry {_quota_retry + 1}/{MAX_QUOTA_RETRIES}). "
                f"Waiting {wait_secs}s before retrying... "
                f"[reset tokens: {qe.reset_tokens}, reset requests: {qe.reset_requests}]"
            )
            await asyncio.sleep(wait_secs)
            return await self.embed_texts(texts, _quota_retry=_quota_retry + 1)

        except Exception as e:
            logger.warning(f"Primary embedder failed: {e}")

            if self.fallback:
                try:
                    logger.info("Falling back to Ollama embedder")
                    return await self.fallback.embed_texts(texts)
                except Exception as fallback_error:
                    logger.error(f"Fallback embedder also failed: {fallback_error}")
                    raise
            else:
                raise

    async def embed_single(self, text: str) -> List[float]:
        """Embed single text."""
        result = await self.embed_texts([text])
        return result[0] if result else []


# Singleton instance
embedder = EmbedderFactory.get_embedder()
