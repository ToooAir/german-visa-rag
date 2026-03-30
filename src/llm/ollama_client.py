"""
Ollama local LLM client for fallback/offline scenarios.
"""

import json
from typing import AsyncIterator, Optional

import httpx

from src.logger import logger


class OllamaClient:
    """Local Ollama LLM client."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=120.0)

    async def call_non_streaming(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Ollama and return complete response."""
        try:
            logger.debug("Calling Ollama %s", self.model)

            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens or 512,
                    },
                },
            )

            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            raise

    async def call_streaming(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Call Ollama with streaming."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens or 512,
                    },
                },
            )

            response.raise_for_status()

            async for line in response.aiter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            logger.error("Ollama streaming failed: %s", e)
            raise

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
