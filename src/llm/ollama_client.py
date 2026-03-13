"""
Ollama local LLM client for fallback/offline scenarios.
"""

from typing import Optional, List, Dict, Any, AsyncIterator
import httpx

from src.config import settings
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
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Ollama and return complete response."""
        try:
            logger.debug(f"Calling Ollama {self.model}")
            
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
            logger.error(f"Ollama call failed: {e}")
            raise

    async def call_streaming(
        self,
        messages: List[Dict[str, str]],
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
                    import json
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
