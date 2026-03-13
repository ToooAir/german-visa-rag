"""
OpenAI API client wrapper with retry logic, token counting, and cost tracking.
Supports both streaming and non-streaming responses.
"""

from typing import Optional, List, Dict, Any, AsyncIterator
import asyncio
from datetime import datetime

from openai import AsyncOpenAI, AsyncAzureOpenAI, OpenAIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    AsyncRetrying,
)
import tiktoken

from src.config import settings
from src.logger import logger


class OpenAIClient:
    """Wrapper for OpenAI API with resilience and observability."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or settings.openai_api_base
        
        if settings.use_azure_openai:
            self.client = AsyncAzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_llm_deployment,
            )
            logger.info(f"Azure OpenAI client initialized (deployment: {settings.azure_llm_deployment})")
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url,
            )
            logger.info("Standard OpenAI client initialized")
        
        # Token counting
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for custom models
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Cost tracking (USD per 1K tokens)
        self.costs = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using approximation")
            return len(text) // 4  # Rough approximation

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate API cost in USD."""
        model_costs = self.costs.get(self.model, {"input": 0, "output": 0})
        
        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]
        
        return input_cost + output_cost

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(OpenAIError),
    )
    async def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        stream: bool = False,
    ) -> Any:
        """
        Call OpenAI API with retry logic.
        """
        try:
            logger.debug(
                f"Calling {self.model}",
                extra={
                    "messages": len(messages),
                    "temperature": temperature,
                    "stream": stream,
                }
            )
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
            )
            
            return response
            
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}", extra={"model": self.model})
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}")
            raise

    async def call_non_streaming(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call OpenAI and return complete response text."""
        response = await self.call(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        
        return response.choices[0].message.content

    async def call_streaming(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Call OpenAI with streaming and yield text chunks.
        """
        response = await self.call(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
