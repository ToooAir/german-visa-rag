"""Token counting and cost estimation utilities."""

from typing import Dict, Any, Optional
import tiktoken

from src.config import settings
from src.logger import logger


class TokenCounter:
    """Utility for token counting and cost estimation."""

    # Model costs (USD per 1K tokens, as of 2024-03)
    MODEL_COSTS = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(self):
        try:
            self.encoding = tiktoken.encoding_for_model(settings.openai_model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_messages(self, messages: list) -> int:
        """Count tokens in a message list."""
        total = 0
        for msg in messages:
            total += 4  # Message overhead
            for key, value in msg.items():
                if isinstance(value, str):
                    total += len(self.encoding.encode(value))
        return total

    def count_text(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return len(text) // 4  # Approximation

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None,
    ) -> float:
        """Estimate API cost in USD."""
        model = model or settings.openai_model
        costs = self.MODEL_COSTS.get(model, {"input": 0, "output": 0})
        
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return round(input_cost + output_cost, 6)

    def estimate_query_cost(
        self,
        query: str,
        response: str,
        model: Optional[str] = None,
    ) -> float:
        """Estimate cost for a query-response pair."""
        input_tokens = self.count_text(query)
        output_tokens = self.count_text(response)
        return self.estimate_cost(input_tokens, output_tokens, model)


# Singleton
_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get token counter instance."""
    global _counter
    if _counter is None:
        _counter = TokenCounter()
    return _counter
