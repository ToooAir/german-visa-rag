"""
OpenAI-compatible chat models and request/response schemas.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Valid message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Chat message."""
    role: MessageRole
    content: str
    name: Optional[str] = None


class Choice(BaseModel):
    """Completion choice."""
    index: int = 0
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    """Chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ChatCompletionStream(BaseModel):
    """Streaming chat completion chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]  # index, delta, finish_reason
