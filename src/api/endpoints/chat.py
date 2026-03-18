from typing import Optional, List, Dict, Any
import time
import uuid

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field

from src.logger import logger
from src.api.auth import auth
from src.rag.answer_generator import AnswerGenerator
from src.api.endpoints.dependencies import get_generator

router = APIRouter(prefix="/v1/chat", tags=["openai-compatible"])

class ChatMessage(BaseModel):
    """Chat message following OpenAI format."""
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(default="gpt-4o-mini", description="Model name")
    messages: List[ChatMessage] = Field(..., description="Message history")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, description="Max output tokens")
    stream: bool = Field(default=False, description="Enable streaming")

class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    generator: AnswerGenerator = Depends(get_generator),
    x_api_key: str = Depends(auth.verify_api_key),
):
    """
    OpenAI-compatible chat completions endpoint.
    """
    logger.info(
        "Chat completion request",
        extra={
            "model": request.model,
            "messages": len(request.messages),
            "stream": request.stream,
        }
    )
    
    if request.stream:
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            generate_chat_stream(request, generator),
            media_type="text/event-stream",
        )
    
    from src.llm.token_counter import get_token_counter
    counter = get_token_counter()
    messages_dicts = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    prompt_tokens = counter.count_messages(messages_dicts)
    
    # Extract the last user message as query
    query = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            query = msg.content
            break
    
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No user message found",
        )
    
    result = await generator.generate_answer(query)
    
    completion_tokens = counter.count_text(result["answer"])
    
    return ChatCompletionResponse(
        id=str(uuid.uuid4()),
        created=int(time.time()),
        model=request.model,
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": result["answer"],
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )

async def generate_chat_stream(request: ChatCompletionRequest, generator: AnswerGenerator):
    """Generate streaming chat completion."""
    # Extract user query
    query = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            query = msg.content
            break
    
    if not query:
        yield f"data: {{'error': 'No user message'}}\n\n"
        return
    
    # Yield streaming response
    async for chunk in generator.generate_answer_streaming(query):
        yield chunk
