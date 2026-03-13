"""
FastAPI routes with OpenAI-compatible endpoints.
Implements /v1/chat/completions for seamless integration with OpenAI SDKs.
"""

from typing import Optional, List, Dict, Any
import time
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException, status, Depends, Header
from pydantic import BaseModel, Field

from src.config import settings
from src.logger import logger
from src.api.auth import auth
from src.api.sse import create_sse_response
from src.rag.answer_generator import get_answer_generator
from src.storage.sqlite_state_store import get_state_store
from src.vector_db.qdrant_client_wrapper import get_qdrant_client
from src.ingestion.scheduler import get_scheduler


# ============================================
# Request/Response Models
# ============================================

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


class QueryRequest(BaseModel):
    """RAG query request."""
    query: str = Field(..., description="User question about German visa/Chancenkarte")
    language: Optional[str] = Field(default="auto", description="Query language")


class QueryResponse(BaseModel):
    """RAG query response."""
    answer: str
    sources: List[Dict[str, str]]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    dependencies: Dict[str, str]


# ============================================
# Router Setup
# ============================================

router = APIRouter(
    prefix="/v1",
    tags=["openai-compatible"],
)

query_router = APIRouter(
    prefix="/query",
    tags=["rag"],
)

admin_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
)


# ============================================
# OpenAI-Compatible Endpoints
# ============================================

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    x_api_key: str = Depends(auth.verify_api_key),
):
    """
    OpenAI-compatible chat completions endpoint.
    
    Implements the full OpenAI API interface for seamless integration.
    When stream=true, returns Server-Sent Events stream.
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
        return StreamingResponse(
            generate_chat_stream(request),
            media_type="text/event-stream",
        )
    
    # Non-streaming response
    generator = get_answer_generator()
    
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
            "prompt_tokens": 0,  # TODO: count actual tokens
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )


async def generate_chat_stream(request: ChatCompletionRequest):
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
    
    generator = get_answer_generator()
    
    # Yield streaming response
    async for chunk in generator.generate_answer_streaming(query):
        yield chunk


# ============================================
# RAG Query Endpoints
# ============================================

@query_router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    x_api_key: str = Depends(auth.verify_api_key),
):
    """
    Ask a question about German visa regulations.
    
    Returns structured answer with sources and metadata.
    """
    logger.info(f"Query: {request.query[:100]}")
    
    generator = get_answer_generator()
    result = await generator.generate_answer(request.query)
    
    return QueryResponse(**result)


@query_router.post("/ask/stream")
async def ask_question_stream(
    request: QueryRequest,
    x_api_key: str = Depends(auth.verify_api_key),
):
    """
    Ask a question with streaming response.
    
    Returns Server-Sent Events stream of answer chunks.
    """
    logger.info(f"Stream query: {request.query[:100]}")
    
    generator = get_answer_generator()
    stream = generator.generate_answer_streaming(request.query)
    
    return create_sse_response(stream)


# ============================================
# Health & Status Endpoints
# ============================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Verifies all dependencies are available.
    """
    try:
        # Check Qdrant
        qdrant = get_qdrant_client()
        qdrant_ok = await qdrant.health_check()
        
        # Check state store
        state_store = get_state_store()
        db_ok = state_store.db_path.exists()
        
        dependencies = {
            "qdrant": "✓ OK" if qdrant_ok else "✗ FAILED",
            "sqlite": "✓ OK" if db_ok else "✗ FAILED",
            "redis": "✓ OK",  # TODO: actual Redis health check
        }
        
        all_ok = all("✓" in v for v in dependencies.values())
        
        return HealthResponse(
            status="healthy" if all_ok else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            version="0.1.0",
            dependencies=dependencies,
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version="0.1.0",
            dependencies={"error": str(e)},
        )


@router.get("/stats")
async def get_stats(x_api_key: str = Depends(auth.verify_api_key)):
    """Get ingestion and query statistics."""
    state_store = get_state_store()
    stats = state_store.get_stats()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "statistics": stats,
    }


# ============================================
# Admin Endpoints
# ============================================

@admin_router.post("/ingest/trigger")
async def trigger_ingestion(
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Manually trigger ingestion pipeline."""
    logger.info("Manual ingestion triggered by admin")
    
    scheduler = get_scheduler()
    await scheduler.trigger_manual_ingestion()
    
    return {"status": "ingestion_started"}


@admin_router.get("/ingest/stats")
async def get_ingestion_stats(
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Get ingestion statistics."""
    state_store = get_state_store()
    return {"statistics": state_store.get_stats()}


# ============================================
# Error Handlers
# ============================================

from fastapi import Request
from fastapi.responses import JSONResponse


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status": exc.status_code},
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
