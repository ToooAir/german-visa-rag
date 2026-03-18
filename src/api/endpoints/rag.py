from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.logger import logger
from src.api.auth import auth
from src.api.sse import create_sse_response
from src.rag.answer_generator import AnswerGenerator
from src.api.endpoints.dependencies import get_generator

router = APIRouter(prefix="/query", tags=["rag"])

class QueryRequest(BaseModel):
    """RAG query request."""
    query: str = Field(..., description="User question about German visa/Chancenkarte")
    language: Optional[str] = Field(default="auto", description="Query language")

class QueryResponse(BaseModel):
    """RAG query response."""
    answer: str
    sources: List[Dict[str, str]]
    metadata: Dict[str, Any]

@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    generator: AnswerGenerator = Depends(get_generator),
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Ask a question about German visa regulations."""
    logger.info(f"Query: {request.query[:100]}")
    result = await generator.generate_answer(request.query)
    return QueryResponse(**result)

@router.post("/ask/stream")
async def ask_question_stream(
    request: QueryRequest,
    generator: AnswerGenerator = Depends(get_generator),
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Ask a question with streaming response."""
    logger.info(f"Stream query: {request.query[:100]}")
    stream = generator.generate_answer_streaming(request.query)
    return create_sse_response(stream)
