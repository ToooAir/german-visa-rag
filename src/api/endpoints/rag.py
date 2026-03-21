from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.logger import logger
from src.api.auth import auth
from src.api.sse import create_sse_response
from src.rag.answer_generator import AnswerGenerator
from src.api.endpoints.dependencies import get_generator, get_qdrant
from src.vector_db.qdrant_client_wrapper import QdrantWrapper

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

class SourceInfo(BaseModel):
    """Source information for Knowledge Base."""
    title: str
    url: str
    authority_level: str
    last_fetched: Optional[str]
    visa_types: List[str]

@router.get("/sources", response_model=List[SourceInfo])
async def get_sources(
    qdrant: QdrantWrapper = Depends(get_qdrant),
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Get unique sources from knowledge base."""
    return await qdrant.get_unique_sources()

@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    generator: AnswerGenerator = Depends(get_generator),
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Ask a question about German visa regulations."""
    logger.info(f"Query: {request.query[:100]}, Language: {request.language}")
    result = await generator.generate_answer(request.query, language=request.language)
    return QueryResponse(**result)

@router.post("/ask/stream")
async def ask_question_stream(
    request: QueryRequest,
    generator: AnswerGenerator = Depends(get_generator),
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Ask a question with streaming response."""
    logger.info(f"Stream query: {request.query[:100]}, Language: {request.language}")
    stream = generator.generate_answer_streaming(request.query, language=request.language)
    return create_sse_response(stream)
